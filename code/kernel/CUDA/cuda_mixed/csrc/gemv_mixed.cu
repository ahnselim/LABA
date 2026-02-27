#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <cstdint>
#include <torch/extension.h>
#include <cuda_fp16.h>

namespace {

constexpr int kThreads = 256;
constexpr int kTileOC = 16;

__device__ inline int min_int(int a, int b) {
  return a < b ? a : b;
}

__global__ void gemv_asym_kernel(
    const half* __restrict__ inputs,
    const uint8_t* __restrict__ qweight,
    const half* __restrict__ scales,
    const uint8_t* __restrict__ qzeros,
    float* __restrict__ outputs,
    int M,
    int IC,
    int OC,
    int groups,
    int group_size,
    int packed_cols,
    int bit_width) {
  int row = blockIdx.y;
  int oc_base = blockIdx.x * kTileOC;
  if (row >= M || oc_base >= OC) {
    return;
  }

  int tile_cols = min_int(kTileOC, OC - oc_base);
  const uint8_t* weight_rows[kTileOC];
  const half* scale_rows[kTileOC];
  const uint8_t* zero_rows[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    if (t < tile_cols) {
      int oc = oc_base + t;
      weight_rows[t] = qweight + oc * packed_cols;
      scale_rows[t] = scales + oc * groups;
      zero_rows[t] = qzeros + oc * groups;
    } else {
      weight_rows[t] = nullptr;
      scale_rows[t] = nullptr;
      zero_rows[t] = nullptr;
    }
  }

  float acc[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    acc[t] = 0.0f;
  }

  const uint8_t code_mask = (bit_width == 3) ? 0x7 : 0xF;

  for (int pack = threadIdx.x; pack < packed_cols; pack += blockDim.x) {
    int k0 = pack * 2;
    int k1 = k0 + 1;
    bool valid0 = k0 < IC;
    bool valid1 = k1 < IC;
    if (!valid0 && !valid1) {
      continue;
    }

    float in0 = valid0 ? __half2float(inputs[row * IC + k0]) : 0.0f;
    float in1 = valid1 ? __half2float(inputs[row * IC + k1]) : 0.0f;

    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      if (t >= tile_cols) {
        break;
      }
      uint8_t byte_val = weight_rows[t][pack];
      if (valid0) {
        uint8_t code0 = byte_val & 0x0F;
        code0 &= code_mask;
        int gid0 = k0 / group_size;
        if (gid0 < groups) {
          float s = __half2float(scale_rows[t][gid0]);
          float z = static_cast<float>(zero_rows[t][gid0]);
          float real = (static_cast<float>(code0) - z) * s;
          acc[t] += real * in0;
        }
      }
      if (valid1) {
        uint8_t code1 = (byte_val >> 4) & 0x0F;
        code1 &= code_mask;
        int gid1 = k1 / group_size;
        if (gid1 < groups) {
          float s = __half2float(scale_rows[t][gid1]);
          float z = static_cast<float>(zero_rows[t][gid1]);
          float real = (static_cast<float>(code1) - z) * s;
          acc[t] += real * in1;
        }
      }
    }
  }

  __shared__ float shm[kThreads][kTileOC];
  int tid = threadIdx.x;
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    shm[tid][t] = acc[t];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int t = 0; t < kTileOC; ++t) {
        shm[tid][t] += shm[tid + stride][t];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      int oc = oc_base + t;
      if (t < tile_cols && oc < OC) {
        outputs[row * OC + oc] = shm[0][t];
      }
    }
  }
}

__global__ void gemv_w2_qtr_kernel(
    const half* __restrict__ inputs,
    const uint8_t* __restrict__ qweight,
    const half* __restrict__ scales,
    float* __restrict__ outputs,
    int M,
    int IC,
    int OC,
    int groups,
    int group_size,
    int packed_cols) {
  int row = blockIdx.y;
  int oc_base = blockIdx.x * kTileOC;
  if (row >= M || oc_base >= OC) {
    return;
  }

  int tile_cols = min_int(kTileOC, OC - oc_base);
  const uint8_t* weight_rows[kTileOC];
  const half* scale_rows[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    if (t < tile_cols) {
      int oc = oc_base + t;
      weight_rows[t] = qweight + oc * packed_cols;
      scale_rows[t] = scales + oc * groups;
    } else {
      weight_rows[t] = nullptr;
      scale_rows[t] = nullptr;
    }
  }

  float acc[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    acc[t] = 0.0f;
  }

  for (int pack = threadIdx.x; pack < packed_cols; pack += blockDim.x) {
    int k_base = pack * 4;
    float in_vals[4];
    bool valid[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int k = k_base + i;
      valid[i] = k < IC;
      in_vals[i] = valid[i] ? __half2float(inputs[row * IC + k]) : 0.0f;
    }
    if (!valid[0] && !valid[1] && !valid[2] && !valid[3]) {
      continue;
    }

    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      if (t >= tile_cols) {
        break;
      }
      uint8_t byte_val = weight_rows[t][pack];
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        if (!valid[lane]) {
          continue;
        }
        uint8_t code = (byte_val >> (lane * 2)) & 0x3;
        int gid = (k_base + lane) / group_size;
        if (gid >= groups) {
          continue;
        }
        float alpha = __half2float(scale_rows[t][gid]);
        float mag = (code & 0x1) ? alpha : 0.0f;
        float sign = (code & 0x2) ? -1.0f : 1.0f;
        acc[t] += sign * mag * in_vals[lane];
      }
    }
  }

  __shared__ float shm[kThreads][kTileOC];
  int tid = threadIdx.x;
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    shm[tid][t] = acc[t];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int t = 0; t < kTileOC; ++t) {
        shm[tid][t] += shm[tid + stride][t];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      int oc = oc_base + t;
      if (t < tile_cols && oc < OC) {
        outputs[row * OC + oc] = shm[0][t];
      }
    }
  }
}

__global__ void gemv_w2_lloyd_kernel(
    const half* __restrict__ inputs,
    const uint8_t* __restrict__ qweight,
    const half* __restrict__ scales,
    float* __restrict__ outputs,
    int M,
    int IC,
    int OC,
    int groups,
    int group_size,
    int packed_cols) {
  int row = blockIdx.y;
  int oc_base = blockIdx.x * kTileOC;
  if (row >= M || oc_base >= OC) {
    return;
  }

  int tile_cols = min_int(kTileOC, OC - oc_base);
  const uint8_t* weight_rows[kTileOC];
  const half* scale_rows[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    if (t < tile_cols) {
      int oc = oc_base + t;
      weight_rows[t] = qweight + oc * packed_cols;
      scale_rows[t] = scales + oc * (groups * 2);
    } else {
      weight_rows[t] = nullptr;
      scale_rows[t] = nullptr;
    }
  }

  float acc[kTileOC];
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    acc[t] = 0.0f;
  }

  for (int pack = threadIdx.x; pack < packed_cols; pack += blockDim.x) {
    int k_base = pack * 4;
    float in_vals[4];
    bool valid[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int k = k_base + i;
      valid[i] = k < IC;
      in_vals[i] = valid[i] ? __half2float(inputs[row * IC + k]) : 0.0f;
    }
    if (!valid[0] && !valid[1] && !valid[2] && !valid[3]) {
      continue;
    }

    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      if (t >= tile_cols) {
        break;
      }
      uint8_t byte_val = weight_rows[t][pack];
      #pragma unroll
      for (int lane = 0; lane < 4; ++lane) {
        if (!valid[lane]) {
          continue;
        }
        uint8_t code = (byte_val >> (lane * 2)) & 0x3;
        int gid = (k_base + lane) / group_size;
        if (gid >= groups) {
          continue;
        }
        int scale_offset = gid * 2;
        float alpha = __half2float(scale_rows[t][scale_offset]);
        float beta = __half2float(scale_rows[t][scale_offset + 1]);
        float mag = (code & 0x1) ? beta : alpha;
        float sign = (code & 0x2) ? -1.0f : 1.0f;
        acc[t] += sign * mag * in_vals[lane];
      }
    }
  }

  __shared__ float shm[kThreads][kTileOC];
  int tid = threadIdx.x;
  #pragma unroll
  for (int t = 0; t < kTileOC; ++t) {
    shm[tid][t] = acc[t];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int t = 0; t < kTileOC; ++t) {
        shm[tid][t] += shm[tid + stride][t];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    #pragma unroll
    for (int t = 0; t < kTileOC; ++t) {
      int oc = oc_base + t;
      if (t < tile_cols && oc < OC) {
        outputs[row * OC + oc] = shm[0][t];
      }
    }
  }
}

}  // namespace

torch::Tensor gemv_wx_asym_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size,
    int bit_width) {
  TORCH_CHECK(_in_feats.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(_kernel.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(_scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(_zeros.is_cuda(), "zeros must be CUDA");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf, "inputs must be fp16");
  TORCH_CHECK(_kernel.dtype() == torch::kByte, "qweight must be uint8");
  TORCH_CHECK(_scales.dtype() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(_zeros.dtype() == torch::kByte, "zeros must be uint8");
  TORCH_CHECK(bit_width == 3 || bit_width == 4, "bit_width must be 3 or 4 for asymmetric kernel");

  auto inputs = _in_feats.contiguous();
  auto kernel = _kernel.contiguous();
  auto scales = _scales.contiguous();
  auto zeros = _zeros.contiguous();

  const int64_t M = inputs.size(0);
  const int64_t IC = inputs.size(1);
  const int64_t OC = kernel.size(0);
  const int64_t packed_cols = kernel.size(1);
  const int groups = static_cast<int>((IC + group_size - 1) / group_size);

  auto out = torch::empty({M, OC}, inputs.options().dtype(torch::kFloat32));
  const half* in_ptr = reinterpret_cast<const half*>(inputs.data_ptr<at::Half>());
  const uint8_t* w_ptr = kernel.data_ptr<uint8_t>();
  const half* s_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  const uint8_t* z_ptr = zeros.data_ptr<uint8_t>();
  float* out_ptr = out.data_ptr<float>();

  dim3 grid((OC + kTileOC - 1) / kTileOC, M);
  dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemv_asym_kernel<<<grid, block, 0, stream>>>(
      in_ptr, w_ptr, s_ptr, z_ptr, out_ptr,
      static_cast<int>(M),
      static_cast<int>(IC),
      static_cast<int>(OC),
      groups,
      group_size,
      static_cast<int>(packed_cols),
      bit_width);
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor gemv_w2_qtr_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    int group_size) {
  TORCH_CHECK(_in_feats.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(_kernel.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(_scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf, "inputs must be fp16");
  TORCH_CHECK(_kernel.dtype() == torch::kByte, "qweight must be uint8");
  TORCH_CHECK(_scales.dtype() == torch::kHalf, "scales must be fp16");

  auto inputs = _in_feats.contiguous();
  auto kernel = _kernel.contiguous();
  auto scales = _scales.contiguous();

  const int64_t M = inputs.size(0);
  const int64_t IC = inputs.size(1);
  const int64_t OC = kernel.size(0);
  const int64_t packed_cols = kernel.size(1);
  const int groups = static_cast<int>((IC + group_size - 1) / group_size);

  auto out = torch::empty({M, OC}, inputs.options().dtype(torch::kFloat32));
  const half* in_ptr = reinterpret_cast<const half*>(inputs.data_ptr<at::Half>());
  const uint8_t* w_ptr = kernel.data_ptr<uint8_t>();
  const half* s_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  float* out_ptr = out.data_ptr<float>();

  dim3 grid((OC + kTileOC - 1) / kTileOC, M);
  dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemv_w2_qtr_kernel<<<grid, block, 0, stream>>>(
      in_ptr, w_ptr, s_ptr, out_ptr,
      static_cast<int>(M),
      static_cast<int>(IC),
      static_cast<int>(OC),
      groups,
      group_size,
      static_cast<int>(packed_cols));
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

torch::Tensor gemv_w2_lloyd_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    int group_size) {
  TORCH_CHECK(_in_feats.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(_kernel.is_cuda(), "qweight must be CUDA");
  TORCH_CHECK(_scales.is_cuda(), "scales must be CUDA");
  TORCH_CHECK(_in_feats.dtype() == torch::kHalf, "inputs must be fp16");
  TORCH_CHECK(_kernel.dtype() == torch::kByte, "qweight must be uint8");
  TORCH_CHECK(_scales.dtype() == torch::kHalf, "scales must be fp16");

  auto inputs = _in_feats.contiguous();
  auto kernel = _kernel.contiguous();
  auto scales = _scales.contiguous();

  const int64_t M = inputs.size(0);
  const int64_t IC = inputs.size(1);
  const int64_t OC = kernel.size(0);
  const int64_t packed_cols = kernel.size(1);
  const int groups = static_cast<int>((IC + group_size - 1) / group_size);

  auto out = torch::empty({M, OC}, inputs.options().dtype(torch::kFloat32));
  const half* in_ptr = reinterpret_cast<const half*>(inputs.data_ptr<at::Half>());
  const uint8_t* w_ptr = kernel.data_ptr<uint8_t>();
  const half* s_ptr = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
  float* out_ptr = out.data_ptr<float>();

  dim3 grid((OC + kTileOC - 1) / kTileOC, M);
  dim3 block(kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemv_w2_lloyd_kernel<<<grid, block, 0, stream>>>(
      in_ptr, w_ptr, s_ptr, out_ptr,
      static_cast<int>(M),
      static_cast<int>(IC),
      static_cast<int>(OC),
      groups,
      group_size,
      static_cast<int>(packed_cols));
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}
