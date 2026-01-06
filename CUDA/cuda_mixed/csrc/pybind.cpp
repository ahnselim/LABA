#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor gemv_wx_asym_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size,
    int bit_width);

torch::Tensor gemv_w2_qtr_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    int group_size);

torch::Tensor gemv_w2_lloyd_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    int group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemv_wx_asym",
        &gemv_wx_asym_forward_cuda,
        "Asymmetric W3/W4 GEMV (CUDA)");
    m.def(
        "gemv_w2_qtr",
        &gemv_w2_qtr_forward_cuda,
        "2-bit QTR GEMV (CUDA)");
    m.def(
        "gemv_w2_lloyd",
        &gemv_w2_lloyd_forward_cuda,
        "2-bit Lloyd GEMV (CUDA)");
}
