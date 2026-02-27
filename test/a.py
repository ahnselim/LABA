import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_uv_intuition_plots(rows=512, cols=512, rank=64, seed=42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 시뮬레이션 데이터 생성 (LLM의 실제 오차 분포 모사)
    # LLM 오차는 특정 채널(Outlier)의 오차가 매우 큰 특성이 있음
    base_error = torch.randn(rows, cols, device=device)
    
    # Outlier 효과 부여 (특정 행/열의 스케일을 키움)
    row_outliers = torch.exp(torch.randn(rows, device=device) * 0.8)
    col_outliers = torch.exp(torch.randn(cols, device=device) * 0.8)
    R = base_error * row_outliers.unsqueeze(1) * col_outliers.unsqueeze(0)

    # ---------------------------------------------------------
    # Case A: No uv (v3 스타일 - SVD만 수행)
    # ---------------------------------------------------------
    U, S, V = torch.linalg.svd(R, full_matrices=False)
    R_approx_v3 = (U[:, :rank] * S[:rank]) @ V[:rank, :]
    resid_v3 = R - R_approx_v3

    # ---------------------------------------------------------
    # Case B: With uv (v2 스타일 - Scaling vectors 학습)
    # ---------------------------------------------------------
    # 간단한 closed-form 추정 (로직 재현)
    u = torch.ones(rows, device=device)
    v = torch.ones(cols, device=device)
    
    # 1회 업데이트 예시 (실제 코드의 update_uv_closed_form 로직 핵심 요약)
    u = torch.sqrt(torch.mean(R**2, dim=1))
    v = torch.sqrt(torch.mean(R**2, dim=0))
    u = u / u.mean() # 정규화
    
    R_norm = R / (u.unsqueeze(1) * v.unsqueeze(0))
    Un, Sn, Vn = torch.linalg.svd(R_norm, full_matrices=False)
    
    # 다시 복원: u * (An * Bn) * v
    R_approx_v2 = (Un[:, :rank] * Sn[:rank] @ Vn[:rank, :]) * (u.unsqueeze(1) * v.unsqueeze(0))
    resid_v2 = R - R_approx_v2

    # ---------------------------------------------------------
    # 시각화 시작
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 5))
    plt.rcParams.update({'font.size': 12})

    # Plot 1: Singular Value Decay (정보 밀도 증명)
    # 정규화된 행렬의 특잇값이 얼마나 더 'Low-rank'에 집중되는지 보여줌
    ax1 = fig.add_subplot(1, 3, 1)
    s_v3 = S.cpu().numpy()
    s_v2 = Sn.cpu().numpy() * (u.mean() * v.mean()).cpu().numpy() # 비교를 위해 스케일 조정
    
    # 누적 설명 분산량(Cumulative Explained Variance)으로 그리면 더 명확함
    cum_v3 = np.cumsum(s_v3**2) / np.sum(s_v3**2)
    cum_v2 = np.cumsum(s_v2**2) / np.sum(s_v2**2)
    
    ax1.plot(cum_v2[:128], label='With $u,v$ (v2)', color='#1f77b4', linewidth=3)
    ax1.plot(cum_v3[:128], label='No $u,v$ (v3)', color='#d62728', linestyle='--', linewidth=2)
    ax1.axvline(x=rank, color='gray', linestyle=':', label=f'Rank {rank}')
    ax1.set_title("1. Information Density\n(Cumulative Explained Variance)", fontweight='bold')
    ax1.set_xlabel("Rank Index")
    ax1.set_ylabel("Cumulative Variance Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residual Heatmap (Outlier 억제 증명)
    # 상위 100x100 영역만 확대하여 시각적 차이 극대화
    ax2 = fig.add_subplot(1, 3, 2)
    diff = (resid_v3.abs().mean() / resid_v2.abs().mean()).item()
    sns.heatmap(resid_v2[:100, :100].cpu().numpy(), cmap='RdBu_r', center=0, 
                cbar=True, ax=ax2, vmin=-5, vmax=5)
    ax2.set_title(f"2. Residual Heatmap (v2)\n(Error is balanced & suppressed)", fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])

    # Plot 3: Error Distribution (정밀도 향상 강조)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(resid_v3.flatten().cpu().numpy(), bins=100, range=(-10, 10), 
             color='#d62728', alpha=0.4, label='No $u,v$ (v3)', density=True)
    ax3.hist(resid_v2.flatten().cpu().numpy(), bins=100, range=(-10, 10), 
             color='#1f77b4', alpha=0.6, label='With $u,v$ (v2)', density=True)
    ax3.set_title("3. Error Distribution\n(More peaked at Zero)", fontweight='bold')
    ax3.set_xlabel("Residual Magnitude")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('uv_scaling_effectiveness.png', dpi=300)
    plt.show()

# 실행
generate_uv_intuition_plots()