"""损失函数：Charbonnier 与 FFT 高频损失。

设计依据
--------
**Charbonnier** — GameSR（Rehman et al., IEEE Transactions on Games 2026）在
游戏超分任务上用它替代 L1/MSE，原文称"it provided more stable convergence than
MSE/MAE"。形式 rho(x) = sqrt(x^2 + eps^2)：小残差时近似 L2（梯度平滑），
大残差时近似 L1（对离群点鲁棒）。相较纯 L1，它在零点附近**可导**，
避免了 |x| 在原点的次梯度抖动。

**FFT 高频** — 纯 L1 的已知问题是其最优解趋向条件均值，表现为输出偏平滑、
高频纹理丢失。这里把预测与真值转到频域，**只在高频段**对幅度谱差异施加
约束。相比 VGG/LPIPS 感知损失，它的优势是：
  1. 不需要任何外部预训练权重（零依赖，可离线运行）
  2. 目标明确——直接针对"锐度"这一个维度
  3. 计算便宜（一次 rfft2）

权重 `fft_weight` 需谨慎：过大会放大压缩伪影与噪声（频域对高频噪声极敏感）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier 损失（L1 的可导平滑近似）。

    eps 取 1e-3：像素值域为 [0,1]，该量级让 |x| < 1e-3 的区域表现为 L2，
    其余区域近似 L1。
    """
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def _radial_mask(h: int, w: int, high_ratio: float, device, dtype):
    """构造频率半径掩码，形状 (h, w)，对应 rfft2 的输出。

    注意 w 是 rfft2 **之后**的宽度（= 原宽 // 2 + 1），故对应原宽为 2*(w-1)；
    这样奇数/偶数宽度都能还原出正确的频率轴。
    """
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(2 * (w - 1), device=device,
                            dtype=dtype).abs().view(1, -1)
    r = torch.sqrt(fy ** 2 + fx ** 2)
    r = r / r.max().clamp(min=1e-8)
    return (r > high_ratio).to(dtype)


def fft_highfreq_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    high_ratio: float = 0.5,
) -> torch.Tensor:
    """频域高频损失：仅约束幅度谱的高频部分。

    pred/target: (B, C, H, W)，值域 [0,1]。
    high_ratio : 归一化频率半径阈值，> 该半径算高频。0.5 表示外圈一半面积。

    用 norm="ortho" 保证与空间域尺度一致，避免 batch 大小影响损失量级。
    """
    p = pred.float()
    t = target.float()
    ap = torch.fft.rfft2(p, norm="ortho").abs()
    at = torch.fft.rfft2(t, norm="ortho").abs()

    h, w = ap.shape[-2], ap.shape[-1]
    mask = _radial_mask(h, w, high_ratio, ap.device, ap.dtype)
    mask = mask.view(1, 1, h, w)

    diff = (ap - at).abs() * mask
    denom = mask.sum().clamp(min=1.0) * ap.shape[0] * ap.shape[1]
    return diff.sum() / denom


def build_reconstruction_loss(
    kind: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_weight: float = 0.1,
    fft_high_ratio: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """按 kind 组装重建损失，返回 (loss, 明细字典)。"""
    detail = {}
    if kind in ("charbonnier", "charbonnier_fft"):
        rec = charbonnier_loss(pred, target)
        detail["charbonnier"] = rec.item()
    else:
        rec = F.l1_loss(pred, target)
        detail["l1"] = rec.item()

    if kind == "charbonnier_fft":
        hf = fft_highfreq_loss(pred, target, fft_high_ratio)
        detail["fft"] = hf.item()
        rec = rec + fft_weight * hf

    detail["total"] = rec.item()
    return rec, detail
