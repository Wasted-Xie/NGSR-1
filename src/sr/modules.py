"""
mHC（流形约束超连接）与高精度旁路。

对应设计文档：
  - 2.6  mHC（记录中称"残差压制"）
  - 2.7  高精度旁路（质量保障）

命名说明（设计文档 9.3 节核实结论）：
    mHC 全称 Manifold-Constrained Hyper-Connections（流形约束超连接），
    机制是把残差混合矩阵投影到 Birkhoff 多胞形（双随机矩阵），
    目的是【恢复恒等映射性质、抑制训练不稳定】，而非"压制残差"。
    Sinkhorn-Knopp 迭代正是双随机矩阵投影的标准算法。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuantScheme
from .quant import QuantConv2d


def sinkhorn_knopp(
    log_alpha: torch.Tensor, num_iters: int = 3, temperature: float = 1.0
) -> torch.Tensor:
    """Sinkhorn-Knopp 迭代：把矩阵投影为双随机矩阵。

    设计文档 2.6 节：Sinkhorn-Knopp 归一化约束残差传播。
    设计文档 9.2 节待定参数：Sinkhorn 迭代次数（此处默认 3）。

    参数
    ----
    log_alpha : (..., n, n) 未归一化的 log 权重
    num_iters : 迭代次数

    返回
    ----
    (..., n, n) 双随机矩阵（行和列均近似为 1）
    """
    log_alpha = log_alpha / temperature
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(
            log_alpha, dim=-1, keepdim=True
        )
        log_alpha = log_alpha - torch.logsumexp(
            log_alpha, dim=-2, keepdim=True
        )
    return log_alpha.exp()


class ManifoldConstrainedHyperConnection(nn.Module):
    """mHC：流形约束超连接。

    设计文档 2.6 节列出的作用：
      - Sinkhorn-Knopp 归一化约束残差传播
      - 限制梯度范数，保障低精度训练稳定性
      - 抑制噪声和压缩伪影，保留高频纹理
      - 支持深层网络平滑收敛

    设计文档 2.6（R53 补充）：
      mHC 根据理解层输出的【退化程度】动态调整约束强度。
    """

    def __init__(
        self,
        dim: int,
        num_streams: int = 4,
        sinkhorn_iters: int = 3,
        init_alpha: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.sinkhorn_iters = sinkhorn_iters

        # 残差混合矩阵（对数域），经 Sinkhorn 投影后变为双随机矩阵
        self.log_alpha = nn.Parameter(
            torch.full((num_streams, num_streams), init_alpha)
        )

        # 退化程度 -> 约束强度的动态映射（R53）
        self.strength_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, degradation: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x: (B, N, C) 单流，或 (B, N, k, C) 多流
        degradation: (B, 1) 或 (B, N, 1) 退化程度估计（0~1），可选

        返回
        ----
        与输入同形状。单流输入时，双随机矩阵退化为 1×1 的单位矩阵，
        故输出与输入完全相等（恒等映射）。

        修复说明
        --------
        原实现只取 `alpha` 的对角均值 `diag_mean` 当标量用，并对残差流做
        `x + strength * (diag_mean - 1) * x`。由于双随机矩阵的对角均值恒
        ≤ 1（仅在置换矩阵时取 1），该式**只能衰减、永不增强**：
        `init_alpha=0.01` 时为 0.25，`strength=0.5` 时每层乘 0.625，
        6 层后约 0.06，等价于逐层遗忘早期特征，与「恢复恒等映射」的
        设计目标相反。

        现改为对 k 条流做真正的混合 `alpha @ streams`，并把约束写成
        相对恒等的残差增量 `streams + strength * (mixed - streams)`。
        单流时 `alpha` 为 1×1 单位阵、`mixed == streams`，输出恒等于输入；
        多流时行为符合设计意图。
        """
        single_stream = x.dim() == 3
        streams = x.unsqueeze(2) if single_stream else x  # (B, N, k, C)
        num_streams = streams.shape[2]

        # 按实际流数取子矩阵再做 Sinkhorn 投影，得到 k×k 双随机矩阵
        alpha = sinkhorn_knopp(
            self.log_alpha[:num_streams, :num_streams], self.sinkhorn_iters
        )
        mixed = torch.einsum("bnkc,kj->bnjc", streams, alpha)

        # 动态约束强度（R53：根据退化程度调整）
        if degradation is not None:
            d = degradation.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
            strength = self.strength_gate(d).reshape(-1, 1, 1, 1)
        else:
            strength = 0.5

        # 相对恒等的残差增量：k=1 时 mixed == streams，输出即恒等
        out = streams + strength * (mixed - streams)
        return out.squeeze(2) if single_stream else out


class HighPrecisionBypass(nn.Module):
    """高精度旁路（质量保障）。

    对应设计文档 2.7 节的结构：
      旁路输入：低精度专家输出 + 浅层细节特征 + 方向估计
        ├─ 边缘重建子模块：方向性卷积 + 边缘残差（FP16）
        ├─ 纹理重建子模块：高频残差 + 纹理记忆（FP16）
        ├─ 1x1 Conv（通道压缩）
        ├─ 3x3 DWConv（空间细节恢复）
        ├─ 1x1 Conv（通道恢复）
        └─ 残差连接
      输出：高精度残差，叠加到低精度专家输出上

    参数（设计文档 2.7）：
      - 参数量占比：总参数量的 8%~12%
      - 激活范围：10%~20% 的质量敏感 Token

    量化（对应设计文档 4.1 节的旁路精度）：
      `quant` 同时作用于本模块全部卷积的权重与输入激活，经 `bypass_proj`
      一并覆盖。可选范围见 `config.validate()`（W8A8 / FP8 / FP16 / FP32）。
      伪量化实现见 `quant.py`；FP32 时退化为普通卷积。
    """

    def __init__(
        self,
        dim: int,
        channels: int = 32,
        kernel_size: int = 3,
        quant: QuantScheme = QuantScheme.FP16,
    ):
        super().__init__()
        self.dim = dim
        self.quant = quant

        # 边缘重建子模块：方向性卷积
        # 用 4 个方向的卷积核近似 Sobel 方向估计
        self.edge_conv = QuantConv2d(
            dim, channels, kernel_size, padding=kernel_size // 2, bias=False,
            quant=quant,
        )
        # 方向估计分支：4 个方向（0/45/90/135 度）
        self.direction = QuantConv2d(
            dim, 4, kernel_size, padding=kernel_size // 2, bias=False,
            quant=quant,
        )

        # 纹理重建子模块
        self.texture_conv = QuantConv2d(
            dim, channels, kernel_size, padding=kernel_size // 2, bias=False,
            quant=quant,
        )

        # 通道压缩 -> DWConv -> 通道恢复
        self.compress = QuantConv2d(channels * 2 + 4, channels, 1, bias=False,
                                    quant=quant)
        self.dw = QuantConv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False,
            quant=quant,
        )
        self.expand = QuantConv2d(channels, dim, 1, bias=False, quant=quant)

        self.act = nn.GELU()
        self.norm = nn.GroupNorm(1, channels)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        shallow: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x       : (B, N, C) 低精度专家输出
        h, w    : 特征图尺寸
        shallow : (B, N, C) 浅层细节特征（可选，需已投影到 C 通道）
        gate    : (B, N, 1) 质量门控概率（可选，来自三级 MoE 的第三级）

        返回
        ----
        (B, N, C) 高精度残差
        """
        b, n, c = x.shape
        assert c == self.dim, (
            f"输入通道数 {c} 与旁路配置 {self.dim} 不匹配"
        )
        xr = x.transpose(1, 2).reshape(b, c, h, w)

        # 边缘分支
        edge = self.act(self.edge_conv(xr))
        direction = self.direction(xr)

        # 纹理分支
        texture = self.act(self.texture_conv(xr))

        # 融合
        feat = torch.cat([edge, texture, direction], dim=1)
        feat = self.compress(feat)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.dw(feat)
        feat = self.act(feat)
        out = self.expand(feat)  # (B, C, h, w)

        # 叠加浅层细节特征（浅层特征需已投影到 C 通道）
        if shallow is not None:
            assert shallow.shape[-1] == c, (
                f"浅层特征通道 {shallow.shape[-1]} 与主干通道 {c} 不匹配"
            )
            sr = shallow.transpose(1, 2).reshape(b, c, h, w)
            out = out + sr

        out = out.flatten(2).transpose(1, 2)  # (B, N, C)

        # 质量门控：只对质量敏感 token 生效
        if gate is not None:
            out = out * gate

        return out


class ShallowDetailPath(nn.Module):
    """浅层细节路。

    对应设计文档 2.1 节：
      浅层细节路：Conv 3x3 x2，保留输入分辨率的高频纹理、边缘、噪声信息
        ├─ Sobel 方向估计分支（0/45/90/135 度）
        └─ 高频残差提取

    设计文档 5.2 节：用 DW+PW 替代标准 3x3 卷积，FLOPs 降低 8~9 倍。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.out_channels = out_channels

        # 两层 DW+PW 卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GELU(),
        )

        # Sobel 方向估计分支（0/45/90/135 度）
        self.sobel = nn.Conv2d(
            out_channels, 4, 3, padding=1, groups=1, bias=False
        )

        # 高频残差提取：原图减去低频
        self.high_freq_proj = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, 3, H, W) 输入图像

        返回
        ----
        feat      : (B, C, H, W) 浅层特征
        direction : (B, 4, H, W) 方向估计
        high_freq : (B, C, H, W) 高频残差
        """
        feat = self.conv2(self.conv1(x))
        direction = self.sobel(feat)

        # 高频残差：用平均池化近似低频，相减得高频
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        high_freq = self.high_freq_proj(high)

        return {"feat": feat, "direction": direction, "high_freq": high_freq}
