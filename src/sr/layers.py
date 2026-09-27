"""
基础网络层。

对应设计文档：
  - 2.3  低分辨率主干 / 空间注意力（单帧）
  - 5.2  计算量控制手段（窗口注意力、DW 卷积、FFN 扩展因子）
  - 5.2.7 算力重算：窗口注意力 FLOPs ∝ N（线性），不是 O(N²)

重要说明（来自设计文档 5.2.7 第一节的重算结论）：
    窗口注意力下，注意力的 FLOPs 对 token 数 N 是【线性】的，
    只有全局注意力才是 O(N²)。本模块实现的是窗口注意力。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, N, C) -> (B, C, H, W)"""
    b, n, c = x.shape
    assert n == h * w, f"token 数 {n} 与 {h}x{w} 不匹配"
    return x.transpose(1, 2).reshape(b, c, h, w)


def to_tokens(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C)"""
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（DW 3x3 + PW 1x1）。

    设计文档 5.2 节：浅层细节路用 DW+PW 替代标准 3x3 卷积，
    FLOPs 降低 8~9 倍。1080p 空间的标准卷积是算力杀手。
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class WindowAttention(nn.Module):
    """窗口多头自注意力（单帧）。

    对应设计文档 2.3 节"空间注意力（单帧）"：
      - 窗口大小：8x8 / 16x16 / 32x32（用户可调）
      - 单帧，无时序组件（R42 单帧约束）

    实现要点：把特征图切分为不重叠窗口，在窗口内做自注意力。
    复杂度对 token 数线性（每个 token 只与窗口内 w² 个 token 交互）。
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"通道数 {dim} 必须能被头数 {num_heads} 整除")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """x: (B, N, C)，h/w 为特征图尺寸。"""
        b, n, c = x.shape
        assert n == h * w, f"token 数 {n} != {h}x{w}"

        ws = self.window_size
        # 需要 H、W 都能被窗口整除（config.validate 已保证）
        assert h % ws == 0 and w % ws == 0, (
            f"特征图 {h}x{w} 不能被窗口 {ws} 整除"
        )

        nh, nw = h // ws, w // ws

        # 切分窗口：(B, N, C) -> (B*nh*nw, ws*ws, C)
        xr = x.reshape(b, nh, ws, nw, ws, c)
        xr = xr.permute(0, 1, 3, 2, 4, 5).contiguous()
        xw = xr.reshape(b * nh * nw, ws * ws, c)

        # 窗口内自注意力
        nw_tokens = xw.shape[1]
        qkv = self.qkv(xw).reshape(
            -1, nw_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B', heads, Nw, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(-1, nw_tokens, c)
        out = self.proj_drop(self.proj(out))

        # 还原窗口：(B*nh*nw, ws*ws, C) -> (B, N, C)
        out = out.reshape(b, nh, nw, ws, ws, c)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.reshape(b, h * w, c)

        return out


class FeedForward(nn.Module):
    """FFN，扩展因子可调。

    设计文档 5.2 节：标准 Transformer FFN 扩展因子 4，
    实时场景降到 2，FFN FLOPs 减半。
    """

    def __init__(self, dim: int, ratio: int = 2, drop: float = 0.0):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class AdaptiveUpsample(nn.Module):
    """自适应上采样。

    对应设计文档 2.8 节（R53 修正版）：
      网络内部固定做 2x（PixelShuffle），最终输出分辨率通过轻量缩放对齐。
      理由：避免为每个超分倍数训练独立的上采样权重。

    流程：
      1. PixelShuffle(2) 做整数倍上采样
      2. 双三次插值 + 1x1 Conv 缩放到目标尺寸
    """

    def __init__(self, in_channels: int, out_channels: int = 3, scale: int = 2):
        super().__init__()
        self.scale = scale
        # PixelShuffle 前需要把通道扩展到 out_channels * scale^2
        self.expand = nn.Conv2d(
            in_channels, out_channels * scale * scale, 1, bias=False
        )
        self.shuffle = nn.PixelShuffle(scale)
        # 轻量缩放：1x1 Conv + 双三次插值
        self.refine = nn.Conv2d(out_channels, out_channels, 1, bias=False)

    def forward(
        self, x: torch.Tensor, target_hw: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        x: (B, C, h, w)
        target_hw: 目标 (H, W)；None 表示只做 PixelShuffle
        """
        y = self.shuffle(self.expand(x))
        if target_hw is not None and y.shape[-2:] != tuple(target_hw):
            # 双三次插值到目标尺寸，再用 1x1 Conv 微调
            y = F.interpolate(
                y, size=target_hw, mode="bicubic", align_corners=False
            )
        return self.refine(y)


class LayerScale(nn.Module):
    """逐通道缩放，用于稳定深层残差训练。"""

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
