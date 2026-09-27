"""伪量化（QAT / 精度仿真）工具。

职责
----
让 `config.QuantScheme` 真正作用于**计算**，而不只是显存估算的标签。

背景：`bypass_quant` 一度是死配置——`HighPrecisionBypass` 没有接收该参数，
改它不影响任何计算结果（见 README 已知限制）。本模块提供统一的伪量化原语，
使各模块的量化方案可被真实执行与验证。

设计约定
--------
1. **伪量化而非真量化**：前向把数值投影到目标格式的表示集合上（cast 或
   round+scale），再转回原 dtype 参与计算。这样无需硬件支持低位宽算子，
   也能在训练时评估精度损失。
2. **STE（直通估计器）**：`round()`/`cast` 的梯度处处为 0，若不做处理权重
   将完全收不到梯度（本项目已在 `TernaryLinear` 上踩过该坑）。所有伪量化
   统一用 `x + (xq - x).detach()`：前向等于 xq，反向梯度直通 x。
3. **train / eval 数值一致**：两种模式下都做同一伪量化，避免训练与推理
   行为分裂。STE 只影响反向，不影响前向数值。
4. **scale 不可学习**：本模块的 INT/三值 scale 由权重统计量动态推出
   （在 `no_grad` 下计算），只有权重本身获得梯度。若需要可学习 scale，
   参见 `moe.TernaryLinear`（那个是为私有专家专门做的）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QuantScheme

# E4M3 的可表示上界（1.75 × 2^8 = 448）
_FP8_E4M3_MAX = 448.0


def _ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """直通估计器：前向取 xq，反向把梯度原样传回 x。

    不可写成 `xq + (x - x).detach()`——该项恒为 0，等同于没有 STE。

    仅在**确实需要梯度**时才走 STE。原因：`x + (xq - x).detach()` 的前向
    在浮点下并不精确等于 xq（`xq - x` 会损失低位），会使量化值散布在目标
    表示集合之外——例如三值量化每通道本应只有 {0, ±scale} 三个取值，
    实测却出现 7 个，且重量化不再幂等。推理阶段不需要 STE，直接返回精确的
    xq 即可，这对精度评估与后续量化导出都更可靠。
    """
    if torch.is_grad_enabled() and x.requires_grad:
        return x + (xq - x).detach()
    return xq


def _scale_dim_for(w: torch.Tensor) -> tuple[int, ...]:
    """权重按输出通道定标时应保留的维度。

    Conv2d 权重形状 (out, in, kh, kw) → 沿 dim=1 归约，保留 (out,1,1,1)。
    Linear 权重形状 (out, in)         → 沿 dim=1 归约，保留 (out,1)。
    """
    return (1,) + (1,) * (w.dim() - 2)


# ---------------------------------------------------------------------------
# 权重伪量化
# ---------------------------------------------------------------------------

def _q_fp16(w: torch.Tensor) -> torch.Tensor:
    return _ste(w, w.to(torch.float16).to(w.dtype))


def _q_fp8(w: torch.Tensor) -> torch.Tensor:
    """E4M3 仿真：先按可表示范围饱和，再用原生 float8 类型舍入。

    用原生 dtype 而非手写位运算，是为了让舍入行为与硬件一致。
    """
    q = w.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    q = q.to(torch.float8_e4m3fn).to(w.dtype)
    return _ste(w, q)


def _q_int_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    """对称 INT 量化，逐输出通道定标。

    bits=8 → 量化级 [-127, 127]；bits=4 → [-7, 7]。
    逐输出通道（而非逐张量）是卷积权重的常规做法：各输出通道的动态范围
    可能相差一个数量级，共用 scale 会让小通道整体被压成 0。
    """
    qmax = float(2 ** (bits - 1) - 1)
    reduce_dims = tuple(range(1, w.dim()))
    with torch.no_grad():
        absmax = w.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        scale = absmax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return _ste(w, q)


def _q_ternary(w: torch.Tensor) -> torch.Tensor:
    """三值量化（-1/0/+1），逐输出通道按绝对值均值定标。

    必须用均值而非固定 scale：若 scale 远大于权重幅值，`|w/scale| < 0.5`
    会把全部权重量化为 0（死区陷阱，本项目已在 TernaryLinear 上踩过）。
    """
    reduce_dims = tuple(range(1, w.dim()))
    with torch.no_grad():
        scale = w.abs().mean(dim=reduce_dims, keepdim=True).clamp(min=1e-5)
    q = torch.clamp(torch.round(w / scale), -1.0, 1.0) * scale
    return _ste(w, q)


def quantize_weight(w: torch.Tensor, scheme: QuantScheme) -> torch.Tensor:
    """按方案对权重做伪量化。"""
    if scheme is QuantScheme.FP32:
        return w
    if scheme is QuantScheme.FP16:
        return _q_fp16(w)
    if scheme is QuantScheme.FP8:
        return _q_fp8(w)
    if scheme is QuantScheme.W8A8:
        return _q_int_symmetric(w, bits=8)
    if scheme is QuantScheme.W4A8:
        return _q_int_symmetric(w, bits=4)
    if scheme is QuantScheme.TERNARY:
        return _q_ternary(w)
    raise ValueError(f"未知量化方案：{scheme!r}")


# ---------------------------------------------------------------------------
# 激活伪量化
# ---------------------------------------------------------------------------

def quantize_activation(x: torch.Tensor, scheme: QuantScheme) -> torch.Tensor:
    """按方案对激活做伪量化。

    浮点方案（FP16/FP8）直接按格式舍入；整型方案用逐张量对称量化
    （等效于运行时动态定标，无需离线校准）。
    """
    if scheme is QuantScheme.FP32:
        return x
    if scheme is QuantScheme.FP16:
        return _ste(x, x.to(torch.float16).to(x.dtype))
    if scheme is QuantScheme.FP8:
        q = x.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        q = q.to(torch.float8_e4m3fn).to(x.dtype)
        return _ste(x, q)
    if scheme in (QuantScheme.W8A8, QuantScheme.W4A8, QuantScheme.TERNARY):
        # 三值方案只约束权重，激活仍按 INT8 处理
        with torch.no_grad():
            absmax = x.abs().amax().clamp(min=1e-8)
            scale = absmax / 127.0
        q = torch.clamp(torch.round(x / scale), -127.0, 127.0) * scale
        return _ste(x, q)
    raise ValueError(f"未知量化方案：{scheme!r}")


# ---------------------------------------------------------------------------
# 量化卷积
# ---------------------------------------------------------------------------

class QuantConv2d(nn.Conv2d):
    """带伪量化的 Conv2d。

    scheme 同时作用于权重与输入激活；`QuantScheme.FP32` 时退化为普通卷积。
    参数量与普通 Conv2d 完全相同（伪量化不引入额外参数）。
    """

    def __init__(self, *args, quant: QuantScheme = QuantScheme.FP32, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = quantize_weight(self.weight, self.quant)
        xq = quantize_activation(x, self.quant)
        return F.conv2d(
            xq, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
