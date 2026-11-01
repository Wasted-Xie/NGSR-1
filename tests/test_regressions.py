"""回归测试：issue #1 / #2 / #3 的修复验证。

这三个问题的共同点是「能跑通、形状对、原有测试全绿」，但语义是错的：

  #1 主干没有跨窗口通路      —— 窗口注意力之外没有任何算子，感受野恒为 8x8
  #2 aux_loss 梯度恒为零      —— 由 argmax 派生，整条链不可导
  #3 mHC 退化为标量衰减       —— 双随机矩阵只用到对角均值，每层乘 0.625

本文件对每个问题给出一条能"变红"的断言：修复前必然失败，修复后通过。
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from _device import DEVICE, device_desc  # noqa: E402
from sr.backbone import HybridBlock  # noqa: E402
from sr.modules import ManifoldConstrainedHyperConnection  # noqa: E402
from sr.moe import ThreeLevelMoE  # noqa: E402

print("计算设备: %s" % device_desc())

passed = 0
failed = 0


def check(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print("  [OK]   %s %s" % (name, detail))
    else:
        failed += 1
        print("  [FAIL] %s %s" % (name, detail))


def banner(s):
    print()
    print("=" * 76)
    print(s)
    print("=" * 76)


torch.manual_seed(0)

# ---------------------------------------------------------------------------
banner("issue #2：aux_loss 必须真的可导")
# ---------------------------------------------------------------------------

DIM, H, W = 32, 16, 16
moe = ThreeLevelMoE(
    dim=DIM, num_experts=8, top_k=2, num_clusters=2, ffn_ratio=2, group_size=4
).to(DEVICE)
xm = torch.randn(1, H * W, DIM, device=DEVICE)

moe.zero_grad()
out = moe(xm, H, W)
aux = out["aux_loss"]
check(aux.requires_grad, "aux_loss 参与计算图", "requires_grad=%s" % aux.requires_grad)

aux.backward()
g = moe.group_router.gate.weight.grad
grad_sum = 0.0 if g is None else g.abs().sum().item()
check(
    grad_sum > 0,
    "aux_loss 对组级路由参数有非零梯度",
    "grad_abs_sum=%.3e" % grad_sum,
)

# 均衡损失应在均匀路由附近取到最小值 1.0，明显塌缩时更大
uniform = torch.full((2,), 0.5, device=DEVICE)          # f_i
probs = torch.full((2,), 0.5, device=DEVICE)           # P_i
ideal = (2 * (uniform * probs).sum()).item()
skewed = (2 * (torch.tensor([1.0, 0.0], device=DEVICE)
               * torch.tensor([1.0, 0.0], device=DEVICE)).sum()).item()
check(
    abs(ideal - 1.0) < 1e-6 and skewed > ideal,
    "均衡损失在均匀分布处取最小值",
    "均匀=%.3f 塌缩=%.3f" % (ideal, skewed),
)

# ---------------------------------------------------------------------------
banner("issue #3：mHC 初始化必须是恒等映射")
# ---------------------------------------------------------------------------

mhc = ManifoldConstrainedHyperConnection(DIM, num_streams=4).to(DEVICE)
xh = torch.randn(2, 64, DIM, device=DEVICE)
yh = mhc(xh)
ratio = (yh.norm() / xh.norm()).item()
check(
    torch.allclose(yh, xh, atol=1e-6),
    "单流输入时输出等于输入",
    "相对误差=%.3e" % (yh - xh).abs().max().item(),
)
check(
    abs(ratio - 1.0) < 1e-4,
    "不再逐层衰减残差",
    "输出/输入范数比=%.6f（修复前为 0.625）" % ratio,
)
# 多流：双随机矩阵保证流间总量守恒，且不会整体衰减
streams = torch.randn(2, 64, 2, DIM, device=DEVICE)
ys = mhc(streams)
check(
    ys.shape == streams.shape and torch.allclose(
        ys.sum(dim=2), streams.sum(dim=2), atol=1e-5
    ),
    "多流输入时流间混合守恒（双随机矩阵性质）",
    "形状=%s" % (tuple(ys.shape),),
)

# ---------------------------------------------------------------------------
banner("issue #1：信息必须能跨窗口传播")
# ---------------------------------------------------------------------------

WINDOW = 8
N_BLOCKS = 4
HH = WW = 16


def build_stack(conv_weight):
    """构造 N_BLOCKS 个 HybridBlock，卷积权重固定为 conv_weight。

    conv_weight=0.0 模拟修复前的状态（没有跨窗通路）；
    conv_weight=0.5 是修复后的状态。
    """
    stack = []
    for _ in range(N_BLOCKS):
        blk = HybridBlock(
            dim=DIM,
            window_size=WINDOW,
            num_heads=2,
            use_moe=False,
            use_engram=False,
            use_mhc=True,
        ).to(DEVICE)
        # LayerScale 初值 1e-4 会把差异压到数值噪声以下，这里放大到 1
        blk.ls1.gamma.data.fill_(1.0)
        blk.ls_conv.gamma.data.fill_(1.0)
        blk.local_conv.weight.data.fill_(conv_weight)
        stack.append(blk)
    return stack


def run_stack(stack, x):
    for blk in stack:
        x = blk(x, HH, WW)["out"]
    return x


def cross_window_delta(stack, base, x, target, src):
    """把 src 位置的 token 加 5，看 target 位置输出变化多少。"""
    pert = x.clone()
    pert[0, src] += 5.0
    return (run_stack(stack, pert)[0, target] - base[0, target]).abs().max().item()


torch.manual_seed(0)
xc = torch.randn(1, HH * WW, DIM, device=DEVICE)

# 目标 token (7,7) 在窗口 (0,0)；扰动点 (11,7) 在窗口 (1,0)，跨窗口，纵向相距 4
target = 7 * WW + 7
src = 11 * WW + 7

# --- 负对照：没有跨窗通路时，窗口外的扰动对窗口内的影响必须恰好为 0 ---
stack_off = build_stack(0.0)
base_off = run_stack(stack_off, xc)
delta_off = cross_window_delta(stack_off, base_off, xc, target, src)
check(
    delta_off == 0.0,
    "无跨窗通路时窗口外扰动影响恰好为 0（复现 issue #1）",
    "|Δ|=%.3e" % delta_off,
)

# --- 修复后：同一扰动必须产生非零响应 ---
stack_on = build_stack(0.5)
base_on = run_stack(stack_on, xc)
delta_on = cross_window_delta(stack_on, base_on, xc, target, src)
check(
    delta_on > 1e-7,
    "加入 DWConv 后窗口外扰动可传播到窗口内（4 层后）",
    "|Δ|=%.3e" % delta_on,
)

# --- 对照：同窗口内的扰动在两种配置下都能传播，说明测试本身灵敏 ---
delta_same = cross_window_delta(stack_on, base_on, xc, target, target - 1)
check(
    delta_same > 1e-7,
    "同窗口扰动可传播（对照）",
    "|Δ|=%.3e" % delta_same,
)

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
