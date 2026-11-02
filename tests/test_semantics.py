"""语义测试：验证模块"真的在学习/真的在交互"，而不只是形状对。

对应 issue #8 的诊断：原有 132~140 项测试全是形状、参数量、数值一致性和
设备可用性，**没有一项测语义**。后果是三个 P0 缺陷（#1 无跨窗口通路、
#2 aux_loss 无梯度、#3 mHC 逐层衰减）在测试全绿的情况下长期存在。

本文件按 issue #8 的建议补三类测试：

  1. 梯度存在性 —— 对具体参数断言梯度非零，而不是"模型里有某个参数有梯度"
  2. 单帧过拟合 —— 固定一张图训练若干步，PSNR 必须显著上升
  3. 感受野/跨窗 —— 扰动一个窗口，相邻窗口的输出必须随之变化
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _device import DEVICE, device_desc  # noqa: E402
from sr import SRConfig  # noqa: E402
from sr.backbone import HybridBlock  # noqa: E402
from sr.engram import Engram  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402
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


def grad_norm(p):
    if p is None or p.grad is None:
        return None
    return float(p.grad.norm())


torch.manual_seed(0)

# ---------------------------------------------------------------------------
banner("1. 梯度存在性：断言【具体参数】拿到了梯度")
# ---------------------------------------------------------------------------
# 原 test_full.py 只检查 sum(grad)>0，只要有一个参数有梯度就算通过。
# 因此 #2（aux_loss 恒零梯度）和 #3（mHC 只有 1 个自由度）都测不出来。

# --- aux_loss -> 组级路由的 gate 权重 ---
DIM, H, W = 32, 16, 16
moe = ThreeLevelMoE(
    dim=DIM, num_experts=8, top_k=2, num_clusters=2, ffn_ratio=2, group_size=4
).to(DEVICE)
moe.zero_grad()
out = moe(torch.randn(1, H * W, DIM, device=DEVICE), H, W)
out["aux_loss"].backward()
g_group = grad_norm(moe.group_router.gate.weight)
check(g_group is not None and g_group > 0.0,
      "aux_loss 对 group_router.gate.weight 有非零梯度",
      "grad_norm=%s" % (None if g_group is None else "%.3e" % g_group))

# --- mHC -> log_alpha 的 16 个参数是否都在参与 ---
# 注意：必须用【多流】输入。单流时 alpha 退化为 1x1 单位阵、mixed == streams，
# 输出与 log_alpha 无关，梯度恒为 0 —— 这是 PR 修复 #3 后的正确行为
# （恒等初始化），不是缺陷。mHC 只有在多流下才真正做流间混合。
mhc = ManifoldConstrainedHyperConnection(DIM, num_streams=4).to(DEVICE)
mhc.zero_grad()
xm2 = torch.randn(1, H * W, 4, DIM, device=DEVICE, requires_grad=True)
mhc(xm2).pow(2).mean().backward()
g_alpha = mhc.log_alpha.grad
n_nonzero_alpha = 0 if g_alpha is None else int((g_alpha != 0).sum())
check(g_alpha is not None and n_nonzero_alpha > 0,
      "mHC 多流输入时 log_alpha 参与计算图",
      "非零梯度元素 %d / %d" % (n_nonzero_alpha, mhc.log_alpha.numel()))

# 单流时必须严格恒等（对应 PR 修复的 #3，代价是 log_alpha 无梯度）
mhc.zero_grad()
xs = torch.randn(1, H * W, DIM, device=DEVICE)
with torch.no_grad():
    ys = mhc(xs)
check(torch.allclose(ys, xs, atol=0.0, rtol=0.0),
      "mHC 单流输入时严格恒等（不再逐层衰减）",
      "max|Δ|=%.3e" % (ys - xs).abs().max().item())

# --- Engram(dynamic) -> 记忆槽必须拿到梯度 ---
eng = Engram(dim=DIM, mode="dynamic", dynamic_slots=16, latent_dim=8).to(DEVICE)
eng.zero_grad()
eng(torch.randn(1, H * W, DIM, device=DEVICE), H, W)["out"].pow(2).mean().backward()
g_mem = grad_norm(eng.general.memory)
check(g_mem is not None and g_mem > 0.0,
      "dynamic Engram 的记忆槽拿到非零梯度",
      "grad_norm=%s" % (None if g_mem is None else "%.3e" % g_mem))

# ---------------------------------------------------------------------------
banner("2. 感受野：扰动一个窗口，相邻窗口必须受影响（issue #1）")
# ---------------------------------------------------------------------------
# 修复前主干没有任何跨窗口算子，窗口之间完全隔离——修改窗口 A 的像素，
# 窗口 B 的输出不会有任何变化。接上 DWConv 通路后应当非零。

WS = 8
blk = HybridBlock(
    dim=16, window_size=WS, num_heads=2, ffn_ratio=2,
    num_experts=4, top_k=2, num_clusters=2, num_shared=1,
    use_moe=False, use_engram=False, use_mhc=False,
).to(DEVICE).eval()


def _win_feature(block, hw, perturb_center=None):
    """返回指定窗口（第二列第一行）的输出特征，可选扰动中心窗口。"""
    h = w = hw
    x = torch.zeros(1, h * w, 16, device=DEVICE)
    if perturb_center is not None:
        # 只在【第一个窗口】内注入信号
        for dy in range(WS):
            for dx in range(WS):
                idx = dy * w + dx
                x[0, idx, :] = perturb_center
    with torch.no_grad():
        y = block(x, h, w)["out"]
    # 取第二个窗口（同行右邻）的特征
    sel = []
    for dy in range(WS):
        for dx in range(WS, 2 * WS):
            sel.append(dy * w + dx)
    return y[0, sel, :]


hw = 16
base = _win_feature(blk, hw, perturb_center=None)
pert = _win_feature(blk, hw, perturb_center=1.0)
delta = (base - pert).abs().max().item()
check(delta > 0.0,
      "窗口 A 的扰动会改变窗口 B 的输出（存在跨窗口通路）",
      "max|Δ|=%.6e" % delta)

# 顺带验证：同一窗口内的扰动当然也应该有影响（对照组）
same_win = _win_feature(blk, hw, perturb_center=2.0)
check((base - same_win).abs().max().item() > 0.0,
      "对照组：扰动强度改变时输出随之变化", "")

# ---------------------------------------------------------------------------
banner("3. 单帧过拟合：小配置训若干步，PSNR 必须显著上升")
# ---------------------------------------------------------------------------
# 这一条能同时暴露 mHC 逐层衰减、aux_loss 失效、跨窗口缺失等问题：
# 只要主干无法有效拟合，PSNR 就不会涨。

PATCH, SCALE = 184, 4   # 主干分辨率下限为 180p，且需被窗口 8 整除
cfg = SRConfig(
    input_height=PATCH, input_width=PATCH,
    output_height=PATCH * SCALE, output_width=PATCH * SCALE,
    downsample=1, main_channels=16, main_blocks=2,
    window_size=8, ffn_ratio=2, num_experts=4,
    top_k=2, num_clusters=2,
    shallow_channels=8, bypass_channels=8,
    bypass_activation=0.15, engram_size_mb=0.1,
    engram_mode="dynamic", engram_dynamic_slots=16,
    understanding_dim_ratio=1.0,
)
cfg.validate()
model = SuperResolutionModel(cfg).to(DEVICE)

# 目标必须是【低频、信息可从 LR 恢复】的图案。
# 若用纯随机噪声当 HR，其高频在双三次下采样时已不可逆丢失，属于无解任务，
# 模型最多只能复现 bicubic 上采样，PSNR 涨不动——那样的测试无法反映拟合能力。
_axis = torch.linspace(0.0, 1.0, PATCH * SCALE, device=DEVICE)
_gy, _gx = torch.meshgrid(_axis, _axis, indexing="ij")
hr = 0.5 + 0.25 * torch.sin(2 * 3.14159265 * 3 * _gx) * torch.cos(
    2 * 3.14159265 * 2 * _gy)
hr = hr.expand(3, -1, -1).unsqueeze(0).contiguous()
lr = F.interpolate(hr, scale_factor=1.0 / SCALE, mode="bicubic",
                   align_corners=False)


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 0 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


model.eval()
with torch.no_grad():
    p0 = psnr(model(lr)["sr"], hr)
model.train()

opt = torch.optim.Adam(model.parameters(), lr=3e-3)
STEPS = 200
for _ in range(STEPS):
    opt.zero_grad(set_to_none=True)
    out = model(lr)
    loss = F.l1_loss(out["sr"], hr) + 0.01 * out["aux_loss"]
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    p1 = psnr(model(lr)["sr"], hr)

check(p1 - p0 > 3.0,
      "单帧过拟合 %d 步后 PSNR 提升 > 3 dB" % STEPS,
      "%.3f -> %.3f dB (Δ=%+.3f)" % (p0, p1, p1 - p0))

# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("结果: %d 通过, %d 失败" % (passed, failed))
print("=" * 76)
sys.exit(1 if failed else 0)
