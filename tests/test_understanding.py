"""
理解层结构优化验证。

针对四个结构缺陷逐一验证是否真正修复：
  1. 空间信息丢失  → 验证 strategy 具有空间分辨力
  2. 策略被平均    → 验证输出是逐 token 而非全局向量
  3. 分支互不交互  → 验证分支间确实互相影响
  4. UI 掩码无感受野 → 验证感受野 > 1 像素
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from _device import DEVICE, device_desc  # noqa: E402
from sr.backbone import UnderstandingLayer, MultiScaleContext, CrossBranchInteraction  # noqa: E402

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
und = UnderstandingLayer(in_channels=3, dim=64).to(DEVICE)
und.eval()

# 输入一张有空间结构的图：左半亮右半暗
H, W = 64, 64
x = torch.zeros(1, 3, H, W, device=DEVICE)
x[:, :, :, : W // 2] = 1.0

# ---------------------------------------------------------------------------
banner("缺陷 1：空间信息丢失")
# ---------------------------------------------------------------------------

with torch.no_grad():
    out = und(x)

strategy = out["strategy"]  # (B, N, C)
b, n, c = strategy.shape

# 检查 strategy 在空间上是否有差异（若被全局池化广播，各 token 会高度相似）
tok_std = strategy.std(dim=1).mean().item()      # 跨 token 的标准差
tok_mean_abs = strategy.abs().mean().item()
spatial_variation = tok_std / (tok_mean_abs + 1e-8)

check(
    spatial_variation > 0.1,
    "strategy 具有空间分辨力",
    "跨 token 变异系数 = %.3f（越大越有空间差异）" % spatial_variation,
)

# 逐位置比较：左半与右半的 strategy 应有明显差异
half = n // 2
left = strategy[:, :half].mean(dim=1)
right = strategy[:, half:].mean(dim=1)
diff = (left - right).norm().item()
scale = (left.norm() + right.norm()).item() / 2
check(
    diff / (scale + 1e-8) > 0.01,
    "左右半区 strategy 可区分",
    "相对差异 = %.4f" % (diff / (scale + 1e-8)),
)

# 退化程度图也应有空间分布
deg = out["degradation_map"]  # (B, N, 1)
check(
    deg.std().item() > 1e-4,
    "退化程度图有空间分布",
    "标准差 = %.6f" % deg.std().item(),
)

# ---------------------------------------------------------------------------
banner("缺陷 2：策略被平均")
# ---------------------------------------------------------------------------

check(
    strategy.dim() == 3 and strategy.shape[1] == n,
    "strategy 为逐 token 形状",
    "%s（token 数 %d）" % (tuple(strategy.shape), n),
)

# 若被 mean(dim=1)，所有 token 应完全相同
identical = torch.allclose(strategy[:, 0], strategy[:, n // 2], atol=1e-6)
check(not identical, "strategy 未被全局平均（各 token 不同）")

# ---------------------------------------------------------------------------
banner("缺陷 3：分支互不交互")
# ---------------------------------------------------------------------------

inter = CrossBranchInteraction(dim=64, num_branches=4).to(DEVICE)
inter.eval()

# 构造 4 个分支输入
br = [torch.randn(1, n, 64, device=DEVICE) for _ in range(4)]
# 只扰动第 1 个分支，看其他分支是否被影响
br_perturbed = [br[0] + 1.0, br[1], br[2], br[3]]

with torch.no_grad():
    out_base = inter(br)
    out_pert = inter(br_perturbed)

# 第 2/3/4 分支的输出应该因第 1 分支的扰动而变化
changed = []
for i in (1, 2, 3):
    d = (out_base[i] - out_pert[i]).abs().mean().item()
    changed.append(d)

check(
    all(d > 1e-6 for d in changed),
    "分支间确实互相影响",
    "分支 2/3/4 的输出变化量 = %s" % ["%.2e" % d for d in changed],
)

# 参数开销检查
# 正确的对照基准是"朴素全混"：把 4 个分支拼成 4C 后用全连接混合，
# 参数量为 (4C)^2。这才是"实现分支交互"的朴素做法。
n_inter = sum(p.numel() for p in inter.parameters())
n_naive_full = (4 * 64) ** 2  # 朴素全混
n_independent = 4 * 64 * 64   # 完全独立（无交互，作为参考）
check(
    n_inter < n_naive_full * 0.5,
    "交叉交互为低秩，显著省于朴素全混",
    "%s 参数（朴素全混 %s，省 %.1f 倍；完全独立参考 %s）"
    % (format(n_inter, ","), format(n_naive_full, ","),
       n_naive_full / max(1, n_inter), format(n_independent, ",")),
)

# ---------------------------------------------------------------------------
banner("缺陷 4：UI 掩码无感受野")
# ---------------------------------------------------------------------------

# 正确检验方式：放一个孤立亮点，检查掩码响应是否在空间上【扩散】。
# 若感受野为 1x1，响应只会出现在亮点对应的单个像素上；
# 若感受野 > 1，响应会扩散到邻域。

x_dot = torch.zeros(1, 3, H, W, device=DEVICE)
x_dot[:, :, H // 2, W // 2] = 1.0

with torch.no_grad():
    mask_dot = und(x_dot)["ui_mask"]  # (B, 1, h, w)
    mask_base = und(torch.zeros(1, 3, H, W, device=DEVICE))["ui_mask"]

# 用"相对基线的响应"衡量，避免未训练时输出偏移的干扰
resp = (mask_dot - mask_base)[0, 0]  # (h, w)
ch, cw = resp.shape[0] // 2, resp.shape[1] // 2

center = resp[ch, cw].abs().item()
# 取 3x3 邻域（不含中心）
neigh = []
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
            continue
        neigh.append(resp[ch + dy, cw + dx].abs().item())
max_neigh = max(neigh)

check(
    center > 0,
    "亮点处有响应",
    "中心响应 = %.3e" % center,
)
check(
    max_neigh > 0,
    "响应扩散到邻域（感受野 > 1 像素）",
    "邻域最大响应 = %.3e，中心/邻域比 = %.1f"
    % (max_neigh, center / (max_neigh + 1e-12)),
)

# 进一步：统计响应非零的像素数。感受野为 1 时应只有 1 个像素。
n_active = (resp.abs() > 1e-9).sum().item()
check(
    n_active > 1,
    "响应覆盖多个像素（确证有感受野）",
    "%d 个像素有响应" % n_active,
)

# 说明：未训练时掩码整体接近 0.5 属正常（末层权重为标准初始化），
# 关键指标是"是否对输入有空间响应"，而非响应幅度。
print()
print("  说明：未训练状态下掩码整体接近 0.5 属正常现象")
print("        （末层权重为标准初始化，输出接近 0，经 sigmoid 后为 0.5）。")
print("        此处检验的是【空间响应能力】，不是响应幅度。")

# ---------------------------------------------------------------------------
banner("参数量与规模")
# ---------------------------------------------------------------------------

n_und = sum(p.numel() for p in und.parameters())
print("  理解层参数量：%s" % format(n_und, ","))
print()
print("  子模块明细：")
for name, mod in und.named_children():
    cnt = sum(p.numel() for p in mod.parameters())
    if cnt > 0:
        print("    %-28s %10s" % (name, format(cnt, ",")))

# ---------------------------------------------------------------------------
banner("前向数值健全性")
# ---------------------------------------------------------------------------

with torch.no_grad():
    o = und(x)

check(torch.isfinite(o["strategy"]).all().item(), "strategy 无 NaN/Inf")
check(torch.isfinite(o["degradation_map"]).all().item(), "退化图无 NaN/Inf")
check(torch.isfinite(o["ui_mask"]).all().item(), "UI 掩码无 NaN/Inf")
check(torch.isfinite(o["scene_logits"]).all().item(), "场景 logits 无 NaN/Inf")
check(
    0.0 <= o["ui_mask"].min().item() and o["ui_mask"].max().item() <= 1.0,
    "UI 掩码在 0~1 范围（已过 sigmoid）",
    "[%.4f, %.4f]" % (o["ui_mask"].min().item(), o["ui_mask"].max().item()),
)

# 梯度可回传
und.train()
xt = torch.randn(1, 3, 32, 32, requires_grad=True, device=DEVICE)
ot = und(xt)
loss = ot["strategy"].sum() + ot["ui_mask"].sum() + ot["degradation_map"].sum()
loss.backward()
n_grad = sum(
    1 for p in und.parameters() if p.grad is not None and p.grad.abs().sum() > 0
)
n_all = sum(1 for p in und.parameters() if p.requires_grad)
check(n_grad > 0, "理解层梯度可回传", "%d/%d 参数有梯度" % (n_grad, n_all))

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
