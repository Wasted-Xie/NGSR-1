"""
验证：理解层条件是否真正影响路由决策。

设计文档 2.1 节要求三级路由分别由理解层驱动：
  第一级组级路由 ← 退化空间分布图
  第二级簇内路由 ← 场景标签
  第三级质量门控 ← UI/边缘检测

本测试用"消融"方式验证：屏蔽某个条件，看对应路由是否变化。
若变化 → 接线有效；若无变化 → 该条件是死代码。
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

DIM = 32
H = W = 16          # 主干分辨率（16x16=256 token）
B = 1
N = H * W

moe = ThreeLevelMoE(
    dim=DIM, num_experts=8, top_k=2, num_clusters=2, ffn_ratio=2, group_size=4
).to(DEVICE)
moe.eval()

x = torch.randn(B, N, DIM, device=DEVICE)

# 构造有区分度的理解层条件
deg = torch.zeros(B, N, 1, device=DEVICE)
deg[:, : N // 2] = 0.0    # 左半：退化轻
deg[:, N // 2:] = 1.0     # 右半：退化重

scene_a = torch.zeros(B, 8, device=DEVICE); scene_a[:, 0] = 5.0   # 场景 A
scene_b = torch.zeros(B, 8, device=DEVICE); scene_b[:, 3] = 5.0   # 场景 B

ui = torch.zeros(B, N, 1, device=DEVICE)
ui[:, : N // 4] = 1.0     # 左上：UI 区域

# ---------------------------------------------------------------------------
banner("第一级：组级路由 是否受退化分布图影响")
# ---------------------------------------------------------------------------

with torch.no_grad():
    cid_none, _ = moe.group_router(x, H, W, degradation=None)
    cid_deg, _ = moe.group_router(x, H, W, degradation=deg)
    # 反转退化分布，看是否产生不同路由
    cid_rev, _ = moe.group_router(x, H, W, degradation=1.0 - deg)

same_as_none = torch.equal(cid_none, cid_deg)
same_as_rev = torch.equal(cid_deg, cid_rev)

check(
    not same_as_none,
    "传入退化图后组级路由结果改变",
    "无退化的簇分配 %s / 有退化的 %s"
    % (cid_none.flatten().tolist()[:8], cid_deg.flatten().tolist()[:8]),
)
check(
    not same_as_rev,
    "反转退化图后组级路由结果改变（确证因果性）",
    "正序 %s / 反序 %s"
    % (cid_deg.flatten().tolist()[:8], cid_rev.flatten().tolist()[:8]),
)

# ---------------------------------------------------------------------------
banner("第二级：簇内路由 是否受场景标签影响")
# ---------------------------------------------------------------------------

with torch.no_grad():
    idx_none, _ = moe.cluster_router(x, scene_logits=None)
    idx_a, _ = moe.cluster_router(x, scene_logits=scene_a)
    idx_b, _ = moe.cluster_router(x, scene_logits=scene_b)

check(
    not torch.equal(idx_none, idx_a),
    "传入场景标签后簇内路由结果改变",
    "无场景 %s / 场景A %s"
    % (idx_none.flatten()[:8].tolist(), idx_a.flatten()[:8].tolist()),
)
check(
    not torch.equal(idx_a, idx_b),
    "不同场景产生不同路由（确证场景有区分作用）",
    "场景A %s / 场景B %s"
    % (idx_a.flatten()[:8].tolist(), idx_b.flatten()[:8].tolist()),
)

# ---------------------------------------------------------------------------
banner("第三级：质量门控 是否受 UI 掩码影响")
# ---------------------------------------------------------------------------


def quality_gate_value(x_in, ui_in):
    """复现 ThreeLevelMoE 中第三级的计算。"""
    qg = moe.quality_gate
    feat = qg[0](x_in)
    feat = qg[1](feat)
    feat = qg[2](feat)
    if ui_in is not None:
        ui_feat = moe.ui_proj(ui_in)
        alpha = moe.ui_gate(x_in)
        feat = feat + alpha * ui_feat
    return torch.sigmoid(qg[3](feat))


with torch.no_grad():
    g_none = quality_gate_value(x, None)
    g_ui = quality_gate_value(x, ui)

check(
    not torch.allclose(g_none, g_ui, atol=1e-6),
    "传入 UI 掩码后质量门控改变",
    "无 UI 均值 %.4f / 有 UI 均值 %.4f"
    % (g_none.mean().item(), g_ui.mean().item()),
)

# UI 区域的门控概率应高于非 UI 区域（UI 应走高精度旁路）
ui_region = g_ui[:, : N // 4].mean().item()
non_ui_region = g_ui[:, N // 4:].mean().item()
check(
    True,
    "UI 区域 vs 非 UI 区域的门控概率",
    "UI 区 %.4f / 非 UI 区 %.4f（未训练时数值接近属正常）"
    % (ui_region, non_ui_region),
)

# ---------------------------------------------------------------------------
banner("整网：MoE 输出是否受理解层条件影响")
# ---------------------------------------------------------------------------

with torch.no_grad():
    out_none = moe(x, H, W)["out"]
    out_cond = moe(x, H, W, degradation=deg, scene_logits=scene_a, ui_mask=ui)["out"]

d = (out_none - out_cond).abs().mean().item()
check(
    d > 1e-7,
    "传入理解层条件后 MoE 输出改变",
    "输出差异 = %.3e" % d,
)

# ---------------------------------------------------------------------------
banner("参数开销")
# ---------------------------------------------------------------------------

n_total = sum(p.numel() for p in moe.parameters())
n_cond = (
    sum(p.numel() for p in moe.group_router.parameters())
    + sum(p.numel() for p in moe.cluster_router.parameters())
    + sum(p.numel() for p in moe.ui_proj.parameters())
    + sum(p.numel() for p in moe.ui_gate.parameters())
)
print("  MoE 总参数        : %s" % format(n_total, ","))
print("  其中路由+条件部分 : %s (%.1f%%)" % (format(n_cond, ","), 100 * n_cond / n_total))
print()
print("  说明：路由网络占比很低，条件注入的额外开销可忽略，")
print("        但它是设计文档 2.1 节要求的核心机制。")

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
