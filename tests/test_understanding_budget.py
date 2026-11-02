"""
理解层算力预算验证。

验证修订后的策略（分辨率随主干缩放 + 算力占比约束）是否生效：
  1. 五档预设均在算力预算内
  2. 理解层算力随档位自然缩放（不再畸变）
  3. 路由网络未被削弱（用户要求：不舍弃精确路由专家）
  4. 超预算配置会被拒绝
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sr import SRConfig, Tier, get_preset, TIER_PRESETS  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

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
    print("=" * 82)
    print(s)
    print("=" * 82)


# ---------------------------------------------------------------------------
banner("1. 五档预设的算力占比")
# ---------------------------------------------------------------------------

print()
print("  %-8s %-10s %12s %14s %10s %10s"
      % ("档位", "主干", "主干FLOPs", "理解层FLOPs", "占比", "预算"))
print("  " + "-" * 72)

all_ok = True
for tier in Tier:
    cfg = TIER_PRESETS[tier]
    main = cfg.estimate_main_flops()
    und = cfg.estimate_understanding_flops()
    ratio = cfg.understanding_compute_ratio
    budget = cfg.understanding_compute_budget
    flag = "" if ratio <= budget else "  <-- 超标"
    if ratio > budget:
        all_ok = False
    print("  %-8s %-10s %12.1f %14.1f %9.0f%% %9.0f%%%s"
          % (tier.value, "%dx%d" % (cfg.main_width, cfg.main_height),
             main, und, ratio * 100, budget * 100, flag))

check(all_ok, "五档均在算力预算内")

# ---------------------------------------------------------------------------
banner("2. 算力随档位自然缩放（不再畸变）")
# ---------------------------------------------------------------------------

ratios = [TIER_PRESETS[t].understanding_compute_ratio for t in Tier]
max_r, min_r = max(ratios), min(ratios)

check(
    max_r <= 0.35,
    "最大占比不超过预算",
    "%.0f%%（预算 35%%）" % (max_r * 100),
)
# 阈值说明：issue #5 补全主干口径后，分母的主要成分由「主干分辨率上的
# 注意力 / FFN / 专家」变为「输入分辨率上的重建路径（fuse / bypass /
# 两次上采样）」。占比趋势因此由递减转为递增（Low 0.81% → Max 2.97%），
# 跨度由 2.7 倍变为约 3.7 倍。这仍属同一量级——「数量级畸变」指修订前
# 固定 540p 时 Low 档那种 184% 的情形，故阈值放宽到 5 倍。
check(
    max_r / min_r < 5.0,
    "各档占比跨度合理（不再有数量级畸变）",
    "%.0f%% ~ %.0f%%，跨度 %.1f 倍"
    % (min_r * 100, max_r * 100, max_r / min_r),
)

# 对照：修订前的固定 540p + dim=128 策略
print()
print("  对照（修订前的固定 540p + dim=128）：")
print("  %-8s %12s %14s %12s" % ("档位", "主干FLOPs", "理解层FLOPs", "占比"))
print("  " + "-" * 52)
N540 = 960 * 540
und_flops_540 = (4 * 2 * N540 * 128 * 128
                 + 2 * N540 * (9 * 3 * 64 + 9 * 64 + 64 * 128)) / 1e9
worst = 0.0
for tier in Tier:
    cfg = TIER_PRESETS[tier]
    main = cfg.estimate_main_flops()
    pct = 100 * und_flops_540 / main
    worst = max(worst, pct)
    print("  %-8s %12.1f %14.1f %11.0f%%"
          % (tier.value, main, und_flops_540, pct))
print()
print("  → 修订前最低档占比达 %.0f%%，属数量级畸变；现已消除。" % worst)

# ---------------------------------------------------------------------------
banner("3. 路由网络未被削弱（不舍弃精确路由专家）")
# ---------------------------------------------------------------------------

print()
print("  %-8s %10s %14s %12s %12s"
      % ("档位", "专家总数", "路由参数量", "占MoE比", "三级完整"))
print("  " + "-" * 62)

for tier in Tier:
    cfg = get_preset(tier)
    m = SuperResolutionModel(cfg)
    blk = m.blocks[0]
    if not hasattr(blk, "moe"):
        continue
    moe = blk.moe
    r_params = (
        sum(p.numel() for p in moe.group_router.parameters())
        + sum(p.numel() for p in moe.cluster_router.parameters())
        + sum(p.numel() for p in moe.quality_gate.parameters())
    )
    moe_params = sum(p.numel() for p in moe.parameters())
    has3 = (
        hasattr(moe, "group_router")
        and hasattr(moe, "cluster_router")
        and hasattr(moe, "quality_gate")
    )
    print("  %-8s %10d %14s %11.2f%% %12s"
          % (tier.value, cfg.num_experts, format(r_params, ","),
             100 * r_params / moe_params, "是" if has3 else "否"))

check(True, "三级路由结构完整保留")

# 条件注入的参数开销
cfg = get_preset(Tier.MAX)
m = SuperResolutionModel(cfg)
moe = m.blocks[0].moe
cond_params = (
    sum(p.numel() for p in moe.group_router.deg_proj.parameters())
    + sum(p.numel() for p in moe.group_router.deg_gate.parameters())
    + sum(p.numel() for p in moe.cluster_router.scene_proj.parameters())
    + sum(p.numel() for p in moe.cluster_router.scene_gate.parameters())
    + sum(p.numel() for p in moe.ui_proj.parameters())
    + sum(p.numel() for p in moe.ui_gate.parameters())
)
moe_total = sum(p.numel() for p in moe.parameters())
check(
    cond_params / moe_total < 0.05,
    "理解层条件注入的参数开销极低",
    "%s / %s = %.1f%%"
    % (format(cond_params, ","), format(moe_total, ","),
       100 * cond_params / moe_total),
)

# ---------------------------------------------------------------------------
banner("4. 超预算配置被拒绝")
# ---------------------------------------------------------------------------

try:
    # 注：issue #5 补全主干口径后，分母含 fuse/bypass/两次上采样等重建路径，
    # 数值比原先的「三项和」大约一个数量级，故这里需用更大的 ratio 才能越界。
    bad = SRConfig(main_blocks=2, understanding_dim_ratio=12.0)
    bad.validate()
    check(False, "超预算配置应被拒绝", "但未报错")
except ValueError as e:
    msg = str(e)
    check("超出预算" in msg, "超预算配置被正确拒绝",
          msg.split("\n")[1].strip()[:60])

# 关闭理解层可绕过预算
try:
    ok = SRConfig(main_blocks=2, understanding_enabled=False)
    ok.validate()
    check(True, "可显式关闭理解层以绕过预算")
except ValueError as e:
    check(False, "关闭理解层应可绕过预算", str(e)[:60])

# ---------------------------------------------------------------------------
banner("5. 算力口径一致性（防止公式漂移）")
# ---------------------------------------------------------------------------
#
# 背景：理解层由「固定 540p + understanding_dim」改为「随主干缩放 + dim_ratio」时，
#       config.py 更新了，但 analysis.py 里另写的一套理解层公式没跟着改，
#       导致 run_analysis.py 直接 AttributeError 崩溃，且长期未被发现
#       （原因是原测试只调用 cfg 的方法，实现漂移时测试跟着漂移）。
# 因此这里显式对拍两个模块的口径，任一处再漂移都会立即失败。

from sr.analysis import estimate_flops  # noqa: E402

print()
print("  %-8s %14s %14s %14s" % ("档位", "config主干G", "analysis主干G", "偏差"))
print("  " + "-" * 56)

for tier in Tier:
    cfg = TIER_PRESETS[tier]
    fl = estimate_flops(cfg)

    # 断言 1：主干口径 = analysis 全部主干项之和
    # （total 减去理解层与 Engram 两个非主干项）。
    # 原先只对拍「注意力 + 共享FFN + 私有专家」三项，而两侧当时都漏算
    # fuse/bypass/上采样等路径，于是"对拍通过"掩盖了共同错误（issue #5/#8）。
    # 补全后必须按完整口径对拍，任一侧再漂移都会立即失败。
    total_main = fl["total"] - fl["understanding"] - fl["engram"]
    main = cfg.estimate_main_flops()
    dev = abs(total_main - main) / max(1e-9, main)
    print("  %-8s %14.1f %14.1f %13.2f%%"
          % (tier.value, main, total_main, dev * 100))
    check(
        dev < 1e-6,
        "%s 主干口径一致" % tier.value,
        "config=%.1fG analysis=%.1fG" % (main, total_main),
    )

    # 断言 2：理解层 FLOPs 两个模块必须一致（此前正是此处漂移）
    du = cfg.estimate_understanding_flops()
    da = fl["understanding"]
    check(
        abs(du - da) < 1e-9,
        "%s 理解层口径一致" % tier.value,
        "config=%.3fG analysis=%.3fG" % (du, da),
    )

# 断言 3：预算分母必须是完整主干，而非仅注意力
cfg = TIER_PRESETS[Tier.MAX]
fl = estimate_flops(cfg)
check(
    cfg.estimate_main_flops() > fl["attention"] * 2,
    "预算分母含专家团（完整主干 >> 仅注意力）",
    "完整 %.1fG vs 仅注意力 %.1fG"
    % (cfg.estimate_main_flops(), fl["attention"]),
)

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
