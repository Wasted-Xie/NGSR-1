"""
冒烟测试：验证模型可构建、可前向传播、形状正确。

用法：
    python tests/test_smoke.py

设备：默认有 CUDA 就用 CUDA；SR_TEST_DEVICE=cpu 可强制回退。
"""

import sys
import os
import traceback

# 强制 UTF-8 输出，避免 Windows 控制台中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from _device import DEVICE, device_desc  # noqa: E402
from sr import SRConfig, Tier, get_preset  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

print("计算设备: %s" % device_desc())


def banner(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


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


# ---------------------------------------------------------------------------
banner("1. 配置系统")
# ---------------------------------------------------------------------------

try:
    cfg = SRConfig()
    cfg.validate()
    check(True, "默认配置校验通过")
except Exception as e:
    check(False, "默认配置校验", str(e))

# 五档预设全部校验
for tier in Tier:
    try:
        c = get_preset(tier)
        ok = True
        detail = "主干 %dp, C=%d, 专家=%d, TopK=%d" % (
            c.main_height, c.main_channels, c.num_experts, c.top_k
        )
    except Exception as e:
        ok = False
        detail = str(e)
    check(ok, "档位 %-7s 校验" % tier.value, detail)

# 非法配置应被拒绝
try:
    bad = SRConfig(input_height=720, downsample=8)  # 720/8 = 90p < 180p
    bad.validate()
    check(False, "非法配置应被拒绝", "但未报错")
except ValueError:
    check(True, "非法配置被正确拒绝", "(90p < 180p)")

# ---------------------------------------------------------------------------
banner("2. 最小配置构建与前向")
# ---------------------------------------------------------------------------

# 用一个精简但合理的配置快速验证（避免测试耗时过长）
# 注意：主干层数不要太少，否则主干算力过小，理解层占比会超预算
#      （理解层有固定的 stem 卷积开销，见 config.understanding_compute_budget）
small = SRConfig(
    input_height=360,
    input_width=640,
    output_height=720,
    output_width=1280,
    downsample=2,          # 360/2 = 180p
    main_channels=16,
    main_blocks=4,         # 层数不能太少，否则主干算力过小
    window_size=4,         # 180 不能被 8 整除，用 4
    num_experts=4,
    top_k=2,
    num_clusters=2,
    shallow_channels=8,
    understanding_dim_ratio=1.25,
    engram_size_mb=1,
    bypass_channels=8,
)
try:
    small.validate()
    check(True, "最小配置校验通过", "主干 180p, 窗口 4")
except Exception as e:
    check(False, "最小配置校验", str(e))

try:
    torch.manual_seed(0)
    model = SuperResolutionModel(small).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    check(True, "模型构建成功", "参数量 {:,}".format(n_param))

    model.eval()
    x = torch.randn(1, 3, 360, 640, device=DEVICE)
    with torch.no_grad():
        out = model(x)

    check("sr" in out, "输出含 sr 字段")
    check(
        out["sr"].shape == (1, 3, 720, 1280),
        "输出形状正确",
        "得到 %s" % (tuple(out["sr"].shape),),
    )
    check(
        torch.isfinite(out["sr"]).all().item(),
        "输出无 NaN/Inf",
    )
    check("aux_loss" in out, "输出含 aux_loss", "%.4f" % out["aux_loss"].item())
    if "hit_rate" in out:
        check(True, "输出含 hit_rate", "%.4f" % out["hit_rate"].item())

except Exception as e:
    check(False, "最小配置前向传播", str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
banner("3. 参数量分布")
# ---------------------------------------------------------------------------

try:
    stats = model.count_parameters()
    total = stats.pop("_total")
    print("  总参数量: %s" % format(total, ","))
    print()
    print("  %-24s %12s %8s" % ("模块", "参数量", "占比"))
    print("  " + "-" * 48)
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print("  %-24s %12s %7.2f%%" % (k, format(v, ","), 100.0 * v / total))
    check(True, "参数量统计完成")
except Exception as e:
    check(False, "参数量统计", str(e))

# ---------------------------------------------------------------------------
banner("4. 五档配置的参数量对比")
# ---------------------------------------------------------------------------

print("  %-8s %10s %10s %8s %8s %8s" % (
    "档位", "主干", "通道", "专家", "TopK", "参数量"))
print("  " + "-" * 60)
for tier in Tier:
    c = get_preset(tier)
    try:
        m = SuperResolutionModel(c)
        n = sum(p.numel() for p in m.parameters())
        print("  %-8s %10s %10d %8d %8d %10s" % (
            tier.value, "%dp" % c.main_height, c.main_channels,
            c.num_experts, c.top_k, format(n, ",")))
    except Exception as e:
        print("  %-8s 构建失败: %s" % (tier.value, str(e)[:50]))

check(True, "五档配置构建完成")

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
