"""
五档配置的分析报告生成。

用法：
    python tests/run_analysis.py            # 全部五档
    python tests/run_analysis.py Max        # 指定档位
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sr import Tier, get_preset  # noqa: E402
from sr.analysis import format_report  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402


def main():
    args = sys.argv[1:]
    tiers = [Tier(a) for a in args] if args else list(Tier)

    for tier in tiers:
        cfg = get_preset(tier)
        model = SuperResolutionModel(cfg)
        model.eval()
        print(format_report(cfg, model))

    # 汇总对比表
    print("=" * 74)
    print("五档汇总对比")
    print("=" * 74)
    print()
    print("  %-8s %8s %6s %6s %6s %12s %10s"
          % ("档位", "主干", "通道", "专家", "TopK", "参数量", "单帧G"))
    print("  " + "-" * 66)
    for tier in Tier:
        cfg = get_preset(tier)
        model = SuperResolutionModel(cfg)
        n = sum(p.numel() for p in model.parameters())
        from sr.analysis import estimate_flops
        fl = estimate_flops(cfg)["total"]
        print("  %-8s %8s %6d %6d %6d %12s %10.1f"
              % (tier.value, "%dp" % cfg.main_height, cfg.main_channels,
                 cfg.num_experts, cfg.top_k, format(n, ","), fl))
    print()


if __name__ == "__main__":
    main()
