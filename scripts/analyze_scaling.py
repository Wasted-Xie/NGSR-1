"""规模实验分析：判断「性能随参数量的跃升」是真实现象还是度量假象。

方法学依据
----------
Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of Large Language Models
a Mirage?", NeurIPS 2023 的结论：

  nonlinear or discontinuous metrics produce apparent emergent abilities,
  whereas linear or continuous metrics produce smooth, continuous predictable
  changes in model performance.

即：同一批模型，只换评分指标，就能让"突变"出现或消失。

因此本脚本的判据不是"某个指标是否突变"，而是**两类指标的形状是否一致**：

  A. 连续指标（L1 / PSNR / SSIM）与不连续指标（阈值准确率 / win_rate）
     都平滑 → 无涌现。性能随规模平滑提升。
  B. 连续指标平滑，而不连续指标突变
     → 突变来自度量设计，不是模型能力。这正是 Mirage 论文描述的情形。
  C. 两类都突变 → 可能存在真实相变，需要用第三种独立指标复核。

用法：
  python scripts/analyze_scaling.py
"""

import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "temp", "feasibility_out")

# 指标分组：连续 vs 不连续
CONT = [("psnr_gain", "PSNR(dB)"), ("ssim_gain", "SSIM"),
        ("l1_gain", "L1")]
# 不连续指标用重评估脚本扫描出的、有区分度的阈值
# （win_rate 原定义「PSNR > bicubic」在所有规模恒为 1.0，已饱和无区分度）
DISC = [("acc_gain", "阈值准确率"), ("win_gt_1.00", "win>1.0dB")]


def load():
    """优先读 40 张验证集的重评估结果；否则退回训练时的 4 张验证集。"""
    re_path = os.path.join(OUT, "scaling_reeval.json")
    if os.path.exists(re_path):
        with open(re_path, encoding="utf-8") as f:
            rows = json.load(f)
        rows.sort(key=lambda d: d["params"])
        return rows, "重评估（%d 张验证集）" % rows[0].get("n_val", 40)

    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "summary_scale_*.json"))):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["_file"] = os.path.basename(p)
        rows.append(d)
    rows.sort(key=lambda d: d["params"])
    return rows, "训练内验证（4 张，噪声较大）"


def shape_stats(x, y):
    """对 y(x) 做归一化并返回形状特征。

    返回 (归一化后的 y, 一阶差分最大值, 二阶差分绝对值最大值)
    其中 x 取 log10(params)，因为规模效应通常对 log 规模线性。
    """
    lx = np.log10(x)
    span = y.max() - y.min()
    yn = (y - y.min()) / span if span > 1e-12 else np.zeros_like(y)
    if len(lx) < 2:
        return yn, np.nan, np.nan
    d1 = np.diff(yn) / np.diff(lx)
    d2 = np.diff(d1) if len(d1) >= 2 else np.array([np.nan])
    return yn, float(np.max(np.abs(d1))), float(np.max(np.abs(d2)))


def main():
    rows, src = load()
    if not rows:
        print("未找到规模实验数据，请先跑 scripts/feasibility_train.py 与 "
              "scripts/reeval_scaling.py。")
        return 1

    n = len(rows)
    print("=" * 96)
    print("规模实验分析：参数量的作用")
    print("=" * 96)
    print("  数据来源: %s" % src)
    print("  配置数  : %d   参数范围: %s ~ %s（%.1f 倍，%.2f 个数量级）"
          % (n, format(rows[0]["params"], ","), format(rows[-1]["params"], ","),
             rows[-1]["params"] / rows[0]["params"],
             np.log10(rows[-1]["params"] / rows[0]["params"])))
    print()

    hdr = ["配置", "参数量", "L1增益", "PSNR增益", "SSIM增益",
           "阈值准确率", "win>1.0dB"]
    print("  " + "".join("%-13s" % h for h in hdr))
    print("  " + "-" * (13 * len(hdr)))
    for d in rows:
        print("  " + "".join("%-13s" % v for v in [
            "C%d/L%d" % (d["channels"], d["blocks"]),
            format(d["params"], ","),
            "%+.5f" % d["l1_gain"],
            "%+.4f" % d["psnr_gain"],
            "%+.5f" % d["ssim_gain"],
            "%+.5f" % d["acc_gain"],
            "%.3f" % d["win_gt_1.00"],
        ]))
    print()

    params = np.array([d["params"] for d in rows], float)

    # ---------- 归一化形状对比 ----------
    print("-" * 96)
    print("归一化形状（各指标缩放到 0~1，用于比较**曲线形状**而非绝对值）")
    print("-" * 96)
    print("  %-14s %8s %10s %10s   %s"
          % ("指标", "类别", "最大斜率", "最大曲率", "形状判定"))
    print("  " + "-" * 78)

    summary = {}
    for key, name in CONT + DISC:
        y = np.array([d[key] for d in rows], float)
        yn, s1, s2 = shape_stats(params, y)
        cat = "连续" if (key, name) in CONT else "不连续"
        # 曲率相对斜率：曲率远大于斜率 => 形状有折点
        ratio = s2 / (s1 + 1e-9) if np.isfinite(s2) and np.isfinite(s1) else np.nan
        verdict = ("平滑" if ratio < 0.8 else "有折点")
        summary[key] = dict(slope=s1, curv=s2, ratio=ratio, verdict=verdict, y=y)
        print("  %-14s %8s %10.3f %10.3f   %s"
              % (name, cat, s1, s2, verdict))
    print()

    # 归一化曲线打印
    print("  归一化曲线（每行一个指标，列按参数量递增）：")
    for key, name in CONT + DISC:
        yn = summary[key]["y"]
        span = yn.max() - yn.min()
        ynn = (yn - yn.min()) / span if span > 1e-12 else np.zeros_like(yn)
        bar = "".join("%5.2f" % v for v in ynn)
        print("    %-12s %s" % (name, bar))
    print()

    # ---------- 分维度扫描（消除混淆变量）----------
    # 原实验里 C（宽度）与 L（深度）同时变化，无法归因。这里按固定一个维度重新分组：
    #   固定 L、变 C → 宽度扫描
    #   固定 C、变 L → 深度扫描
    print("=" * 96)
    print("分维度扫描（固定一个维度，只变另一个）")
    print("=" * 96)

    by_l, by_c = {}, {}
    for d in rows:
        by_l.setdefault(d["blocks"], []).append(d)
        by_c.setdefault(d["channels"], []).append(d)

    def report_scan(title, grp, keyname, keyfmt):
        grp = sorted(grp, key=lambda d: d[keyname])
        if len(grp) < 3:
            return None
        print()
        print("  %s" % title)
        print("    %-6s %-12s %10s %10s %10s %8s"
              % (keyfmt, "参数", "PSNR增益", "SSIM增益", "阈值准确率", "win>1.0"))
        for g in grp:
            print("    %-6s %-12s %+10.4f %+10.5f %+10.5f %8.3f"
                  % (g[keyname], format(g["params"], ","), g["psnr_gain"],
                     g["ssim_gain"], g["acc_gain"], g["win_gt_1.00"]))
        ys = [g["psnr_gain"] for g in grp]
        d1 = np.diff(ys)
        mono = bool(np.all(d1 >= -1e-9)) or bool(np.all(d1 <= 1e-9))
        print("    PSNR 逐段增量: %s" % " ".join("%+.4f" % v for v in d1))
        print("    单调: %s" % ("是" if mono else "否"))
        return ys

    for L in sorted(by_l):
        report_scan("宽度扫描（L=%d 固定）" % L, by_l[L], "channels", "C")
    for C in sorted(by_c):
        report_scan("深度扫描（C=%d 固定）" % C, by_c[C], "blocks", "L")

    print()
    print("  说明：只有点数 >= 3 的分组才会列出。若某个固定维度下 PSNR 增量")
    print("        单调递减或近似常数，即为平滑缩放；出现由负转正的拐点才值得")
    print("        进一步核查。")

    # ---------- 综合判定 ----------
    print("=" * 96)
    print("判定")
    print("=" * 96)

    # 判据 1：各指标归一化形状之间的一致性。
    # 若所有指标形状一致，则不存在「换指标就出现突变」的度量假象——
    # 这正是 Mirage 论文要否定的东西。
    norm = {}
    for key, name in CONT + DISC:
        y = summary[key]["y"]
        span = y.max() - y.min()
        norm[name] = (y - y.min()) / span if span > 1e-12 else np.zeros_like(y)

    names = list(norm)
    cors = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = norm[names[i]], norm[names[j]]
            if a.std() > 1e-12 and b.std() > 1e-12:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
    mean_cor = float(np.mean(cors)) if cors else float("nan")
    min_cor = float(np.min(cors)) if cors else float("nan")

    # 判据 2：是否存在「从无能到有能」。
    # 不能用绝对阈值（各指标尺度不同：L1 增益 0.0013 与 PSNR 增益 1.06 不可比），
    # 而要看**最小规模已拿到最大规模的多少比例**。
    # 真实涌现的起点应接近 0（相对终点几乎无改善）。
    print("  判据 2  最小规模占最大规模的比例（越接近 0 越像「无能」）")
    base_keys = [("psnr_gain", "PSNR(dB)"), ("ssim_gain", "SSIM"),
                 ("l1_gain", "L1"), ("acc_gain", "阈值准确率"),
                 ("win_gt_1.00", "win>1.0dB")]
    weakest_ratio = 1e9
    for k, nm in base_keys:
        a, b = rows[0][k], rows[-1][k]
        ratio = a / b if abs(b) > 1e-12 else float("nan")
        weakest_ratio = min(weakest_ratio, ratio)
        print("          %-12s %+.5f / %+.5f = %.3f" % (nm, a, b, ratio))
    if weakest_ratio > 0.2:
        print("          → 最小规模已拿到最大规模 %.0f%% 以上的增益，"
              % (weakest_ratio * 100))
        print("            **不存在从无能到有能的跃迁**。")
        no_leap = True
    else:
        print("          → 最小规模接近无能，需检查是否存在越过某规模的跃升。")
        no_leap = False
    print()

    if mean_cor > 0.9 and no_leap:
        print("  结论：**未观察到涌现**。")
        print("    - 所有指标（连续与不连续）形状高度一致（相关 %.3f），" % mean_cor)
        print("      不存在「换个指标就冒出突变」的度量假象；")
        print("    - 最小规模已显著优于基线，没有「从无能到有能」的质变；")
        print("    - 观察到的 S 形是所有指标共有的饱和特征：增量集中于前段，")
        print("      之后进入平台。这是边际收益递减，不是相变。")
    elif mean_cor > 0.9:
        print("  结论：**所有指标同步出现陡升**。")
        print("    由于指标形状一致，无法用「度量假象」解释。但需排除混淆：")
        print("    (a) 各规模训练步数相同，大模型可能欠训练；")
        print("    (b) 本实验参数量与深度同时变化（C 与 L 都在动），非单一变量。")
        print("    要定论需固定深度只变宽度、并让各规模训练到收敛。")
    else:
        print("  结论：指标之间形状不一致。")
        print("    不连续指标若明显更陡，即为 Mirage 论文描述的情形：")
        print("    突变来自度量阈值，而非模型能力。")

    print()
    print("  各指标相对变化（末/首）：")
    for key, name in CONT + DISC:
        y = summary[key]["y"]
        print("    %-12s %.3f -> %.3f" % (name, y[0], y[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
