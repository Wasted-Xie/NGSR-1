"""分析涌现实验：exact_match 的陡升，是模型能力相变，还是乘积放大？

核心检验
--------
多标签分类里，若各标签判断近似独立，则

    exact_match ≈ Π p_i        （p_i = 第 i 个标签的准确率）

所以只要各标签准确率 p_i 平滑上升，乘积 Π p_i 就会呈现出"先平、后陡、再饱和"
的 S 形——**看起来像能力突变，实际上是乘法**。

本脚本做两件事：
  1. 对比「连续指标 mean_acc」与「不连续指标 exact_match」的曲线形状；
  2. 用实测的各标签准确率算出 Π p_i，与实测 exact_match 对照。
     若二者吻合，则 exact_match 不含任何超出各标签准确率的信息，
     "跃升"完全由评分方式决定 —— 即 Schaeffer et al. (NeurIPS 2023)
     所描述的度量假象。

用法：
  python scripts/analyze_emergence.py
"""

import argparse
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
OUT = os.path.join(ROOT, "temp", "emergence_out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="指定 clf_sweep_s*.json；默认取步数最大的一个")
    args = ap.parse_args()

    if args.file:
        p = args.file
    else:
        cands = glob.glob(os.path.join(OUT, "clf_sweep*.json"))
        if not cands:
            print("未找到规模扫描结果，请先跑 emergence_clf.py --scale-sweep")
            return 1
        # 取步数最大的那份
        def steps_of(f):
            base = os.path.basename(f)
            try:
                return int(base.split("_s")[-1].split(".")[0])
            except Exception:
                return 0
        p = max(cands, key=steps_of)
    if not os.path.exists(p):
        print("未找到 %s" % p)
        return 1
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    rows.sort(key=lambda d: d["params"])
    n = len(rows)
    print("  数据源: %s" % os.path.basename(p))

    print("=" * 94)
    print("涌现实验分析：exact_match 陡升的来源")
    print("=" * 94)
    print("  配置数  : %d   参数范围: %s ~ %s（%.0f 倍，%.2f 个数量级）"
          % (n, format(rows[0]["params"], ","),
             format(rows[-1]["params"], ","),
             rows[-1]["params"] / rows[0]["params"],
             np.log10(rows[-1]["params"] / rows[0]["params"])))
    print("  步数    : %d（全部规模相同）" % rows[0]["steps"])
    print()

    print("  %-9s %-11s %10s %12s %10s %12s"
          % ("配置", "参数", "mean_acc", "exact_match", "MAE", "Πp_i（推算）"))
    print("  " + "-" * 74)
    for r in rows:
        prod = float(np.prod(r["per_label"]))
        print("  %-9s %-11s %10.4f %12.4f %10.4f %12.4f"
              % (r["tag"], format(r["params"], ","), r["mean_acc"],
                 r["exact_match"], r["mae"], prod))
    print()

    params = np.array([r["params"] for r in rows], float)
    acc = np.array([r["mean_acc"] for r in rows], float)
    ex = np.array([r["exact_match"] for r in rows], float)
    prod = np.array([float(np.prod(r["per_label"])) for r in rows], float)

    # ---------- 形状对比 ----------
    print("-" * 94)
    print("形状对比（归一化到 0~1）")
    print("-" * 94)
    lx = np.log10(params)

    def norm(y):
        s = y.max() - y.min()
        return (y - y.min()) / s if s > 1e-12 else np.zeros_like(y)

    na, ne = norm(acc), norm(ex)
    print("  %-14s %s" % ("mean_acc", "".join("%6.2f" % v for v in na)))
    print("  %-14s %s" % ("exact_match", "".join("%6.2f" % v for v in ne)))
    print("  %-14s %s" % ("log10(参数)", "".join("%6.2f" % v for v in norm(lx))))
    print()
    print("  逐段斜率（每十倍参数的变化量）")
    d1a = np.diff(na) / np.diff(lx)
    d1e = np.diff(ne) / np.diff(lx)
    print("    mean_acc   : %s" % " ".join("%+.2f" % v for v in d1a))
    print("    exact_match: %s" % " ".join("%+.2f" % v for v in d1e))
    print("    → exact_match 的斜率峰值 / mean_acc 的斜率峰值 = %.2f 倍"
          % (np.max(np.abs(d1e)) / max(1e-9, np.max(np.abs(d1a)))))
    print()

    # ---------- 乘积验证 ----------
    # 注意：必须按**命中数**过滤。exact_match 是 600 个验证样本里的命中比例，
    # 当它只有 0.015 时命中仅 9 个，统计噪声会主导，相关系数被无意义地拉低。
    # 实测就是如此：小模型相对差 60~80%，而大模型仅 1~10%。
    NVAL = 600.0
    hits = ex * NVAL
    print("-" * 94)
    print("乘积验证：实测 exact_match  vs  Π p_i（由各标签准确率推算）")
    print("-" * 94)
    print("  %-9s %8s %12s %12s %11s %8s"
          % ("配置", "命中数", "实测 exact", "Π p_i", "相对差", "纳入"))
    print("  " + "-" * 68)
    ok = hits >= 12
    for r, a, b, h, o in zip(rows, ex, prod, hits, ok):
        print("  %-9s %8.1f %12.4f %12.4f %10.1f%% %8s"
              % (r["tag"], h, a, b, 100 * abs(a - b) / max(a, 1e-6),
                 "是" if o else "否"))
    print()

    if ok.sum() >= 3:
        c_ok = float(np.corrcoef(ex[ok], prod[ok])[0, 1])
        rel_ok = float((np.abs(ex[ok] - prod[ok])
                        / np.maximum(ex[ok], 1e-6)).mean())
        print("  仅统计命中数 >= 12 的 %d 个配置：" % int(ok.sum()))
        print("    相关系数 %.5f   平均相对偏差 %.1f%%" % (c_ok, 100 * rel_ok))
    else:
        c_ok, rel_ok = float("nan"), float("nan")
        print("  命中数 >= 12 的配置不足 3 个，无法做有效统计。")
    c_all = float(np.corrcoef(ex, prod)[0, 1]) if ex.std() > 0 else float("nan")
    print("  （对照）全部 %d 个配置一起算：相关 %.5f —— 被小模型的采样噪声拉低"
          % (n, c_all))
    print()

    # ---------- 判定 ----------
    print("=" * 94)
    print("判定")
    print("=" * 94)
    ratio = float(np.max(np.abs(d1e)) / max(1e-9, np.max(np.abs(d1a))))

    # 先检查单调性。若两个指标在最大规模处同步回落，说明大模型欠训练，
    # 此时任何"规模效应"的结论都不成立（不是评分问题，是训练不足）。
    peak_a, peak_e = int(np.argmax(acc)), int(np.argmax(ex))
    print("  mean_acc    峰值在 %-9s (%.4f)" % (rows[peak_a]["tag"], acc[peak_a]))
    print("  exact_match 峰值在 %-9s (%.4f)" % (rows[peak_e]["tag"], ex[peak_e]))
    fell_a = acc[-1] < acc[peak_a] - 0.005
    fell_e = ex[-1] < ex[peak_e] - 0.002
    if fell_a and fell_e:
        print("  → 两指标在最大规模处**同步回落**（mean_acc %.4f -> %.4f，"
              % (acc[peak_a], acc[-1]))
        print("    exact_match %.4f -> %.4f）。" % (ex[peak_e], ex[-1]))
        print("    这不是评分造成的，而是模型本身变差：固定 %d 步训练下"
              % rows[0]["steps"])
        print("    大模型欠训练。**本次扫描不能用于判定规模是否带来跃升**——")
        print("    必须为每个规模训练到收敛，或给相同的算力预算。")
    print()

    if np.isfinite(c_ok) and c_ok > 0.95 and rel_ok < 0.15:
        print("  乘积验证通过（命中数足够的 %d 个配置：相关 %.5f，偏差 %.1f%%）。"
              % (int(ok.sum()), c_ok, 100 * rel_ok))
        print("  → exact_match 不含超出各标签准确率的额外信息，")
        print("    它就是 12 个近似独立判断的乘积 Π p_i。")
        print()
        print("  归一化形状的斜率峰值比（exact / mean_acc）= %.2f" % ratio)
        if ratio > 1.3:
            print("  → exact_match 明显更陡：乘积把平滑的 p 放大成了跃升，")
            print("    即 Mirage 论文描述的度量假象。")
        else:
            print("  → 两者形状接近，本实验未出现额外的陡升。")
            print("    原因是绝对水平仍低（mean_acc 仅 %.2f~%.2f），"
                  % (acc.min(), acc.max()))
            print("    12 次幂要 mean_acc 到 0.9 以上才会有可观的 exact_match。")
    elif np.isfinite(c_ok):
        print("  乘积验证未通过（相关 %.5f，偏差 %.1f%%）。" % (c_ok, 100 * rel_ok))
        print("  各标签判断不独立，乘积模型不成立；exact_match 含标签间相关性")
        print("  带来的额外信息，不能简单归因于度量假象。")
    else:
        print("  命中数不足，无法判定乘积关系是否成立。")
    print()
    print("  参考：末段各标签准确率（最大规模）")
    ks = rows[-1].get("kinds") or ["blur", "noise", "jpeg", "downscale",
                                   "sharpen", "banding"]
    for k, v in zip(ks, rows[-1]["per_label"]):
        print("    %-12s %.4f" % (k, v))
    print("    Π p_i = %.4f   （实测 exact_match = %.4f，共 %d 个标签）"
          % (prod[-1], ex[-1], len(ks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
