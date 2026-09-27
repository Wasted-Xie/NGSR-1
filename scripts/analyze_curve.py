"""分析训练曲线，客观定位增益的突变点（拐点）。

判据说明
--------
"量变引起质变"在一条训练曲线上会表现为斜率的结构性变化，而不是数值变大。
因此这里不只看 gain 本身，重点看**分段速率**：

  速率 r_i = Δgain / Δstep（每 1000 步的 PSNR 增益）

并给出三种客观判据：
  1. 速率峰值点：r 最大的位置（加速最快）
  2. 曲率极值点：r 的二阶差分最大处（加速的加速度最大）
  3. 分段均值突变：把曲线等分若干段，比较各段平均速率

对幂律收敛（gain ~ step^a）而言 r 单调递减，不会有拐点；
若出现 r 先升后降，则存在"启动期"，其峰值即为突变位置。

用法：
  python scripts/analyze_curve.py temp/feasibility_out/curve_long1.csv
"""

import argparse
import csv
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402


def load(path):
    steps, pg, sg, tl = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            pg.append(float(row["psnr_gain"]))
            sg.append(float(row["ssim_gain"]))
            v = row.get("train_loss", "")
            tl.append(float(v) if v else np.nan)
    return (np.array(steps, float), np.array(pg), np.array(sg), np.array(tl))


def rate(steps, gain, lo, hi):
    """[lo, hi] 区间内每 1000 步的增益速率。"""
    m = (steps >= lo) & (steps <= hi)
    if m.sum() < 2:
        return np.nan
    ds = steps[m][-1] - steps[m][0]
    return (gain[m][-1] - gain[m][0]) / ds * 1000.0 if ds > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    steps, pg, sg, tl = load(args.csv)
    n = len(steps)

    print("=" * 84)
    print("训练曲线分析：%s" % os.path.basename(args.csv))
    print("=" * 84)
    print("  评估点数 : %d（step %d ~ %d）" % (n, steps[0], steps[-1]))
    print("  PSNR 增益: 起 %.3f -> 末 %.3f dB（总计 %+.3f）"
          % (pg[0], pg[-1], pg[-1] - pg[0]))
    print("  SSIM 增益: 起 %.4f -> 末 %.4f（总计 %+.4f）"
          % (sg[0], sg[-1], sg[-1] - sg[0]))
    print()

    # ---------- 分段速率 ----------
    print("-" * 84)
    print("分段速率（每 1000 步的 PSNR 增益）")
    print("-" * 84)
    edges = np.linspace(steps[0], steps[-1], args.segments + 1)
    rates = []
    print("  %-18s %12s %12s" % ("区间", "PSNR 速率", "SSIM 速率"))
    print("  " + "-" * 46)
    for i in range(args.segments):
        lo, hi = edges[i], edges[i + 1]
        rp = rate(steps, pg, lo, hi)
        rs = rate(steps, sg, lo, hi)
        rates.append(rp)
        print("  %-18s %12.4f %12.5f"
              % ("%d-%d" % (lo, hi), rp, rs))
    rates = np.array(rates)
    print()

    # ---------- 逐点速率与曲率 ----------
    print("-" * 84)
    print("逐点速率（相邻评估点之间，每 1000 步）")
    print("-" * 84)
    dstep = np.diff(steps)
    dgain = np.diff(pg)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dstep > 0, dgain / dstep * 1000.0, np.nan)
    mid = (steps[:-1] + steps[1:]) / 2

    print("  %-12s %12s %12s" % ("step", "PSNR 速率", "增量"))
    print("  " + "-" * 40)
    for i in range(len(r)):
        if not np.isfinite(r[i]):
            continue
        # 长训练下只打印变化明显的点，避免刷屏
        if n > 20 and i % max(1, len(r) // 20) != 0:
            continue
        print("  %-12d %12.4f %12.4f" % (mid[i], r[i], dgain[i]))
    print()

    # ---------- 三种判据 ----------
    print("-" * 84)
    print("突变点判据")
    print("-" * 84)
    valid = np.isfinite(r)
    if valid.sum() >= 3:
        rv = r[valid]
        mv = mid[valid]

        i_max = int(np.argmax(rv))
        print("  1) 速率峰值      : step ~%d，速率 %.4f /1000步"
              % (mv[i_max], rv[i_max]))

        dr = np.diff(rv)
        if len(dr) >= 1:
            i_acc = int(np.argmax(dr))
            print("  2) 曲率极值      : step ~%d，加速度 %.4f"
                  % (mv[i_acc + 1], dr[i_acc]))

        seg = max(1, len(rv) // 3)
        a, b, c = (np.nanmean(rv[:seg]), np.nanmean(rv[seg:2 * seg]),
                   np.nanmean(rv[2 * seg:]))
        print("  3) 三段平均速率  : %.4f -> %.4f -> %.4f" % (a, b, c))

        if np.all(np.diff(rv) <= 1e-9):
            print()
            print("  → 速率单调递减：曲线是平滑幂律收敛，**不存在突变点**。")
            print("    这符合「按损失训练」的预期——梯度下降不会自发产生相变，")
            print("    相变需要改变的是模型容量或数据，而不是步数。")
        else:
            print()
            print("  → 速率非单调：存在加速段，拐点见上。")
    else:
        print("  评估点太少（<3），无法判断。")

    # ---------- 拟合 ----------
    print()
    print("-" * 84)
    print("幂律拟合  gain = a * step^b")
    print("-" * 84)
    ok = (steps > 0) & np.isfinite(pg)
    if ok.sum() >= 3:
        x = np.log(steps[ok])
        y = np.log(np.maximum(pg[ok], 1e-9))
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print("  b = %.4f   a = %.4f   R^2 = %.5f" % (b, a, r2))
        if b > 0:
            print("  b > 0：增益随步数增长（未饱和）")
        else:
            print("  b < 0：增益随步数递减（趋于饱和，符合幂律收敛）")
        print()
        print("  实测末段 vs 幂律外推：")
        for frac in (0.5, 0.7, 0.9):
            s = steps[int((len(steps) - 1) * frac)]
            idx = int(np.argmin(np.abs(steps - s)))
            print("    step %6d  实测 %+.4f  幂律 %.4f"
                  % (s, pg[idx], np.exp(a + b * np.log(max(s, 1)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
