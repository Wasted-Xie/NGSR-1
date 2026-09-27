"""涌现实验：离散多标签退化分类，用 exact match 评分。

为什么换这个任务
----------------
前一阶段用逐像素 L1 回归找涌现，结论是"没有"。根本原因是任务形式：
密集回归的每个像素独立贡献损失，梯度信号平滑，不存在"全对才算对"的门槛。

本实验改为具备涌现条件的设定：
  - **离散标签**：6 种退化各自"存在/不存在"
  - **组合结构**：每张图随机叠加 0~4 种退化
  - **全或无评分**：exact match 要求 6 个标签全部判对

设计依据 Schaeffer, Miranda & Koyejo (NeurIPS 2023)：
  不连续/非线性指标会制造出表观突变。这里 exact_match ≈ Π p_i 是各标签
  准确率的**乘积**；当 p 随规模平滑上升时，乘积在中间段会显得陡峭。
  因此本实验要同时记录两类指标并对比形状：
    连续  : 每标签平均准确率 mean_acc
    不连续: exact_match（6 个全对的比例）

预期：若 mean_acc 平滑而 exact_match 出现"跃升"，则跃升来自评分方式，
而非模型能力的相变。

用法：
  python scripts/emergence_clf.py --steps 2000 --channels 32 --blocks 3
  python scripts/emergence_clf.py --scale-sweep        # 跑完整规模扫描
"""

import argparse
import io
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "temp", "emergence_out")
HR_DIR = os.path.join(DATA, "train", "hr")

KINDS = ["blur_w", "blur_s", "noise_w", "noise_s", "jpeg_w", "jpeg_s",
         "down_w", "down_s", "sharp_w", "sharp_s", "band_w", "band_s"]
N_LABEL = len(KINDS)


# ---------------------------------------------------------------------------
# 退化算子（每种含弱/强两档，对应两个独立标签）
# ---------------------------------------------------------------------------
#
# 为什么拆弱/强：只判"某退化是否存在"对 2k 参数的小模型都太容易
# （实测 500 步即达 mean_acc 0.76），最小规模压不到随机水平，就看不到跃升。
# 拆出强度后判别更难，且 exact_match = Πp_i 从 6 次幂变成 12 次幂，
# p 的微小提升会被放大得更剧烈。

def deg_blur(arr, rng, strong):
    r = float(rng.uniform(1.8, 3.0)) if strong else float(rng.uniform(0.6, 1.2))
    return np.asarray(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(r)))


def deg_noise(arr, rng, strong):
    s = float(rng.uniform(18, 32)) if strong else float(rng.uniform(5, 12))
    return np.clip(arr.astype(np.float32) + rng.normal(0, s, arr.shape), 0, 255) \
        .astype(np.uint8)


def deg_jpeg(arr, rng, strong):
    q = int(rng.integers(8, 20)) if strong else int(rng.integers(35, 60))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def deg_downscale(arr, rng, strong):
    f = float(rng.uniform(0.25, 0.40)) if strong else float(rng.uniform(0.5, 0.7))
    im = Image.fromarray(arr)
    w, h = im.size
    small = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BICUBIC)
    return np.asarray(small.resize((w, h), Image.BICUBIC))


def deg_sharpen(arr, rng, strong):
    p = int(rng.integers(250, 400)) if strong else int(rng.integers(100, 180))
    return np.asarray(Image.fromarray(arr).filter(
        ImageFilter.UnsharpMask(radius=2, percent=p, threshold=2)))


def deg_banding(arr, rng, strong):
    """量化到较少色阶，产生色带（模拟低位深/强压缩）。"""
    levels = int(rng.integers(4, 8)) if strong else int(rng.integers(10, 20))
    q = 255.0 / (levels - 1)
    return np.clip(np.round(arr.astype(np.float32) / q) * q, 0, 255).astype(np.uint8)


KIND_FUNCS = [deg_blur, deg_noise, deg_jpeg, deg_downscale, deg_sharpen, deg_banding]


def apply_label(arr, rng, i):
    """按标签索引施加对应退化：i//2 选种类，i%2 选强弱。"""
    return KIND_FUNCS[i // 2](arr, rng, (i % 2) == 1)


def make_sample(arr, rng, p_deg=0.4):
    """随机叠加若干退化，返回 (图, 12 维二值标签)。"""
    labels = (rng.random(N_LABEL) < p_deg).astype(np.float32)
    if labels.sum() == 0:
        labels[rng.integers(0, N_LABEL)] = 1.0
    if labels.sum() > 5:
        idx = rng.permutation(N_LABEL)[:5]
        labels = np.zeros(N_LABEL, dtype=np.float32)
        labels[idx] = 1.0
    out = arr
    for i in np.where(labels > 0)[0]:
        out = apply_label(out, rng, int(i))
    return out, labels


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

class PoolDS(torch.utils.data.Dataset):
    """从预生成池取样。

    关键：池在生成阶段就完成 permute 与 /255 归一化并存为 float32，
    因此 __getitem__ 退化为**纯切片**，没有任何 CPU 运算。
    早期版本把 permute/float/div 放在 __getitem__ 里、且 num_workers=0，
    导致数据准备与 GPU 前向在主进程串行——实测 GPU 利用率仅 57~67%、
    功耗 61W（该卡可上 100W+），即约 1/3 时间在等 CPU。
    """

    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class GpuPool:
    """把整个样本池常驻显存，训练时只做 GPU 上索引切片。

    为什么这么做：分类模型很小（最大约 1M 参数、4MB），而池仅 2.2GB，
    显存充裕。池常驻后，数据准备与 H2D 传输**完全消失**，
    GPU 不必再等 CPU。原实现用 DataLoader(num_workers=0) 时
    实测 GPU 利用率仅 57~67%、功耗 61W。

    迭代接口与 DataLoader 一致（__iter__ / __len__），便于替换。
    """

    def __init__(self, X, Y, dev, batch, seed=0):
        self.X = X.to(dev, non_blocking=True)
        self.Y = Y.to(dev, non_blocking=True)
        self.bs = batch
        self.n = len(X)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        return max(1, self.n // self.bs)

    def __iter__(self):
        perm = torch.randperm(self.n, generator=self.gen)   # CPU 上洗牌即可
        for i in range(0, self.n - self.bs + 1, self.bs):
            idx = perm[i:i + self.bs].to(self.X.device, non_blocking=True)
            yield self.X[idx], self.Y[idx]


def build_pool(n, patch, seed=0, n_img=120):
    """离线生成 n 个 (3,H,W) float32 样本 + 多标签。"""
    files = sorted(os.listdir(HR_DIR))[:-40]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(n_img, len(files)), replace=False)
    srcs = []
    for i in pick:
        im = Image.open(os.path.join(HR_DIR, files[int(i)])).convert("RGB")
        srcs.append(np.asarray(im))
    X = np.zeros((n, 3, patch, patch), dtype=np.float32)
    Y = np.zeros((n, N_LABEL), dtype=np.float32)
    for k in range(n):
        a = srcs[int(rng.integers(0, len(srcs)))]
        h, w, _ = a.shape
        y = int(rng.integers(0, h - patch + 1))
        x = int(rng.integers(0, w - patch + 1))
        img, lb = make_sample(a[y:y + patch, x:x + patch].copy(), rng)
        X[k] = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        Y[k] = lb
    return torch.from_numpy(X), torch.from_numpy(Y)


def build_val(patch=96, n=600, seed=12345):
    """固定验证集：确定性样本，避免评估噪声掩盖规模效应。"""
    files = sorted(os.listdir(HR_DIR))[-40:]
    cache = [np.asarray(Image.open(os.path.join(HR_DIR, f)).convert("RGB"))
             for f in files]
    xs, ys = [], []
    rng = np.random.default_rng(seed)
    for i in range(n):
        full = cache[i % len(cache)]
        h, w, _ = full.shape
        p = patch
        # 固定位置（中心区域），减少位置带来的方差
        y = (h - p) // 2 + int(rng.integers(-16, 17))
        x = (w - p) // 2 + int(rng.integers(-16, 17))
        y = max(0, min(h - p, y))
        x = max(0, min(w - p, x))
        arr = full[y:y + p, x:x + p].copy()
        img, labels = make_sample(arr, rng)
        xs.append(torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1))
        ys.append(torch.from_numpy(labels))
    return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(1, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(1, ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return self.act(x + h)


class DegClassifier(nn.Module):
    """规模由 channels（宽）与 blocks（深）控制，用于单一变量的规模扫描。"""

    def __init__(self, channels=32, blocks=3, n_label=N_LABEL):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, channels), nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, channels), nn.GELU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Linear(channels * 2, n_label)

    def forward(self, x):
        h = self.body(self.stem(x))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        # 同时接入最大池化，保留"局部是否存在该退化"的证据
        m = F.adaptive_max_pool2d(self.stem(x), 1).flatten(1)
        return self.head(torch.cat([h, m], dim=1))


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, xv, yv, dev, bs=64):
    model.eval()
    preds = []
    for i in range(0, len(xv), bs):
        preds.append(torch.sigmoid(model(xv[i:i + bs].to(dev))).cpu())
    p = torch.cat(preds)
    y = yv
    bit = (p > 0.5).float()
    per_label = (bit == y).float().mean(dim=0)          # 每个标签的准确率
    mean_acc = per_label.mean().item()
    exact = (bit == y).all(dim=1).float().mean().item()  # 6 个全对
    # 另给一个连续型指标：平均绝对误差（越小越好）
    mae = (p - y).abs().mean().item()
    model.train()
    return {"mean_acc": mean_acc, "exact_match": exact, "mae": mae,
            "per_label": per_label.tolist()}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_one(args, channels, blocks, tag, xv, yv, dev, quiet=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = DegClassifier(channels=channels, blocks=blocks).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    dl = GpuPool(args.pool[0], args.pool[1], dev, args.batch, seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if not quiet:
        print("  参数量 %s   验证集 %d 固定样本" % (format(nparam, ","), len(xv)))

    r0 = evaluate(model, xv, yv, dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    it = iter(dl)
    hist = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        # GpuPool 已保证数据驻留显存，无需任何 H2D 传输
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())

        if not quiet and step % args.log_every == 0:
            print("    step %5d/%d  loss %.4f  %.1f step/s"
                  % (step, args.steps, np.mean(hist[-args.log_every:]),
                     step / (time.perf_counter() - t0)))

    sec = time.perf_counter() - t0
    r1 = evaluate(model, xv, yv, dev)
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2
            if dev.type == "cuda" else 0.0)
    res = {"tag": tag, "channels": channels, "blocks": blocks,
           "params": nparam, "steps": args.steps, "patch": args.patch,
           "sec": sec, "peak_mb": peak,
           "mean_acc": r1["mean_acc"], "exact_match": r1["exact_match"],
           "mae": r1["mae"],
           "mean_acc_gain": r1["mean_acc"] - r0["mean_acc"],
           "exact_gain": r1["exact_match"] - r0["exact_match"],
           "mae_gain": r0["mae"] - r1["mae"],
           "per_label": r1["per_label"],
           "kinds": KINDS,
           "acc_before": r0["mean_acc"], "exact_before": r0["exact_match"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pool-n", type=int, default=16000,
                    help="预生成训练池样本数")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--scale-sweep", action="store_true")
    ap.add_argument("--val-n", type=int, default=600)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("涌现实验：多标签复合退化分类（%d 个标签，exact match 需全部判对）"
          % N_LABEL)
    print("=" * 88)
    print("  标签  : %s" % ", ".join(KINDS))
    print("  设备  : %s" % (torch.cuda.get_device_name(0)
                            if dev.type == "cuda" else "cpu"))
    print("  步数  : %d    batch %d    patch %d    lr %.0e"
          % (args.steps, args.batch, args.patch, args.lr))
    xv, yv = build_val(patch=args.patch, n=args.val_n)
    print("  验证集: %d 个固定样本（来自末尾 40 张 HR）" % len(xv))
    t0 = time.perf_counter()
    args.pool = build_pool(args.pool_n, args.patch, seed=args.seed)
    print("  训练池: %d 个样本，生成耗时 %.1fs，占用 %.0f MB"
          % (args.pool_n, time.perf_counter() - t0,
             args.pool[0].numel() / 1024 ** 2))
    print()

    if args.scale_sweep:
        # 参数量大致 ∝ C²·(18L+9)，本列表覆盖约 2.7 个数量级
        sweeps = [(8, 1), (12, 1), (16, 1), (16, 2), (24, 2), (32, 2),
                  (32, 3), (48, 3), (64, 3), (96, 2), (128, 3)]
        results = []
        print("  %-10s %-12s %10s %12s %10s %12s" %
              ("配置", "参数", "mean_acc", "exact_match", "MAE", "耗时s"))
        print("  " + "-" * 74)
        for c, l in sweeps:
            r = run_one(args, c, l, "C%dL%d" % (c, l), xv, yv, dev, quiet=True)
            results.append(r)
            print("  %-10s %-12s %10.4f %12.4f %10.4f %12.0f"
                  % ("C%d/L%d" % (c, l), format(r["params"], ","),
                     r["mean_acc"], r["exact_match"], r["mae"], r["sec"]))
        path = os.path.join(OUT, "clf_sweep_n%d_s%d.json"
                            % (N_LABEL, args.steps))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print()
        print("  已写入 %s" % path)
        return 0

    r = run_one(args, args.channels, args.blocks,
                args.out or "clf", xv, yv, dev)
    tag = r["tag"]
    with open(os.path.join(OUT, "clf_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 88)
    print("结果")
    print("=" * 88)
    print("  连续指标  mean_acc    : %.4f（训练前 %.4f，%+.4f）"
          % (r["mean_acc"], r["acc_before"], r["mean_acc_gain"]))
    print("  连续指标  MAE         : %.4f（越小越好，改善 %+.4f）"
          % (r["mae"], r["mae_gain"]))
    print("  不连续    exact_match : %.4f（训练前 %.4f，%+.4f）"
          % (r["exact_match"], r["exact_before"], r["exact_gain"]))
    print("  峰值显存  : %.0f MB" % r["peak_mb"])
    print()
    print("  各标签准确率：")
    for k, v in zip(KINDS, r["per_label"]):
        print("    %-12s %.4f" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
