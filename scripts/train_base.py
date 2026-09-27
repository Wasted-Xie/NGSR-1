"""基座模型训练：长时间训练，寻找能力边界（性能饱和点）。

与 feasibility_train.py 的区别
------------------------------
1. **用数据集真实 LR**（data/train/lr，引擎渲染的 480x270），而不是从 HR
   bicubic 下采样合成。原因：bicubic 伪退化下 L1 的最优解就是 bicubic 本身，
   模型无处可学，能力边界会被人为压到 bicubic 水平（已实测：增益仅 +0.02dB）。
   数据集提供的是真实 4x 配对（lr 与 hr 文件名一一对应）。
2. **长时间训练 + cosine 退火**，以逼近饱和点，而非只看几百步的趋势。
3. **持续 checkpoint 与曲线落盘**，中断不丢进度。

能力边界的判定
--------------
不是看训练 loss（每步换样本，方差大），而是在**固定验证集**上看 PSNR 的
滑动均值：连续 `--patience` 次评估提升均 < `--min-delta` 时判定饱和。
验证集取 data/train 末尾 40 张（训练只按 files[i % N] 取样，未触及末尾）。

用法：
  python scripts/train_base.py --steps 60000 --channels 48 --blocks 4
"""

import argparse
import copy
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
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from sr import SRConfig  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "temp", "base_out")
SCALE = 4


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class RealPair(Dataset):
    """数据集自带的真实 4x 配对（lr 480x270 / hr 1920x1080）。"""

    def __init__(self, patch=192, length=100000, seed=0):
        self.lr_dir = os.path.join(DATA, "train", "lr")
        self.hr_dir = os.path.join(DATA, "train", "hr")
        self.files = sorted(os.listdir(self.lr_dir))
        self.p = patch
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1000003 + i)
        f = self.files[i % len(self.files)]
        lr = np.asarray(Image.open(os.path.join(self.lr_dir, f)).convert("RGB"))
        hr = np.asarray(Image.open(os.path.join(self.hr_dir, f)).convert("RGB"))
        h, w, _ = lr.shape
        p = min(self.p, h, w)
        y = int(rng.integers(0, h - p + 1))
        x = int(rng.integers(0, w - p + 1))
        lr_p = lr[y:y + p, x:x + p]
        hr_p = hr[y * SCALE:(y + p) * SCALE, x * SCALE:(x + p) * SCALE]
        a = torch.from_numpy(lr_p.astype(np.float32) / 255.0).permute(2, 0, 1)
        b = torch.from_numpy(hr_p.astype(np.float32) / 255.0).permute(2, 0, 1)
        return a, b


def build_val(patch=192, n=40, scale=SCALE):
    """固定验证集：末尾 n 张图的中心区域（确定性，跨步可比）。"""
    lr_dir = os.path.join(DATA, "train", "lr")
    hr_dir = os.path.join(DATA, "train", "hr")
    files = sorted(os.listdir(lr_dir))[-n:]
    pairs = []
    for f in files:
        lr = np.asarray(Image.open(os.path.join(lr_dir, f)).convert("RGB"))
        hr = np.asarray(Image.open(os.path.join(hr_dir, f)).convert("RGB"))
        h, w, _ = lr.shape
        p = min(patch, h, w)
        y = (h - p) // 2
        x = (w - p) // 2
        lr_p = lr[y:y + p, x:x + p]
        hr_p = hr[y * scale:(y + p) * scale, x * scale:(x + p) * scale]
        pairs.append((
            torch.from_numpy(lr_p.astype(np.float32) / 255.0).permute(2, 0, 1),
            torch.from_numpy(hr_p.astype(np.float32) / 255.0).permute(2, 0, 1)))
    return pairs


def _gauss_win(size, sigma, device):
    c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def ssim(a, b, size=11, sigma=1.5):
    ch = a.shape[1]
    win = _gauss_win(size, sigma, a.device).expand(ch, 1, size, size).contiguous()
    mu1 = F.conv2d(a, win, groups=ch)
    mu2 = F.conv2d(b, win, groups=ch)
    s1 = F.conv2d(a * a, win, groups=ch) - mu1 ** 2
    s2 = F.conv2d(b * b, win, groups=ch) - mu2 ** 2
    s12 = F.conv2d(a * b, win, groups=ch) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return m.mean().item()


@torch.no_grad()
def evaluate(model, val, dev):
    model.eval()
    po, pb, so = [], [], []
    for lr, hr in val:
        lr = lr.unsqueeze(0).to(dev)
        hr = hr.unsqueeze(0).to(dev)
        o = model(lr)["sr"].clamp(0, 1)
        bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic",
                            align_corners=False).clamp(0, 1)
        po.append(psnr(o, hr))
        pb.append(psnr(bic, hr))
        so.append(ssim(o, hr))
    model.train()
    return (float(np.mean(po)), float(np.mean(pb)), float(np.mean(so)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=8,
                    help="连续多少次评估未提升则判定饱和")
    ap.add_argument("--min-delta", type=float, default=0.01,
                    help="视为提升的最小 dB 幅度")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="base")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SRConfig(
        input_height=args.patch, input_width=args.patch,
        output_height=args.patch * SCALE, output_width=args.patch * SCALE,
        downsample=2,                      # 自适应会按输入高度决定是否跳过
        main_channels=args.channels, main_blocks=args.blocks,
        window_size=8, ffn_ratio=2, num_experts=args.experts,
        top_k=2, num_clusters=2,
        shallow_channels=max(8, args.channels // 2), bypass_channels=8,
        bypass_activation=0.10, engram_size_mb=4, understanding_dim_ratio=1.25,
    )
    cfg.validate()

    print("=" * 86)
    print("基座训练：寻找能力边界")
    print("=" * 86)
    print("  设备      : %s" % (torch.cuda.get_device_name(0)
                               if dev.type == "cuda" else "cpu"))
    print("  任务      : x%d 超分（数据集真实 LR-HR 配对，非 bicubic 合成）" % SCALE)
    print("  配对      : LR %dx%d -> HR %dx%d"
          % (args.patch, args.patch, args.patch * SCALE, args.patch * SCALE))
    print("  主干      : %dx%d（实际下采样 %dx）, C=%d, L=%d, 专家 %d"
          % (cfg.main_height, cfg.main_width, cfg.effective_downsample,
             args.channels, args.blocks, args.experts))
    print("  训练      : %d 步, batch %d, lr %.0e -> %.0e (cosine)"
          % (args.steps, args.batch, args.lr, args.min_lr))
    print("  饱和判据  : 连续 %d 次评估提升 < %.3f dB" % (args.patience, args.min_delta))

    model = SuperResolutionModel(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print("  参数量    : %s" % format(nparam, ","))

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=dev)
        model.load_state_dict(sd["model"])
        start_step = sd.get("step", 0)
        print("  已恢复    : %s（step %d）" % (args.resume, start_step))

    ds = RealPair(patch=args.patch, length=args.steps * args.batch + 64,
                  seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    persistent_workers=(args.workers > 0))
    val = build_val(patch=args.patch)
    print("  验证集    : %d 张（末尾 40 张中心区域）" % len(val))
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.min_lr)

    p0, b0, s0 = evaluate(model, val, dev)
    print("  初始 PSNR : 模型 %.3f dB | bicubic %.3f dB（模型 %+.3f）"
          % (p0, b0, p0 - b0))
    print()

    curve = [(0, p0 - b0, s0)]
    best = p0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    ckpt = os.path.join(OUT, "ckpt_%s.pt" % args.tag)

    def save_curve():
        p = os.path.join(OUT, "curve_%s.csv" % args.tag)
        with open(p, "w", encoding="utf-8") as f:
            f.write("step,psnr_gain,ssim\n")
            for st, g, s in curve:
                f.write("%d,%.6f,%.6f\n" % (st, g, s))

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    it = iter(dl)
    t0 = time.perf_counter()
    hist = []
    stopped = "达到步数上限"
    for step in range(start_step + 1, args.steps + 1):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)

        out = model(xb)
        loss = F.l1_loss(out["sr"], yb) + 0.01 * out["aux_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(loss.item())

        if step % args.log_every == 0:
            el = time.perf_counter() - t0
            print("  step %6d/%d  loss %.5f  lr %.2e  %.1f step/s  %.0fmin  峰值%.0fMB"
                  % (step, args.steps, np.mean(hist[-args.log_every:]),
                     sched.get_last_lr()[0], step / el, el / 60,
                     torch.cuda.max_memory_allocated() / 1024 ** 2
                     if dev.type == "cuda" else 0))

        if args.eval_every and step % args.eval_every == 0:
            p, b, s = evaluate(model, val, dev)
            curve.append((step, p - b, s))
            save_curve()
            torch.save({"model": model.state_dict(), "step": step,
                        "cfg": {"patch": args.patch, "channels": args.channels,
                                "blocks": args.blocks, "experts": args.experts}},
                       ckpt)
            gain = p - b
            gain_abs = p
            print("        └ 验证 PSNR %.4f dB（bicubic %.4f，%+.4f）  SSIM %.5f"
                  % (p, b, gain, s))
            if gain_abs > best + args.min_delta:
                best = gain_abs
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                print("          新最优 %.4f dB" % best)
            else:
                no_improve += 1
                print("          未提升 %d/%d（最优 %.4f dB）"
                      % (no_improve, args.patience, best))
                if no_improve >= args.patience:
                    stopped = ("饱和：连续 %d 次评估提升 < %.3f dB"
                               % (args.patience, args.min_delta))
                    print()
                    print("  **停止：%s**" % stopped)
                    break

    total = time.perf_counter() - t0
    model.load_state_dict(best_state)
    p1, b1, s1 = evaluate(model, val, dev)
    save_curve()

    print()
    print("=" * 86)
    print("结果")
    print("=" * 86)
    print("  停止原因  : %s" % stopped)
    print("  执行步数  : %d（耗时 %.1f 分钟，%.1f step/s）"
          % (step - start_step, total / 60, (step - start_step) / total))
    print("  训练前    : 模型 %.4f dB | bicubic %.4f dB" % (p0, b0))
    print("  最优模型  : 模型 %.4f dB | bicubic %.4f dB（%+.4f dB）"
          % (p1, b1, p1 - b1))
    print("  最优 SSIM : %.5f" % s1)
    print("  相对提升  : %.4f dB（训练前 %+.4f -> 最优 %+.4f）"
          % (p1 - p0, p0 - b0, p1 - b1))
    if dev.type == "cuda":
        print("  峰值显存  : %.0f MB" % (torch.cuda.max_memory_allocated() / 1024 ** 2))

    torch.save({"model": best_state, "step": step,
                "best_psnr": p1, "bicubic": b1}, ckpt)
    with open(os.path.join(OUT, "summary_%s.json" % args.tag), "w",
              encoding="utf-8") as f:
        json.dump({"task": "x%d super-resolution (real LR)" % SCALE,
                   "patch": args.patch, "channels": args.channels,
                   "blocks": args.blocks, "experts": args.experts,
                   "params": nparam, "steps_run": step - start_step,
                   "minutes": total / 60, "stopped": stopped,
                   "psnr_before": p0, "psnr_bicubic": b0,
                   "psnr_best": p1, "ssim_best": s1,
                   "gain_over_bicubic": p1 - b1,
                   "curve": [[int(a), float(b_), float(c)] for a, b_, c in curve]},
                  f, ensure_ascii=False, indent=2)
    print("  权重      : %s" % ckpt)
    print("  曲线      : %s" % os.path.join(OUT, "curve_%s.csv" % args.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
