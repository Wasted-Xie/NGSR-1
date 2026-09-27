"""可行性训练：验证模型能否在真实游戏数据上跑通训练。

**目的只有一个：跑通。**
不做指标优化、不做调参、不追求 PSNR。设计文档里的基座模型要求
（main_height >= 180、理解层算力占比、领域采样权重等）属于后续基座训练阶段，
本脚本不受其约束——这里只是为了确认数据、模型、反传、显存这条链没有断。

数据：data/train/hr（14,431 张 1920x1080 PNG）
  按 --scale 从 HR 现场下采样生成 LR 配对，因此 2x / 4x 均可，
  不依赖 data/train/lr（那是 4x 的固定配对）。

说明：LR 由 bicubic 下采样合成，属于"伪退化"。真实游戏 LR 是引擎渲染的
（锐利、无噪声），与伪退化分布不同——这会影响最终质量，但不影响"能否跑通"。

用法：
  python scripts/feasibility_train.py --steps 30
  python scripts/feasibility_train.py --steps 300 --scale 2 --patch 384
"""

import argparse
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
OUT = os.path.join(ROOT, "temp", "feasibility_out")


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class GameSRPatch(Dataset):
    """从 HR 图随机裁 patch，bicubic 下采样得到 LR。"""

    def __init__(self, patch=384, scale=2, length=2000, seed=0):
        self.hr_dir = os.path.join(DATA, "train", "hr")
        self.files = sorted(os.listdir(self.hr_dir))
        self.patch = patch
        self.scale = scale
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1000003 + i)
        f = self.files[i % len(self.files)]
        p = self.patch * self.scale

        hr = Image.open(os.path.join(self.hr_dir, f)).convert("RGB")
        w, h = hr.size
        if w < p or h < p:
            hr = hr.resize((max(w, p), max(h, p)), Image.BICUBIC)
            w, h = hr.size
        x = int(rng.integers(0, w - p + 1))
        y = int(rng.integers(0, h - p + 1))
        hr_p = hr.crop((x, y, x + p, y + p))
        hr.close()

        lr_p = hr_p.resize((self.patch, self.patch), Image.BICUBIC)
        return to_tensor(lr_p), to_tensor(hr_p)


def make_val_set(patch=384, scale=2, n=4):
    """固定验证样本：头几张图的中心区域，确定性，便于跨步比较。"""
    hr_dir = os.path.join(DATA, "train", "hr")
    files = sorted(os.listdir(hr_dir))[:n]
    p = patch * scale
    pairs = []
    for f in files:
        hr = Image.open(os.path.join(hr_dir, f)).convert("RGB")
        w, h = hr.size
        x = max(0, (w - p) // 2)
        y = max(0, (h - p) // 2)
        hr_p = hr.crop((x, y, x + p, y + p))
        hr.close()
        lr_p = hr_p.resize((patch, patch), Image.BICUBIC)
        pairs.append((to_tensor(lr_p), to_tensor(hr_p)))
    return pairs


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def _gauss_win(size, sigma, device):
    """归一化二维高斯窗（窗口和必须为 1）。

    注意：必须先对一维核归一化再做外积。若写成
        (g[:,None] * g[None,:]) / g.sum()
    则二维窗口的和等于 sum(g) 而非 1，局部均值被放大、方差项被错误缩放，
    会使 SSIM 在加噪时反而大于 1（自比较仍为 1，因而掩盖该错误）。
    """
    c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


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
    """返回多类指标，用于区分「真实涌现」与「度量假象」。

    指标分类依据 Schaeffer et al., NeurIPS 2023（Are Emergent Abilities of
    LLMs a Mirage?）：非线性/不连续指标会**制造**出突变，线性/连续指标则给出
    平滑曲线。因此必须同时记录两类，再对比：

      连续/线性  : l1           逐像素绝对误差
      连续/非线性: psnr, ssim   对数尺度与结构化度量
      不连续     : acc          逐像素误差 <= 1/255 的占比（阈值化）
                   win_rate     整图 PSNR 超过 bicubic 的图占比（0/1 平均）
    """
    model.eval()
    acc_m = {"l1": [], "b_l1": [], "psnr": [], "b_psnr": [], "ssim": [],
             "b_ssim": [], "acc": [], "b_acc": []}
    wins = 0
    for lr, hr in val:
        lr = lr.unsqueeze(0).to(dev)
        hr = hr.unsqueeze(0).to(dev)
        out = model(lr)["sr"].clamp(0, 1)
        bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic",
                            align_corners=False).clamp(0, 1)
        po, pb = psnr(out, hr), psnr(bic, hr)
        acc_m["l1"].append(F.l1_loss(out, hr).item())
        acc_m["b_l1"].append(F.l1_loss(bic, hr).item())
        acc_m["psnr"].append(po)
        acc_m["b_psnr"].append(pb)
        acc_m["ssim"].append(ssim(out, hr))
        acc_m["b_ssim"].append(ssim(bic, hr))
        acc_m["acc"].append(((out - hr).abs() <= 1.0 / 255).float().mean().item())
        acc_m["b_acc"].append(((bic - hr).abs() <= 1.0 / 255).float().mean().item())
        if po > pb:
            wins += 1
    model.train()
    r = {k: float(np.mean(v)) for k, v in acc_m.items()}
    r["win_rate"] = wins / len(val)
    r["psnr_gain"] = r["psnr"] - r["b_psnr"]
    r["l1_gain"] = r["b_l1"] - r["l1"]
    return r


def save_compare(model, lr, hr, dev, path):
    model.eval()
    with torch.no_grad():
        out = model(lr.unsqueeze(0).to(dev))["sr"][0].clamp(0, 1).cpu()
    bic = F.interpolate(lr.unsqueeze(0), size=hr.shape[-2:], mode="bicubic",
                        align_corners=False)[0].clamp(0, 1)

    def to_img(t):
        return Image.fromarray(
            (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

    imgs = [to_img(bic), to_img(out), to_img(hr.clamp(0, 1))]
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * 3 + 16, h), (255, 255, 255))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * (w + 8), 0))
    canvas.save(path)
    model.train()
    return path


def build_cfg(args):
    return SRConfig(
        input_height=args.patch,
        input_width=args.patch,
        output_height=args.patch * args.scale,
        output_width=args.patch * args.scale,
        downsample=args.downsample,
        main_channels=args.channels,
        main_blocks=args.blocks,
        window_size=8,
        ffn_ratio=2,
        num_experts=args.experts,
        top_k=2,
        num_clusters=2,
        shallow_channels=max(8, args.channels // 2),
        bypass_channels=8,
        bypass_activation=0.10,
        engram_size_mb=4,
        understanding_dim_ratio=1.25,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=384, help="LR patch 边长")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = build_cfg(args)
    try:
        cfg.validate()
    except ValueError as e:
        print("配置校验失败:\n%s" % e)
        return 1

    print("=" * 78)
    print("可行性训练（目标：跑通）")
    print("=" * 78)
    print("  设备      : %s" % (
        torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"))
    print("  超分倍数  : x%d" % args.scale)
    print("  配对      : LR %dx%d  ->  HR %dx%d"
          % (args.patch, args.patch, args.patch * args.scale,
             args.patch * args.scale))
    ds_note = "%dx" % cfg.effective_downsample
    if cfg.effective_downsample != args.downsample:
        ds_note += "（配置 %dx，低分辨率自适应跳过）" % args.downsample
    print("  主干      : %dx%d（下采样 %s）, C=%d, L=%d, 专家 %d"
          % (cfg.main_height, cfg.main_width, ds_note,
             args.channels, args.blocks, args.experts))
    print("  数据      : %s" % os.path.join(DATA, "train", "hr"))

    model = SuperResolutionModel(cfg).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    nbuf = sum(b.numel() * b.element_size() for b in model.buffers())
    print("  参数量    : %s   缓冲区 %.1f MB"
          % (format(nparam, ","), nbuf / 1024 ** 2))

    ds = GameSRPatch(patch=args.patch, scale=args.scale,
                     length=max(args.steps * args.batch + 64, 128),
                     seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(dev.type == "cuda"),
                    persistent_workers=(args.workers > 0))
    val = make_val_set(patch=args.patch, scale=args.scale, n=4)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    r0 = evaluate(model, val, dev)
    print("\n  初始指标 : PSNR 模型 %.3f | bicubic %.3f  (%+.3f dB)"
          % (r0["psnr"], r0["b_psnr"], r0["psnr_gain"]))
    print("             SSIM 模型 %.4f | bicubic %.4f  (%+.4f)"
          % (r0["ssim"], r0["b_ssim"], r0["ssim"] - r0["b_ssim"]))
    print("             阈值准确率 模型 %.4f | bicubic %.4f"
          % (r0["acc"], r0["b_acc"]))
    print()

    curve = [(0, r0["psnr_gain"], r0["ssim"] - r0["b_ssim"],
              r0["acc"] - r0["b_acc"], r0["l1_gain"], float("nan"))]
    tag = args.out or ("x%d_s%d_c%d_l%d" % (args.scale, args.steps,
                                            args.channels, args.blocks))
    csv_path = os.path.join(OUT, "curve_%s.csv" % tag)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    it = iter(dl)
    hist = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            lr, hr = next(it)
        except StopIteration:
            it = iter(dl)
            lr, hr = next(it)
        lr = lr.to(dev, non_blocking=True)
        hr = hr.to(dev, non_blocking=True)

        out = model(lr)
        loss = F.l1_loss(out["sr"], hr) + 0.01 * out["aux_loss"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hist.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            dt = time.perf_counter() - t0
            tail = hist[-args.log_every:]
            msg = ("  step %5d/%d  loss %.5f  (近%d步 %.5f)  grad %.2f  "
                   "%.1f step/s  %.0fs"
                   % (step, args.steps, loss.item(), len(tail),
                      sum(tail) / len(tail), float(gn), step / dt, dt))
            if dev.type == "cuda":
                msg += "  峰值 %.0fMB" % (
                    torch.cuda.max_memory_allocated() / 1024 ** 2)
            print(msg)

        if args.eval_every and step % args.eval_every == 0:
            rv = evaluate(model, val, dev)
            recent = float(np.mean(hist[-args.eval_every:]))
            curve.append((step, rv["psnr_gain"], rv["ssim"] - rv["b_ssim"],
                          rv["acc"] - rv["b_acc"], rv["l1_gain"], recent))
            print("        └ PSNR %+.3f  SSIM %+.4f  阈值准确率 %+.4f  win %.2f"
                  % (rv["psnr_gain"], rv["ssim"] - rv["b_ssim"],
                     rv["acc"] - rv["b_acc"], rv["win_rate"]))
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("step,psnr_gain,ssim_gain,acc_gain,l1_gain,train_loss\n")
                for st, pg, sg, ag, lg, tl in curve:
                    f.write("%d,%.6f,%.6f,%.6f,%.6f,%s\n"
                            % (st, pg, sg, ag, lg,
                               "" if tl != tl else "%.6f" % tl))

    total = time.perf_counter() - t0
    r1 = evaluate(model, val, dev)

    print()
    print("=" * 78)
    print("结果")
    print("=" * 78)
    print("  跑通      : 是（%d 步无异常）" % args.steps)
    print("  耗时      : %.1fs（%.1f step/s）" % (total, args.steps / total))
    print("  loss      : %.5f -> %.5f"
          % (np.mean(hist[:min(20, len(hist))]), np.mean(hist[-20:])))
    print("  连续指标  : PSNR %+.3f dB | SSIM %+.4f | L1 %+.5f"
          % (r1["psnr_gain"], r1["ssim"] - r1["b_ssim"], r1["l1_gain"]))
    print("  不连续指标: 阈值准确率 %+.4f | win_rate %.2f"
          % (r1["acc"] - r1["b_acc"], r1["win_rate"]))
    if dev.type == "cuda":
        print("  峰值显存  : %.0f MB / 总 %.1f GB"
              % (torch.cuda.max_memory_allocated() / 1024 ** 2,
                 torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

    img = save_compare(model, val[0][0], val[0][1], dev,
                       os.path.join(OUT, "compare_%s.png" % tag))
    torch.save(model.state_dict(), os.path.join(OUT, "ckpt_%s.pt" % tag))
    with open(os.path.join(OUT, "summary_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({"scale": args.scale, "steps": args.steps,
                   "patch": args.patch, "downsample": cfg.effective_downsample,
                   "channels": args.channels, "blocks": args.blocks,
                   "experts": args.experts,
                   "params": nparam, "sec": total,
                   "psnr_gain": r1["psnr_gain"],
                   "ssim_gain": r1["ssim"] - r1["b_ssim"],
                   "l1_gain": r1["l1_gain"],
                   "acc_gain": r1["acc"] - r1["b_acc"],
                   "win_rate": r1["win_rate"],
                   "loss_first": float(np.mean(hist[:min(20, len(hist))])),
                   "loss_last": float(np.mean(hist[-20:])),
                   "curve": [[int(c[0]), float(c[1]), float(c[2]),
                              float(c[3]), float(c[4])] for c in curve]},
                  f, ensure_ascii=False, indent=2)
    print("\n  对比图    : %s" % img)
    print("  曲线      : %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
