"""评估已训练模型的上采样效果：定量指标 + 可目视的局部放大对比图。

评估集刻意取 data/train/hr 的**末尾若干张**——训练时按 files[i % 14431] 取用，
i 只到 steps*batch（数百），因此末尾的图未被训练触及。

用法：
  python scripts/eval_sr.py --ckpt temp/feasibility_out/ckpt_big2x.pt \
      --scale 2 --patch 384 --channels 48 --blocks 4
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from sr import SRConfig  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "temp", "feasibility_out")


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def to_img(t):
    return Image.fromarray(
        (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def psnr(a, b):
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def _gauss_win(size, sigma, device):
    """归一化二维高斯窗（窗口和必须为 1）。

    必须先归一化一维核再做外积，否则窗口和等于 sum(g) 而非 1，
    方差项被错误缩放，SSIM 会在加噪时反而大于 1。
    """
    c = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(a, b, size=11, sigma=1.5):
    ch = a.shape[1]
    win = _gauss_win(size, sigma, a.device)
    win = win.expand(ch, 1, size, size).contiguous()
    mu1 = F.conv2d(a, win, groups=ch)
    mu2 = F.conv2d(b, win, groups=ch)
    s1 = F.conv2d(a * a, win, groups=ch) - mu1 ** 2
    s2 = F.conv2d(b * b, win, groups=ch) - mu2 ** 2
    s12 = F.conv2d(a * b, win, groups=ch) - mu1 * mu2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
        (mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return m.mean().item()


def best_crop(hr, bic, size=96):
    """找 bicubic 与 HR 差异最大的区域（信息量最高，最能体现差距）。"""
    d = (hr - bic).abs().mean(dim=0, keepdim=True)
    k = size
    pooled = F.avg_pool2d(d.unsqueeze(0), k, stride=k // 2)
    idx = int(pooled.flatten().argmax())
    gh, gw = pooled.shape[-2], pooled.shape[-1]
    iy, ix = divmod(idx, gw)
    y = min(iy * (k // 2), hr.shape[-2] - k)
    x = min(ix * (k // 2), hr.shape[-1] - k)
    return y, x, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--patch", type=int, default=384)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SRConfig(
        input_height=args.patch, input_width=args.patch,
        output_height=args.patch * args.scale,
        output_width=args.patch * args.scale,
        downsample=args.downsample,
        main_channels=args.channels, main_blocks=args.blocks,
        window_size=8, ffn_ratio=2, num_experts=args.experts,
        top_k=2, num_clusters=2,
        shallow_channels=max(8, args.channels // 2), bypass_channels=8,
        bypass_activation=0.10, engram_size_mb=4, understanding_dim_ratio=1.25,
    )
    cfg.validate()
    model = SuperResolutionModel(cfg).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    hr_dir = os.path.join(DATA, "train", "hr")
    files = sorted(os.listdir(hr_dir))
    picks = files[-args.n:]

    print("=" * 78)
    print("上采样效果评估（x%d，评估图为训练未触及的末尾 %d 张）" % (args.scale, args.n))
    print("=" * 78)
    print("  权重: %s" % os.path.relpath(args.ckpt, ROOT))
    print()
    print("  %-14s %12s %12s %10s" % ("样本", "bicubic PSNR", "模型 PSNR", "增益"))
    print("  " + "-" * 52)

    rows, agg_b, agg_m, agg_bs, agg_ms = [], [], [], [], []
    p = args.patch * args.scale
    for f in picks:
        hr_img = Image.open(os.path.join(hr_dir, f)).convert("RGB")
        w, h = hr_img.size
        x = max(0, (w - p) // 2)
        y = max(0, (h - p) // 2)
        hr_p = hr_img.crop((x, y, x + p, y + p))
        hr_img.close()
        lr_p = hr_p.resize((args.patch, args.patch), Image.BICUBIC)

        lr = to_tensor(lr_p).unsqueeze(0).to(dev)
        hr = to_tensor(hr_p).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(lr)["sr"]
        bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic",
                            align_corners=False)

        pb, pm = psnr(bic, hr), psnr(out, hr)
        sb, sm = ssim(bic, hr), ssim(out, hr)
        agg_b.append(pb); agg_m.append(pm)
        agg_bs.append(sb); agg_ms.append(sm)
        rows.append((f, lr[0].cpu(), bic[0].cpu(), out[0].cpu(), hr[0].cpu()))
        print("  %-14s %11.3f %12.3f %+9.3f" % (f[:12], pb, pm, pm - pb))

    print("  " + "-" * 52)
    print("  %-14s %11.3f %12.3f %+9.3f" % ("平均 PSNR",
                                           np.mean(agg_b), np.mean(agg_m),
                                           np.mean(agg_m) - np.mean(agg_b)))
    print("  %-14s %11.4f %12.4f %+9.4f" % ("平均 SSIM",
                                            np.mean(agg_bs), np.mean(agg_ms),
                                            np.mean(agg_ms) - np.mean(agg_bs)))
    print()

    # 拼图：两行（全图 / 局部放大），三列（bicubic / 模型 / HR）
    tiles = []
    for f, lr_t, bic_t, out_t, hr_t in rows[:4]:
        y, x, k = best_crop(hr_t, bic_t, size=96)
        z = args.zoom
        row = []
        for t in (bic_t, out_t, hr_t):
            full = to_img(t)
            crop = full.crop((x, y, x + k, y + k)).resize(
                (k * z, k * z), Image.NEAREST)
            row.append((full, crop, (x, y, k, z)))
        tiles.append((f, row))

    full_w = tiles[0][1][0][0].width
    crop_w = 96 * args.zoom
    pad, gap = 20, 8
    W = full_w * 3 + gap * 2 + pad * 2
    H = pad + full_w + gap + crop_w + pad + 18
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)

    for ci, name in enumerate(["bicubic", "模型输出", "HR 真值"]):
        cx = pad + ci * (full_w + gap)
        d.text((cx + 4, 4), name, fill=(0, 0, 0))

    # 用第一张做全图+放大展示
    f, row = tiles[0]
    for ci, (full, crop, (x, y, k, z)) in enumerate(row):
        cx = pad + ci * (full_w + gap)
        cy = pad + 14
        canvas.paste(full, (cx, cy))
        if ci == 0:
            d.rectangle([cx + x, cy + y, cx + x + k, cy + y + k],
                        outline=(255, 0, 0), width=3)
        cy2 = cy + full_w + gap
        canvas.paste(crop, (cx, cy2))
        d.rectangle([cx, cy2, cx + crop_w - 1, cy2 + crop_w - 1],
                    outline=(180, 180, 180))
    d.text((pad, pad + full_w + gap + crop_w + 2),
           "下排 = 上排红框区域放大 %dx（最近邻，便于看像素级差异）  |  %s"
           % (args.zoom, f), fill=(60, 60, 60))

    path = os.path.join(OUT, "eval_%s.png" % args.out)
    canvas.save(path)
    print("  对比图: %s" % path)
    print("  （上排全图，下排为左图红框区域的 %dx 放大）" % args.zoom)
    return 0


if __name__ == "__main__":
    sys.exit(main())
