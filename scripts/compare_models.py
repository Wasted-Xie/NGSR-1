"""对比多个基座模型的超分效果（定量指标 + 目视放大图）。

与 scripts/eval_sr.py 的三点区别
--------------------------------
1. **用数据集自带的真实 LR-HR 配对**（与 train_base.py 的训练口径一致）。
   eval_sr.py 是从 HR 双三次下采样合成 LR，与训练分布不符，评估会失真。
2. **支持 train_base.py 保存的 {"model":..., "step":...} 包装格式**
   （eval_sr.py 只接受裸 state_dict）。
3. **逐模型指定 engram_size_mb**（新旧版的 Engram 表大小不同）。

用法：
  python scripts/compare_models.py ^
      --a "base48:models/ckpt_base48.pt:48:4:4" ^
      --b "base64:temp/base_out/ckpt_base64_e026.pt:64:4:0.26" ^
      --n 20

规格格式： 名称:权重路径:通道数:层数:Engram_MB
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
OUT = os.path.join(ROOT, "temp", "base_out")
SCALE = 4


def _gauss_win(size, sigma, device):
    """归一化二维高斯窗（窗口和必须为 1，否则 SSIM 会被错误缩放）。"""
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
    return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2))
            / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))).mean().item()


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def to_img(t):
    return Image.fromarray(
        (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def make_lpips(dev):
    """构造 LPIPS 评估器。输入须归一化到 [-1,1]，故内部做 (x*2-1)。

    LPIPS 与 PSNR 常呈反向关系：感知类改动会提升 LPIPS、拉低 PSNR。
    只有两者并列报告，才能判断"观感是否真的变好"。
    """
    import lpips as _lp
    fn = _lp.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in fn.parameters():
        p.requires_grad_(False)

    def _calc(pred, target):
        with torch.no_grad():
            return float(fn(pred * 2 - 1, target * 2 - 1).mean())
    return _calc


def build_val(patch, n):
    """真实 LR-HR 配对，取末尾 n 张的中心区域（与 train_base.py 同源）。"""
    lr_dir = os.path.join(DATA, "train", "lr")
    hr_dir = os.path.join(DATA, "train", "hr")
    files = sorted(os.listdir(lr_dir))[-n:]
    out = []
    for f in files:
        lr = Image.open(os.path.join(lr_dir, f)).convert("RGB")
        hr = Image.open(os.path.join(hr_dir, f)).convert("RGB")
        w, h = lr.size
        p = min(patch, h, w)
        x, y = (w - p) // 2, (h - p) // 2
        lr_p = lr.crop((x, y, x + p, y + p))
        hr_p = hr.crop((x * SCALE, y * SCALE,
                        (x + p) * SCALE, (y + p) * SCALE))
        out.append((f, to_tensor(lr_p), to_tensor(hr_p)))
        lr.close(); hr.close()
    return out


def load_model(spec, dev, engram_mode="static", dynamic_slots=64):
    name, ckpt, C, L, eng, hb = spec
    cfg = SRConfig(
        input_height=192, input_width=192,
        output_height=192 * SCALE, output_width=192 * SCALE,
        downsample=2,
        main_channels=C, main_blocks=L,
        window_size=8, ffn_ratio=2, num_experts=8,
        top_k=2, num_clusters=2,
        shallow_channels=max(8, C // 2), bypass_channels=8,
        bypass_activation=0.10, engram_size_mb=eng,
        engram_hash_bits=hb,
        engram_mode=engram_mode,
        engram_dynamic_slots=dynamic_slots,
        understanding_dim_ratio=1.25,
    )
    cfg.validate()
    m = SuperResolutionModel(cfg).to(dev)
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(ROOT, ckpt)
    obj = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd)
    m.eval()
    step = obj.get("step", "?") if isinstance(obj, dict) else "?"
    return name, m, step


def best_crop(hr, bic, size=96):
    """取 bicubic 与 HR 差异最大的区域（信息量最高）。"""
    d = (hr - bic).abs().mean(dim=0, keepdim=True)
    k = size
    pooled = F.avg_pool2d(d.unsqueeze(0), k, stride=k // 2)
    idx = int(pooled.flatten().argmax())
    gw = pooled.shape[-1]
    iy, ix = divmod(idx, gw)
    y = min(iy * (k // 2), hr.shape[-2] - k)
    x = min(ix * (k // 2), hr.shape[-1] - k)
    return y, x, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="名称:路径:C:L:EngramMB")
    ap.add_argument("--b", required=True, help="名称:路径:C:L:EngramMB")
    ap.add_argument("--n", type=int, default=20, help="验证图张数")
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--out", default="cmp")
    ap.add_argument("--lpips", action="store_true",
                    help="额外计算 LPIPS（感知指标）。与 PSNR 常反向，"
                         "两者并列才能判断观感是否真的改善")
    ap.add_argument("--engram-mode", default="static",
                    choices=["static", "dynamic"],
                    help="被比较模型所用的 Engram 模式，必须与训练时一致")
    ap.add_argument("--engram-dynamic-slots", type=int, default=64)
    ap.add_argument("--loss-label", default="",
                    help="可选：打印在标题上的损失类型说明")
    args = ap.parse_args()

    def parse(s):
        p = s.split(":")
        assert len(p) >= 5, "规格格式应为 名称:路径:C:L:EngramMB[:HashBits]"
        hb = int(p[5]) if len(p) > 5 else 16
        return (p[0], p[1], int(p[2]), int(p[3]), float(p[4]), hb)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = [parse(args.a), parse(args.b)]
    models = [load_model(s, dev, args.engram_mode,
                         args.engram_dynamic_slots) for s in specs]
    val = build_val(192, args.n)

    print("=" * 92)
    print("基座模型超分效果对比（x%d，真实 LR-HR 配对，末尾 %d 张未训练图）"
          % (SCALE, args.n))
    print("=" * 92)
    for (name, _, step), sp in zip(models, specs):
        print("  %-10s C=%-3d L=%-2d Engram=%-6s step=%-7s  %s"
              % (name, sp[2], sp[3], sp[4], step, os.path.relpath(sp[1], ROOT)))
    print()

    res = {m[0]: {"psnr": [], "ssim": [], "lpips": []} for m in models}
    bic_p, bic_s, bic_l = [], [], []
    store = []
    lpips_fn = make_lpips(dev) if args.lpips else None
    if lpips_fn is not None:
        print("  LPIPS: 已启用（net=alex，输入归一化到 [-1,1]）")
        print()

    hdr = "  %-10s %12s" % ("样本", "bicubic")
    for name, _, _ in models:
        hdr += " %16s" % (name + " PSNR")
    hdr += " %10s" % "A-B 差"
    print(hdr)
    print("  " + "-" * (22 + 17 * len(models)))

    for f, lr, hr in val:
        lr_d = lr.unsqueeze(0).to(dev)
        hr_d = hr.unsqueeze(0).to(dev)
        bic = F.interpolate(lr_d, size=hr_d.shape[-2:], mode="bicubic",
                            align_corners=False)
        pb, sb = psnr(bic, hr_d), ssim(bic, hr_d)
        bic_p.append(pb); bic_s.append(sb)

        outs = []
        for name, m, _ in models:
            with torch.no_grad():
                o = m(lr_d)["sr"]
            res[name]["psnr"].append(psnr(o, hr_d))
            res[name]["ssim"].append(ssim(o, hr_d))
            if lpips_fn is not None:
                res[name]["lpips"].append(lpips_fn(o, hr_d))
            outs.append(o)
        if lpips_fn is not None:
            bic_l.append(lpips_fn(bic, hr_d))
        store.append((f, lr, bic[0].cpu(), [o[0].cpu() for o in outs], hr))

        line = "  %-10s %11.3f" % (f[:10], pb)
        for name, _, _ in models:
            line += " %16.3f" % res[name]["psnr"][-1]
        line += " %+9.3f" % (res[models[0][0]]["psnr"][-1]
                             - res[models[1][0]]["psnr"][-1])
        print(line)

    print("  " + "-" * (22 + 17 * len(models)))
    line = "  %-10s %11.3f" % ("平均 PSNR", np.mean(bic_p))
    for name, _, _ in models:
        line += " %16.3f" % np.mean(res[name]["psnr"])
    line += " %+9.3f" % (np.mean(res[models[0][0]]["psnr"])
                         - np.mean(res[models[1][0]]["psnr"]))
    print(line)

    line = "  %-10s %11.4f" % ("平均 SSIM", np.mean(bic_s))
    for name, _, _ in models:
        line += " %16.4f" % np.mean(res[name]["ssim"])
    line += " %+9.4f" % (np.mean(res[models[0][0]]["ssim"])
                         - np.mean(res[models[1][0]]["ssim"]))
    print(line)
    print()
    for name, _, _ in models:
        line = ("  %-10s 相对 bicubic: PSNR %+.4f dB   SSIM %+.5f"
                % (name, np.mean(res[name]["psnr"]) - np.mean(bic_p),
                   np.mean(res[name]["ssim"]) - np.mean(bic_s)))
        if lpips_fn is not None:
            # LPIPS 越小越好
            line += ("   LPIPS %.4f (bicubic %.4f, %+.4f)"
                     % (np.mean(res[name]["lpips"]), np.mean(bic_l),
                        np.mean(res[name]["lpips"]) - np.mean(bic_l)))
        print(line)
    if lpips_fn is not None:
        print()
        print("  注：LPIPS 越小越好；它与 PSNR 常呈反向关系。")

    # ---------- 拼图：四列（bicubic / A / B / HR）----------
    z = args.zoom
    ncol = 2 + len(models)
    f0, lr0, bic0, outs0, hr0 = store[0]
    y, x, k = best_crop(hr0, bic0, size=96)
    cols = [bic0] + outs0 + [hr0]
    labels = ["bicubic"] + [m[0] for m in models] + ["HR 真值"]

    W = (hr0.shape[-1]) * ncol + 8 * (ncol - 1) + 40
    H = 20 + hr0.shape[-2] + 8 + k * z + 40
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    for ci, (t, lab) in enumerate(zip(cols, labels)):
        cx = 20 + ci * (hr0.shape[-1] + 8)
        d.text((cx + 4, 4), lab, fill=(0, 0, 0))
        full = to_img(t)
        canvas.paste(full, (cx, 20))
        if ci == 0:
            d.rectangle([cx + x, 20 + y, cx + x + k, 20 + y + k],
                        outline=(255, 0, 0), width=3)
        crop = full.crop((x, y, x + k, y + k)).resize((k * z, k * z),
                                                      Image.NEAREST)
        cy2 = 20 + hr0.shape[-2] + 8
        canvas.paste(crop, (cx, cy2))
        d.rectangle([cx, cy2, cx + k * z - 1, cy2 + k * z - 1],
                    outline=(180, 180, 180))
    d.text((20, H - 30),
           "下排 = 左图红框区域放大 %dx（最近邻）  |  样本 %s" % (z, f0),
           fill=(60, 60, 60))

    path = os.path.join(OUT, "compare_%s.png" % args.out)
    canvas.save(path)
    print()
    print("  对比图: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
