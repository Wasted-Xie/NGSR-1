"""用更大的评估集重算规模实验的指标，并扫描多个 win_rate 阈值。

为什么需要这个脚本
------------------
训练脚本的验证集只有 4 张图（make_val_set(n=4)），评估噪声可达 ±0.1 dB，
足以掩盖相邻规模之间的真实差异——实测中 C=48(436k) 的 PSNR 增益反而低于
C=32(205k)，很可能就是噪声所致。

另外原 win_rate 定义为「模型 PSNR > bicubic」，所有规模恒为 1.00，
完全饱和、无区分度。这里改为扫描多个更严的阈值。

评估集取自 data/train/hr 的**末尾 40 张**：训练按 files[i % 14431] 取用，
i 最大为 steps*batch = 2000*2 = 4000，因此末尾 40 张未被训练触及。

用法：
  python scripts/reeval_scaling.py
"""

import glob
import json
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
from PIL import Image  # noqa: E402

from sr import SRConfig  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "temp", "feasibility_out")

N_VAL = 40
PSNR_THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 1.5]


def to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def _gauss_win(size, sigma, device):
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


def build_val(patch, scale, n=N_VAL):
    hr_dir = os.path.join(DATA, "train", "hr")
    files = sorted(os.listdir(hr_dir))[-n:]
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


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(os.path.join(OUT, "ckpt_scale_*.pt")))
    if not ckpts:
        print("未找到 ckpt_scale_*.pt")
        return 1

    results = []
    for ck in ckpts:
        tag = os.path.basename(ck)[len("ckpt_"):-len(".pt")]
        sj = os.path.join(OUT, "summary_%s.json" % tag)
        if not os.path.exists(sj):
            continue
        with open(sj, encoding="utf-8") as f:
            s = json.load(f)

        cfg = SRConfig(
            input_height=s["patch"], input_width=s["patch"],
            output_height=s["patch"] * s["scale"],
            output_width=s["patch"] * s["scale"],
            downsample=s["downsample"],
            main_channels=s["channels"], main_blocks=s["blocks"],
            window_size=8, ffn_ratio=2, num_experts=s["experts"],
            top_k=2, num_clusters=2,
            shallow_channels=max(8, s["channels"] // 2), bypass_channels=8,
            bypass_activation=0.10, engram_size_mb=4,
            understanding_dim_ratio=1.25,
        )
        model = SuperResolutionModel(cfg).to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()

        val = build_val(s["patch"], s["scale"])
        acc_o, acc_b, l1_o, l1_b, ps_o, ps_b, ss_o, ss_b = [], [], [], [], [], [], [], []
        wins = {t: 0 for t in PSNR_THRESHOLDS}
        with torch.no_grad():
            for lr, hr in val:
                lr = lr.unsqueeze(0).to(dev)
                hr = hr.unsqueeze(0).to(dev)
                o = model(lr)["sr"].clamp(0, 1)
                bic = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic",
                                    align_corners=False).clamp(0, 1)
                po, pb = psnr(o, hr), psnr(bic, hr)
                acc_o.append(((o - hr).abs() <= 1 / 255).float().mean().item())
                acc_b.append(((bic - hr).abs() <= 1 / 255).float().mean().item())
                l1_o.append(F.l1_loss(o, hr).item())
                l1_b.append(F.l1_loss(bic, hr).item())
                ps_o.append(po)
                ps_b.append(pb)
                ss_o.append(ssim(o, hr))
                ss_b.append(ssim(bic, hr))
                for t in PSNR_THRESHOLDS:
                    if po - pb > t:
                        wins[t] += 1

        r = {
            "tag": tag, "params": s["params"],
            "channels": s["channels"], "blocks": s["blocks"],
            "psnr_gain": float(np.mean(ps_o) - np.mean(ps_b)),
            "ssim_gain": float(np.mean(ss_o) - np.mean(ss_b)),
            "l1_gain": float(np.mean(l1_b) - np.mean(l1_o)),
            "acc_gain": float(np.mean(acc_o) - np.mean(acc_b)),
            "psnr_model": float(np.mean(ps_o)),
            "psnr_bicubic": float(np.mean(ps_b)),
            "n_val": len(val),
        }
        for t in PSNR_THRESHOLDS:
            r["win_gt_%.2f" % t] = wins[t] / len(val)
        results.append(r)
        print("  %-18s 参数 %-10s PSNR %+.4f  SSIM %+.5f  acc %+.5f  win>0.5 %.2f"
              % (tag, format(s["params"], ","), r["psnr_gain"],
                 r["ssim_gain"], r["acc_gain"], r["win_gt_0.50"]))
        del model
        torch.cuda.empty_cache()

    results.sort(key=lambda d: d["params"])
    with open(os.path.join(OUT, "scaling_reeval.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 96)
    print("重评估（验证集 %d 张，训练未触及）" % N_VAL)
    print("=" * 96)
    print("  %-9s %-10s %10s %10s %10s %10s %10s"
          % ("规模", "参数", "PSNR增益", "SSIM增益", "L1增益",
             "阈值准确率", "win>0.5dB"))
    print("  " + "-" * 76)
    for r in results:
        print("  %-9s %-10s %+10.4f %+10.5f %+10.5f %+10.5f %10.2f"
              % ("C%d/L%d" % (r["channels"], r["blocks"]),
                 format(r["params"], ","), r["psnr_gain"], r["ssim_gain"],
                 r["l1_gain"], r["acc_gain"], r["win_gt_0.50"]))
    print()
    print("  win_rate 阈值扫描（多个阈值，用于寻找有区分度的不连续指标）：")
    print("  %-9s %s" % ("规模", "".join("%12s" % (">%.2fdB" % t)
                                        for t in PSNR_THRESHOLDS)))
    for r in results:
        print("  %-9s %s" % ("C%d/L%d" % (r["channels"], r["blocks"]),
                             "".join("%12.2f" % r["win_gt_%.2f" % t]
                                     for t in PSNR_THRESHOLDS)))
    print()
    print("  已写入 %s" % os.path.join(OUT, "scaling_reeval.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
