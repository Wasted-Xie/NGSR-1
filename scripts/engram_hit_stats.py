"""统计 Engram 表的命中分布（只读，不修改任何文件）。

动机
----
EngramLayer._hash() 用 latent_dim(=16) 个二进制位构成索引：

    bits = (proj > 0)
    idx  = sum(bits * 2**i) % num_slots

因此**索引空间上限为 2^16 = 65536**。若 num_slots 超过该值，
超出的槽位永远不可能被命中。本脚本用真实数据前向，统计：
  - 实际命中的不同槽位数与覆盖率
  - 索引取值范围（验证 65536 上限）
  - 命中集中度（前 10% 槽位占多少流量）
  - 零命中槽位占比

用法：
  python scripts/engram_hit_stats.py            # 同时测两个 ckpt
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sr import SRConfig  # noqa: E402
from sr.engram import EngramLayer  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LR_DIR = os.path.join(ROOT, "data", "train", "lr")
SCALE = 4
PATCH = 192


def build_cfg(C, L, eng):
    cfg = SRConfig(
        input_height=PATCH, input_width=PATCH,
        output_height=PATCH * SCALE, output_width=PATCH * SCALE,
        downsample=2, main_channels=C, main_blocks=L,
        window_size=8, ffn_ratio=2, num_experts=8,
        top_k=2, num_clusters=2,
        shallow_channels=max(8, C // 2), bypass_channels=8,
        bypass_activation=0.10, engram_size_mb=eng,
        understanding_dim_ratio=1.25,
    )
    cfg.validate()
    return cfg


def run(dev, name, ckpt, C, L, eng, n_img=16):
    cfg = build_cfg(C, L, eng)
    model = SuperResolutionModel(cfg).to(dev)
    obj = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # 记录每个 EngramLayer 实例的查询索引
    records = {}
    orig_hash = EngramLayer._hash

    def patched(self, x, h, w):
        # 与 EngramLayer._hash 的新签名保持一致（issue #4：N-Gram 需要 h/w
        # 来构造空间邻域键）
        idx = orig_hash(self, x, h, w)
        records.setdefault(id(self), {"slots": self.num_slots,
                                      "chunks": []})
        records[id(self)]["chunks"].append(idx.detach().reshape(-1).cpu())
        return idx

    EngramLayer._hash = patched
    files = sorted(os.listdir(LR_DIR))[:n_img]
    with torch.no_grad():
        for f in files:
            im = Image.open(os.path.join(LR_DIR, f)).convert("RGB")
            w, h = im.size
            p = min(PATCH, h, w)
            x, y = (w - p) // 2, (h - p) // 2
            im = im.crop((x, y, x + p, y + p))
            t = torch.from_numpy(
                np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            model(t.unsqueeze(0).to(dev))
            im.close()
    EngramLayer._hash = orig_hash

    print("=" * 88)
    print("%s   C=%d L=%d  engram_size_mb=%.2f" % (name, C, L, eng))
    print("  表总槽位: %s" % format(model.engram.total_slots, ","))
    print("  理论哈希空间上限: 2^16 = %s" % format(2 ** 16, ","))
    print("=" * 88)
    tot_q = 0
    for lid, rec in records.items():
        slots = rec["slots"]
        idx = torch.cat(rec["chunks"]).numpy().astype(np.int64)
        tot_q += len(idx)
        uniq = np.unique(idx)
        cnt = np.bincount(idx, minlength=slots)
        order = np.sort(cnt)[::-1]
        k10 = max(1, slots // 10)
        top10 = order[:k10].sum() / max(1, cnt.sum())
        print("  [num_slots=%s]" % format(slots, ","))
        print("    查询总次数        : %s" % format(len(idx), ","))
        print("    索引实际取值范围  : [%d, %d]" % (idx.min(), idx.max()))
        print("    命中的不同槽位    : %s  (覆盖率 %.2f%%)"
              % (format(len(uniq), ","), 100 * len(uniq) / slots))
        print("    零命中槽位        : %s  (%.2f%%)"
              % (format(int((cnt == 0).sum()), ","),
                 100 * (cnt == 0).mean()))
        print("    前 10%% 热门槽位   : 占流量 %.1f%%" % (100 * top10))
        print("    单槽位最高频次    : %s" % format(int(cnt.max()), ","))
        print()
    print("  合计查询次数: %s" % format(tot_q, ","))
    print()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(dev, "base64 (小表 0.26MB)",
        os.path.join(ROOT, "temp", "base_out", "ckpt_base64_e026.pt"),
        64, 4, 0.26)
    run(dev, "base48 (大表 4MB)",
        os.path.join(ROOT, "models", "ckpt_base48.pt"),
        48, 4, 4.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
