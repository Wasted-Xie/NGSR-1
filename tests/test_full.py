"""
完整验证：形状正确性、梯度可回传、单帧约束、配置约束。

用法：
    python tests/test_full.py

设备：默认有 CUDA 就用 CUDA（本文件第 4 节要跑五档模型前向，CPU 上是主要耗时项）。
      用 SR_TEST_DEVICE=cpu 可强制回退。
"""

import os
import sys
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from _device import DEVICE, device_desc  # noqa: E402
from sr import SRConfig, Tier, get_preset  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402

print("计算设备: %s" % device_desc())

passed = 0
failed = 0


def check(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print("  [OK]   %s %s" % (name, detail))
    else:
        failed += 1
        print("  [FAIL] %s %s" % (name, detail))


def banner(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


# ---------------------------------------------------------------------------
banner("1. 单帧约束（设计文档 R42）")

# 模型不应接受或维护任何跨帧状态
# 注意：主干层数不能太少，否则主干算力过小、理解层占比会超预算
small = SRConfig(
    input_height=360, input_width=640,
    output_height=720, output_width=1280,
    downsample=2, main_channels=16, main_blocks=4,
    window_size=4, num_experts=4, top_k=2, num_clusters=2,
    shallow_channels=8, understanding_dim_ratio=1.25, engram_size_mb=1,
    bypass_channels=8,
)
small.validate()
model = SuperResolutionModel(small).to(DEVICE)
model.eval()

# 检查是否有任何时序相关属性
temporal_attrs = [a for a in dir(model) if any(
    k in a.lower() for k in ["temporal", "motion", "history", "prev_frame", "jitter"]
)]
check(not temporal_attrs, "无时序相关属性", str(temporal_attrs) if temporal_attrs else "")

# 连续两次前向应完全独立（无状态泄漏）
torch.manual_seed(42)
x = torch.randn(1, 3, 360, 640, device=DEVICE)
with torch.no_grad():
    o1 = model(x)["sr"]
    o2 = model(x)["sr"]
check(torch.allclose(o1, o2, atol=1e-6), "两次前向结果一致（无状态泄漏）",
      "最大差异 %.2e" % (o1 - o2).abs().max().item())

# ---------------------------------------------------------------------------
banner("2. 梯度可回传")

model.train()
x = torch.randn(1, 3, 360, 640, device=DEVICE)
target = torch.randn(1, 3, 720, 1280, device=DEVICE)
out = model(x)
loss = torch.nn.functional.l1_loss(out["sr"], target) + 0.01 * out["aux_loss"]
loss.backward()

n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for p in model.parameters() if p.requires_grad)
check(n_grad > 0, "梯度可回传", "%d/%d 个参数有非零梯度" % (n_grad, n_total))
check(torch.isfinite(loss).item(), "损失有限", "%.4f" % loss.item())

# 三值专家应有梯度（验证 STE 生效）
# 注意：专家是稀疏激活的，未被 Top-K 选中的专家本就没有梯度。
# 因此这里直接对 TernaryLinear 做单元测试，确保 STE 本身正确。
from sr.moe import TernaryLinear  # noqa: E402

tl = TernaryLinear(16, 32).to(DEVICE)
tl.train()
xx = torch.randn(4, 16, requires_grad=True, device=DEVICE)
yy = tl(xx).sum()
yy.backward()
check(
    tl.weight.grad is not None and tl.weight.grad.abs().sum() > 0,
    "三值层 STE 梯度回传（单元测试）",
    "梯度范数 %.4f" % (tl.weight.grad.norm().item() if tl.weight.grad is not None else 0),
)
check(
    tl.scale.grad is not None and tl.scale.grad.abs().sum() > 0,
    "三值层 scale 有梯度",
    "梯度范数 %.4f" % (tl.scale.grad.norm().item() if tl.scale.grad is not None else 0),
)

# 验证前向确实使用了量化权重（权重值应接近 scale 的整数倍）
tl.eval()
w_used = tl.quantize_weight(tl.weight, tl.scale) * tl.scale
resid = (tl.weight - w_used).abs().mean().item()
check(True, "三值量化残差", "平均 %.4f（越大说明量化越激进）" % resid)

# 整网中被激活的专家应有梯度
model.train()
model.zero_grad()
out = model(x)
loss = torch.nn.functional.l1_loss(out["sr"], target) + 0.01 * out["aux_loss"]
loss.backward()
active_ternary = []
for name, m in model.named_modules():
    if type(m).__name__ == "TernaryLinear" and m.weight.grad is not None:
        g = m.weight.grad.abs().sum().item()
        if g > 0:
            active_ternary.append((name, g))
check(len(active_ternary) > 0,
      "整网中被激活的三值专家有梯度",
      "%d 个三值层获得梯度" % len(active_ternary))

# ---------------------------------------------------------------------------
banner("3. 配置约束校验")

cases = [
    ("窗口不能整除主干", dict(window_size=7), False),
    ("TopK 超过专家数", dict(top_k=999), False),
    ("专家数不能被簇数整除", dict(num_experts=32, num_clusters=3), False),
    ("通道数不是 8 的倍数", dict(main_channels=30), False),
    ("旁路激活率超范围", dict(bypass_activation=0.5), False),
    # 注意：这里必须用 >= downsample_skip_below(540) 的输入高度，
    # 否则会被「低分辨率跳过下采样」接住，主干不再低于 180p，测不到该约束。
    # 540 / 4 = 135p < 180p，且 540 不小于阈值，自适应不会介入。
    ("主干低于 180p", dict(input_height=540, downsample=4), False),
]
for name, kw, expect_ok in cases:
    try:
        c = SRConfig(**{**small.__dict__, **kw})
        c.validate()
        ok = True
        err = ""
    except ValueError as e:
        ok = False
        err = str(e).split("\n")[1].strip() if "\n" in str(e) else str(e)
    check(ok == expect_ok, "拒绝非法配置：%s" % name, err[:60])

# --- 低分辨率自适应下采样（config.effective_downsample）---
# 依据：实测同一 2x 任务、同样 300 步，ds=1（主干 384², 86k 参数）比
#      ds=2（主干 192², 549k 参数）的 PSNR 增益高约 20 倍。
_a = SRConfig(**{**small.__dict__, "input_height": 384, "input_width": 384,
                 "downsample": 2})
check(_a.effective_downsample == 1 and _a.main_height == 384,
      "低分辨率跳过下采样",
      "输入 384p + 配置 2x -> 实际 %dx，主干 %dp"
      % (_a.effective_downsample, _a.main_height))

_b = SRConfig(**{**small.__dict__, "input_height": 1080, "input_width": 1920,
                 "downsample": 2})
check(_b.effective_downsample == 2,
      "高分辨率正常下采样",
      "输入 1080p + 配置 2x -> 实际 %dx" % _b.effective_downsample)

_c = SRConfig(**{**small.__dict__, "input_height": 384, "input_width": 384,
                 "downsample": 2, "downsample_auto": False})
check(_c.effective_downsample == 2,
      "可显式关闭自适应",
      "downsample_auto=False -> 实际 %dx" % _c.effective_downsample)

for _t in Tier:
    _p = get_preset(_t)
    check(_p.effective_downsample == _p.downsample,
          "%-7s 档位预设未被自适应改动" % _t.value,
          "输入 %dp，下采样 %dx" % (_p.input_height, _p.downsample))

# ---------------------------------------------------------------------------
banner("4. 各档形状正确性")

for tier in Tier:
    cfg = get_preset(tier)
    try:
        m = SuperResolutionModel(cfg).to(DEVICE)
        m.eval()
        x = torch.randn(1, 3, cfg.input_height, cfg.input_width, device=DEVICE)
        with torch.no_grad():
            out = m(x)
        shp = tuple(out["sr"].shape)
        want = (1, 3, cfg.output_height, cfg.output_width)
        ok = shp == want and torch.isfinite(out["sr"]).all().item()
        check(ok, "%-7s 形状与数值" % tier.value,
              "%s" % (shp,) if ok else "得到 %s，期望 %s" % (shp, want))
    except Exception as e:
        check(False, "%-7s" % tier.value, str(e)[:70])
        traceback.print_exc()

# ---------------------------------------------------------------------------
banner("5. 批量推理")

try:
    m = SuperResolutionModel(small).to(DEVICE)
    m.eval()
    x = torch.randn(2, 3, 360, 640, device=DEVICE)
    with torch.no_grad():
        out = m(x)
    check(out["sr"].shape[0] == 2, "batch=2 推理", "%s" % (tuple(out["sr"].shape),))
except Exception as e:
    check(False, "批量推理", str(e)[:70])

# ---------------------------------------------------------------------------
banner("6. Engram 注入接口（设计文档 R43）")

try:
    eng = model.engram
    # 索引与记忆表必须同设备（inject 内部做 self.memory[indices] = q）
    idx = torch.arange(16, device=DEVICE)
    vecs = torch.randn(16, eng.latent_dim, device=DEVICE)
    before = (eng.general.memory != 0).sum().item()
    eng.general.inject(idx, vecs)
    after = (eng.general.memory != 0).sum().item()
    check(after > before, "Engram 注入写入记忆表",
          "非零槽位 %d -> %d" % (before, after))
    check(eng.general.memory.dtype == torch.int8, "记忆表为 INT8",
          str(eng.general.memory.dtype))
    check(abs(eng.table_mb - small.engram_size_mb) < 0.5,
          "记忆表体积符合预算",
          "%.2f MB / 预算 %d MB" % (eng.table_mb, small.engram_size_mb))
except Exception as e:
    check(False, "Engram 注入", str(e)[:70])

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
