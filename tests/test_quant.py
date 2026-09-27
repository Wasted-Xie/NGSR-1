"""
伪量化（QAT / 精度仿真）与 bypass_quant 接线验证。

背景：`bypass_quant` 曾是**死配置**——`HighPrecisionBypass` 未接收该参数，
修改它不影响任何计算。本测试同时验证两件事：
  1. `quant.py` 的伪量化原语本身正确（幂等、STE、饱和、无死区）
  2. 该配置真的作用于模型计算（不同方案 → 不同输出）
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from _device import DEVICE, device_desc  # noqa: E402
from sr import SRConfig, QuantScheme  # noqa: E402
from sr.model import SuperResolutionModel  # noqa: E402
from sr.quant import (  # noqa: E402
    QuantConv2d,
    quantize_activation,
    quantize_weight,
)

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
    print("=" * 80)
    print(s)
    print("=" * 80)


ALL = [QuantScheme.FP32, QuantScheme.FP16, QuantScheme.FP8,
       QuantScheme.W8A8, QuantScheme.W4A8, QuantScheme.TERNARY]

torch.manual_seed(0)

# ---------------------------------------------------------------------------
banner("1. 权重伪量化原语")
# ---------------------------------------------------------------------------

w = torch.randn(32, 16, 3, 3, device=DEVICE) * 0.2

# 每输出通道允许的最大不同取值数——位宽的直接体现
MAX_LEVELS = {
    QuantScheme.TERNARY: 3,     # {-scale, 0, +scale}
    QuantScheme.W4A8: 15,       # [-7, 7]
    QuantScheme.W8A8: 255,      # [-127, 127]
}

print()
print("  %-8s %12s %12s %12s %8s" % ("方案", "最大误差", "相对误差", "每通道取值", "幂等"))
print("  " + "-" * 60)

for s in ALL:
    q = quantize_weight(w, s)
    err = (q - w).abs().max().item()
    rel = err / (w.abs().max().item() + 1e-12)
    nuniq = max(len(torch.unique(q[i])) for i in range(q.shape[0]))
    idem = torch.equal(q, quantize_weight(q, s))
    print("  %-8s %12.3e %12.3e %12d %8s"
          % (s.value, err, rel, nuniq, "是" if idem else "否"))
    check(torch.isfinite(q).all().item(), "%s 量化结果有限" % s.value)
    if s in MAX_LEVELS:
        check(
            nuniq <= MAX_LEVELS[s],
            "%s 取值数符合位宽" % s.value,
            "每通道 %d ≤ %d" % (nuniq, MAX_LEVELS[s]),
        )
    # 幂等性只对"纯格式舍入"成立：其表示集合与数据无关。
    # TERNARY / W8A8 / W4A8 用动态定标（scale 由权重统计量推出），重量化时
    # 统计量本身已改变，故不保证幂等——这是定标方式的固有性质，非缺陷。
    # 其中 TERNARY 尤甚：absmean 定标在量化后必变（非零元素比例 < 1）。
    if s in (QuantScheme.FP32, QuantScheme.FP16, QuantScheme.FP8):
        check(idem, "%s 格式舍入幂等" % s.value)

check(torch.equal(quantize_weight(w, QuantScheme.FP32), w),
      "FP32 为恒等变换（不量化）")

# FP8 必须落在 E4M3 可表示集合内，且超出 ±448 被饱和
big = torch.tensor([1e-3, 0.5, 448.0, 500.0, -600.0], device=DEVICE)
qb = quantize_weight(big, QuantScheme.FP8)
check(qb.abs().max().item() <= 448.0,
      "FP8 饱和到 E4M3 上界 448",
      "输入 500/-600 → %s" % qb.tolist())

# 三值：非零比例应合理（死区陷阱回归）
tq = quantize_weight(w, QuantScheme.TERNARY)
nz = (tq != 0).float().mean().item()
check(0.3 < nz < 1.0,
      "三值量化未落入死区（非零比例合理）",
      "非零 %.1f%%（若 scale 取值失当会全为 0）" % (nz * 100))

# 逐输出通道定标：不同通道的量程不应被互相污染
wc = torch.zeros(4, 8, 1, 1, device=DEVICE)
wc[0] = 0.01
wc[1] = 10.0
qc = quantize_weight(wc, QuantScheme.W8A8)
check(qc[0].abs().max().item() > 0,
      "逐输出通道定标（小量程通道未被压成 0）",
      "通道0 最大 %.3e / 通道1 最大 %.3f"
      % (qc[0].abs().max().item(), qc[1].abs().max().item()))

# ---------------------------------------------------------------------------
banner("2. STE：伪量化后梯度仍可回传")
# ---------------------------------------------------------------------------
#
# 伪量化的 round()/cast 梯度处处为 0。正确实现用
#     x + (xq - x).detach()
# 使前向取 xq、反向梯度恒为 1（直通）。若写成 xq + (x - x).detach()
# 则该项恒为 0，等同于没有 STE，权重将完全收不到梯度。

print()
print("  %-8s %16s %16s" % ("方案", "梯度是否全 1", "最大偏差"))
print("  " + "-" * 44)

for s in ALL:
    if s is QuantScheme.FP32:
        continue
    wg = torch.randn(16, 8, 3, 3, device=DEVICE, requires_grad=True)
    qg = quantize_weight(wg, s)
    qg.sum().backward()
    ones = torch.ones_like(wg.grad)
    dev = (wg.grad - ones).abs().max().item()
    ok = dev < 1e-6
    print("  %-8s %16s %16.3e" % (s.value, "是" if ok else "否", dev))
    check(ok, "%s STE 梯度直通" % s.value)

# 推理路径（no_grad）下即使张量 requires_grad，也必须走精确分支。
# 回归点：`_ste` 曾在任何情况下都走 `x + (xq-x).detach()`，其前向受浮点
# 舍入影响并不精确等于 xq，使三值量化每通道出现 7 个取值而非 3 个。
wr = (torch.randn(8, 8, 3, 3, device=DEVICE) * 0.2).requires_grad_(True)
with torch.no_grad():
    qr = quantize_weight(wr, QuantScheme.TERNARY)
nu = max(len(torch.unique(qr[i])) for i in range(qr.shape[0]))
check(nu == 3, "no_grad 下量化值精确落在表示集合内",
      "每通道 %d 个取值（应为 3）" % nu)

# ---------------------------------------------------------------------------
banner("3. QuantConv2d")
# ---------------------------------------------------------------------------

conv = QuantConv2d(8, 16, 3, padding=1, bias=False, quant=QuantScheme.FP32).to(DEVICE)
plain = nn.Conv2d(8, 16, 3, padding=1, bias=False).to(DEVICE)
plain.load_state_dict({"weight": conv.weight.detach().clone()})
conv.eval(); plain.eval()

xin = torch.randn(2, 8, 16, 16, device=DEVICE)
with torch.no_grad():
    o_q = conv(xin)
    o_p = plain(xin)

check(torch.allclose(o_q, o_p, atol=1e-6),
      "FP32 时 QuantConv2d 等价于普通 Conv2d",
      "最大差异 %.2e" % (o_q - o_p).abs().max().item())

# 非 FP32 应与原卷积不同（证明量化真的介入了计算）
for s in (QuantScheme.FP16, QuantScheme.FP8, QuantScheme.W8A8):
    c2 = QuantConv2d(8, 16, 3, padding=1, bias=False, quant=s).to(DEVICE)
    c2.load_state_dict({"weight": conv.weight.detach().clone()})
    c2.eval()
    with torch.no_grad():
        o2 = c2(xin)
    d = (o2 - o_p).abs().max().item()
    check(d > 0, "%s 使卷积输出改变" % s.value, "最大差异 %.3e" % d)

check(sum(p.numel() for p in conv.parameters())
      == sum(p.numel() for p in plain.parameters()),
      "伪量化不引入额外参数",
      "%s" % format(sum(p.numel() for p in conv.parameters()), ","))

# 深度可分离卷积（groups=C）也需正常工作
dw = QuantConv2d(8, 8, 3, padding=1, groups=8, bias=False,
                 quant=QuantScheme.W8A8).to(DEVICE)
dw.eval()
with torch.no_grad():
    od = dw(xin)
check(od.shape == (2, 8, 16, 16) and torch.isfinite(od).all().item(),
      "DWConv（groups=C）在量化下正常", "%s" % (tuple(od.shape),))

# ---------------------------------------------------------------------------
banner("4. 模型级接线：bypass_quant 真正影响计算")
# ---------------------------------------------------------------------------

SMALL = dict(
    input_height=360, input_width=640,
    output_height=720, output_width=1280,
    downsample=2, main_channels=16, main_blocks=4,
    window_size=4, num_experts=4, top_k=2, num_clusters=2,
    shallow_channels=8, understanding_dim_ratio=1.25,
    engram_size_mb=1, bypass_channels=8,
)

xm = torch.randn(1, 3, 360, 640, device=DEVICE)

base_state = None
outs = {}
nparams = {}
for s in (QuantScheme.FP32, QuantScheme.FP16, QuantScheme.FP8,
          QuantScheme.W8A8):
    cfg = SRConfig(**SMALL, bypass_quant=s)
    cfg.validate()
    m = SuperResolutionModel(cfg).to(DEVICE)
    m.eval()
    if base_state is None:
        base_state = m.state_dict()
    else:
        m.load_state_dict(base_state)
    with torch.no_grad():
        outs[s] = m(xm)["sr"]
    nparams[s] = sum(p.numel() for p in m.parameters())

print()
print("  %-8s %16s %14s" % ("方案", "相对 FP32 最大差", "参数量"))
print("  " + "-" * 42)
ref = outs[QuantScheme.FP32]
for s in (QuantScheme.FP32, QuantScheme.FP16, QuantScheme.FP8,
          QuantScheme.W8A8):
    d = (outs[s] - ref).abs().max().item()
    print("  %-8s %16.3e %14s" % (s.value, d, format(nparams[s], ",")))
    if s is not QuantScheme.FP32:
        check(d > 0, "%s 改变模型输出（接线生效）" % s.value,
              "最大差异 %.3e" % d)

check(len(set(nparams.values())) == 1,
      "参数量与量化方案无关", "%s" % format(nparams[QuantScheme.FP32], ","))

# 旁路路径上的量化卷积应继承配置
cfg = SRConfig(**SMALL, bypass_quant=QuantScheme.W8A8)
mm = SuperResolutionModel(cfg)
qconvs = {n: mod.quant for n, mod in mm.named_modules()
          if isinstance(mod, QuantConv2d) and n.startswith("bypass")}
check(len(qconvs) == 7 and all(v is QuantScheme.W8A8 for v in qconvs.values()),
      "旁路路径 7 个卷积全部继承 bypass_quant",
      "含 bypass_proj；%s" % sorted(qconvs))

# 梯度仍可回传到旁路（STE 在整网下生效）
cfg = SRConfig(**SMALL, bypass_quant=QuantScheme.W8A8)
mt = SuperResolutionModel(cfg).to(DEVICE)
mt.train()
loss = torch.nn.functional.l1_loss(
    mt(xm)["sr"], torch.randn(1, 3, 720, 1280, device=DEVICE)
)
loss.backward()
bw = mt.bypass.edge_conv.weight.grad
check(bw is not None and bw.abs().sum().item() > 0,
      "整网反传后旁路量化卷积有梯度",
      "edge_conv 梯度范数 %.4f" % (bw.norm().item() if bw is not None else 0))

# ---------------------------------------------------------------------------
banner("5. 配置校验：旁路不接受激进位宽")
# ---------------------------------------------------------------------------

print()
for s, expect_ok in [(QuantScheme.FP32, True), (QuantScheme.FP16, True),
                     (QuantScheme.FP8, True), (QuantScheme.W8A8, True),
                     (QuantScheme.W4A8, False), (QuantScheme.TERNARY, False)]:
    try:
        SRConfig(**SMALL, bypass_quant=s).validate()
        ok = True
    except ValueError:
        ok = False
    check(ok == expect_ok,
          "bypass_quant=%-8s %s" % (s.value, "通过" if expect_ok else "被拒绝"))

# ---------------------------------------------------------------------------
banner("6. 激活伪量化")
# ---------------------------------------------------------------------------

print()
print("  %-8s %16s %14s" % ("方案", "最大绝对误差", "相对误差"))
print("  " + "-" * 42)
xa = torch.randn(4, 32, 16, 16, device=DEVICE) * 3.0
for s in ALL:
    qa = quantize_activation(xa, s)
    err = (qa - xa).abs().max().item()
    rel = err / (xa.abs().max().item() + 1e-12)
    print("  %-8s %16.3e %14.3e" % (s.value, err, rel))
    check(torch.isfinite(qa).all().item(), "%s 激活量化无 NaN/Inf" % s.value)

# 全零输入不应产生 NaN（absmax 的 clamp 保护）
z = torch.zeros(2, 4, 8, 8, device=DEVICE)
for s in ALL:
    qz = quantize_activation(z, s)
    check(torch.isfinite(qz).all().item() and qz.abs().sum().item() == 0,
          "%s 全零输入不产生 NaN" % s.value)

# ---------------------------------------------------------------------------
banner("汇总")
print("  通过: %d" % passed)
print("  失败: %d" % failed)
print()
sys.exit(0 if failed == 0 else 1)
