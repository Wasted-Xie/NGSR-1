"""验证 CUDA 版 PyTorch 是否真正可用（RTX 5060 / sm_120）。"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

print("=" * 74)
print("PyTorch CUDA 可用性验证")
print("=" * 74)
print()
print("  torch 版本      :", torch.__version__)
print("  CUDA 编译版本   :", torch.version.cuda)
print("  cuDNN 版本      :", torch.backends.cudnn.version())
print("  CUDA 可用       :", torch.cuda.is_available())
print("  可见 GPU 数     :", torch.cuda.device_count())
print()

if not torch.cuda.is_available():
    print("  !! CUDA 不可用，后续验证无法进行")
    sys.exit(1)

print("=" * 74)
print("GPU 详情")
print("=" * 74)
print()
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("  [%d] %s" % (i, p.name))
    print("      计算能力   : sm_%d%d" % (p.major, p.minor))
    print("      显存       : %.2f GB" % (p.total_memory / 1024 ** 3))
    print("      多处理器   : %d" % p.multi_processor_count)
    print("      L2 缓存    : %.1f MB" % (p.L2_cache_size / 1024 ** 2))
print()

# 关键：计算能力必须是 12.0（RTX 5060 = Blackwell）
cap = torch.cuda.get_device_capability(0)
print("  本机计算能力: sm_%d%d" % cap)
print("  期望（RTX 5060）: sm_120")
print("  匹配:", cap == (12, 0))
print()

print("=" * 74)
print("实际运算测试（验证 kernel 是否真的能跑 sm_120）")
print("=" * 74)
print()

try:
    # 1. 基础张量运算
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print("  [OK] 矩阵乘法          %s" % (tuple(c.shape),))

    # 2. FP16（设计文档要求的推理精度）
    ah = a.half()
    bh = b.half()
    ch = ah @ bh
    torch.cuda.synchronize()
    print("  [OK] FP16 矩阵乘法     %s" % (tuple(ch.shape),))

    # 3. 卷积（主干与理解层大量使用）
    conv = torch.nn.Conv2d(32, 64, 3, padding=1).cuda()
    x = torch.randn(2, 32, 64, 64, device="cuda")
    y = conv(x)
    torch.cuda.synchronize()
    print("  [OK] 2D 卷积           %s" % (tuple(y.shape),))

    # 4. 深度可分离卷积（DW 卷积，设计文档 5.2 要求）
    dw = torch.nn.Conv2d(32, 32, 3, padding=1, groups=32).cuda()
    ydw = dw(x)
    torch.cuda.synchronize()
    print("  [OK] DW 卷积           %s" % (tuple(ydw.shape),))

    # 5. FP16 卷积（混合精度训练必备）
    # 注意：用 .half() 创建独立副本，避免原地转换影响后续步骤
    conv_h = torch.nn.Conv2d(32, 64, 3, padding=1).cuda().half()
    xh = x.half()
    yh = conv_h(xh)
    torch.cuda.synchronize()
    print("  [OK] FP16 卷积         %s" % (tuple(yh.shape),))

    # 6. 注意力核心算子（本项目的窗口注意力）
    q = torch.randn(8, 4, 256, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(8, 4, 256, 16, device="cuda", dtype=torch.float16)
    attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
    torch.cuda.synchronize()
    print("  [OK] FP16 注意力       %s" % (tuple(attn.shape),))

    # 7. 自动混合精度（AMP）
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y_amp = conv(x)
    torch.cuda.synchronize()
    print("  [OK] AMP 自动混合精度  %s" % (tuple(y_amp.shape),))

    # 8. 梯度回传
    xg = torch.randn(2, 32, 64, 64, device="cuda", requires_grad=True)
    loss = conv(xg).sum()
    loss.backward()
    torch.cuda.synchronize()
    print("  [OK] 梯度回传          grad norm = %.2f" % xg.grad.norm().item())

except Exception as e:
    print("  [FAIL] 运算失败:", e)
    raise

print()
print("=" * 74)
print("显存分配测试")
print("=" * 74)
print()
torch.cuda.empty_cache()
free_before, total = torch.cuda.mem_get_info()
print("  分配前可用: %.2f GB / %.2f GB" % (free_before / 1024 ** 3, total / 1024 ** 3))

big = torch.zeros(1024, 1024, 512, dtype=torch.float16, device="cuda")  # 1 GB
torch.cuda.synchronize()
alloc = torch.cuda.memory_allocated() / 1024 ** 3
print("  分配 1GB FP16 张量成功")
print("  已分配    : %.2f GB" % alloc)
print("  峰值      : %.2f GB" % (torch.cuda.max_memory_allocated() / 1024 ** 3))
del big
torch.cuda.empty_cache()
print("  释放成功")

print()
print("=" * 74)
print("结论")
print("=" * 74)
print()
print("  CUDA 可用，sm_%d%d 内核正常，FP16 / DW卷积 / AMP / 反向传播均通过。" % cap)
print("  可以在此环境上运行本项目模型。")
