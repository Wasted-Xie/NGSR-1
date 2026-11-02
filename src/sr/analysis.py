"""
模型分析工具：参数量、FLOPs、显存估算。

用途：
  1. 校验实现与《超分算法设计文档》5.2.7 节的重算值是否一致
  2. 为档位设计提供实测数据

FLOPs 计算模型（与设计文档 5.2.7 节保持一致）：
    窗口注意力  : QKV+O 投影 8NC^2  +  窗口内注意力 4Nw^2C   （对 N 线性）
    共享专家FFN : 4N·r·C^2
    Top-K 专家  : K x 4N·r·C^2
    浅层细节路  : 输入分辨率上 2x(DW3x3 + PW1x1)
    输出后处理  : 2160p 上 1x1 Conv (C->3)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import BITS_PER_WEIGHT, SRConfig


def count_parameters(model: nn.Module) -> dict[str, int]:
    """按顶层子模块统计参数量。

    注意：注册为 buffer 的 Engram 记忆表不计入参数量
    （它是 INT8 查找表，非可训练参数），但会单独统计其体积。
    """
    stats = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            stats[name] = n
    stats["_total"] = sum(p.numel() for p in model.parameters())
    return stats


def buffer_bytes(model: nn.Module) -> int:
    """统计所有 buffer 的字节数（含 Engram INT8 记忆表）。"""
    total = 0
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def estimate_weight_memory_mb(cfg: SRConfig, n_params: int) -> float:
    """按量化方案估算权重显存（MB）。

    简化处理：整体按专家量化方案估算（实际是分层混合量化）。
    """
    bits = BITS_PER_WEIGHT[cfg.expert_quant]
    return n_params * bits / 8 / 1024 / 1024


# ---------------------------------------------------------------------------
# FLOPs 估算（解析式，与设计文档 5.2.7 节同口径）
# ---------------------------------------------------------------------------

def estimate_flops(cfg: SRConfig) -> dict[str, float]:
    """按设计文档 5.2.7 节的解析模型估算单帧 FLOPs（单位 G = 1e9）。

    这是【解析估算】，不含实现细节开销，用于与设计文档对齐。
    """
    n_main = cfg.main_height * cfg.main_width
    n_in = cfg.input_height * cfg.input_width
    n_out = cfg.output_height * cfg.output_width

    c = cfg.main_channels
    w = cfg.window_size
    r = cfg.ffn_ratio
    L = cfg.main_blocks
    L_moe = cfg.effective_moe_layers
    K = cfg.top_k

    # 窗口注意力：QKV+O 投影 + 窗口内注意力
    attn = L * (8 * n_main * c * c + 4 * n_main * w * w * c)

    # 共享专家 FFN（每层都有）
    shared = L * 4 * n_main * r * c * c

    # 私有专家（仅 MoE 层）
    private = L_moe * K * 4 * n_main * r * c * c

    # 浅层细节路：输入分辨率上的 2x(DW3x3 + PW1x1)
    cs = cfg.shallow_channels
    shallow = 2 * (2 * 9 * n_in * cs + 2 * n_in * cs * cs)

    # MoE 共享专家：HybridBlock 内除独立 FFN 外，MoE 内部另有 num_shared 个
    # 共享专家，原公式只算了前者（issue #5 第 6 项）
    moe_shared = L_moe * cfg.num_shared_experts * 4 * n_main * r * c * c

    # ---- 以下为 issue #5 指出的漏算路径，全部在输入分辨率上执行 ----
    # 主干上采样到输入分辨率：Conv1x1(c -> cs*4) + PixelShuffle + Conv1x1(cs -> cs)
    main_up = n_main * 2 * c * (cs * 4) + n_in * 2 * cs * cs

    # 融合：Conv1x1(2cs -> c) + GELU + DW3x3(c) + PW1x1(c)
    fuse = n_in * (2 * (cs * 2) * c + 2 * 9 * c + 2 * c * c)

    # 高精度旁路：edge/texture(c -> ch, 3x3) + direction(c -> 4, 3x3)
    #            + compress(2ch+4 -> ch, 1x1) + DW3x3(ch) + expand(ch -> c, 1x1)
    ch = cfg.bypass_channels
    bypass = n_in * (
        2 * 9 * c * ch * 2
        + 2 * 9 * c * 4
        + 2 * (ch * 2 + 4) * ch
        + 2 * 9 * ch
        + 2 * ch * c
    )
    bypass_proj = n_in * 2 * c * c

    # 输出上采样：Conv1x1(c -> 12) + PixelShuffle + Conv1x1(3 -> 3)
    upsampler = n_in * 2 * c * 12 + n_out * 2 * 9

    # 输出后处理：Conv1x1(3 -> 3)。
    # 原公式为 2*n_out*c*3（按 c->3 的 1x1 算），与实现中的 Conv2d(3,3,1) 不符。
    outpost = n_out * 2 * 9

    # 理解层：直接复用 config 的权威实现，避免此处与 config 各写一套公式后漂移
    # （单位已是 G）
    und = cfg.estimate_understanding_flops()

    # Engram 检索（哈希 + 解码器，粗估）
    engram = 2 * n_main * 16 * c * 2

    G = 1e9
    total = (attn + shared + private + moe_shared + shallow
             + main_up + fuse + bypass + bypass_proj + upsampler + outpost
             + und * G + engram)

    return {
        "attention": attn / G,
        "shared_ffn": shared / G,
        "private_experts": private / G,
        "moe_shared_experts": moe_shared / G,
        "shallow": shallow / G,
        "main_upsample": main_up / G,
        "fuse": fuse / G,
        "bypass": bypass / G,
        "bypass_proj": bypass_proj / G,
        "upsampler": upsampler / G,
        "output_post": outpost / G,
        "understanding": und,
        "engram": engram / G,
        "total": total / G,
    }


def estimate_runtime_memory_mb(cfg: SRConfig, n_params: int) -> dict[str, float]:
    """运行时显存估算。

    对应设计文档 5.6.2 节的"基线 + 浮动"模型。
    注意：此处按【单帧全帧推理】估算，FP16 激活。
    """
    fp16 = 2  # 字节

    def feat(res_h, res_w, ch, n=1):
        return n * res_h * res_w * ch * fp16 / 1024 / 1024

    mh, mw = cfg.main_height, cfg.main_width

    items = {
        "权重（按量化方案）": estimate_weight_memory_mb(cfg, n_params),
        "Engram 表（INT8）": cfg.engram_size_mb,
        "主干激活（2 个特征图）": feat(mh, mw, cfg.main_channels, 2),
        "浅层特征（输入分辨率）": feat(
            cfg.input_height, cfg.input_width, cfg.shallow_channels
        ),
        "输出缓冲（RGB FP16）": feat(
            cfg.output_height, cfg.output_width, 3
        ),
        "Vulkan 运行时开销": 125.0,
    }
    baseline = sum(items.values())
    return {**items, "基线合计": baseline, "峰值（+15%）": baseline * 1.15}


def format_report(cfg: SRConfig, model: nn.Module) -> str:
    """生成可读的分析报告。"""
    lines = []
    add = lines.append

    stats = count_parameters(model)
    total = stats.pop("_total", 0)
    bufs = buffer_bytes(model)

    add("=" * 74)
    add("模型分析报告")
    add("=" * 74)
    add("")
    add("【配置】")
    add("  输入分辨率    : %d x %d" % (cfg.input_width, cfg.input_height))
    add("  输出分辨率    : %d x %d（固定）" % (cfg.output_width, cfg.output_height))
    add("  下采样倍数    : %dx%s"
        % (cfg.effective_downsample,
           "（低分辨率自适应：配置值 %dx）" % cfg.downsample
           if cfg.effective_downsample != cfg.downsample else ""))
    add("  主干分辨率    : %d x %d" % (cfg.main_width, cfg.main_height))
    add("  主干通道/层数 : %d / %d" % (cfg.main_channels, cfg.main_blocks))
    add("  窗口大小      : %d" % cfg.window_size)
    add("  专家总数      : %d（Top-%d，%d 簇）"
        % (cfg.num_experts, cfg.top_k, cfg.num_clusters))
    add("  超分倍数      : %.2fx" % cfg.scale_factor)
    add("")

    add("【参数量】")
    add("  可训练参数    : %s" % format(total, ","))
    add("  缓冲区        : %.2f MB（含 Engram INT8 记忆表）"
        % (bufs / 1024 / 1024))
    add("")
    add("  %-24s %14s %9s" % ("模块", "参数量", "占比"))
    add("  " + "-" * 50)
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        add("  %-24s %14s %8.2f%%" % (k, format(v, ","), 100.0 * v / max(1, total)))
    add("")

    add("【FLOPs 估算】（解析式，设计文档 5.2.7 节同口径）")
    fl = estimate_flops(cfg)
    for k in ["attention", "shared_ffn", "private_experts", "shallow",
              "output_post", "understanding", "engram"]:
        add("  %-20s %10.1f G" % (k, fl[k]))
    add("  " + "-" * 34)
    add("  %-20s %10.1f G" % ("单帧合计", fl["total"]))
    add("")
    add("  %-20s %10.0f GFLOPS" % ("30fps 需求", fl["total"] * 30))
    add("  %-20s %10.0f GFLOPS" % ("60fps 需求", fl["total"] * 60))
    add("")

    add("【运行时显存估算】（单帧全帧推理，FP16 激活）")
    mem = estimate_runtime_memory_mb(cfg, total)
    for k, v in mem.items():
        add("  %-28s %8.1f MB" % (k, v))
    add("")

    return "\n".join(lines)
