"""
档位与参数配置系统。

对应设计文档：
  - 5.1  五档模型族（Low / Middle / High / XHigh / Max）
  - 5.4  用户可自定义滑块参数体系
  - 5.1.0 "专家数"统一为专家总数口径

所有默认值取自 R52 版档位表（设计文档 5.1.2 节）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Tuple


class Tier(str, Enum):
    """五档模型族。"""

    LOW = "Low"
    MIDDLE = "Middle"
    HIGH = "High"
    XHIGH = "XHigh"
    MAX = "Max"


class QuantScheme(str, Enum):
    """量化方案。

    注意：设计文档 4.1 节已确认 W8A8 在本项目中指 INT8，
    与 DeepSeek 的 FP8-W8A8 不是同一格式体系。
    """

    TERNARY = "ternary"  # 三值 -1/0/+1，约 1.58 bit
    W4A8 = "w4a8"  # 4bit 权重 / 8bit 激活
    W8A8 = "w8a8"  # 8bit 权重 / 8bit 激活（INT8）
    FP8 = "fp8"  # 8bit 浮点权重（E4M3/E5M2，需硬件支持）
    FP16 = "fp16"  # 半精度浮点
    FP32 = "fp32"  # 单精度浮点（不量化）


# 各量化方案的每权重比特数（用于显存估算）
BITS_PER_WEIGHT: Dict[QuantScheme, float] = {
    QuantScheme.TERNARY: 1.58,
    QuantScheme.W4A8: 4.0,
    QuantScheme.W8A8: 8.0,
    QuantScheme.FP8: 8.0,
    QuantScheme.FP16: 16.0,
    QuantScheme.FP32: 32.0,
}


@dataclass
class SRConfig:
    """超分模型完整配置。

    字段分组对应设计文档 5.4 节的滑块体系：
      分辨率 / 主干 / MoE / Engram / 量化 / 后处理
    """

    # ---------- 分辨率（R52 修正：固定输出 + 可调输入）----------
    input_height: int = 1080  # 输入分辨率（720p/810p/900p/1080p）
    input_width: int = 1920
    output_height: int = 2160  # 固定输出
    output_width: int = 3840

    # ---------- 主干 ----------
    # 下采样倍数：主干实际分辨率 = 输入 / downsample
    #
    # 低分辨率自适应（见 effective_downsample）：
    #   当输入分辨率本身很低时，下采样会先破坏细节、之后还要靠上采样重建，
    #   是纯损失。此时自动跳过下采样（用 1）。
    downsample: int = 3
    # 是否启用「低分辨率跳过下采样」
    downsample_auto: bool = True
    # 输入高度低于此值则不下采样。默认 540：
    #   五档预设的输入为 736/816/912/1080/1088，全部 >= 540，因此不受影响；
    #   而 patch 训练（192~384）会自动走 downsample=1。
    downsample_skip_below: int = 540
    main_channels: int = 48  # 主干通道数
    main_blocks: int = 4  # Hybrid Block 层数（R32: 各档为 2/3/4/5/6）
    window_size: int = 8  # 窗口注意力边长（8/16/32）
    ffn_ratio: int = 2  # FFN 扩展因子（记录建议由标准 4 降到 2）

    # ---------- MoE ----------
    # 注意：按设计文档 5.1.0，此处为"专家总数"，不是"专家数/簇"
    num_experts: int = 32
    top_k: int = 4
    num_shared_experts: int = 1
    num_clusters: int = 4  # 需整除 num_experts（32/4=8）
    # MoE 作用于主干的层数；None 表示每个 Block 都是 MoE Block
    moe_layers: int | None = None

    # ---------- Engram ----------
    engram_size_mb: int = 25
    engram_ngram: int = 3  # N-Gram 长度
    engram_heads: int = 4  # 哈希头数
    engram_quant: QuantScheme = QuantScheme.W8A8

    # ---------- 理解层 ----------
    # R53：理解层置于第一层
    #
    # 分辨率与规模策略（修订，取代设计文档 2.2.4 的「固定 540p + 参数占比 15%~20%」）
    # ------------------------------------------------------------------------
    # 原设计要求「理解层在 540p 空间运行，参数量占总参数 15%~20%」。
    # 经算力核算，这两条在 540p 全分辨率下互相冲突并产生严重畸变：
    #   理解层算力 ∝ N_理解层 × dim²，主干算力 ∝ N_主干 × C²
    #   而 540p 的 token 数是 Low 档主干（184p）的 17 倍。实测固定
    #   540p + dim=128 时，理解层算力达主干算力的 6226%（Low 档），
    #   即「理解层比整个主干还贵 62 倍」。
    #
    # 修订方案（已与用户确认）：
    #   1. 理解层分辨率随主干缩放（understanding_scale），不再固定 540p
    #   2. 规模约束改为「算力占比」，不再用「参数占比 15%~20%」
    # 效果（分母 = 完整主干算力，见 estimate_main_flops）：
    #   理解层算力占主干比 = 12.3% / 12.9% / 6.2% / 5.7% / 4.7%（Low→Max），
    #   不再是数量级畸变值。
    #
    # 注意：分辨率固定为 540p 是畸变的根因；但理解层参数占比仍随档位下降
    # （8.53% → 1.59%），因为理解层是稠密计算（F/P ∝ N）而主干是稀疏 MoE
    # （F/P ∝ K/专家数）。按用户决定，本期不动规模，先验证结构。
    understanding_enabled: bool = True
    # 理解层分辨率 = 主干分辨率 × understanding_scale
    understanding_scale: float = 1.0
    # 理解层通道数 = 主干通道数 × understanding_dim_ratio
    #
    # 取值依据（实测，含 stem 卷积开销）：
    #   取全局统一值 1.25，无需为每档手工指定
    #   （符合「档位不作为开发阶段处理标准」的原则）。
    #   在补全后的完整主干口径下，该值离 35% 预算尚有较大余量
    #   （最高档位仅 12.9%）；ratio 上限留给后续实验探测。
    understanding_dim_ratio: float = 1.25
    # 理解层算力占【完整主干】算力的预算上限（用于校验，超出则报错）
    understanding_compute_budget: float = 0.35
    # 显式覆盖（None 表示按上面的比例派生）
    understanding_dim: int | None = None

    # ---------- 浅层细节路 ----------
    shallow_channels: int = 32

    # ---------- 高精度旁路 ----------
    bypass_activation: float = 0.15  # 旁路激活率 0~0.25
    bypass_channels: int = 32

    # ---------- 量化 ----------
    # 可选范围（设计文档 4.1 节量化策略表）：
    #   expert_quant : TERNARY / W4A8 / W8A8 / FP16
    #   main_quant   : W4A8 / W8A8 / FP16
    #   bypass_quant : W8A8 / FP8 / FP16
    #
    # 旁路精度说明（用户放宽，取代设计文档原「旁路建议 FP16」）：
    #   旁路【不强制 FP16】。FP8 的精度损失在一定程度上可接受，
    #   在支持 cooperative_matrix FP8 的硬件（RTX 40/50、RDNA4）上
    #   可作为省带宽/省算力的选项。默认仍取 FP16（质量优先）。
    #
    # 接线状态：bypass_quant 已接入 HighPrecisionBypass 与 bypass_proj
    #   （伪量化实现见 quant.py）。可选范围见 validate()：旁路是质量保障
    #   路径，不接受 TERNARY / W4A8 这类激进位宽。
    expert_quant: QuantScheme = QuantScheme.TERNARY
    main_quant: QuantScheme = QuantScheme.W8A8
    bypass_quant: QuantScheme = QuantScheme.FP16

    # ---------- 后处理 ----------
    sharpness: float = 0.5  # 锐度 0~1
    antialiasing: float = 0.5  # 抗锯齿强度 0~1

    # ---------- 派生属性 ----------
    @property
    def effective_downsample(self) -> int:
        """实际生效的下采样倍数。

        低输入分辨率下返回 1（不下采样）。依据为本项目的实测结果：
        同一个 2x 任务、同样 300 步、显存相近：

          ds=2（主干 192²，549k 参数）：验证 PSNR 增益 +0.021 dB，SSIM +0.0003
          ds=1（主干 384²， 86k 参数）：验证 PSNR 增益 +0.411 dB，SSIM +0.0068

        即参数少 6 倍却拿到约 20 倍增益——因为 ds=2 时 384p 被压到 192p，
        主干与理解层都在 192² 上工作，细节先被破坏再靠上采样重建，纯属损失。
        下采样省算力的前提是「有分辨率可省」；输入本来就低时并不成立。
        """
        if self.downsample_auto and self.input_height < self.downsample_skip_below:
            return 1
        return self.downsample

    @property
    def main_height(self) -> int:
        """主干分辨率高度 = 输入高度 / 实际下采样倍数。"""
        return self.input_height // self.effective_downsample

    @property
    def main_width(self) -> int:
        """主干分辨率宽度。"""
        return self.input_width // self.effective_downsample

    @property
    def effective_moe_layers(self) -> int:
        """含私有专家的 Block 数。"""
        return self.main_blocks if self.moe_layers is None else min(
            self.moe_layers, self.main_blocks
        )

    @property
    def scale_factor(self) -> float:
        """超分倍数 = 输出 / 输入。"""
        return self.output_height / self.input_height

    # ---------- 理解层派生量 ----------
    @property
    def understanding_height(self) -> int:
        """理解层运行高度 = 主干高度 × understanding_scale。"""
        return max(16, int(self.main_height * self.understanding_scale))

    @property
    def understanding_width(self) -> int:
        """理解层运行宽度。"""
        return max(16, int(self.main_width * self.understanding_scale))

    @property
    def effective_understanding_dim(self) -> int:
        """理解层通道数。

        默认由主干通道数派生（dim_ratio），也可显式覆盖。
        派生规则保证理解层随档位自然缩放，无需为每档手工指定。
        """
        if self.understanding_dim is not None:
            return self.understanding_dim
        dim = int(self.main_channels * self.understanding_dim_ratio)
        # 取 8 的倍数，便于与主干的通道对齐
        return max(8, (dim // 8) * 8)

    def estimate_understanding_flops(self) -> float:
        """理解层单帧 FLOPs 粗估（G）。

        构成：
          stem    : Conv3x3(3→d/2) + DWConv3x3(d/2) + PW(d/2→d)
          线性层  : 等效 4 个 d×d（context / shared_proj / heads / strategy）
        """
        if not self.understanding_enabled:
            return 0.0
        h = self.understanding_height
        w = self.understanding_width
        n = h * w
        d = self.effective_understanding_dim
        half = max(1, d // 2)

        stem = 2 * n * (9 * 3 * half + 9 * half + half * d)
        linear = 4 * 2 * n * d * d
        return (stem + linear) / 1e9

    def estimate_main_flops(self) -> float:
        """主干单帧 FLOPs（G），口径同设计文档 5.2.7 节。

        含三部分，即「留给重建主干（含专家团）」的完整算力：
          窗口注意力   : L × (8Nc² + 4Nw²c)
          共享专家 FFN : L × 4Nrc²
          私有专家     : L_moe × K × 4Nrc²

        口径修订说明（已与用户确认）：
          本方法原先只统计窗口注意力，但 understanding_compute_budget 的语义是
          「理解层不吃掉留给重建主干的算力」，分母若漏掉 FFN 与专家就低估了主干
          规模、把预算卡得过紧。补全后理解层算力占比由 21%~32% 变为 4.7%~12.9%，
          仍远低于 35% 预算（说明原口径下理解层规模被无谓压小）。
        """
        n = self.main_height * self.main_width
        c = self.main_channels
        w = self.window_size
        r = self.ffn_ratio
        k = self.top_k
        layers = self.main_blocks
        layers_moe = self.effective_moe_layers

        attn = layers * (8 * n * c * c + 4 * n * w * w * c)
        shared = layers * 4 * n * r * c * c
        private = layers_moe * k * 4 * n * r * c * c
        return (attn + shared + private) / 1e9

    @property
    def understanding_compute_ratio(self) -> float:
        """理解层算力 / 完整主干算力（注意力 + 共享 FFN + 私有专家）。"""
        main = self.estimate_main_flops()
        if main <= 0:
            return 0.0
        return self.estimate_understanding_flops() / main

    def validate(self) -> None:
        """校验参数合法性。

        对应设计文档 5.4 节"参数联动约束"。
        """
        errors = []

        # 约束：输入分辨率 ÷ 下采样倍数 ≥ 180p
        if self.main_height < 180:
            errors.append(
                f"主干分辨率 {self.main_height}p 低于下限 180p"
                f"（输入 {self.input_height}p ÷ 下采样 "
                f"{self.effective_downsample}x）"
            )

        # 约束：窗口大小 ≤ 主干分辨率
        if self.window_size > self.main_height:
            errors.append(
                f"窗口大小 {self.window_size} 超过主干分辨率 {self.main_height}p"
            )

        # 约束：Top-K ≤ 专家数
        if self.top_k > self.num_experts:
            errors.append(f"Top-K {self.top_k} 超过专家总数 {self.num_experts}")

        # 约束：专家簇数 ≤ 专家数
        if self.num_clusters > self.num_experts:
            errors.append(
                f"专家簇数 {self.num_clusters} 超过专家总数 {self.num_experts}"
            )

        # 约束：窗口大小必须整除主干分辨率
        if self.main_height % self.window_size != 0:
            errors.append(
                f"窗口大小 {self.window_size} 不能整除主干高度 {self.main_height}"
            )
        if self.main_width % self.window_size != 0:
            errors.append(
                f"窗口大小 {self.window_size} 不能整除主干宽度 {self.main_width}"
            )

        # 约束：通道数应为 8 的倍数（便于三值打包与 cooperative_matrix 对齐）
        if self.main_channels % 8 != 0:
            errors.append(f"主干通道数 {self.main_channels} 不是 8 的倍数")

        # 约束：旁路激活率范围
        if not 0.0 <= self.bypass_activation <= 0.25:
            errors.append(
                f"旁路激活率 {self.bypass_activation} 超出 0~0.25"
                "（设计文档 5.4：超过 25% 会抵消低精度收益）"
            )

        # 约束：旁路量化方案范围
        # 旁路是质量保障路径，只接受较保守的位宽；FP32 表示不做伪量化。
        allowed_bypass = {
            QuantScheme.W8A8,
            QuantScheme.FP8,
            QuantScheme.FP16,
            QuantScheme.FP32,
        }
        if self.bypass_quant not in allowed_bypass:
            errors.append(
                f"旁路量化方案 {self.bypass_quant.value} 不在允许范围内"
                f"（{'/'.join(sorted(s.value for s in allowed_bypass))}）"
                "——旁路为质量保障路径，不接受三值/4bit 等激进位宽"
            )

        # 约束：下采样倍数应为整数且能整除输入（按实际生效的倍数校验）
        if self.input_height % self.effective_downsample != 0:
            errors.append(
                f"下采样倍数 {self.effective_downsample} 不能整除输入高度"
                f" {self.input_height}"
            )
        if self.input_width % self.effective_downsample != 0:
            errors.append(
                f"下采样倍数 {self.effective_downsample} 不能整除输入宽度"
                f" {self.input_width}"
            )

        # 约束：专家总数必须能被专家簇数整除（每个簇的专家数需为整数）
        if self.num_experts % self.num_clusters != 0:
            errors.append(
                f"专家总数 {self.num_experts} 不能被簇数 {self.num_clusters} 整除"
                f"（每簇专家数需为整数）"
            )

        # 约束：注意力头数必须能整除通道数
        # 实现中头数取 max(1, C // 16)，故需校验该值能整除 C
        num_heads = max(1, self.main_channels // 16)
        if self.main_channels % num_heads != 0:
            errors.append(
                f"通道数 {self.main_channels} 不能被注意力头数 {num_heads} 整除"
                f"（头数由 max(1, C//16) 推出）"
            )

        # 约束：理解层算力不得超过主干算力的预算上限
        # （取代原「参数占比 15%~20%」——该指标在 540p 下会畸变，见字段注释）
        if self.understanding_enabled:
            ratio = self.understanding_compute_ratio
            if ratio > self.understanding_compute_budget:
                errors.append(
                    f"理解层算力占主干 {ratio * 100:.0f}%，超出预算 "
                    f"{self.understanding_compute_budget * 100:.0f}%。"
                    f"可降低 understanding_scale（当前 "
                    f"{self.understanding_scale}）或 understanding_dim_ratio"
                    f"（当前 {self.understanding_dim_ratio}）"
                )

        if errors:
            raise ValueError("配置校验失败:\n  - " + "\n  - ".join(errors))

    def with_overrides(self, **kwargs) -> "SRConfig":
        """返回覆盖部分字段后的新配置（用于用户滑块调节）。"""
        return replace(self, **kwargs)


# ---------------------------------------------------------------------------
# 五档预设（取自设计文档 5.1.2 R52 版档位表）
#
# 注意：设计文档的档位表未考虑"窗口必须整除主干分辨率"这一实现约束，
# 直接套用会导致 Low/Middle/High 三档无法构建。此处对输入分辨率与通道数
# 做了最小幅度调整，使全部档位自洽。调整记录见各档注释。
# ---------------------------------------------------------------------------

TIER_PRESETS: Dict[Tier, SRConfig] = {
    # Low: 设计值 720p/4x=180p，180 不能被窗口 8 整除 → 改用 736p/4x=184p（184=23x8）
    #      通道 28 不是 8 的倍数 → 改用 24
    Tier.LOW: SRConfig(
        input_height=736,
        input_width=1280,
        downsample=4,  # 736 / 4 = 184p = 23 x 8
        main_blocks=2,  # R32: 各档 Block 数 2/3/4/5/6
        main_channels=24,
        num_experts=16,
        top_k=2,
        num_clusters=2,
        bypass_activation=0.04,
        shallow_channels=16,
        engram_size_mb=10,
        window_size=8,
    ),
    # Middle: 设计值 810p/3x=270p，270 不能被 8 整除 → 改用 816p/3x=272p（272=34x8）
    #         通道 36 不是 8 的倍数 → 改用 32
    #         专家 32 不能被簇数 3 整除 → 簇数改用 4（32/4=8）
    Tier.MIDDLE: SRConfig(
        input_height=816,
        input_width=1440,
        downsample=3,  # 816 / 3 = 272p = 34 x 8
        main_blocks=3,  # R32: 各档 Block 数 2/3/4/5/6
        main_channels=32,
        num_experts=32,
        top_k=2,
        num_clusters=4,  # 32 / 4 = 8 专家/簇
        bypass_activation=0.11,
        shallow_channels=24,
        engram_size_mb=25,
        window_size=8,
    ),
    # High: 设计值 900p/3x=300p、宽 1600/3=533（533 不能被 8 整除）
    #       → 改用 912p/3x=304p（304=38x8）、宽 1632/3=544（544=68x8）
    #       专家 32 不能被簇数 3 整除 → 簇数改用 4
    Tier.HIGH: SRConfig(
        input_height=912,
        input_width=1632,
        downsample=3,  # 912/3 = 304p = 38x8;  1632/3 = 544 = 68x8
        main_blocks=4,  # R32: 各档 Block 数 2/3/4/5/6
        main_channels=48,
        num_experts=32,
        top_k=4,
        num_clusters=4,  # 32 / 4 = 8 专家/簇
        bypass_activation=0.15,
        shallow_channels=32,
        engram_size_mb=25,
        window_size=8,
    ),
    # XHigh: 设计值 1080p/3x=360p（360=45x8，可被 8 整除，宽 1920/3=640=80x8 亦可）
    #        通道 56 使头数 c//16=3，56 不能被 3 整除 → 通道改用 48
    #        （也可保持 56 并把头数固定为 4；此处选通道 48 以保持 8 的倍数且头数=3 可整除）
    Tier.XHIGH: SRConfig(
        input_height=1080,
        input_width=1920,
        downsample=3,  # 1080/3 = 360p = 45x8;  1920/3 = 640 = 80x8
        main_blocks=5,  # R32: 各档 Block 数 2/3/4/5/6
        main_channels=64,  # 64 可被 8 整除，且 64/16=4 头可整除
        num_experts=48,
        top_k=4,
        num_clusters=4,  # 48 / 4 = 12 专家/簇
        bypass_activation=0.18,
        shallow_channels=40,
        engram_size_mb=40,
        window_size=8,
    ),
    # Max: 设计值 1080p/2x=540p，540 不能被 8 整除 → 改用 1088p/2x=544p（544=68x8）
    Tier.MAX: SRConfig(
        input_height=1088,
        input_width=1920,
        downsample=2,  # 1088 / 2 = 544p = 68 x 8
        main_blocks=6,  # R32: 各档 Block 数 2/3/4/5/6
        main_channels=64,
        num_experts=64,
        top_k=4,
        num_clusters=4,
        bypass_activation=0.20,
        shallow_channels=48,
        engram_size_mb=50,
        window_size=8,
    ),
}


def get_preset(tier: Tier, **overrides) -> SRConfig:
    """获取指定档位的预设配置，可附加覆盖。

    参数
    ----
    tier : Tier
        档位。
    **overrides
        需要覆盖的字段，对应用户滑块调节。

    返回
    ----
    SRConfig
        已校验的配置对象。
    """
    cfg = TIER_PRESETS[tier].with_overrides(**overrides)
    cfg.validate()
    return cfg
