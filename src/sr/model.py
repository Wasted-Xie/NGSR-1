"""
完整超分模型组装。

对应设计文档 2.1 节的数据流（R53 修正版，理解层前置）：

    输入分辨率（720p~1080p，用户可调）
      │
      ├─ 浅层细节路：Conv 3x3 x2，保留输入分辨率的高频信息
      │
      └─ 理解层（第一层激活，540p 空间运行，轻量）
           ├─ 场景理解 → 全局条件向量
           ├─ 退化理解 → 退化类型 + 空间分布图
           ├─ 意图理解 → 艺术风格 + UI 掩码
           └─ 策略决策 → 路由条件 + 旁路条件 + 量化条件
      │
      └─ 下采样（2x/3x/4x，用户可调）→ 180p~540p
           └─ 低分辨率主干（全帧、不分块）
                ├─ 空间注意力
                ├─ 三级路由 MoE
                ├─ Engram 条件记忆
                └─ mHC 残差连接
      │
      └─ 两路在输入分辨率空间融合
      └─ 自适应上采样（网络内部固定 2x + 轻量缩放）
      └─ 固定输出分辨率（2160p/1440p）

    输出 = 双三次上采样 + 主干低频残差 + 浅层高频残差 + 旁路修正残差
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import HybridBlock, UnderstandingLayer
from .config import SRConfig
from .engram import Engram
from .layers import AdaptiveUpsample, DepthwiseSeparableConv, to_2d, to_tokens
from .modules import HighPrecisionBypass, ShallowDetailPath
from .quant import QuantConv2d


class SuperResolutionModel(nn.Module):
    """自研实时超分模型。

    设计文档 2.1 节的完整数据流实现。
    """

    def __init__(self, config: SRConfig):
        super().__init__()
        config.validate()
        self.config = config

        c = config.main_channels

        # ---------- 浅层细节路（输入分辨率）----------
        self.shallow = ShallowDetailPath(
            in_channels=3, out_channels=config.shallow_channels
        )
        # 把浅层特征投影到主干通道数，便于融合
        self.shallow_proj = nn.Conv2d(
            config.shallow_channels, c, 1, bias=False
        )

        # ---------- 理解层（第一层激活，分辨率随主干缩放）----------
        if config.understanding_enabled:
            self.understanding = UnderstandingLayer(
                in_channels=3, dim=config.effective_understanding_dim
            )
            # 理解层的策略条件投影到主干通道
            self.strategy_proj = nn.Linear(
                config.effective_understanding_dim, c
            )
        else:
            self.understanding = None

        # ---------- 下采样 + 主干输入投影 ----------
        # 用 effective_downsample：输入分辨率过低时自动跳过下采样（见 config）
        ds = config.effective_downsample
        self.downsample = nn.Conv2d(3, c, ds, stride=ds, bias=False)
        self.main_norm = nn.LayerNorm(c)

        # ---------- Engram（主干共享）----------
        self.engram = Engram(
            dim=c,
            size_mb=config.engram_size_mb,
            hash_bits=config.engram_hash_bits,
            mode=config.engram_mode,
            dynamic_slots=config.engram_dynamic_slots,
            ngram=config.engram_ngram,
            num_heads=config.engram_heads,
        )

        # ---------- 主干 Hybrid Blocks ----------
        self.blocks = nn.ModuleList()
        for i in range(config.main_blocks):
            # 按 config.moe_layers 决定哪些层含私有专家
            use_moe = i < config.effective_moe_layers
            self.blocks.append(
                HybridBlock(
                    dim=c,
                    window_size=config.window_size,
                    num_heads=max(1, c // 16),
                    ffn_ratio=config.ffn_ratio,
                    num_experts=config.num_experts,
                    top_k=config.top_k,
                    num_clusters=config.num_clusters,
                    num_shared=config.num_shared_experts,
                    use_moe=use_moe,
                )
            )
        self.final_norm = nn.LayerNorm(c)

        # ---------- 融合（在输入分辨率空间）----------
        # 主干输出上采样到输入分辨率后与浅层特征融合
        self.main_upsample = AdaptiveUpsample(
            c, config.shallow_channels, scale=2
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(
                config.shallow_channels * 2, c, 1, bias=False
            ),
            nn.GELU(),
            DepthwiseSeparableConv(c),
        )

        # ---------- 高精度旁路 ----------
        # 旁路精度由 config.bypass_quant 控制（伪量化，见 quant.py）。
        # bypass_proj 同属旁路路径（把旁路残差投回主干空间），故共用该方案。
        self.bypass = HighPrecisionBypass(
            dim=c, channels=config.bypass_channels, quant=config.bypass_quant
        )
        self.bypass_proj = QuantConv2d(
            c, c, 1, bias=False, quant=config.bypass_quant
        )

        # ---------- 自适应上采样到输出分辨率 ----------
        # 网络内部固定 2x（R53 修正），最终缩放对齐
        self.upsampler = AdaptiveUpsample(c, 3, scale=2)
        self.post = nn.Sequential(
            nn.Conv2d(3, 3, 1, bias=False),
        )

        # 全局残差：双三次上采样基线
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        x: (B, 3, H_in, W_in) 低分辨率输入

        返回
        ----
        sr        : (B, 3, H_out, W_out) 超分结果
        aux_loss  : MoE 负载均衡损失
        hit_rate  : Engram 命中率
        gate_prob : 平均质量门控概率
        """
        cfg = self.config
        b, _, h_in, w_in = x.shape
        out_hw = (cfg.output_height, cfg.output_width)

        # ========== 1. 浅层细节路 ==========
        shallow_out = self.shallow(x)
        shallow_feat = shallow_out["feat"]  # (B, Cs, H, W)

        # ========== 2. 理解层（第一层激活）==========
        strategy_tokens = None
        degradation_map = None
        scene_logits = None
        ui_mask_full = None
        ui_token_map = None
        uh = uw = 0
        if self.understanding is not None:
            # 理解层分辨率 = 主干分辨率 × understanding_scale（随档位缩放）
            uh = cfg.understanding_height
            uw = cfg.understanding_width
            x_und = F.interpolate(
                x, size=(uh, uw), mode="bilinear", align_corners=False
            )
            und = self.understanding(x_und)
            strategy_tokens = und["strategy"]          # (B, N_u, C_u) 逐 token
            degradation_map = und["degradation_map"]   # (B, N_u, 1)
            scene_logits = und["scene_logits"]         # (B, K)
            ui_mask_full = und["ui_mask"]              # (B, 1, h_u, w_u)

        # ========== 3. 下采样到主干分辨率 ==========
        main = self.downsample(x)  # (B, C, h_m, w_m)
        h_m, w_m = main.shape[-2:]

        # 按 config.validate 保证窗口能整除
        ws = cfg.window_size
        if h_m % ws != 0 or w_m % ws != 0:
            raise ValueError(
                f"主干分辨率 {h_m}x{w_m} 不能被窗口 {ws} 整除。"
                f"请调整 input_height/width 或 downsample。"
            )

        tokens = to_tokens(main)  # (B, N, C)
        tokens = self.main_norm(tokens)

        # ---------- 把理解层条件对齐到主干分辨率 ----------
        # 理解层在 540p 运行，主干在更低分辨率，需插值对齐。
        # 设计文档 2.1 节要求三级路由分别由三种条件驱动：
        #   退化分布图 → 第一级组级路由
        #   场景标签   → 第二级簇内路由
        #   UI/边缘    → 第三级质量门控
        deg_tokens = None   # (B, N, 1)
        ui_tokens = None    # (B, N, 1)

        if strategy_tokens is not None:
            b_s, n_u, cu = strategy_tokens.shape
            # (B, N_u, C_u) -> (B, C_u, h_u, w_u) -> 插值到主干尺寸
            s_map = strategy_tokens.transpose(1, 2).reshape(b_s, cu, uh, uw)
            s_map = F.interpolate(
                s_map, size=(h_m, w_m), mode="bilinear", align_corners=False
            )
            tokens = tokens + self.strategy_proj(to_tokens(s_map))

        if degradation_map is not None:
            b_s, n_u, _ = degradation_map.shape
            d_map = degradation_map.transpose(1, 2).reshape(b_s, 1, uh, uw)
            d_map = F.interpolate(
                d_map, size=(h_m, w_m), mode="bilinear", align_corners=False
            )
            deg_tokens = to_tokens(d_map)              # (B, N, 1)

        if ui_mask_full is not None:
            ui_map = F.interpolate(
                ui_mask_full, size=(h_m, w_m),
                mode="bilinear", align_corners=False,
            )
            ui_tokens = to_tokens(ui_map)              # (B, N, 1)

        # 退化信息有两种用途，形状要求不同，需分开传递：
        #   deg_tokens : (B, N, 1) 逐 token 空间分布图 → 第一级组级路由
        #   deg_global : (B, 1)    全局退化程度标量   → mHC 动态约束强度
        deg_global = None
        if degradation_map is not None:
            deg_global = degradation_map.mean(dim=1, keepdim=True)  # (B, 1)

        # ========== 4. 主干（全帧、不分块）==========
        aux_loss = tokens.new_zeros(())
        hit_rates = []
        gate_probs = []

        for blk in self.blocks:
            res = blk(
                tokens, h_m, w_m,
                engram=self.engram,
                degradation=deg_tokens,      # 空间分布图 → 组级路由
                deg_global=deg_global,       # 全局标量   → mHC
                scene_logits=scene_logits,
                ui_mask=ui_tokens,
            )
            tokens = res["out"]
            aux_loss = aux_loss + res["aux_loss"]
            if "hit_rate" in res:
                hit_rates.append(res["hit_rate"])
            if "gate_prob" in res:
                gate_probs.append(res["gate_prob"])

        tokens = self.final_norm(tokens)
        main_out = to_2d(tokens, h_m, w_m)  # (B, C, h_m, w_m)

        # ========== 5. 主干上采样到输入分辨率并融合 ==========
        main_up = self.main_upsample(main_out, target_hw=(h_in, w_in))
        fused = torch.cat([main_up, shallow_feat], dim=1)
        fused = self.fuse(fused)  # (B, C, H, W)

        # ========== 6. 高精度旁路 ==========
        fused_tokens = to_tokens(fused)
        # 平均门控概率
        if gate_probs:
            gate = torch.stack(gate_probs, dim=0).mean(dim=0)  # (B, N_m, 1)
            gate = F.interpolate(
                gate.transpose(1, 2).reshape(b, 1, h_m, w_m),
                size=(h_in, w_in),
                mode="bilinear",
                align_corners=False,
            )
            gate = to_tokens(gate)  # (B, N, 1)
        else:
            gate = None

        shallow_tokens = to_tokens(self.shallow_proj(shallow_feat))
        bypass_res = self.bypass(
            fused_tokens, h_in, w_in, shallow=shallow_tokens, gate=gate
        )
        bypass_res = to_2d(bypass_res, h_in, w_in)
        fused = fused + self.bypass_proj(bypass_res)

        # ========== 7. 自适应上采样到输出分辨率 ==========
        # 网络内部固定 2x（R53），再缩放对齐输出
        sr = self.upsampler(fused, target_hw=out_hw)
        sr = self.post(sr)

        # ========== 8. 全局残差（双三次基线）==========
        base = F.interpolate(x, size=out_hw, mode="bicubic", align_corners=False)
        sr = base + self.residual_scale * sr

        # 应用后处理参数（锐度 / 抗锯齿强度）
        if cfg.sharpness != 0.5:
            blur = F.avg_pool2d(sr, 3, stride=1, padding=1)
            sr = sr + (cfg.sharpness - 0.5) * 2.0 * (sr - blur)

        out = {
            "sr": sr,
            "aux_loss": aux_loss,
        }
        if hit_rates:
            out["hit_rate"] = torch.stack(hit_rates).mean()
        if gate is not None:
            out["gate_prob"] = gate.mean()
        if ui_mask_full is not None:
            # 对外输出理解层的 UI 掩码（原始理解层分辨率）
            out["ui_mask"] = ui_mask_full

        return out

    # ------------------------------------------------------------------
    # 统计工具
    # ------------------------------------------------------------------
    def count_parameters(self) -> dict[str, int]:
        """统计各模块参数量。

        设计文档 3.6 节指标：PSNR/参数（参数效率）。
        """
        stats = {}
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            if n > 0:
                stats[name] = n
        stats["_total"] = sum(p.numel() for p in self.parameters())
        return stats
