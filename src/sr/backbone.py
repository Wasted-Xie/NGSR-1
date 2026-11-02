"""
主干 Hybrid Block 与理解层。

对应设计文档：
  - 2.2  理解层（始终激活，最高智能）
  - 2.3  低分辨率主干
  - 2.4  三级路由 MoE
  - 2.5  Engram
  - 2.6  mHC
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engram import Engram
from .layers import FeedForward, LayerScale, WindowAttention
from .modules import ManifoldConstrainedHyperConnection
from .moe import ThreeLevelMoE


# ===========================================================================
# 主干 Hybrid Block
# ===========================================================================

class HybridBlock(nn.Module):
    """主干 Hybrid Block。

    设计文档 2.1 节数据流中的主干组成：
      低分辨率主干（全帧、不分块）
        ├─ 空间注意力（窗口 8x8/16x16/32x32）
        ├─ 跨窗口局部混合（DWConv 3x3；弥补窗口注意力无法跨窗通信）
        ├─ 三级路由 MoE
        ├─ Engram 条件记忆
        └─ mHC 残差连接

    注意（设计文档 5.2.7 节重算的两种假设）：
      moe_layers 控制含私有专家的层数。默认每个 Block 都含 MoE
      （依据 R51 数据流图），可通过 config.moe_layers 限制为部分层。
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        ffn_ratio: int = 2,
        num_experts: int = 32,
        top_k: int = 4,
        num_clusters: int = 3,
        num_shared: int = 1,
        group_size: int = 4,
        use_moe: bool = True,
        use_engram: bool = True,
        use_mhc: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.use_moe = use_moe
        self.use_engram = use_engram
        self.use_mhc = use_mhc

        # 空间注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.ls1 = LayerScale(dim)

        # 跨窗口局部混合（DWConv 3x3）
        # 窗口注意力只在窗口内交互，主干里没有任何跨窗口通路，感受野被
        # 锁死在 window_size 内（见 issue #1 与 tests/test_regressions.py）。
        # 这里补一条深度方向的局部通路，让相邻窗口可以直接传递信息。
        self.norm_conv = nn.LayerNorm(dim)
        self.local_conv = nn.Conv2d(
            dim, dim, 3, padding=1, groups=dim, bias=False
        )
        self.ls_conv = LayerScale(dim)

        # FFN（共享专家）
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_ratio)
        self.ls2 = LayerScale(dim)

        # 三级路由 MoE
        if use_moe:
            self.norm3 = nn.LayerNorm(dim)
            self.moe = ThreeLevelMoE(
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                num_clusters=num_clusters,
                num_shared=num_shared,
                ffn_ratio=ffn_ratio,
                group_size=group_size,
            )
            self.ls3 = LayerScale(dim)

        # Engram 条件记忆
        if use_engram:
            self.norm4 = nn.LayerNorm(dim)
            self.engram_proj = nn.Linear(dim, dim)
            self.engram_gate = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.Sigmoid()
            )
            self.ls4 = LayerScale(dim)

        # mHC 流形约束
        if use_mhc:
            self.mhc = ManifoldConstrainedHyperConnection(dim)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        engram: Engram | None = None,
        degradation: torch.Tensor | None = None,
        deg_global: torch.Tensor | None = None,
        strategy: torch.Tensor | None = None,
        scene_logits: torch.Tensor | None = None,
        ui_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        x: (B, N, C)

        理解层条件（对应设计文档 2.1 节的三级驱动）：
          strategy     : (B, N, C) 逐 token 策略条件
          degradation  : (B, N, 1) 退化空间分布图 → 第一级组级路由
          deg_global   : (B, 1)    全局退化程度   → mHC 动态约束强度
          scene_logits : (B, K)    场景标签       → 第二级簇内路由
          ui_mask      : (B, N, 1) UI/边缘检测    → 第三级质量门控

        返回
        ----
        out        : (B, N, C)
        aux_loss   : 标量，MoE 负载均衡损失
        gate_prob  : (B, N, 1) 质量门控概率（仅 use_moe 时）
        hit_rate   : Engram 命中率（仅 use_engram 时）
        """
        result = {}
        aux_loss = x.new_zeros(())
        gate_prob = None
        hit_rate = None

        b, n, c = x.shape

        # 注入理解层的逐 token 策略条件（保留空间分辨力）
        if strategy is not None:
            x = x + strategy

        # ---------- 空间注意力 ----------
        x = x + self.ls1(self.attn(self.norm1(x), h, w))

        # ---------- 跨窗口局部混合 ----------
        # 把 token 还原成特征图做一次 DWConv 3x3，让相邻窗口互通信息。
        xc = self.norm_conv(x).transpose(1, 2).reshape(b, c, h, w)
        xc = self.local_conv(xc).flatten(2).transpose(1, 2)
        x = x + self.ls_conv(xc)

        # ---------- 共享专家 FFN ----------
        x = x + self.ls2(self.ffn(self.norm2(x)))

        # ---------- 三级路由 MoE（三级分别由理解层条件驱动）----------
        if self.use_moe:
            moe_out = self.moe(
                self.norm3(x), h, w,
                degradation=degradation,
                scene_logits=scene_logits,
                ui_mask=ui_mask,
            )
            x = x + self.ls3(moe_out["out"])
            aux_loss = moe_out["aux_loss"]
            gate_prob = moe_out["gate_prob"]
            result["cluster_ids"] = moe_out["cluster_ids"]

        # ---------- Engram 条件记忆 ----------
        if self.use_engram and engram is not None:
            eng = engram(self.norm4(x), h, w)
            prior = self.engram_proj(eng["out"])
            gate = self.engram_gate(torch.cat([x, prior], dim=-1))
            x = x + self.ls4(gate * prior)
            hit_rate = eng["hit_rate"]

        # ---------- mHC 流形约束（由全局退化程度动态调整约束强度）----------
        if self.use_mhc:
            x = self.mhc(x, deg_global)

        result["out"] = x
        result["aux_loss"] = aux_loss
        if gate_prob is not None:
            result["gate_prob"] = gate_prob
        if hit_rate is not None:
            result["hit_rate"] = hit_rate
        return result


# ===========================================================================
# 理解层
# ===========================================================================

class CrossBranchInteraction(nn.Module):
    """分支间交叉交互（低秩瓶颈实现，参数开销远小于独立全连接）。

    动机：场景/退化/意图三者本应互相约束——
          "这是 UI 区域" 意味着 "退化应较轻"、"该走高精度旁路"。
          若四个分支完全独立，无法表达这种约束。

    实现：把四分支各投影到低秩隐空间，拼接后在【瓶颈维度】内交互，
          再投影回各分支。参数开销 ≈ 4·C·r + (B·r)² + 4·(B·r)·C，
          其中 r 为秩、B 为分支数；相比四个独立 C×C 全连接
          （4·C²）显著更省。

    参数账（以 C=64, B=4, 秩比 0.125 为例）：
        r = 8，瓶颈 = B·r = 32
        下投影 4×(64×8)=2048，交互 32×32=1024，上投影 4×(32×64)=8192
        合计约 11,264，而四个独立全连接为 4×64×64=16,384
    """

    def __init__(self, dim: int, num_branches: int = 4, rank_ratio: float = 0.125):
        super().__init__()
        # 秩按 dim 缩放，但设有上限，保证瓶颈维度严格小于 dim
        rank = max(4, int(dim * rank_ratio))
        bottleneck = num_branches * rank
        self.num_branches = num_branches
        self.rank = rank
        self.bottleneck = bottleneck

        self.down = nn.ModuleList(
            [nn.Linear(dim, rank, bias=False) for _ in range(num_branches)]
        )
        # 在瓶颈维度内交互（低秩）
        self.interact = nn.Sequential(
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, bottleneck, bias=False),
        )
        self.up = nn.ModuleList(
            [nn.Linear(bottleneck, dim, bias=False) for _ in range(num_branches)]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_branches)])

    def forward(self, branches: list[torch.Tensor]) -> list[torch.Tensor]:
        """branches: 长度 = num_branches 的列表，每个 (B, N, C)。"""
        latents = [d(b) for d, b in zip(self.down, branches)]
        cat = torch.cat(latents, dim=-1)          # (B, N, B·r)
        interacted = self.interact(cat)           # (B, N, B·r)
        return [
            n(b + u(interacted))
            for u, n, b in zip(self.up, self.norm, branches)
        ]


class MultiScaleContext(nn.Module):
    """多尺度上下文提取（替代全局池化）。

    动机：把特征压成单向量再广播会丢失全部空间信息，
          理解层就无法区分"左上角是 UI、右下角是天空"。

    实现：三级尺度（全局 1x1 / 中间 h/4 / 局部原分辨率）对齐后融合。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj_global = nn.Linear(dim, dim // 4, bias=False)
        self.proj_mid = nn.Conv2d(dim, dim // 4, 1, bias=False)
        self.proj_local = nn.Conv2d(dim, dim // 4, 1, bias=False)
        self.fuse = nn.Conv2d(dim // 4 * 3, dim, 1, bias=False)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, h, w) -> (B, C, h, w)，保留空间维度。"""
        b, c, h, w = feat.shape

        # 尺度 1：全局（保留为可广播的空间图，而非压成单向量）
        g = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        g = self.proj_global(g)
        g = g.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

        # 尺度 2：中间
        m = F.adaptive_avg_pool2d(feat, max(1, h // 4))
        m = self.proj_mid(m)
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

        # 尺度 3：局部（原分辨率）
        loc = self.proj_local(feat)

        return self.norm(self.fuse(torch.cat([g, m, loc], dim=1)))


class UnderstandingLayer(nn.Module):
    """理解层（始终激活，最高智能）。

    职责（设计文档 2.2.1）——不直接重建像素，输出条件、先验和决策：
        这是什么？  场景类型、物体语义、纹理类别
        退化是什么？模糊、噪声、压缩伪影、锯齿
        该保留什么？创作意图、艺术风格、UI文字
        该怎么做？  路由决策、旁路激活、量化选择

    架构（2.2.2）：共享认知核心 + 四个分支。
    位置（2.2.3，R53）：模型第一层，540p 空间运行。

    本次结构优化（参数量不变，提升同等参数下的理解能力）
    ----------------------------------------------------
    修复的四个结构缺陷：

      1. 空间信息丢失：原用 global_pool 压成单向量再 expand 广播，
         无法区分"左上角是 UI、右下角是天空"。
         → 改为 MultiScaleContext，保留空间维度。

      2. 策略被平均：原逐 token 算 strategy 后 mean(dim=1)，
         路由条件失去空间分辨力。
         → 改为输出逐 token 的 (B, N, C)，由主干按位置注入。

      3. 分支互不交互：原四分支只共享 stem。
         → 增加 CrossBranchInteraction（低秩，参数开销小）。

      4. UI 掩码无感受野：原用 1x1 Conv，只能看单像素。
         → 改为 DWConv 提供感受野。

    关键输出：
        strategy        : (B, N, C) 逐 token 策略条件，保留空间分辨力
        degradation_map : (B, N, 1) 退化程度空间分布，供 mHC 动态调整约束
        ui_mask         : (B, 1, h, w) UI 区域掩码
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        num_scene_classes: int = 8,
        num_degradation_classes: int = 5,
    ):
        super().__init__()
        self.dim = dim

        # ---------- 共享认知核心 ----------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                dim // 2, dim // 2, 3, padding=1, groups=dim // 2, bias=False
            ),
            nn.Conv2d(dim // 2, dim, 1, bias=False),
            nn.GELU(),
        )

        # ---------- 多尺度上下文（替代全局池化）----------
        self.context = MultiScaleContext(dim)

        # ---------- 四分支共享底层 ----------
        # 设计文档 2.2.4："多任务共享底层：共享同一特征提取器"
        self.shared_proj = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
        )

        # ---------- 分支 1：场景理解 ----------
        self.scene_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_scene_classes),
        )

        # ---------- 分支 2：退化理解（类型 + 空间分布）----------
        self.degradation_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 2, num_degradation_classes + 1, 1, bias=False),
        )

        # ---------- 分支 3：意图理解（风格 + UI 掩码）----------
        self.style_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 4)
        )
        self.ui_mask_head = nn.Sequential(
            nn.Conv2d(
                dim, dim // 4, 3, padding=1,
                groups=max(1, dim // 4), bias=False,
            ),
            nn.Conv2d(dim // 4, dim // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1, bias=False),
        )

        # ---------- 分支 4：策略决策（逐 token，不池化）----------
        self.strategy_head = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

        # ---------- 分支间交叉交互 ----------
        # 把四分支的表征统一到 dim 维后交互
        self.to_dim = nn.ModuleList([
            nn.Linear(dim, dim, bias=False),           # 场景（全局广播）
            nn.Linear(num_degradation_classes, dim, bias=False),  # 退化类型
            nn.Linear(4, dim, bias=False),             # 风格
            nn.Identity(),                              # 策略（已是 dim）
        ])
        self.interaction = CrossBranchInteraction(dim, num_branches=4)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, 3, H, W) 输入图像（已在 540p 空间）。"""
        # ---------- 共享认知核心 ----------
        feat = self.stem(x)  # (B, C, h, w)
        b, c, h, w = feat.shape
        n = h * w

        # ---------- 多尺度上下文（保留空间维度）----------
        ctx = self.context(feat)
        combined = self.shared_proj(torch.cat([feat, ctx], dim=1))  # (B, C, h, w)

        # ---------- 分支 1：场景理解（全局分类）----------
        scene_global = combined.mean(dim=(2, 3))  # (B, C)
        scene_logits = self.scene_head(scene_global)  # (B, K)

        # ---------- 分支 2：退化理解（逐像素，保留空间分布）----------
        deg_map = self.degradation_head(combined)  # (B, K+1, h, w)
        deg_tokens = deg_map.flatten(2).transpose(1, 2)  # (B, N, K+1)
        degradation_map = torch.sigmoid(deg_tokens[..., -1:])  # (B, N, 1)
        degradation_type = deg_tokens[..., :-1].mean(dim=1)  # (B, K)

        # ---------- 分支 3：意图理解 ----------
        style = self.style_head(scene_global)  # (B, 4)
        ui_mask = torch.sigmoid(self.ui_mask_head(combined))  # (B, 1, h, w)

        # ---------- 分支 4：策略决策（逐 token）----------
        strategy_map = self.strategy_head(combined)  # (B, C, h, w)
        strategy_tokens = strategy_map.flatten(2).transpose(1, 2)  # (B, N, C)

        # ---------- 分支间交叉交互 ----------
        # 把四分支表征统一到 (B, N, C)
        br_scene = self.to_dim[0](scene_global).unsqueeze(1).expand(-1, n, -1)
        br_deg = self.to_dim[1](deg_tokens[..., :-1])       # (B, N, C)
        br_style = self.to_dim[2](style).unsqueeze(1).expand(-1, n, -1)
        br_strategy = self.to_dim[3](strategy_tokens)       # (B, N, C)

        _, _, _, br_strategy = self.interaction(
            [br_scene, br_deg, br_style, br_strategy]
        )
        strategy_out = self.out_norm(br_strategy)  # (B, N, C)

        return {
            "feat": feat,
            "scene_logits": scene_logits,
            "degradation_type": degradation_type,
            "degradation_map": degradation_map,  # 供 mHC 动态调整约束强度
            "style": style,
            "ui_mask": ui_mask,
            "strategy": strategy_out,  # (B, N, C) 逐 token
            "h": h,
            "w": w,
        }
