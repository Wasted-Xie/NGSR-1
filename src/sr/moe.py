"""
MoE 模块：三级路由。

对应设计文档：
  - 2.4    三级路由 MoE（组级路由 → 簇内路由 → 质量门控）
  - 2.4.2  专家数量往多的方向（三值量化使多专家可行）
  - 4.1    量化策略：路由网络用 W8A8/FP16，私有专家用三值
  - 5.1.0  "专家数"统一为专家总数口径

三级语义（R21 修正后）：
  第一级：组级路由  → 低精度专家簇（W4A8/三值）
  第二级：簇内路由  → 低精度专家（Top-K）
  第三级：质量门控  → 高精度旁路（FP16）

注意：R19 原始设计中第三级是"零专家槽位（直接跳过计算）"，
      R21 将其修正为"质量门控 → 高精度旁路"（不是跳过，而是走旁路）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupRouter(nn.Module):
    """第一级：组级路由。

    设计文档 2.4.1 节：
      把 token 按空间位置划分为若干组，每组做一次粗粒度路由，
      决定该组激活哪个专家簇（Top-1 簇）。

    价值：路由决策数从 12.9 万降到约 8000（16 倍减少），
          且决策在空间上连续，避免碎片化。

    设计文档 2.1 节要求：
      第一级组级路由【基于理解层的退化空间分布图】。
      因此本模块接收 degradation_map 作为条件输入——
      退化重的区域应路由到"强重建"簇，退化轻的区域路由到"精细"簇。
    """

    def __init__(self, dim: int, num_clusters: int, group_size: int = 4):
        super().__init__()
        self.num_clusters = num_clusters
        self.group_size = group_size
        # 组级路由：对每个组的 pooled 特征做分类
        self.norm = nn.LayerNorm(dim)

        # 退化条件融合（理解层 → 组级路由）
        # 退化图是逐 token 的标量，投影到 dim 后与 pooled 特征相加
        self.deg_proj = nn.Linear(1, dim, bias=False)
        self.deg_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        self.gate = nn.Linear(dim, num_clusters)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        degradation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, C)
        degradation: (B, N, 1) 理解层输出的退化程度空间分布（可选）

        返回
        ----
        cluster_ids : (B, G) 每组的专家簇编号
        group_logits: (B, G, num_clusters) 用于负载均衡损失
        """
        b, n, c = x.shape
        gs = self.group_size
        gh, gw = h // gs, w // gs

        # (B, N, C) -> (B, gh, gs, gw, gs, C) -> (B, G, gs*gs, C)
        xr = x.reshape(b, gh, gs, gw, gs, c)
        xr = xr.permute(0, 1, 3, 2, 4, 5).contiguous()
        xg = xr.reshape(b, gh * gw, gs * gs, c)

        # 组内平均池化 → 组级表征
        pooled = xg.mean(dim=2)  # (B, G, C)

        # ---------- 注入理解层的退化条件 ----------
        if degradation is not None:
            # 退化图同样按组池化：(B, N, 1) -> (B, G, 1)
            dr = degradation.reshape(b, gh, gs, gw, gs, 1)
            dr = dr.permute(0, 1, 3, 2, 4, 5).contiguous()
            dg = dr.reshape(b, gh * gw, gs * gs, 1).mean(dim=2)  # (B, G, 1)
            deg_feat = self.deg_proj(dg)                      # (B, G, C)
            # 门控融合：让退化信息按需影响路由
            alpha = self.deg_gate(pooled)                     # (B, G, 1)
            pooled = pooled + alpha * deg_feat

        logits = self.gate(self.norm(pooled))  # (B, G, num_clusters)
        cluster_ids = logits.argmax(dim=-1)  # (B, G)

        return cluster_ids, logits


class ClusterRouter(nn.Module):
    """第二级：簇内路由。

    设计文档 2.4.1 节：
      在选定的专家簇内，做 token 级 Top-K 路由。
      每个 token 从簇内 M 个专家中选择 Top-K。

    设计文档 2.1 节要求：
      第二级簇内路由【基于理解层的场景标签】。
      因此本模块接收 scene_logits 作为条件输入——
      不同场景（UI/自然/动漫等）应偏好不同的专家。
    """

    def __init__(
        self,
        dim: int,
        experts_per_cluster: int,
        top_k: int,
        num_scene_classes: int = 8,
    ):
        super().__init__()
        self.experts_per_cluster = experts_per_cluster
        self.top_k = min(top_k, experts_per_cluster)
        self.norm = nn.LayerNorm(dim)

        # 场景条件融合（理解层 → 簇内路由）
        self.scene_proj = nn.Linear(num_scene_classes, dim, bias=False)
        self.scene_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        self.gate = nn.Linear(dim, experts_per_cluster, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        scene_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, C)
        scene_logits: (B, num_scene_classes) 理解层输出的场景分类（可选）

        返回
        ----
        topk_idx   : (B, N, K) 簇内专家索引
        topk_weight: (B, N, K) 归一化权重
        """
        feat = self.norm(x)

        # ---------- 注入理解层的场景条件 ----------
        if scene_logits is not None:
            # 场景标签是全局的，广播到所有 token
            scene_feat = self.scene_proj(scene_logits)        # (B, C)
            scene_feat = scene_feat.unsqueeze(1)              # (B, 1, C)
            alpha = self.scene_gate(feat)                     # (B, N, 1)
            feat = feat + alpha * scene_feat

        logits = self.gate(feat)  # (B, N, E_c)
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)
        topk_weight = F.softmax(topk_logits, dim=-1)
        return topk_idx, topk_weight


class Expert(nn.Module):
    """单个专家：两层 FFN。"""

    def __init__(self, dim: int, ratio: int = 2):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TernaryLinear(nn.Module):
    """三值权重线性层（-1 / 0 / +1）。

    对应设计文档 4.2 节：三值量化专家层，约 1.58 bit/权重。

    实现说明：此处用 FP32 存储权重 + 前向时量化到 {-1,0,+1}（QAT 风格），
    便于开发阶段验证。真实部署时权重会以 TQ2_0 格式打包。

    scale 初始化（关键）
    --------------------
    三值量化的死区陷阱：若 scale 初值远大于权重幅值，则 |w/scale| < 0.5，
    全部权重被量化为 0，输出恒为 0 且梯度消失。
    因此 scale 必须按权重的实际幅值初始化，使量化后约有一半权重落在 ±1。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # 每输出通道一个可学习 scale
        self.scale = nn.Parameter(torch.ones(out_features, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_scale()

    @torch.no_grad()
    def reset_scale(self) -> None:
        """按权重幅值初始化 scale，避免死区陷阱。

        取每输出通道权重绝对值的均值作为 scale，
        使 |w/scale| 的典型值约为 1，量化后落在 {-1, 0, +1} 上分布合理。
        """
        absmean = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        self.scale.copy_(absmean)

    @staticmethod
    def quantize_weight(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """把权重投影到 {-1, 0, +1}。

        使用可学习 scale 做阈值判断：|w/scale| < 0.5 归零。
        """
        normalized = w / (scale.abs() + 1e-8)
        return torch.clamp(torch.round(normalized), -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量化感知训练（QAT）——STE（直通估计器）
        #
        # 背景：q_unit ∈ {-1,0,+1} 由 round() 得到，而 round() 的梯度处处为 0。
        #       若直接使用 w_q = q_unit * scale，权重将完全收不到梯度。
        #
        # 标准 STE 写法（已实测验证）：
        #     w_used = q_unit.detach() * scale + weight - weight.detach()
        #   - 前向：q_unit.detach()*scale + 0 = 量化后的权重
        #   - 反向：d/d(weight) = 1（直通），d/d(scale) 经由 q_unit.detach()*scale
        #   两者都能获得梯度。
        #
        # 注意：不能用 `w_q + (weight - weight).detach()`——该项恒为 0，
        #       梯度也是 0，等同于没有 STE。
        q_unit = self.quantize_weight(self.weight, self.scale)  # {-1,0,+1}

        if self.training:
            w_used = q_unit.detach() * self.scale + self.weight - self.weight.detach()
        else:
            w_used = q_unit * self.scale

        return F.linear(x, w_used, self.bias)


class TernaryExpert(nn.Module):
    """三值量化专家（私有专家使用）。

    设计文档 4.1 节：一级/二级私有专家用三值量化，
    理由是参数量大、误差容忍度高。
    """

    def __init__(self, dim: int, ratio: int = 2):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = TernaryLinear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = TernaryLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ThreeLevelMoE(nn.Module):
    """三级路由 MoE。

    设计文档 2.4.1 节的三级语义：
      第一级 组级路由   → 专家簇选择（Top-1）
      第二级 簇内路由   → Top-K 私有专家
      第三级 质量门控   → 高精度旁路（由外部 HighPrecisionBypass 处理）

    本模块实现前两级 + 第三级的门控信号输出。
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 32,
        top_k: int = 4,
        num_clusters: int = 3,
        num_shared: int = 1,
        ffn_ratio: int = 2,
        group_size: int = 4,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_clusters = num_clusters
        self.top_k = top_k

        # 专家按簇划分
        if num_experts % num_clusters != 0:
            raise ValueError(
                f"专家总数 {num_experts} 必须能被簇数 {num_clusters} 整除"
            )
        self.experts_per_cluster = num_experts // num_clusters

        # 第一级：组级路由
        self.group_router = GroupRouter(dim, num_clusters, group_size)

        # 第二级：簇内路由
        self.cluster_router = ClusterRouter(dim, self.experts_per_cluster, top_k)

        # 私有专家（三值量化）
        expert_cls = TernaryExpert if use_ternary else Expert
        self.experts = nn.ModuleList(
            [expert_cls(dim, ffn_ratio) for _ in range(num_experts)]
        )

        # 共享专家（始终激活，设计文档 4.1：用 W8A8）
        self.shared_experts = nn.ModuleList(
            [Expert(dim, ffn_ratio) for _ in range(num_shared)]
        )

        # 第三级：质量门控（输出每个 token 走旁路的概率）
        # 设计文档 2.1 节要求：第三级质量门控【基于理解层的 UI/边缘检测】。
        # 因此接收 ui_mask 作为条件输入——UI/文字区域应走高精度旁路。
        self.quality_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        # UI 掩码融合（理解层 → 质量门控）
        self.ui_proj = nn.Linear(1, dim // 2, bias=False)
        self.ui_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        degradation: torch.Tensor | None = None,
        scene_logits: torch.Tensor | None = None,
        ui_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        x: (B, N, C)

        理解层条件（可选，对应设计文档 2.1 节的三级驱动）：
          degradation : (B, N, 1) 退化空间分布图 → 第一级组级路由
          scene_logits: (B, K)    场景标签       → 第二级簇内路由
          ui_mask     : (B, N, 1) UI/边缘检测    → 第三级质量门控

        返回字典：
          out          : (B, N, C) MoE 输出
          gate_prob    : (B, N, 1) 质量门控概率（供高精度旁路使用）
          aux_loss     : 标量，负载均衡损失
          cluster_ids  : (B, G) 组级路由结果
        """
        b, n, c = x.shape
        gs = self.group_router.group_size
        gh, gw = h // gs, w // gs
        num_groups = gh * gw

        # ---------- 共享专家（始终激活）----------
        shared_out = sum(se(x) for se in self.shared_experts)

        # ---------- 第一级：组级路由（由退化分布图驱动）----------
        cluster_ids, group_logits = self.group_router(
            x, h, w, degradation=degradation
        )

        # ---------- 第二级：簇内路由（由场景标签驱动）----------
        topk_idx, topk_weight = self.cluster_router(x, scene_logits=scene_logits)

        # 把"簇内索引"映射为"全局专家索引"
        # 每个 token 属于哪个组，就用哪个组的簇
        token_cluster = cluster_ids.repeat_interleave(gs * gs, dim=1)  # (B, N)
        global_idx = (
            token_cluster.unsqueeze(-1) * self.experts_per_cluster + topk_idx
        )  # (B, N, K)

        # ---------- 专家计算（稀疏激活）----------
        # 实现说明：不用 `out[mask] += ...` 的掩码赋值（该写法走 index_put，
        # 梯度无法正确回传到 expert_out）。改为 flat index + index_add，
        # 既保持稀疏计算（只对被选中的 token 调用专家），又保证梯度通路完整。
        out = torch.zeros(b * n, c, device=x.device, dtype=x.dtype)
        for e in range(self.num_experts):
            # 找出所有选中专家 e 的 (batch, token, k) 位置
            hit = global_idx == e  # (B, N, K)
            if not hit.any():
                continue
            b_idx, n_idx, k_idx = hit.nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                continue

            # 只对被选中的 token 做专家前向（保持稀疏）
            sel = x[b_idx, n_idx]  # (M, C)
            expert_out = self.experts[e](sel)  # (M, C)

            # 按路由权重加权
            w_sel = topk_weight[b_idx, n_idx, k_idx].unsqueeze(-1)  # (M, 1)
            contrib = w_sel * expert_out  # (M, C)

            # flat index 累加，梯度可正确回传
            flat_idx = b_idx * n + n_idx  # (M,)
            out = out.index_add(0, flat_idx, contrib)

        out = out.view(b, n, c) + shared_out

        # ---------- 第三级：质量门控（由 UI/边缘检测驱动）----------
        # 理解层输出的 ui_mask 指示 UI/文字区域，这些区域应走高精度旁路。
        gate_feat = self.quality_gate[0](x)          # LayerNorm
        gate_feat = self.quality_gate[1](gate_feat)  # Linear
        gate_feat = self.quality_gate[2](gate_feat)  # GELU

        if ui_mask is not None:
            # ui_mask: (B, N, 1) -> 投影到 gate_feat 的维度
            ui_feat = self.ui_proj(ui_mask)          # (B, N, dim//2)
            alpha = self.ui_gate(x)                  # (B, N, 1)
            gate_feat = gate_feat + alpha * ui_feat

        gate_prob = torch.sigmoid(self.quality_gate[3](gate_feat))  # (B, N, 1)

        # ---------- 负载均衡辅助损失 ----------
        # 参考 Switch Transformer 的 aux loss：num_clusters * Σ f_i * P_i
        # 其中 f_i 是硬分配的频率（不可导），P_i 是路由概率的均值（可导）。
        #
        # 原实现只用了 f_i，且 f_i 来自 argmax -> one_hot，整条链不可导，
        # aux_loss 是常量、梯度恒为零，负载均衡从未生效（见 issue #2）。
        # 这里补上可导的 P_i 项，并保留硬频率作为目标分布。
        cluster_frac = F.one_hot(
            cluster_ids, self.num_clusters
        ).float().mean(dim=(0, 1))  # (num_clusters,)
        cluster_prob = group_logits.softmax(dim=-1).mean(dim=(0, 1))
        aux_loss = self.num_clusters * (cluster_frac * cluster_prob).sum()

        return {
            "out": out,
            "gate_prob": gate_prob,
            "aux_loss": aux_loss,
            "cluster_ids": cluster_ids,
        }
