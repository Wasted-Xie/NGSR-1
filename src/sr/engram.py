"""
Engram 条件记忆（分层）。

对应设计文档：
  - 2.5  Engram 条件记忆（通用层 / 游戏层 / 用户层）
  - R19  与路由的协同：N-Gram 哈希检索为组级路由提供退化类型标签先验
  - R28  NTC 理念借鉴：存储低维潜在向量而非完整原型，检索后经轻量 MLP 解码
  - R43  Engram 注入：一键采集、聚类、构建（哈希写入，非反向传播）

分层结构（设计文档 2.5）：
  通用层（全局共享，10~15MB）：基础边缘方向模板、基础纹理原型、
                                常见 UI 控件形态、常见字体字形
  游戏层（每游戏特化，5~15MB）：该游戏特有纹理风格、UI 字体控件、
                                渲染特征、特效模式
  用户层（可选）：用户偏好锐度模式、自定义纹理先验

实现要点（重要）
----------------
R28 的权重表存在一处内部不一致：
    "Engram 记忆表 | INT8 | 10M 条目 | 10MB"
若"条目"指 16 维潜在向量，则 10M 条 = 160MB，与表中 10MB 矛盾。
只有当"条目"指【单个 INT8 数值】时，10M 条目 x 1 字节 = 10MB 才成立。

因此本实现按 R28 同一段的另一句话来定架构：
    "Engram 不存储完整纹理原型，而是存储低维潜在向量（如 16 维 INT8），
     检索后经过轻量 MLP 解码。解码器与 MoE 私有专家共享，避免独立存储。"

即：
    - 记忆表 = INT8 缓冲区（非可训练参数），由 inject() 写入
    - 解码器 = 可训练参数（这才是模型参数的一部分）
这样既满足显存预算（size_mb 即 INT8 表体积），
又不会让记忆表以 FP32 形式膨胀 4 倍。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EngramLayer(nn.Module):
    """单层 Engram 记忆表。

    结构：
      1. N-Gram 哈希把 token 特征映射到槽位
      2. 查 INT8 表得到低维潜在向量，反量化
      3. 轻量 MLP 解码为纹理先验
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 4096,
        latent_dim: int = 16,
        ngram: int = 3,
        num_heads: int = 4,
        decode_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.latent_dim = latent_dim
        self.ngram = ngram
        self.num_heads = num_heads

        # 哈希投影：把 dim 维特征映射到 num_heads 个哈希桶（可训练）
        self.hash_proj = nn.Linear(dim, num_heads * latent_dim, bias=False)

        # 记忆表：INT8 缓冲区，不参与反向传播
        # （对应设计文档 R43：由"Engram 注入"流程按哈希写入）
        self.register_buffer(
            "memory",
            torch.zeros(num_slots, latent_dim, dtype=torch.int8),
            persistent=True,
        )
        # 全局反量化 scale（标量，开销可忽略）
        self.register_buffer("mem_scale", torch.tensor(1.0))

        # 轻量解码器：潜在向量 -> 纹理先验（可训练参数）
        hidden = decode_dim or dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    @property
    def table_bytes(self) -> int:
        """记忆表占用的字节数（INT8，即 size_mb 的原始口径）。"""
        return self.num_slots * self.latent_dim

    def _hash(self, x: torch.Tensor) -> torch.Tensor:
        """计算哈希槽位索引。

        x: (B, N, C) -> (B, N, num_heads)
        """
        proj = self.hash_proj(x)  # (B, N, heads*latent)
        b, n, _ = proj.shape
        proj = proj.reshape(b, n, self.num_heads, self.latent_dim)
        # 用投影向量的符号做哈希（LSH 风格）
        bits = (proj > 0).float()
        weights = 2 ** torch.arange(
            self.latent_dim, device=x.device, dtype=torch.float32
        )
        idx = (bits * weights).sum(dim=-1).long()  # (B, N, heads)
        return idx % self.num_slots

    def inject(
        self, indices: torch.Tensor, vectors: torch.Tensor
    ) -> None:
        """向记忆表写入潜在向量（对应设计文档 R43 的 Engram 注入）。

        参数
        ----
        indices : (M,) 槽位索引
        vectors : (M, latent_dim) 待写入的潜在向量（浮点）

        说明：按全局 scale 量化为 INT8 后写入。真实流程中这一步由
              "一键采集 -> 离线聚类 -> 哈希表生成"自动完成，不走反向传播。
        """
        if vectors.numel() == 0:
            return
        with torch.no_grad():
            absmax = vectors.abs().max().clamp(min=1e-8)
            scale = absmax / 127.0
            q = torch.clamp(torch.round(vectors / scale), -127, 127).to(torch.int8)
            self.memory[indices] = q
            # 用最新的 scale（简化处理；生产环境应按层或按槽位维护）
            self.mem_scale.fill_(scale.item())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, C)

        返回
        ----
        prior    : (B, N, C) 检索到的纹理先验
        hit_rate : 标量，记忆命中率（设计文档 3.6 节指标）
        """
        idx = self._hash(x)  # (B, N, heads)

        # 查表 + 反量化：INT8 -> 计算精度
        # 注意：必须跟随输入 dtype，不能硬编码 .float()。
        # 若整网以 FP16/混合精度运行（AMP 推理或 .half()），
        # 硬编码 .float() 会使反量化结果与 half 权重 dtype 冲突而报错。
        # 这与「旁路用何精度」无关，是整网 dtype 适配问题。
        dtype = x.dtype
        retrieved = (
            self.memory[idx].to(dtype) * self.mem_scale.to(dtype)
        )  # (B,N,heads,latent)
        latent = retrieved.mean(dim=2)  # (B, N, latent)

        prior = self.decoder(latent)
        prior = self.norm(prior)

        # 命中率：以"槽位是否已被写入（非零）"为代理指标
        with torch.no_grad():
            nonzero = (self.memory[idx] != 0).float().mean()
        return prior, nonzero


class Engram(nn.Module):
    """分层 Engram：通用层 + 游戏层 + 用户层。

    设计文档 2.5 节的检索与融合：
      Token 特征
        ├─ 通用层检索 → 基础先验
        ├─ 游戏层检索 → 特化先验
        └─ 融合（加权求和 + 门控）→ 最终纹理先验

    容量分配：size_mb 为 INT8 记忆表总预算，
              通用层约占 60%，游戏层约 40%（对应 2.5 节的 10~15MB / 5~15MB）。
    """

    def __init__(
        self,
        dim: int,
        size_mb: int = 25,
        ngram: int = 3,
        num_heads: int = 4,
        latent_dim: int = 16,
        enable_user_layer: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim

        # 槽位数由 INT8 预算反推：每个槽位占 latent_dim 字节
        bytes_total = size_mb * 1024 * 1024
        total_slots = max(256, bytes_total // latent_dim)

        general_slots = int(total_slots * 0.6)
        game_slots = max(128, total_slots - general_slots)

        self.general = EngramLayer(
            dim, general_slots, latent_dim, ngram, num_heads
        )
        self.game = EngramLayer(dim, game_slots, latent_dim, ngram, num_heads)

        if enable_user_layer:
            self.user = EngramLayer(
                dim, max(64, total_slots // 8), latent_dim, ngram, num_heads
            )
        else:
            self.user = None

        # 融合门控：决定通用层与游戏层的权重（可训练）
        self.fuse_gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1),
        )
        self.out_norm = nn.LayerNorm(dim)

        self.total_slots = general_slots + game_slots

    @property
    def table_mb(self) -> float:
        """记忆表实际占用（MB，INT8 口径）。"""
        b = self.general.table_bytes + self.game.table_bytes
        if self.user is not None:
            b += self.user.table_bytes
        return b / 1024 / 1024

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, N, C)

        返回
        ----
        out      : (B, N, C) 融合后的纹理先验
        hit_rate : 平均命中率
        """
        prior_general, hr_g = self.general(x)
        prior_game, hr_gm = self.game(x)

        # 门控融合
        concat = torch.cat([prior_general, prior_game], dim=-1)
        weights = self.fuse_gate(concat)  # (B, N, 2)

        out = (
            weights[..., 0:1] * prior_general
            + weights[..., 1:2] * prior_game
        )

        hit_rate = (hr_g + hr_gm) / 2
        if self.user is not None:
            prior_user, hr_u = self.user(x)
            out = out + 0.1 * prior_user
            hit_rate = (hit_rate + hr_u) / 2

        return {"out": self.out_norm(out), "hit_rate": hit_rate}
