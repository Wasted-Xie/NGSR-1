"""
自研实时超分算法 —— 参考实现。

本包按《超分算法设计文档》的架构搭建，模块对应关系：

    config.py    5.1 五档模型族 / 5.4 滑块参数体系
    layers.py    2.3 空间注意力 / 5.2 算力控制手段 / 2.8 自适应上采样
    moe.py       2.4 三级路由 MoE / 4.2 三值量化专家
    engram.py    2.5 Engram 条件记忆（分层）
    modules.py   2.6 mHC / 2.7 高精度旁路 / 浅层细节路
    backbone.py  2.2 理解层 / 主干 Hybrid Block
    model.py     2.1 完整数据流
    analysis.py  参数量与 FLOPs 统计（校验设计文档 5.2.7 重算值）

设计约束（来自设计文档）：
    - 单帧推理，无时序组件（R42）
    - 全帧推理，不分块（R9）
    - 固定输出分辨率 + 可调输入分辨率（R52）
    - 理解层置于第一层（R53）
"""

from .config import (
    BITS_PER_WEIGHT,
    SRConfig,
    QuantScheme,
    Tier,
    TIER_PRESETS,
    get_preset,
)
from .model import SuperResolutionModel

__all__ = [
    "SRConfig",
    "QuantScheme",
    "Tier",
    "TIER_PRESETS",
    "BITS_PER_WEIGHT",
    "get_preset",
    "SuperResolutionModel",
]

__version__ = "0.1.0"
