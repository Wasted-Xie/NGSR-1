# Next-Generation Super-Resolution (NGSR-1)

> **这是个人对超分算法的一种探索，不是成熟产品或生产级实现。**
> 目的是把一套自研的架构设计落到可运行的代码上：验证设计假设、暴露实现陷阱、
> 记录哪些想法在实测中站得住、哪些站不住。「-1」表示第一代。

按《[超分算法设计文档](docs/超分算法设计文档.md)》的架构搭建的 PyTorch 参考实现。

## 当前状态

**核心骨架已完成并通过验证**（156 项断言全部通过）。

| 模块 | 状态 | 对应设计文档 |
|---|---|---|
| 配置系统（五档 + 滑块约束） | ✅ | 5.1 五档模型族 / 5.4 滑块参数 |
| 窗口注意力（单帧） | ✅ | 2.3 空间注意力 |
| FFN（扩展因子可调） | ✅ | 5.2 算力控制 |
| 三级路由 MoE | ✅ | 2.4 三级路由 |
| **理解层 → 三级路由条件驱动** | ✅ | **2.1 三级分别由理解层驱动** |
| **理解层分辨率随主干缩放** | ✅ | **修订 2.2.4（见下）** |
| 三值量化专家（STE） | ✅ | 4.2 三值量化 |
| Engram 分层条件记忆 | ✅ | 2.5 Engram |
| mHC 流形约束 | ✅ | 2.6 mHC |
| 高精度旁路 | ✅ | 2.7 高精度旁路 |
| 旁路量化（W8A8 / FP8 / FP16） | ✅ | 4.1 量化策略 |
| 浅层细节路 | ✅ | 2.1 数据流 |
| 理解层（四分支 + 跨分支交互） | ✅ | 2.2 理解层 |
| 自适应上采样 | ✅ | 2.8 自适应上采样 |
| 完整模型组装 | ✅ | 2.1 完整数据流 |
| 参数量 / FLOPs / 显存分析 | ✅ | 5.2.7 算力重算 / 5.6 显存模型 |

**未做**：量化导出（TQ2_0 打包）、Vulkan 部署。训练脚本已实现（见 `scripts/`），
但受算力限制，本仓库**不发布模型权重**。

## 对设计文档 2.2.4 的一处修订

**原要求**：理解层「在 540p 空间运行，参数量占总参数 15%~20%」。

**问题**：这两条在 540p 全分辨率下**互相冲突并产生数量级畸变**。理解层算力 ∝ N × dim²，而 540p 的 token 数是 Low 档主干（184p）的 17 倍：

> 口径说明：下文「主干 FLOPs」= 窗口注意力 + 共享专家 FFN + 私有专家，即**完整主干**算力
> （`config.estimate_main_flops`）。早先该口径漏掉了 FFN 与专家，详见本节末尾的修订记录。

| 档位 | 完整主干 FLOPs | 理解层@固定540p+dim128 | 占比 |
|---|---|---|---|
| Low | 2.9 G | 78.8 G | **2724%** |
| Middle | 16.0 G | 78.8 G | 491% |
| High | 81.3 G | 78.8 G | 97% |
| XHigh | 245.4 G | 78.8 G | 32% |
| Max | 667.4 G | 78.8 G | 12% |

即「理解层在 Low 档比整个主干还贵 27 倍」。且若强求参数占比 15%，**算力最弱的档位反而需要更大的理解层**，形成反向约束。

**修订**（已确认）：

1. 理解层分辨率**随主干缩放**（`understanding_scale = 1.0`），不再固定 540p
2. 规模约束改为**算力占比**（`understanding_compute_budget = 0.35`），不再用参数占比
3. 通道数由主干派生（`understanding_dim_ratio = 1.25`），**不需要为每档手工指定**

修订后效果：

| 档位 | 完整主干 FLOPs | 理解层 FLOPs | 算力占比 | 理解层参数占比 |
|---|---|---|---|---|
| Low | 44.2 G | 0.36 G | 0.80% | 8.31% |
| Middle | 89.7 G | 2.07 G | 2.31% | 5.48% |
| High | 231.9 G | 5.00 G | 2.16% | 3.75% |
| XHigh | 553.0 G | 13.93 G | 2.52% | 2.45% |
| Max | 1063.7 G | 31.59 G | 2.97% | 1.59% |

算力占比 **0.80%~2.97%**（跨度 3.7 倍），畸变消除，且离 35% 预算有较大余量。

**口径二次修订（issue #5）**：上表分母已再次补全。此前 `estimate_main_flops`
只含「窗口注意力 + 共享 FFN + 私有专家」，漏掉了 `fuse`、`HighPrecisionBypass`、
`bypass_proj`、`main_upsample`、`upsampler` 以及 MoE 内部的共享专家——这些路径
**在输入分辨率上执行**，开销随分辨率线性增长，是重建路径的大头。补全后各档主干
FLOPs 上升约 1.0~13.6 倍（Low 档 2.9 → 44.2 G），理解层占比由 4.7%~12.9%
降至 0.80%~2.97%。

需要注意的是**占比趋势发生了反转**：旧口径下随档位递减（12.3% → 4.7%），
新口径下递增（0.80% → 2.97%）。原因是分母的主要成分从「主干分辨率上的
注意力/FFN」变成了「输入分辨率上的重建路径」。这对早期"理解层太贵必须砍"
的判断有直接影响——那个判断建立在**低估了约一个数量级**的分母之上。

**口径修订记录（共两次）**：

1. 最初只统计窗口注意力，占比显示 21%~32%，看似逼近上限。补入共享 FFN 与私有专家后
   降至 4.7%~12.9%。
2. issue #5 指出仍漏算重建路径（`fuse` / `HighPrecisionBypass` / `bypass_proj` /
   `main_upsample` / `upsampler` / MoE 共享专家），再次补全后降至 **0.80%~2.97%**。

`tests/test_understanding_budget.py` 第 5 节现显式对拍 `config` 与 `analysis` 两处口径。
**该对拍曾长期"假通过"**：两侧当时漏掉的是同一批模块，公式虽各写一套却同样残缺，
于是偏差为 0、错误被掩盖——这正是 issue #8 指出的测试方法论问题。补全后改为按
**完整主干口径**（`total − understanding − engram`）对拍，任一侧再漂移都会立即失败。

**遗留说明**：理解层**参数**占比随档位单调下降（8.31% → 1.59%），这是结构性的——
理解层是稠密计算（F/P ∝ N），主干是稀疏 MoE（F/P ∝ K/专家数），要提高参数占比
必然同步抬升算力占比。按当前决定，规模暂不动，先验证结构。

**路由未被削弱**：三级路由结构完整保留，路由参数量占 MoE 的 0.41%~2.34%；理解层条件注入的额外参数仅 **0.1%**（Max 档 803 / 1,081,832）。

## 测试

```bash
python tests/test_smoke.py                  # 冒烟：构建 + 前向 + 参数量        (16 项)
python tests/test_full.py                   # 完整：单帧/梯度/配置/各档形状     (31 项)
python tests/test_understanding.py          # 理解层四项结构缺陷验证            (16 项)
python tests/test_conditioning.py           # 理解层条件是否真驱动三级路由      (7 项)
python tests/test_understanding_budget.py   # 理解层算力预算 / 路由 / 口径一致性 (18 项)
python tests/test_quant.py                  # 伪量化原语与 bypass_quant 接线      (52 项)
python tests/test_regressions.py            # P0 回归：跨窗通路 / aux_loss / mHC  (9 项)
python tests/test_semantics.py              # 语义：梯度存在性 / 感受野 / 过拟合  (7 项)
python tests/test_cuda.py                   # CUDA 环境与算子可用性
python tests/test_gpu_model.py              # 五档模型在 GPU 上的显存与耗时
python tests/run_analysis.py                # 分析报告（参数量 / FLOPs / 显存）
python tests/run_analysis.py Max            # 只看 Max 档
```

合计 **156 项断言**。

**后两个测试文件是 issue #8 的产物**。该 issue 指出：原有测试全是形状、参数量、
数值一致性与设备可用性，**没有一项测语义**，因此三个 P0 缺陷（主干无跨窗口通路、
`aux_loss` 梯度恒零、mHC 逐层衰减 0.625）能在"测试全绿"的情况下长期存在。

- `test_regressions.py`：为每个 P0 各附一条**修复前必然失败**的断言（如无跨窗通路时
  窗外扰动影响恰好为 `0.000e+00`）
- `test_semantics.py`：梯度存在性（断言**具体参数**而非"模型里有某个参数有梯度"）、
  感受野跨窗、**单帧过拟合**（小配置训练 200 步，PSNR 须提升 > 3 dB）

单帧过拟合是最强的整体检验：它同时覆盖主干拟合能力、跨窗口通路与残差缩放是否正常。
首版该测试用随机噪声当 HR，结果只涨 0.65 dB——因为噪声的高频在双三次下采样时已
不可逆丢失，属于**无解任务**；改用低频正弦图案后提升 **+9.90 dB**，验证通过。

### 计算设备

前五个测试默认**自动选择**：有 CUDA 就用 CUDA，否则 CPU。可用环境变量覆盖：

```bash
SR_TEST_DEVICE=cpu    python tests/test_full.py   # 强制 CPU（复现设备无关基准）
SR_TEST_DEVICE=cuda   python tests/test_full.py   # 强制 CUDA（不可用则报错，不静默回退）
SR_TEST_DEVICE=cuda:1 python tests/test_full.py   # 指定卡号
```

实测（RTX 5060 Laptop）：五个测试合计 **23.7s（GPU）vs 46.6s（CPU）**。
收益高度集中——`test_full` 从 39.4s 降到 15.0s（第 4 节要跑五档模型全分辨率前向），
其余四个测试本体耗时可忽略，其 1.5~2.0s 几乎全是解释器与 torch 导入开销，
GPU 化反而因 CUDA context 初始化各慢约 0.5s。之所以仍统一走设备入口，
是为了后续测试规模变大时自动受益。

### 基本用法

```python
from sr import Tier, get_preset
from sr.model import SuperResolutionModel

cfg = get_preset(Tier.HIGH)
model = SuperResolutionModel(cfg)
model.eval()

import torch
x = torch.randn(1, 3, cfg.input_height, cfg.input_width)
out = model(x)
print(out["sr"].shape)      # (1, 3, 2160, 3840)
print(out["aux_loss"])      # MoE 负载均衡损失
print(out["hit_rate"])      # Engram 命中率
```

## 目录结构

```
LICENSE             Apache-2.0（适用于代码）
NOTICE              第三方组件归属声明

src/sr/             算法实现
├── config.py       配置系统：五档预设 + 参数校验
├── layers.py       基础层：窗口注意力、FFN、DW卷积、自适应上采样
├── quant.py        伪量化原语（QAT / 精度仿真）+ QuantConv2d
├── moe.py          三级路由 MoE + 三值量化专家
├── engram.py       分层 Engram 条件记忆
├── modules.py      mHC、高精度旁路、浅层细节路
├── backbone.py     主干 Hybrid Block + 理解层
├── model.py        完整模型组装
└── analysis.py     参数量 / FLOPs / 显存分析

scripts/            训练与实验脚本
├── train_base.py         基座训练（真实 LR-HR 配对，长时间训练）
├── feasibility_train.py  可行性验证训练
├── eval_sr.py            评估（PSNR / SSIM + 对比图）
├── analyze_curve.py      训练曲线分析（判定饱和点）
├── analyze_scaling.py    规模扫描分析（分维度）
├── reeval_scaling.py     规模模型重评估
├── emergence_clf.py      涌现实验：多标签退化分类
└── analyze_emergence.py  涌现实验分析

tests/              测试（156 项断言，可直接运行复现）
├── _device.py         计算设备选择（默认自动 CUDA，SR_TEST_DEVICE 可覆盖）
├── test_smoke.py      冒烟测试
├── test_full.py       完整验证
├── test_understanding.py          理解层结构缺陷验证
├── test_conditioning.py           理解层条件驱动路由的消融验证
├── test_understanding_budget.py   算力预算 / 路由完整性 / 口径一致性
├── test_quant.py                  伪量化原语 / STE / bypass_quant 接线
├── test_cuda.py                   CUDA 环境与算子可用性
├── test_gpu_model.py              五档模型 GPU 显存与耗时
└── run_analysis.py    分析报告

docs/               设计文档（CC BY-SA 4.0，见 docs/LICENSE-docs）
├── 超分算法设计文档.md
├── 待决问题清单.md
└── LICENSE-docs
```

数据不随仓库分发，目录约定见上文「数据」章节。
```

## 实现中修正的设计约束

搭建过程中发现设计文档的档位表缺少若干**可整除性约束**，直接套用会导致模型无法构建。已做最小幅度调整：

| 约束 | 问题 | 修正 |
|---|---|---|
| 窗口必须整除主干分辨率 | Low 档 180p ÷ 8 = 22.5 | 输入改 736p → 主干 184p |
| 同上 | Max 档 540p ÷ 8 = 67.5 | 输入改 1088p → 主干 544p |
| 同上 | High 档宽 533 ÷ 8 | 输入宽改 1632 → 主干宽 544 |
| 通道数为 8 的倍数 | Low 28、Middle 36 | 改为 24、32 |
| 专家数能被簇数整除 | 32 不能被 3 整除 | 簇数改为 4 |
| 头数能整除通道数 | XHigh 通道 56，头数 3 | 通道改为 64 |

这些约束已写入 `config.validate()`，非法配置会在构建前报错。

## 实现中的关键陷阱（已修复）

**三值量化的死区陷阱**：若 `scale` 初值为 1.0 而权重幅值约 0.25，则 `|w/scale| < 0.5`，**全部权重被量化为 0**，模型输出恒为 0 且梯度消失。

修复：`scale` 按权重绝对值的均值初始化（`reset_scale()`），使量化后分布约为 -1:40% / 0:26% / +1:34%。

**STE 写法**：`round()` 梯度处处为 0，必须用直通估计器。正确写法是

```python
w_used = q_unit.detach() * scale + weight - weight.detach()
```

错误写法 `w_q + (weight - weight).detach()` 中第二项恒为 0，等同于没有 STE，会导致权重完全收不到梯度。

**STE 的浮点保真度陷阱**：即便写法正确，`x + (xq - x).detach()` 的**前向**在浮点下
并不精确等于 `xq`——`xq - x` 会损失低位，回加后无法精确复原。后果是量化值散布在
目标表示集合之外：三值量化每输出通道本应只有 `{0, ±scale}` 三个取值，实测出现 **7 个**，
且对已量化权重再量化不再幂等。

修复：仅在确实需要梯度时才走 STE，否则直接返回精确的 `xq`：

```python
def _ste(x, xq):
    if torch.is_grad_enabled() and x.requires_grad:
        return x + (xq - x).detach()
    return xq
```

推理阶段本就不需要 STE，这样既修掉保真度问题，也让"量化值落在表示集合内"成为可断言的
性质（`tests/test_quant.py` 第 1、2 节）。

## 与设计文档的对应关系

| 设计文档 | 实现位置 |
|---|---|
| 2.1 完整数据流 | `model.py::SuperResolutionModel.forward` |
| 2.2 理解层（四分支） | `backbone.py::UnderstandingLayer` |
| 2.3 空间注意力（单帧） | `layers.py::WindowAttention` |
| 2.4 三级路由 MoE | `moe.py::ThreeLevelMoE` |
| 2.5 Engram（分层） | `engram.py::Engram` |
| 2.6 mHC 流形约束 | `modules.py::ManifoldConstrainedHyperConnection` |
| 2.7 高精度旁路 | `modules.py::HighPrecisionBypass` |
| 2.8 自适应上采样 | `layers.py::AdaptiveUpsample` |
| 4.1 旁路量化（W8A8 / FP8 / FP16） | `quant.py::QuantConv2d` + `modules.py::HighPrecisionBypass` |
| 4.2 三值量化 | `moe.py::TernaryLinear` / `TernaryExpert` |
| 5.1 五档模型族 | `config.py::TIER_PRESETS` |
| 5.4 滑块与联动约束 | `config.py::SRConfig.validate` |

## 已知限制

1. **理解层规模约束已修订**：设计文档 2.2.4 原要求「参数占比 15%~20%」，
   因在 540p 下产生数量级畸变，已改为「算力占比 ≤ 35%」（详见上文修订章节）。
   分母为**完整主干**（含重建路径，见 issue #5 的二次口径修订），当前实测 **0.80%~2.97%**。

2. **理解层参数占比仍随档位下降**（8.31% → 1.59%），且与本条无关地低于设计文档
   原定的 15%~20%。原因是理解层为稠密计算而主干为稀疏 MoE，二者不可兼得：
   要把参数占比提到 10%，Max 档理解层就要吃掉完整主干约 30% 的算力。
   按当前决定暂不动规模。

   **补充（issue #5）**：由于分母此前被低估约一个数量级，「理解层算力占比」的真实
   值远低于早期认知（0.80%~2.97%，而非 12.9% 量级）。这意味着调低
   `understanding_dim_ratio` 所能释放的算力远小于预期——当初"理解层太贵必须砍"
   的判断依据不成立，该参数有较大上调空间，可供后续实验探测。

3. **旁路量化是伪量化仿真，不是真实低位宽算子**：`bypass_quant` 已接入
   `HighPrecisionBypass` 与 `bypass_proj`（共 7 个卷积，权重与输入激活都量化），
   可选 W8A8 / FP8 / FP16 / FP32；三值与 W4A8 被 `validate()` 拒绝
   （旁路是质量保障路径）。但实现方式是把数值投影到目标格式的表示集合后
   **转回原 dtype 计算**，并非真的用 INT8/FP8 算子执行。因此它可用于评估
   精度损失与做 QAT，但**省不了显存与带宽**——那取决于后续的量化导出与
   Vulkan 后端，见第 7 条。

   两个实测结论值得注意：

   - **FP8 的精度低于 W8A8**。本项目的权重动态范围下，旁路输出相对 FP32 的
     偏差：FP16 `1.8e-05` < W8A8 `9.5e-04` < FP8 `5.2e-03`。E4M3 只有 3 位
     尾数，而 INT8 是均匀 8 位。FP8 的价值在动态范围与硬件吞吐，不在精度——
     旁路若以质量为先，FP16 或 W8A8 都优于 FP8。
   - 量化是**动态定标**（scale 由权重统计量推出，不可学习），因此对已量化权重
     再做一次量化不保证幂等；其中三值方案的 absmean 定标必然改变（非零元素
     比例 < 1）。这是定标方式的固有性质，只量化一次时无影响。

4. **显存实测超设计文档预算**（开发阶段放宽，压制归后期优化）：

   | 档位 | 实测峰值（PyTorch CUDA） | 文档预算 |
   |---|---|---|
   | Low | 1154 MB | 330~380 MB |
   | Middle | 1672 MB | 420~480 MB |
   | High | 2617 MB | 520~580 MB |
   | XHigh | 4409 MB | 620~700 MB |
   | Max | 5153 MB | 760~850 MB |

   差异主因**不是**框架开销，而是**融合路径在输入分辨率上使用了主干通道数**：
   `fuse` / `shallow_proj` / `bypass` / `bypass_proj` 四个张量都是
   `(1, C主干, H_in, W_in)`，以 High 档为例每个 272 MB。
   而设计文档 5.6.2 的显存模型是按「1080p 浅层 32 通道 = 132 MB」口径估算的。
   另需注意两者口径不同：文档预算含 Vulkan 运行时，本测量是 PyTorch 侧分配。

5. **FLOPs 为解析估算**：与设计文档 5.2.7 节同口径，未含实现细节开销。

6. **MoE 层数假设**：默认每个 Block 都含私有专家（依据 R51 数据流图），
   可通过 `config.moe_layers` 限制。

7. **未实现**：量化导出（TQ2_0 打包）、Vulkan 部署、离线 Engram 注入流程。
   训练循环与数据加载已在 `scripts/` 下提供（可复现的实验脚本，非框架）。

8. **主干量化未实现**（issue #6）：主干全部为 FP32 的 `nn.Linear` / `nn.Conv2d`，
   真正接线的量化只有旁路（`bypass_quant`）与专家（`expert_quant`）。原 `main_quant`
   配置项只有声明与校验、零引用，已删除以避免「配置存在即已实现」的误读；同理删除的
   还有 `engram_quant`（记忆表恒为 int8）与 `antialiasing`（后处理只消费 `sharpness`）。
   注意 `estimate_weight_memory_mb` 仍按 `expert_quant` 位宽估算**全模型**权重显存，
   在本条解决前该数值偏乐观。

9. **Engram 的静态表在训练中不会被写入**：`engram.py::inject()` 的调用点只有测试，
   训练脚本从不调用；`memory` 是 `register_buffer`，优化器也不会更新它。实测两个已训练
   checkpoint 的记忆表**非零元素均为 0**，`hash_proj` 因索引路径不可导而梯度恒为 `None`，
   模块实际退化为「常数偏置 + 可学习门控」。已另行实现 `engram_mode="dynamic"`
   （可训练记忆槽 + 可微软注意力，端到端可训练）作为替代路径；
   `engram_hash_bits` 参数化则解决了原 16 位地址空间导致大表大量槽位零命中的问题。

10. **mHC 在单流架构下不产生实际作用**：模块本意是「k 条残差流之间的混合」，但主干
    始终是单流 `(B,N,C)`。修复 issue #3（原实现每层乘 0.625、主动破坏恒等映射）之后，
    单流输入时输出严格等于输入、`log_alpha` 梯度恒为 0——即从"有害"变为"无害但也无用"。
    若要 mHC 真正起作用，需要把主干改为多流结构；否则按 issue #3 的建议，应将其
    如实降级为标量门控（或删除 Sinkhorn 与那 16 个参数）。

11. **`bypass_activation` 现已接线，但为软约束**：该配置原先只有声明与 validate 校验、
    从未参与计算（issue #6）。现接入训练损失（`--gate-weight`），以均方误差约束平均
    门控概率逼近目标激活率。这是**软约束**而非硬性 top-k 选择，门控分布仍由学习决定；
    若需要确定性激活率，应改用 top-k 硬选择（issue #7 的建议）。

## 运行环境

本机已验证的 GPU 环境（`python tests/test_cuda.py` 可复核）：

| 项 | 值 |
|---|---|
| PyTorch | 2.14.1+cu130 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 计算能力 | sm_120（Blackwell） |
| 显存 | 7.96 GB |

**安装要点**：RTX 50 系（sm_120）**必须用 CUDA 12.8 及以上的构建**，
cu126 及更早不含 sm_120 内核。可用的组合：

| CUDA | 最高 torch | 支持 sm_120 | 说明 |
|---|---|---|---|
| cu126 | 2.14.0 | ❌ | 不可用于 RTX 50 系 |
| cu128 | 2.11.0 | ✅ | |
| cu130 | 2.14.0 | ✅ | 当前使用 |

安装包下载地址（Python 3.12 / Windows）：

```
https://download.pytorch.org/whl/cu130/torch-2.14.0%2Bcu130-cp312-cp312-win_amd64.whl
```

下载后校验（官方 sha256）：

```
6a20371bf5af2af933a6149e5ec13f7ea7ad506276137dae9386f98d236c02ff
```

> **关于 Vulkan**：PyTorch **没有** Vulkan 后端（官方仅 CUDA / ROCm / MPS / XPU / CPU）。
> Vulkan 是**部署目标**，由 ExecuTorch 的 Vulkan 后端承载（设计文档 7.1 节）。

## 数据

本参考实现使用**一个**公开数据集：

| 数据集 | 许可 | 说明 |
|---|---|---|
| [Super Resolution in Video Games 2](https://www.kaggle.com/competitions/super-resolution-in-video-games-2) | Apache-2.0 | Unreal Engine 采集的游戏画面真值配对：LR 480×270 → HR 1920×1080（**4x**），训练集 14,431 对 |

**数据集不随本仓库分发**——其体积约 51.4 GB，且适用条款独立于本仓库（下载需登录 Kaggle 并接受该竞赛规则）。

### 目录约定

脚本按以下结构读取数据，`lr` 与 `hr` 中的文件必须**同名一一对应**：

```
data/
└── train/
    ├── lr/     # 低分辨率 480×270
    └── hr/     # 高分辨率 1920×1080
```

## 许可

| 对象 | 许可 |
|---|---|
| 代码（`src/`、`scripts/`、`tests/`） | [Apache License 2.0](LICENSE) |
| 文档（`docs/`） | [CC BY-SA 4.0](docs/LICENSE-docs) |
| 模型权重 | **不发布** |
| 数据集 | 不随仓库分发 |

第三方组件的归属声明见 [`NOTICE`](NOTICE)。

**文档为何用 CC BY-SA 4.0**：允许任何人（含商业使用者）阅读、引用、改编这些文档，
但**必须署名原作者**，且**改编后的文档必须同样以 CC BY-SA 4.0 开放**。
这既不妨碍正常的技术使用与商业应用，也阻止了把文档改头换面后闭源据为己有的行为。
> 正确分工是：**PyTorch 开发验证 → 导出 ExecuTorch → Vulkan 部署**。
