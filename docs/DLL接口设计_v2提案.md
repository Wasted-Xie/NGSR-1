# DLL 接口设计 v2 提案（DLSS 形状）

对应设计文档 7.4 节的修订提案。头文件见 [`include/sr_dll.h`](../include/sr_dll.h)。

## 为什么改

7.4 的接口（`srCreate` / `srEvaluate` / `srRelease` + `SrInput`）方向是对的：单帧、无时序字段、DLL 即插即用。问题在于**契约里没有给未来的自己留位置**：

| 7.4 现状 | 会出问题的地方 |
|---|---|
| `SrInput` 是裸结构体，无 `structSize` | 加一个字段就得让所有调用方重编译；DLL 与宿主版本错配时行为未定义 |
| `width/height` 与 `outWidth/outHeight` 混在输入结构里 | 语义上分不清"哪个是资源尺寸、哪个是逻辑尺寸"，图集/切片场景会出错 |
| 没有 reset | 场景切换、分辨率剧变后内部缓存（分区统计、LUT、自适应阈值）残留 |
| 没有显存查询 | 调用方无法在 create 前预留，8GB 卡上容易 create 成功、首帧 OOM |
| `sharpness` 在主路径 | 与 DLSS 的 deprecated 结论相反，会把画质问题掩盖成"调个锐度" |
| 档位（tier）与倍率耦合 | "任意分辨率到指定分辨率"之后，档位不能再代表倍率 |

DLSS 的证据：4 代把 CNN 换成 Transformer，**调用方一行没改**；`Sharpness` 参数被标为 deprecated；`outWidth` 是参数、`renderWidth` 由 `slDLSSGetOptimalSettings` 反推，开 DRS 时逐帧变化。

## 已确认的三条（2026-11-13）

1. **输入与输出长宽比必须一致**，只允许等比缩放。
   - 契约层：`srExecute` 校验 `outWidth*inHeight == inWidth*outHeight`，不满足返回 `SR_ERR_ASPECT_MISMATCH`，DLL 不做 letterbox / 拉伸。
   - 设计收益：倍率退化成**一个**标量有理数 `p/q`（1920→2560 就是 4/3），重采样核只依赖这一个数 → 可以实现成**多相（polyphase）查表**，每个输出像素只查一次预计算权重，比二维逐像素核便宜一个量级。这是"限制"换来的效率。
   - 如果以后要做非等比（比如方块图集），不要改这个约束，另开一个 `srExecuteAniso`，避免主路径被两套坐标逻辑污染。
2. **格式语义值暂不维护**。`format` 用后端原生枚举（VkFormat / DXGI_FORMAT），实现侧自检不支持就返回 `SR_ERR_UNSUPPORTED`，不许静默按别的格式解释。`SrFormat` 保留为占位，等 Vulkan 后端跑通后再冻结映射表。
3. **tier 与 quality 都做成运行中可变**：
   - `srSetTier` / `srGetTier` 补回（7.4 原有）；tier 是能力档上限，quality 是逐帧预算，两者正交。
   - 切换 tier 会重建内部管线/权重变体，允许有一次卡顿，但必须原子生效（不能在一帧中途）、并在 `SrStats.lastAppliedTier` 里可见。
   - 切换后内部缓存失效，语义等价于 `srReset(SR_RESET_TIER_CHANGED)`（新增的 reset 原因）。

## 逐条改动

| 项 | v2 | 理由 |
|---|---|---|
| 版本 | 每个结构体 `structSize` + `version` | 唯一能防返工的字段 |
| 尺寸 | `inWidth/inHeight`、`outWidth/outHeight` 每帧传 | 支持任意非整数比（同长宽比）、DRS、窗口拖拽；比例不再烘焙进上下文 |
| 资源 | `SrResource{type, format, handle, layout}` | 头文件不依赖 `vulkan.h` / `d3d12.h`；Vk / D3D12 / 主机内存三种 |
| 算力 | `SrQuality` 枚举 | 描述"花多少算力"，与"放大多少倍"解耦 |
| 模型 | `SrModelDesc{path, contentHash, preferPrecision}` + `srApplyModel` | 换网络/换量化不重建上下文；结构信息不进契约 |
| 显存 | `srEstimateMemory(desc, outW, outH)` | create 前问，避免首帧 OOM |
| 状态 | `srReset(ctx, reason)` | 单帧也有内部状态：缓存、统计、阈值 |
| 调试 | `SrTuning{sharpnessLegacy, detailBias}` | 把锐度赶出主路径，只留对照组 |
| 统计 | `srGetStats` 含 `lastQuantizationPath` | 用来验证"厂商扩展优先、逐一回退"那条回退链真的走到了哪一档 |

## 刻意保留 7.4 的东西

- 前缀仍用 `sr`、仍叫 `create/evaluate(执行)/release` 的语义，减少你和现有 Python 侧命名的割裂。
- 仍然**没有** `depth` / `motionVector` / `jitter` / `history`（R42 已定，7.6 视频扩展已废弃）。
- 仍然是"调用方建资源、DLL 只记录命令"，不接管生命周期。

## 实现方必须回答的三个问题

1. **格式映射什么时候冻结？** 现在 `format` 是后端原生值、不受契约保护。等 Vulkan 路径跑通后，用一次版本号提升把语义表定死（`SR_FORMAT_*` → `VkFormat`），之后不允许再漂。
2. **多相 LUT 的相位表怎么建？** 等比约束让倍率变成 `p/q`，可以预计算 `q` 组相位权重；但 `p/q` 是运行时才定的（窗口尺寸可变），所以要支持"倍率变化时重建相位表"，并把它算进 `srReset(SR_RESET_RESOLUTION_CHANGED)` 的成本里。
3. **tier 切换的卡顿预算**：切档要重建管线，允许卡一帧还是不允许？建议允许，但要求实现能在切换前预建新管线（后台线程或首帧 warmup），否则在游戏里切档会明显掉帧。

## 迁移步骤（不破坏现有代码）

1. 先只实现 `SR_RESOURCE_HOST_MEM` 路径（Python/离线对比图用），把 `srExecute` 的语义跑通。
2. 接 Vulkan：`SR_RESOURCE_VK_IMAGE` + `SrSyncDesc`，用现有 `src/sr` 的 PyTorch 侧导出权重，DLL 只做推理。
3. 加 `srReset` 的触发点（场景切换、分辨率变化）——这是唯一需要"游戏侧配合"的地方。

## 未决

- `sharpness` 到底保留多久：建议正式路径永远 0，只在回归对照里用，三个版本后删除。
- 是否需要一个 `srSubmit`（DLL 自己提交）而不是只记录命令：目前选"只记录"，因为 Vulkan 的 queue 所有权在调用方。
