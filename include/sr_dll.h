/*
 * sr_dll.h —— NGSR 超分 DLL 的对外契约（v2 提案，形状取自 DLSS/Streamline）
 *
 * 设计原则（为什么这么定，来自 DLSS 换了四代模型、对外契约却基本没变的教训）
 * ------------------------------------------------------------------
 * 1. 每个结构体都带 structSize + version。
 *    DLSS 4 把 CNN 换成 Transformer，调用方一行没改——因为契约里没有"网络"的痕迹。
 *    你后面还要换主干、换量化、换 Engram 方案，这个字段是唯一能保证不返工的东西。
 *
 * 2. 输入/输出尺寸每次 execute 单独传，不在 create 时固定。
 *    DLSS 的 outWidth 是参数，renderWidth 由 slDLSSGetOptimalSettings 反推；
 *    开 DRS 时渲染分辨率逐帧变化。你要支持"任意分辨率上采样到指定分辨率"，
 *    就不能把比例烘焙进上下文。
 *
 * 3. 没有 depth / motionVector / jitter / history。
 *    这是 R42 已定的单帧约束；DLL 内部不维护跨帧图像状态。
 *    注意：没有跨帧图像状态 ≠ 没有内部状态（缓存、统计、自适应阈值都有），
 *    所以仍然需要 srReset()。
 *
 * 4. sharpness 不在主路径里。
 *    DLSS 的 Sharpness 参数已被标记 deprecated——网络训对了就不该靠末端锐化补。
 *    保留在 SrTuning 里只作为调试/回归对照，不参与正式画质路径。
 *
 * 5. quality 是"算力预算"，不是"倍率"。
 *    倍率 = out/in，由两个 extent 决定；quality 只告诉实现"这帧最多花多少算力"。
 *    把两者混在一起的接口（FSR 的 Quality/Balanced 档位）在任意分辨率下会自相矛盾。
 *
 * 6. 资源以 type-erased 句柄传入，头文件不依赖 vulkan.h / d3d12.h。
 *    调用方负责创建资源、设置 layout/state；DLL 不接管生命周期，只在 execute 期间使用。
 */

#ifndef SR_DLL_H
#define SR_DLL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* 版本                                                                */
/* ------------------------------------------------------------------ */
#define SR_API_VERSION_MAJOR 2
#define SR_API_VERSION_MINOR 0
#define SR_MAKE_VERSION(major, minor) \
    ((uint32_t)(((uint32_t)(major) << 16) | (uint32_t)(minor)))
#define SR_API_VERSION SR_MAKE_VERSION(SR_API_VERSION_MAJOR, SR_API_VERSION_MINOR)

/* 所有带 structSize 的结构体都填 sizeof(结构体)；新增字段只能追加在尾部，
 * 并通过 version 区分语义变化。调度器按 structSize 判断调用方给了多少字段。 */
#define SR_STRUCT_VERSION 1u

/* ------------------------------------------------------------------ */
/* 结果码                                                              */
/* ------------------------------------------------------------------ */
typedef enum SrResult {
    SR_OK = 0,
    SR_ERR_INVALID_ARGUMENT   = -1,
    SR_ERR_UNSUPPORTED        = -2,  /* 设备/格式不支持 */
    SR_ERR_OUT_OF_MEMORY      = -3,
    SR_ERR_DEVICE_LOST        = -4,
    SR_ERR_MODEL_LOAD         = -5,  /* 权重缺失/校验失败 */
    SR_ERR_STATE              = -6,  /* 上下文状态不允许（如未配置执行） */
    SR_ERR_ASPECT_MISMATCH    = -7,  /* in/out 长宽比不一致（已定约束，见 SrFrameDesc） */
} SrResult;

/* ------------------------------------------------------------------ */
/* 算力预算（不是倍率）                                                */
/* ------------------------------------------------------------------ */
typedef enum SrQuality {
    SR_QUALITY_MIN_LATENCY = 0,  /* 最小算力：降通道/跳过细节分支，画质让位 */
    SR_QUALITY_BALANCED    = 1,  /* 默认 */
    SR_QUALITY_QUALITY     = 2,  /* 满通道 */
    SR_QUALITY_REFERENCE   = 3,  /* 离线对照：允许慢，用于生成对比图 */
} SrQuality;

/* ------------------------------------------------------------------ */
/* 帧标志                                                              */
/* ------------------------------------------------------------------ */
enum {
    SR_FLAG_NON_LINEAR_COLOR   = 1u << 0, /* 输入非线性色（DLSS 同样有这个开关） */
    SR_FLAG_UI_MASK_IN         = 1u << 1, /* uiMaskIn 有效 */
    SR_FLAG_UI_MASK_OUT        = 1u << 2, /* 请求回写 uiMaskOut（调试/后续阶段用） */
    SR_FLAG_DEBUG_VIEW         = 1u << 3, /* 输出调试图（分区/退化图），不参与正式画质 */
    SR_FLAG_ALLOW_APPROXIMATE  = 1u << 4, /* 允许实现降精度/降采样内部路径 */
};

/* ------------------------------------------------------------------ */
/* 资源描述（type-erased）                                             */
/* ------------------------------------------------------------------ */
typedef enum SrResourceType {
    SR_RESOURCE_NONE = 0,
    SR_RESOURCE_VK_IMAGE   = 1, /* handle = VkImage，layout 字段填 VkImageLayout */
    SR_RESOURCE_D3D12_TEX  = 2, /* handle = ID3D12Resource*，layout 字段填 D3D12_RESOURCE_STATES */
    SR_RESOURCE_HOST_MEM   = 3, /* handle = 主机指针，用于测试/离线批处理 */
} SrResourceType;

typedef struct SrResource {
    uint32_t structSize;
    uint32_t version;
    uint32_t type;      /* SrResourceType */
    uint32_t format;    /* 像素格式，vulkan/dxgi 各自枚举值，由调用方与实现约定 */
    void*    handle;
    uint32_t layout;    /* 使用期间的 layout/state；DLL 不负责转换 */ 
    uint32_t reserved;
} SrResource;

/* 像素格式约定
 * ------------------------------------------------------------------
 * 已决定（2026-11）：**语义值暂不维护**。format 取后端的原生枚举值
 * （VkFormat / DXGI_FORMAT），由实现侧与调用方约定，跨版本不保证稳定。
 * SrFormat 保留为占位，等 Vulkan 后端跑通后再定语义表并冻结映射。
 * 在此之前，实现必须自检 inFormat/outFormat 是否被自己支持，不支持返回
 * SR_ERR_UNSUPPORTED，不要静默按别的格式解释。 */
typedef enum SrFormat {
    SR_FORMAT_UNKNOWN = 0,
    SR_FORMAT_RGBA8_UNORM = 1,     /* 占位 */
    SR_FORMAT_RGBA8_SRGB  = 2,     /* 占位 */
    SR_FORMAT_RGBA16_SFLOAT = 3,   /* 占位（HDR 输入） */
    SR_FORMAT_R8_UNORM = 4,        /* 占位（UI 掩码） */
    SR_FORMAT_RGBA8_RGB_OUT = 5,   /* 占位（输出，仅 RGB 有效） */
} SrFormat;

/* ------------------------------------------------------------------ */
/* 上下文创建                                                          */
/* ------------------------------------------------------------------ */
typedef struct SrAllocator {
    uint32_t structSize;
    uint32_t version;
    void* user;
    void* (*alloc)(void* user, size_t size, size_t alignment);
    void  (*free)(void* user, void* ptr);
} SrAllocator;

/* 模型描述：权重与网络结构解耦。
 * DLSS 4 换 Transformer 时调用方无感，原因就是权重不是一个"结构体里的字段集合"。
 * 这里只传 blob + 校验和，实现自己解释。 */
typedef struct SrModelDesc {
    uint32_t structSize;
    uint32_t version;
    const char* path;        /* 权重文件路径（TQ2_0 / INT8 / FP16 由实现嗅探） */
    uint64_t    contentHash; /* 0 = 不校验；非 0 = 与权重头部记录的哈希比对 */
    uint32_t    preferPrecision; /* SrPrecision：0 = 自动 */
} SrModelDesc;

typedef enum SrPrecision {
    SR_PRECISION_AUTO = 0,
    SR_PRECISION_FP32 = 1,
    SR_PRECISION_FP16 = 2,
    SR_PRECISION_INT8 = 3,
    SR_PRECISION_TERNARY = 4,
} SrPrecision;

typedef struct SrContextDesc {
    uint32_t structSize;
    uint32_t version;
    const SrAllocator* allocator;  /* NULL = 默认 malloc */
    const SrModelDesc* model;      /* 可在 create 之后用 srApplyModel 替换 */
    SrQuality defaultQuality;
    SrPrecision preferPrecision;
    uint32_t  tier;                /* 初始档位；0 = 自动探测。运行中用 srSetTier 改 */
    uint64_t  flags;               /* 能力开关，见 SrContextFlags */
} SrContextDesc;

enum {
    SR_CTX_ENABLE_DEBUG_VIEW = 1ull << 0,
    SR_CTX_ENABLE_PROFILING  = 1ull << 1,
};

typedef struct SrContext SrContext;  /* 不透明句柄 */

/* ------------------------------------------------------------------ */
/* 能力查询（在任何 create 之前调用）                                   */
/* ------------------------------------------------------------------ */
typedef struct SrCapability {
    uint32_t structSize;
    uint32_t version;
    uint32_t maxInputWidth,  maxInputHeight;
    uint32_t maxOutputWidth, maxOutputHeight;
    /* 长宽比约束（已决定）：输入与输出必须同比例，只允许等比缩放。
     * 因此"倍率"是一个标量有理数 = outWidth/inWidth = outHeight/inHeight，
     * 允许非整数（如 1920→2560 即 4/3），但不允许 x/y 比例不同。
     * 这两个字段给出支持的最大放大倍数（有理数表达，如 4/1）。 */
    uint32_t maxUpscaleNum, maxUpscaleDen;
    uint32_t supportHdr;
    uint32_t supportedPrecisions;   /* 位掩码：1<<SrPrecision */
    uint32_t reserved[8];
} SrCapability;

/* ------------------------------------------------------------------ */
/* 帧描述                                                              */
/* ------------------------------------------------------------------ */
typedef struct SrFrameDesc {
    uint32_t structSize;
    uint32_t version;

    /* 资源。colorIn 的 extent 与 colorOut 的 extent 各自独立：
     * 比例 = outWidth/inWidth = outHeight/inHeight，允许非整数（4/3、2.93 都行），
     * 但必须**同比例**（已定约束）。不一致时 srExecute 返回 SR_ERR_ASPECT_MISMATCH，
     * 不做自动 letterbox / 拉伸 —— 那属于调用方的事。 */
    SrResource colorIn;
    SrResource colorOut;
    SrResource uiMaskIn;   /* 可选，SR_FLAG_UI_MASK_IN */
    SrResource uiMaskOut;  /* 可选，SR_FLAG_UI_MASK_OUT */

    /* 显式 extent：不依赖资源的创建尺寸，因为调用方可能用更大的图集/数组切片 */
    uint32_t inWidth,  inHeight;
    uint32_t outWidth, outHeight;

    /* 格式见 SrFormat 的说明：当前为后端原生值，语义表待定 */
    SrFormat inFormat;
    SrFormat outFormat;
    SrFormat maskFormat;

    SrQuality quality;   /* 本帧预算；SR_QUALITY_BALANCED 可用默认值覆盖 */
    uint32_t  flags;
    uint64_t  frameId;   /* 单调递增，仅用于统计/调试，不参与算法 */
} SrFrameDesc;

/* 同步与提交。DLL 只记录命令，不提交、不等待。
 * 调用方保证：cmdBuffer 处于 recording 状态，queue 与资源同族。 */
typedef struct SrSyncDesc {
    uint32_t structSize;
    uint32_t version;
    void* queue;      /* VkQueue / ID3D12CommandQueue* */
    void* cmdBuffer;  /* VkCommandBuffer / ID3D12GraphicsCommandList* */
    uint32_t reserved[4];
} SrSyncDesc;

/* ------------------------------------------------------------------ */
/* 统计                                                                */
/* ------------------------------------------------------------------ */
typedef struct SrStats {
    uint32_t structSize;
    uint32_t version;
    uint64_t framesProcessed;
    uint64_t lastFrameCpuMicros;
    uint64_t peakDeviceMemoryBytes;   /* 内部工作集峰值，不含调用方资源 */
    uint64_t scratchMemoryBytes;      /* 当前工作集 */
    uint32_t lastAppliedQuality;
    uint32_t lastAppliedTier;
    uint32_t lastQuantizationPath;    /* 实际走通的精度路径，用于验证回退链 */
    uint32_t reserved[4];
} SrStats;

/* ------------------------------------------------------------------ */
/* 配置（可在运行中调整）                                              */
/* ------------------------------------------------------------------ */
typedef struct SrTuning {
    uint32_t structSize;
    uint32_t version;
    /* 仅调试用。DLSS 已把 Sharpness 标为 deprecated：
     * 网络输出正确时不需要末端锐化，开了会在边缘加 halo。
     * 保留此字段是为了做"有/无锐化"的回归对照，正式路径应保持 0.0f。 */
    float    sharpnessLegacy;
    float    detailBias;        /* 纹理先验强度，-1..1，调试用 */
    uint32_t reserved[6];
} SrTuning;

/* ------------------------------------------------------------------ */
/* 上下文生命周期                                                       */
/* ------------------------------------------------------------------ */
uint32_t srGetApiVersion(void);

/* 主版本不匹配时该 DLL 应拒绝加载（由宿主检查）。 */
SrResult srQueryCapability(const void* physicalDevice, SrCapability* out);

/* create 之前可查询内部工作集需求，供调用方预留显存。
 * DLSS 的 VRAM query 是同一个思路：先问、再建。 */
SrResult srEstimateMemory(const SrContextDesc* desc,
                          uint32_t maxOutWidth, uint32_t maxOutHeight,
                          uint64_t* scratchBytes, uint64_t* persistentBytes);

SrResult srCreate(const SrContextDesc* desc, SrContext** out);

/* 换权重不重建上下文。网络结构变化时实现内部自行重建管线。 */
SrResult srApplyModel(SrContext* ctx, const SrModelDesc* model);

SrResult srApplyTuning(SrContext* ctx, const SrTuning* tuning);

/* 档位（已决定：运行中可变）。
 * tier 是"设备能力档"，决定模型规模上限与精度路径（设计文档 5.1 的五档）。
 * 允许在运行中切换，但实现必须保证：
 *   - 切换是原子的，不能在 execute 中途生效；
 *   - 切换会导致内部管线/权重变体重建，可能有一次卡顿，需在 stats 里可见；
 *   - 切换后内部缓存失效，语义等价于 srReset(SR_RESET_MODEL_CHANGED)。
 * quality 与本函数正交：quality 是逐帧预算，tier 是上限。 */
SrResult srSetTier(SrContext* ctx, uint32_t tier);
SrResult srGetTier(SrContext* ctx, uint32_t* tier);

/* 内容发生突变时调用：场景切换、窗口尺寸剧变、视频 seek、分辨率档位切换。
 * 单帧方案没有历史帧，但内部缓存（分区统计、自适应阈值、LUT）会失效。
 * DLSS 的 reset 语义同理——它不是给"时域历史"用的，是给"内部状态"用的。 */
typedef enum SrResetReason {
    SR_RESET_CONTENT_CHANGED = 0,
    SR_RESET_RESOLUTION_CHANGED = 1,
    SR_RESET_MODEL_CHANGED = 2,
    SR_RESET_DEVICE_LOST = 3,
    SR_RESET_TIER_CHANGED = 4,
} SrResetReason;

SrResult srReset(SrContext* ctx, uint32_t reason);

SrResult srExecute(SrContext* ctx, const SrFrameDesc* frame, const SrSyncDesc* sync);

SrResult srGetStats(SrContext* ctx, SrStats* out);

void srRelease(SrContext* ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SR_DLL_H */
