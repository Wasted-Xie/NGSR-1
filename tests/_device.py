"""测试用计算设备选择。

策略
----
默认自动：有 CUDA 就用 CUDA（GPU 加速），否则回退 CPU。
可用环境变量 SR_TEST_DEVICE 显式覆盖：

    SR_TEST_DEVICE=cpu        强制 CPU（用于复现设备无关的基准结果）
    SR_TEST_DEVICE=cuda       强制 CUDA（不可用则直接报错，不静默回退）
    SR_TEST_DEVICE=cuda:1     指定卡号

设计说明
--------
之所以保留 CPU 通路：这些测试验证的是**逻辑正确性**（形状、梯度可达性、
因果性），不应依赖具体硬件。GPU 只是加速手段，因此提供强制回退开关，
以便在怀疑设备相关数值差异时做对照。
"""

import os

import torch


def _resolve() -> torch.device:
    want = os.environ.get("SR_TEST_DEVICE", "").strip().lower()

    if want:
        if want.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "SR_TEST_DEVICE=%s，但当前环境 CUDA 不可用" % want
                )
            return torch.device(want)
        return torch.device(want)

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _resolve()


def device_desc() -> str:
    """人类可读的设备描述，用于测试输出标注。"""
    if DEVICE.type == "cuda":
        idx = DEVICE.index or 0
        return "cuda:%d (%s)" % (idx, torch.cuda.get_device_name(idx))
    return "cpu"
