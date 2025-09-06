# NVTX标记 装饰器函数，用于Nsight system 性能分析

import torch
import functools
import os

# 通过环境变量控制是否启用 NVTX
ENABLE_NVTX = os.getenv("ENABLE_NVTX", "1").lower() in ("1", "true", "on")

def nvtx_range(name=None, color=None):
    """
    NVTX 装饰器：为函数添加性能分析标记
    使用示例：
        @nvtx_range()
        def forward(x): ...

        @nvtx_range("My Forward", color="green")
        def custom_func(): ...
    """
    def decorator(func):
        # 如果禁用 NVTX，直接返回原函数（零开销）
        if not ENABLE_NVTX:
            return func

        # 确定范围名称
        range_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 构造带颜色的名称（Nsight Systems 支持 ###color 语法）
            if color:
                full_name = f"{range_name}###{color}"
            else:
                full_name = range_name

            torch.cuda.nvtx.range_push(full_name)
            try:
                return func(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        return wrapper
    return decorator