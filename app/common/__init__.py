#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 模块初始化文件
"""

__all__ = []

# 便捷导出日志工具
try:
    from .logger import get_logger  # type: ignore[F401]
except Exception:
    # 在早期初始化或工具缺失时安全忽略
    pass
