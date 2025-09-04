#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 日志工具，提供带请求上下文的日志获取方法
"""

import contextvars
import logging
from typing import Optional


# 全局请求ID上下文变量
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


class ContextRequestIdFilter(logging.Filter):
    """从contextvars注入request_id到日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        request_id = request_id_ctx.get()
        if not hasattr(record, "request_id"):
            record.request_id = request_id or ""
        return True


def get_logger(name: str) -> logging.Logger:
    """获取带ContextFilter的日志器。"""
    logger = logging.getLogger(name)
    # 确保至少一个过滤器存在（避免重复添加多个相同过滤器）
    has_filter = any(isinstance(f, ContextRequestIdFilter) for f in logger.filters)
    if not has_filter:
        logger.addFilter(ContextRequestIdFilter())
    return logger


