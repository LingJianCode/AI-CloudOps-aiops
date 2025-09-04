#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 日志配置模块
"""

import logging
import sys
from typing import Any, Optional

from app.config.settings import config


class RequestIdFilter(logging.Filter):
    """为日志记录注入 request_id 字段（若无则从contextvars获取）"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            if not hasattr(record, "request_id") or not record.request_id:
                from app.common.logger import request_id_ctx

                rid = request_id_ctx.get()
                record.request_id = rid or ""
        except Exception:
            if not hasattr(record, "request_id"):
                record.request_id = ""
        return True


def setup_logging(app: Optional[Any] = None) -> None:
    """设置日志配置（统一格式、按模块分类、支持级别配置）"""

    # 统一日志格式（包含模块与请求ID占位符）
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    console_handler.setLevel(getattr(logging, config.log_level.upper()))

    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper()))

    # 清理旧处理器，避免重复输出
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(console_handler)

    # 按模块分类日志器（仅设置级别，复用根处理器）
    module_loggers = {
        "aiops": config.log_level.upper(),
        "aiops.services": config.log_level.upper(),
        "aiops.core": config.log_level.upper(),
        "aiops.api": config.log_level.upper(),
    }
    for name, level in module_loggers.items():
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))

    # 第三方库降噪
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    logging.getLogger("aiops.logging").info(
        f"日志系统初始化完成，级别: {config.log_level.upper()}"
    )
