#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 日志配置模块 - 为整个应用程序提供集中的日志设置和配置管理
"""

import logging
import sys
from typing import Optional

from app.config.settings import config


def setup_logging(app: Optional["FastAPI"] = None) -> None:
    """设置日志配置"""

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, config.log_level.upper()))

    # 根日志器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper()))

    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)

    # FastAPI应用不需要特殊的日志配置，它使用Python标准logging

    # 设置特定日志器的级别
    aiops_logger = logging.getLogger("aiops")
    aiops_logger.setLevel(getattr(logging, config.log_level.upper()))

    # 抑制一些第三方库的冗余日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    logger = logging.getLogger("aiops.logging")
    logger.info(f"日志系统初始化完成，级别: {config.log_level.upper()}")