#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能小助手API路由模块，提供智能问答功能
"""

import logging

# 创建日志器
logger = logging.getLogger("aiops.api.assistant")

# 从重构后的模块导入所需组件
from app.api.routes.assistant_routes import assistant_bp
from app.api.routes.assistant_init import get_assistant_agent, init_assistant_in_background

# 重导出必要的组件，保持向后兼容
__all__ = [
    'assistant_bp',
    'get_assistant_agent'
]

# 应用启动时自动初始化
init_assistant_in_background()

logger.info("已加载重构后的小助手API路由模块")
