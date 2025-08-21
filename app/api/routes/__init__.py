#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 模块初始化文件
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger("aiops.routes")

# 创建API v1路由器
api_v1 = APIRouter(prefix="/api/v1", tags=["api_v1"])

# 导入所有路由器
try:
    from .health import router as health_router

    api_v1.include_router(health_router)
    logger.info("已注册健康检查路由")
except Exception as e:
    logger.warning(f"注册健康检查路由失败: {str(e)}")

try:
    from .predict import router as predict_router

    api_v1.include_router(predict_router, prefix="/predict")
    logger.info("已注册预测路由")
except Exception as e:
    logger.warning(f"注册预测路由失败: {str(e)}")

try:
    from .rca import router as rca_router

    api_v1.include_router(rca_router, prefix="/rca")
    logger.info("已注册根因分析路由")
except Exception as e:
    logger.warning(f"注册根因分析路由失败: {str(e)}")

try:
    from .autofix import router as autofix_router

    api_v1.include_router(autofix_router, prefix="/autofix")
    logger.info("已注册自动修复路由")
except Exception as e:
    logger.warning(f"注册自动修复路由失败: {str(e)}")

try:
    from .assistant import router as assistant_router

    api_v1.include_router(assistant_router, prefix="/assistant")
    logger.info("已注册智能助手路由")
except Exception as e:
    logger.warning(f"注册智能助手路由失败: {str(e)}")


def register_routes(app):

    # 注册API v1路由
    app.include_router(api_v1)

    # 根路径端点
    @app.get("/", tags=["root"])
    async def root():
        from ...common.constants import get_api_info

        return get_api_info()
