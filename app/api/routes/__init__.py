#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: API路由模块初始化文件，负责注册和管理所有FastAPI端点
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
    api_v1.include_router(predict_router)
    logger.info("已注册预测路由")
except Exception as e:
    logger.warning(f"注册预测路由失败: {str(e)}")

try:
    from .rca import router as rca_router
    api_v1.include_router(rca_router)
    logger.info("已注册根因分析路由")
except Exception as e:
    logger.warning(f"注册根因分析路由失败: {str(e)}")

try:
    from .autofix import router as autofix_router
    api_v1.include_router(autofix_router)
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
    """注册所有FastAPI路由"""
    
    # 注册API v1路由
    app.include_router(api_v1)
    
    # 根路径端点
    @app.get("/", tags=["root"])
    async def root():
        """根路径端点，返回平台信息"""
        return {
            "service": "AIOps Platform",
            "version": "1.0.0",
            "status": "running",
            "description": "智能云原生运维平台",
            "endpoints": {
                "health": "/api/v1/health",
                "health_components": "/api/v1/health/components",
                "health_metrics": "/api/v1/health/metrics",
                "health_ready": "/api/v1/health/ready",
                "health_live": "/api/v1/health/live",
                "prediction": {
                    "predict": "/api/v1/predict",
                    "trend": "/api/v1/predict/trend",
                    "health": "/api/v1/predict/health",
                    "ready": "/api/v1/predict/ready",
                    "info": "/api/v1/predict/info"
                },
                "rca": {
                    "analyze": "/api/v1/rca",
                    "metrics": "/api/v1/rca/metrics",
                    "config": "/api/v1/rca/config",
                    "health": "/api/v1/rca/health",
                    "ready": "/api/v1/rca/ready",
                    "info": "/api/v1/rca/info"
                },
                "autofix": {
                    "fix": "/api/v1/autofix",
                    "diagnose": "/api/v1/autofix/diagnose",
                    "health": "/api/v1/autofix/health",
                    "ready": "/api/v1/autofix/ready",
                    "info": "/api/v1/autofix/info"
                },
                "assistant": {
                    "query": "/api/v1/assistant/query",
                    "session": "/api/v1/assistant/session",
                    "refresh": "/api/v1/assistant/refresh",
                    "health": "/api/v1/assistant/health",
                    "ready": "/api/v1/assistant/ready",
                    "info": "/api/v1/assistant/info"
                },
            },
            "features": [
                "智能负载预测",
                "根因分析",
                "自动修复",
                "智能问答",
                "健康检查"
            ]
        }
