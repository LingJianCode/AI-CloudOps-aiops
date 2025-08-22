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

    # 整体健康检查端点
    @app.get("/health", tags=["health"])
    async def overall_health():
        """整体应用健康检查"""
        from ...services.startup import startup_service
        from datetime import datetime
        
        try:
            # 获取所有服务状态
            services_status = await startup_service.get_services_status()
            startup_info = startup_service.get_startup_info()
            
            # 计算整体健康状态
            all_healthy = all(
                status.get("healthy", False) if status.get("type") == "BaseService" 
                else status.get("available", True)
                for status in services_status.values()
            )
            
            response = {
                "status": "healthy" if all_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "uptime": startup_info["uptime"],
                "services": services_status,
                "summary": {
                    "total_services": len(services_status),
                    "healthy_services": sum(
                        1 for status in services_status.values() 
                        if status.get("healthy", status.get("available", False))
                    ),
                    "initialized": startup_info["initialized"],
                    "managed_services": startup_info.get("managed_services", 0)
                }
            }
            
            return {"success": True, "data": response, "message": "健康检查完成"}
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "success": False, 
                "data": {
                    "status": "unhealthy", 
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, 
                "message": "健康检查失败"
            }
    
    # 就绪检查端点
    @app.get("/ready", tags=["health"])
    async def overall_ready():
        """整体应用就绪检查"""
        from ...services.startup import startup_service
        from datetime import datetime
        
        try:
            services_status = await startup_service.get_services_status()
            startup_info = startup_service.get_startup_info()
            
            # 检查关键服务是否就绪
            is_ready = (
                startup_info["initialized"] and 
                startup_info["healthy"] and
                len(services_status) > 0
            )
            
            if not is_ready:
                return {
                    "success": False, 
                    "data": {
                        "ready": False, 
                        "message": "应用未就绪",
                        "timestamp": datetime.now().isoformat()
                    }, 
                    "message": "应用未就绪"
                }
            
            return {
                "success": True, 
                "data": {
                    "ready": True, 
                    "message": "应用已就绪",
                    "timestamp": datetime.now().isoformat(),
                    "uptime": startup_info["uptime"]
                }, 
                "message": "应用已就绪"
            }
            
        except Exception as e:
            logger.error(f"就绪检查失败: {e}")
            return {
                "success": False, 
                "data": {
                    "ready": False, 
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }, 
                "message": "就绪检查失败"
            }
