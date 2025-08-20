#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 系统健康检查FastAPI模块 - 提供系统级健康监控和状态检查功能
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.response_models import APIResponse
from .health_manager import health_manager

logger = logging.getLogger("aiops.health")

# 创建路由器
router = APIRouter(tags=["health"])

# 响应模型
class HealthResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


@router.get("/health", response_model=HealthResponse, summary="系统综合健康检查")
async def health_check() -> HealthResponse:
    """
    系统综合健康检查API
    
    Returns:
        JSON: 系统整体健康状态
    """
    try:
        health_data = health_manager.get_overall_health()
        return HealthResponse(
            code=0,
            message="健康检查完成",
            data=health_data
        )

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"健康检查失败: {str(e)}"
        )


@router.get("/health/components", response_model=HealthResponse, summary="组件健康检查")
async def components_health() -> HealthResponse:
    """
    各组件健康检查API
    
    Returns:
        JSON: 各组件的健康状态详情
    """
    try:
        components_data = health_manager.get_components_health()
        return HealthResponse(
            code=0,
            message="组件健康检查完成",
            data=components_data
        )

    except Exception as e:
        logger.error(f"组件健康检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"组件健康检查失败: {str(e)}"
        )


@router.get("/health/metrics", response_model=HealthResponse, summary="系统指标检查")
async def metrics_health() -> HealthResponse:
    """
    系统指标检查API
    
    Returns:
        JSON: 系统资源使用指标
    """
    try:
        metrics_data = health_manager.get_system_metrics()
        return HealthResponse(
            code=0,
            message="系统指标检查完成",
            data=metrics_data
        )

    except Exception as e:
        logger.error(f"系统指标检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"系统指标检查失败: {str(e)}"
        )


@router.get("/health/ready", response_model=HealthResponse, summary="就绪状态检查")
async def readiness_check() -> HealthResponse:
    """
    就绪状态检查API - 检查服务是否准备好接收请求
    
    Returns:
        JSON: 服务就绪状态
    """
    try:
        ready_status = health_manager.check_readiness()
        
        # 如果未就绪，返回503状态
        if not ready_status.get("ready", False):
            raise HTTPException(
                status_code=503,
                detail="服务未就绪"
            )
        
        return HealthResponse(
            code=0,
            message="服务已就绪",
            data=ready_status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"就绪检查失败: {str(e)}"
        )


@router.get("/health/live", response_model=HealthResponse, summary="存活状态检查")
async def liveness_check() -> HealthResponse:
    """
    存活状态检查API - 检查服务是否正在运行
    
    Returns:
        JSON: 服务存活状态
    """
    try:
        live_status = health_manager.check_liveness()
        
        return HealthResponse(
            code=0,
            message="服务存活",
            data=live_status
        )

    except Exception as e:
        logger.error(f"存活检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"存活检查失败: {str(e)}"
        )


@router.get("/health/startup", response_model=HealthResponse, summary="启动状态检查")
async def startup_check() -> HealthResponse:
    """
    启动状态检查API - 检查服务启动是否完成
    
    Returns:
        JSON: 服务启动状态
    """
    try:
        startup_status = health_manager.check_startup()
        
        # 如果启动未完成，返回503状态
        if not startup_status.get("started", False):
            raise HTTPException(
                status_code=503,
                detail="服务启动未完成"
            )
        
        return HealthResponse(
            code=0,
            message="服务启动完成",
            data=startup_status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"启动检查失败: {str(e)}"
        )


@router.get("/health/dependencies", response_model=HealthResponse, summary="依赖服务检查")
async def dependencies_check() -> HealthResponse:
    """
    依赖服务健康检查API
    
    Returns:
        JSON: 依赖服务的健康状态
    """
    try:
        deps_status = health_manager.check_dependencies()
        
        return HealthResponse(
            code=0,
            message="依赖检查完成",
            data=deps_status
        )

    except Exception as e:
        logger.error(f"依赖检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"依赖检查失败: {str(e)}"
        )


@router.get("/health/detail", response_model=HealthResponse, summary="详细健康检查")
async def detailed_health() -> HealthResponse:
    """
    详细健康检查API - 包含所有健康检查信息
    
    Returns:
        JSON: 完整的系统健康状态
    """
    try:
        detailed_data = {
            "overall": health_manager.get_overall_health(),
            "components": health_manager.get_components_health(),
            "metrics": health_manager.get_system_metrics(),
            "dependencies": health_manager.check_dependencies(),
            "timestamp": datetime.utcnow().isoformat(),
            "check_duration": "N/A"  # 可以添加计时逻辑
        }
        
        return HealthResponse(
            code=0,
            message="详细健康检查完成",
            data=detailed_data
        )

    except Exception as e:
        logger.error(f"详细健康检查失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"详细健康检查失败: {str(e)}"
        )


# 导出
__all__ = ["router"]