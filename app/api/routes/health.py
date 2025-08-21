#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查API接口
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.decorators import api_response, log_api_call
from app.common.response import ResponseWrapper
from app.services.health_service import HealthService

logger = logging.getLogger("aiops.api.health")

router = APIRouter(tags=["health"])
health_service = HealthService()


@router.get("/health", summary="系统综合健康检查")
@api_response("系统健康检查")
@log_api_call(log_request=True)
async def health_check() -> Dict[str, Any]:
    await health_service.initialize()
    health_data = await health_service.get_overall_health()
    return ResponseWrapper.success(data=health_data, message="success")


@router.get("/health/components", summary="组件健康检查")
@api_response("组件健康检查")
async def components_health() -> Dict[str, Any]:
    await health_service.initialize()
    components_data = await health_service.get_components_health()
    return ResponseWrapper.success(data=components_data, message="success")


@router.get("/health/metrics", summary="系统指标检查")
@api_response("系统指标检查")
async def metrics_health() -> Dict[str, Any]:
    await health_service.initialize()
    metrics_data = await health_service.get_system_metrics()
    return ResponseWrapper.success(data=metrics_data, message="success")


@router.get("/health/ready", summary="就绪状态检查")
@api_response("就绪状态检查")
async def readiness_check() -> Dict[str, Any]:
    await health_service.initialize()
    ready_status = await health_service.check_readiness()

    if not ready_status.get("ready", False):
        raise HTTPException(status_code=503, detail="服务未就绪")

    return ResponseWrapper.success(data=ready_status, message="success")


@router.get("/health/live", summary="存活状态检查")
@api_response("存活状态检查")
async def liveness_check() -> Dict[str, Any]:
    await health_service.initialize()
    live_status = await health_service.check_liveness()
    return ResponseWrapper.success(data=live_status, message="success")


@router.get("/health/startup", summary="启动状态检查")
@api_response("启动状态检查")
async def startup_check() -> Dict[str, Any]:
    await health_service.initialize()
    startup_status = await health_service.check_startup()

    if not startup_status.get("started", False):
        raise HTTPException(status_code=503, detail="服务启动未完成")

    return ResponseWrapper.success(data=startup_status, message="success")


@router.get("/health/dependencies", summary="依赖服务检查")
@api_response("依赖服务检查")
async def dependencies_check() -> Dict[str, Any]:
    await health_service.initialize()
    deps_status = await health_service.check_dependencies()
    return ResponseWrapper.success(data=deps_status, message="success")


@router.get("/health/detail", summary="详细健康检查")
@api_response("详细健康检查")
async def detailed_health() -> Dict[str, Any]:
    await health_service.initialize()
    detailed_data = await health_service.get_detailed_health()
    return ResponseWrapper.success(data=detailed_data, message="success")


__all__ = ["router"]
