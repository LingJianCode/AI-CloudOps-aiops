#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI-CloudOps智能根因分析API接口
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.api.decorators import api_response, log_api_call
from app.common.constants import (
    AppConstants,
    ErrorMessages,
    HttpStatusCodes,
    ServiceConstants,
)
from app.common.response import ResponseWrapper
from app.models import ListResponse
from app.models.rca_models import (
    RCAAnalysisResponse,
    RCAAnalyzeRequest,
    RCAHealthResponse,
)
from app.services.rca_service import RCAService

logger = logging.getLogger("aiops.api.rca")

router = APIRouter(tags=["rca"])
rca_service = RCAService()


@router.post("/analyze", summary="AI-CloudOps执行根因分析")
@api_response("AI-CloudOps执行根因分析")
@log_api_call(log_request=True)
async def analyze_root_cause(
    request: RCAAnalyzeRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """执行根因分析"""
    await rca_service.initialize()

    analysis_result = await rca_service.analyze_root_cause(
        namespace=request.namespace,
        time_window_hours=request.time_window_hours,
        metrics=request.metrics,
    )

    # 后台任务：缓存分析结果
    background_tasks.add_task(rca_service.cache_analysis_result, analysis_result)

    # 使用统一的响应模型
    response = RCAAnalysisResponse(
        namespace=request.namespace,
        analysis_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        time_window_hours=request.time_window_hours,
        root_causes=analysis_result.get("root_causes", []),
        anomalies=analysis_result.get("anomalies", {}),
        correlations=analysis_result.get("correlations", []),
        recommendations=analysis_result.get("recommendations", []),
        confidence_score=analysis_result.get("confidence_score", 0.0),
        status=analysis_result.get("status", "completed"),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/metrics", summary="AI-CloudOps获取所有可用的Prometheus指标")
@api_response("AI-CloudOps获取所有可用的Prometheus指标")
async def get_all_prometheus_metrics() -> Dict[str, Any]:
    """获取Prometheus指标"""
    await rca_service.initialize()

    try:
        # 获取所有可用的指标
        available_metrics = await rca_service.get_all_available_metrics()

        # 使用统一的列表响应格式
        metrics_response = ListResponse[str](
            items=available_metrics, total=len(available_metrics)
        )
        return ResponseWrapper.success(
            data=metrics_response.dict(),
            message="success",
        )
    except Exception as e:
        logger.error(f"获取Prometheus指标失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.RCA_SERVICE_ERROR,
        )


@router.get("/health", summary="AI-CloudOps健康检查")
@api_response("AI-CloudOps健康检查")
async def health_check() -> Dict[str, Any]:
    """检查RCA服务及其依赖的健康状态"""
    await rca_service.initialize()

    health_status = await rca_service.get_health_status()

    # 使用统一的响应模型
    from datetime import datetime

    response = RCAHealthResponse(
        status=health_status.get("status", "healthy"),
        prometheus_connected=health_status.get("prometheus_connected", False),
        kubernetes_connected=health_status.get("kubernetes_connected", False),
        redis_connected=health_status.get("redis_connected", False),
        last_check_time=datetime.now().isoformat(),
        version=health_status.get("version"),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/ready", summary="AI-CloudOps RCA服务就绪检查")
@api_response("AI-CloudOps RCA服务就绪检查")
async def rca_ready() -> Dict[str, Any]:
    """检查RCA服务是否已就绪可以提供服务"""
    try:
        await rca_service.initialize()
        health_status = await rca_service.get_health_status()

        is_ready = health_status.get("status") == "healthy"

        if not is_ready:
            raise HTTPException(status_code=503, detail="RCA服务未就绪")

        return ResponseWrapper.success(
            data={
                "ready": True,
                "service": "rca",
                "timestamp": datetime.now().isoformat(),
            },
            message="服务就绪",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail="RCA就绪检查失败")


@router.get("/config", summary="AI-CloudOps获取RCA配置")
@api_response("AI-CloudOps获取RCA配置")
async def get_rca_config() -> Dict[str, Any]:
    """获取RCA服务的配置信息"""
    try:
        await rca_service.initialize()
        config_info = await rca_service.get_config_info()
        return ResponseWrapper.success(data=config_info, message="配置获取成功")
    except Exception as e:
        logger.error(f"获取RCA配置失败: {str(e)}")
        return ResponseWrapper.error(message=f"配置获取失败: {str(e)}")


@router.get("/quick-diagnosis", summary="AI-CloudOps快速诊断")
@api_response("AI-CloudOps快速诊断")
async def quick_diagnosis(namespace: str) -> Dict[str, Any]:
    """快速问题诊断，返回最近1小时内的关键问题"""
    await rca_service.initialize()

    diagnosis_result = await rca_service.quick_diagnosis(namespace=namespace)

    return ResponseWrapper.success(data=diagnosis_result, message="success")


@router.get("/event-patterns", summary="AI-CloudOps事件模式分析")
@api_response("AI-CloudOps事件模式分析")
async def get_event_patterns(
    namespace: str,
    hours: float = Query(1.0, ge=0.1, le=24, description="分析时间范围（小时）"),
) -> Dict[str, Any]:
    """分析事件模式和趋势"""
    await rca_service.initialize()

    patterns_result = await rca_service.get_event_patterns(
        namespace=namespace, hours=hours
    )

    return ResponseWrapper.success(data=patterns_result, message="success")


@router.get("/error-summary", summary="AI-CloudOps错误摘要")
@api_response("AI-CloudOps错误摘要")
async def get_error_summary(
    namespace: str,
    hours: float = Query(1.0, ge=0.1, le=24, description="分析时间范围（小时）"),
) -> Dict[str, Any]:
    """汇总错误信息"""
    await rca_service.initialize()

    summary_result = await rca_service.get_error_summary(
        namespace=namespace, hours=hours
    )

    return ResponseWrapper.success(data=summary_result, message="success")


@router.get("/info", summary="AI-CloudOps RCA服务信息")
@api_response("AI-CloudOps RCA服务信息")
async def rca_info() -> Dict[str, Any]:
    """获取RCA服务的详细信息"""
    info = {
        "service": "根因分析",
        "version": AppConstants.APP_VERSION,
        "description": "多数据源智能根因分析服务，整合指标、事件和日志进行深度分析",
        "capabilities": [
            "根因分析",
            "指标异常检测",
            "事件关联分析",
            "日志模式识别",
            "快速问题诊断",
            "趋势分析",
            "智能建议",
        ],
        "endpoints": {
            "analyze": "/rca/analyze",
            "metrics": "/rca/metrics",
            "health": "/rca/health",
            "quick_diagnosis": "/rca/quick-diagnosis",
            "event_patterns": "/rca/event-patterns",
            "error_summary": "/rca/error-summary",
            "info": "/rca/info",
        },
        "data_sources": ["Prometheus指标", "Kubernetes事件", "Pod日志"],
        "analysis_methods": [
            "统计异常检测",
            "时间序列分析",
            "模式匹配",
            "关联分析",
            "因果推理",
        ],
        "constraints": {
            "max_time_window_hours": 24,
            "min_time_window_hours": 0.1,
            "max_log_lines": 1000,
            "default_log_lines": 100,
            "timeout": (
                ServiceConstants.RCA_TIMEOUT
                if hasattr(ServiceConstants, "RCA_TIMEOUT")
                else 300
            ),
        },
        "status": "available" if rca_service else "unavailable",
    }

    return ResponseWrapper.success(data=info, message="success")


__all__ = ["router"]
