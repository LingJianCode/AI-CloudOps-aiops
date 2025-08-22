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
from app.common.constants import AppConstants, ErrorMessages, HttpStatusCodes
from app.common.response import ResponseWrapper
from app.models import (
    ErrorSummaryResponse,
    EventPatternsResponse,
    ListResponse,
    QuickDiagnosisResponse,
    ServiceConfigResponse,
    ServiceInfoResponse,
    ServiceReadyResponse,
)
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

    background_tasks.add_task(rca_service.cache_analysis_result, analysis_result)
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
        available_metrics = await rca_service.get_all_available_metrics()
        return ResponseWrapper.success_list(
            items=available_metrics,
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
    """检查RCA服务就绪状态"""
    try:
        await rca_service.initialize()
        health_status = await rca_service.get_health_status()

        is_ready = health_status.get("status") == "healthy"

        if not is_ready:
            raise HTTPException(status_code=503, detail="RCA服务未就绪")

        response = ServiceReadyResponse(
            ready=True,
            service="rca",
            timestamp=datetime.now().isoformat(),
            message="服务就绪"
        )
        return ResponseWrapper.success(
            data=response.dict(),
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
    """获取RCA配置信息"""
    try:
        await rca_service.initialize()
        config_info = await rca_service.get_config_info()
        
        response = ServiceConfigResponse(
            service="rca",
            config=config_info,
            timestamp=datetime.now().isoformat()
        )
        return ResponseWrapper.success(data=response.dict(), message="配置获取成功")
    except Exception as e:
        logger.error(f"获取RCA配置失败: {str(e)}")
        return ResponseWrapper.error(message=f"配置获取失败: {str(e)}")


@router.get("/quick-diagnosis", summary="AI-CloudOps快速诊断")
@api_response("AI-CloudOps快速诊断")
async def quick_diagnosis(namespace: str) -> Dict[str, Any]:
    """快速问题诊断"""
    await rca_service.initialize()

    diagnosis_result = await rca_service.quick_diagnosis(namespace=namespace)
    
    response = QuickDiagnosisResponse(
        namespace=namespace,
        status=diagnosis_result.get("status", "completed"),
        critical_issues=diagnosis_result.get("critical_issues", []),
        warnings=diagnosis_result.get("warnings", []),
        recommendations=diagnosis_result.get("recommendations", []),
        timestamp=datetime.now().isoformat(),
        analysis_duration=diagnosis_result.get("analysis_duration", 0.0)
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/event-patterns", summary="AI-CloudOps事件模式分析")
@api_response("AI-CloudOps事件模式分析")
async def get_event_patterns(
    namespace: str,
    hours: float = Query(1.0, ge=0.1, le=24, description="分析时间范围（小时）"),
) -> Dict[str, Any]:
    """分析事件模式"""
    await rca_service.initialize()

    patterns_result = await rca_service.get_event_patterns(
        namespace=namespace, hours=hours
    )
    
    response = EventPatternsResponse(
        namespace=namespace,
        time_range_hours=hours,
        patterns=patterns_result.get("patterns", []),
        trending_events=patterns_result.get("trending_events", []),
        anomalous_events=patterns_result.get("anomalous_events", []),
        timestamp=datetime.now().isoformat()
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


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
    
    response = ErrorSummaryResponse(
        namespace=namespace,
        time_range_hours=hours,
        total_errors=summary_result.get("total_errors", 0),
        error_categories=summary_result.get("error_categories", {}),
        top_errors=summary_result.get("top_errors", []),
        error_timeline=summary_result.get("error_timeline", []),
        timestamp=datetime.now().isoformat()
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/info", summary="AI-CloudOps RCA服务信息")
@api_response("AI-CloudOps RCA服务信息")
async def rca_info() -> Dict[str, Any]:
    """获取RCA服务信息"""
    info = {
        "service": "根因分析",
        "version": AppConstants.APP_VERSION,
        "description": "智能根因分析服务",
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
        "analysis_methods": ["异常检测", "时间序列分析", "关联分析"],
        "constraints": {
            "max_time_window_hours": 24,
            "min_time_window_hours": 0.1,
            "max_log_lines": 1000,
            "default_log_lines": 100,
            "timeout": 300,
        },
        "status": "available" if rca_service else "unavailable",
    }

    response = ServiceInfoResponse(
        service=info["service"],
        version=info["version"],
        description=info["description"],
        capabilities=info["capabilities"],
        endpoints=info["endpoints"],
        constraints=info["constraints"],
        status=info["status"]
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


__all__ = ["router"]
