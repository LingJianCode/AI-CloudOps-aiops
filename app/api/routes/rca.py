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
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.decorators import api_response, log_api_call
from app.common.constants import AppConstants, ErrorMessages, HttpStatusCodes
from app.common.response import ResponseWrapper
from app.models import (
    ErrorSummaryResponse,
    EventPatternsResponse,
    QuickDiagnosisResponse,
    ServiceConfigResponse,
    ServiceInfoResponse,
    ServiceReadyResponse,
)
from app.models.rca_models import (
    RCAAnalysisResponse,
    RCAAnalyzeRequest,
    RCAHealthResponse,
    RCAMetricsDataRequest,
    RCAEventsDataRequest, 
    RCALogsDataRequest,
    RCADataResponse,
    RCACacheStatsResponse,
    RCAClearCacheResponse,
    RCAClearNamespaceCacheRequest,
    RCAClearOperationCacheRequest,
    RCAQuickDiagnosisRequest,
    RCAEventPatternsRequest,
    RCAErrorSummaryRequest,
)
from app.services.rca_service import RCAService

logger = logging.getLogger("aiops.api.rca")

router = APIRouter(tags=["rca"])
rca_service = RCAService()


# 工具函数：统一的时间解析
def parse_iso_timestamp(timestamp_str: Optional[str], field_name: str) -> Optional[datetime]:
    """
    解析ISO格式时间戳的统一工具函数
    """
    if not timestamp_str:
        return None
    
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        raise HTTPException(
            status_code=HttpStatusCodes.BAD_REQUEST,
            detail=f"无效的{field_name}格式，请使用ISO格式"
        )


# 工具函数：统一的异常处理
def handle_service_error(operation: str, error: Exception) -> Dict[str, Any]:
    """
    统一的服务异常处理工具函数
    
    Args:
        operation: 操作名称
        error: 异常对象
    
    Returns:
        标准化的错误响应
    """
    error_message = f"{operation}失败: {str(error)}"
    logger.error(error_message, exc_info=True)
    
    # 根据异常类型判断HTTP状态码
    if isinstance(error, ValueError):
        status_code = HttpStatusCodes.BAD_REQUEST
    elif isinstance(error, ConnectionError):
        status_code = HttpStatusCodes.SERVICE_UNAVAILABLE
    elif isinstance(error, TimeoutError):
        status_code = HttpStatusCodes.REQUEST_TIMEOUT
    else:
        status_code = HttpStatusCodes.INTERNAL_SERVER_ERROR
    
    raise HTTPException(
        status_code=status_code,
        detail=error_message,
    )


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
    
    # 动态生成分析ID和时间戳
    analysis_id = str(uuid.uuid4())
    current_timestamp = datetime.now().isoformat()
    
    # 构建Response，使用实际分析结果的数据
    # 确保timestamp是字符串格式
    result_timestamp = analysis_result.get("timestamp", current_timestamp)
    if isinstance(result_timestamp, datetime):
        result_timestamp = result_timestamp.isoformat()
    
    response = RCAAnalysisResponse(
        namespace=request.namespace,
        analysis_id=analysis_id,
        timestamp=result_timestamp,
        time_window_hours=request.time_window_hours,
        root_causes=analysis_result.get("root_causes", []),
        anomalies=analysis_result.get("anomalies", {}),
        correlations=analysis_result.get("correlations", []),
        recommendations=analysis_result.get("recommendations", []),
        confidence_score=analysis_result.get("confidence_score", 0.0),
        status="completed" if analysis_result.get("success", True) else "failed",
    )

    return ResponseWrapper.success(data=response.model_dump(), message="根因分析完成")


@router.get("/metrics", summary="AI-CloudOps获取所有可用的Prometheus指标")
@api_response("AI-CloudOps获取所有可用的Prometheus指标")
async def get_all_prometheus_metrics() -> Dict[str, Any]:
    """获取Prometheus指标"""
    await rca_service.initialize()

    try:
        available_metrics = await rca_service.get_all_available_metrics()
        return ResponseWrapper.success_list(
            items=available_metrics,
            message="获取指标成功",
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
        last_check_time=health_status.get("timestamp", datetime.now()).isoformat() if isinstance(health_status.get("timestamp"), datetime) else datetime.now().isoformat(),
        version=health_status.get("version"),
    )

    return ResponseWrapper.success(data=response.model_dump(), message="健康检查完成")


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
            message="服务就绪",
        )
        return ResponseWrapper.success(
            data=response.model_dump(),
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
            service="rca", config=config_info, timestamp=datetime.now().isoformat()
        )
        return ResponseWrapper.success(data=response.model_dump(), message="配置获取成功")
    except Exception as e:
        logger.error(f"获取RCA配置失败: {str(e)}")
        return ResponseWrapper.error(message=f"配置获取失败: {str(e)}")


@router.post("/quick-diagnosis", summary="AI-CloudOps快速诊断")
@api_response("AI-CloudOps快速诊断")
async def quick_diagnosis(request: RCAQuickDiagnosisRequest) -> Dict[str, Any]:
    """快速问题诊断"""
    await rca_service.initialize()

    diagnosis_result = await rca_service.quick_diagnosis(namespace=request.namespace)

    # 计算分析时长（如果服务未提供）
    analysis_duration = diagnosis_result.get("analysis_duration", 0.0)
    if analysis_duration == 0.0 and "diagnosis_time" in diagnosis_result:
        # 使用当前时间与诊断时间的差值估算
        try:
            diagnosis_dt = datetime.fromisoformat(diagnosis_result["diagnosis_time"].replace('Z', '+00:00'))
            current_dt = datetime.now(diagnosis_dt.tzinfo)
            analysis_duration = (current_dt - diagnosis_dt).total_seconds()
        except:
            analysis_duration = 0.0
    
    # 优化：使用实际的诊断时间戳，而不是硬编码
    diagnosis_timestamp = diagnosis_result.get("diagnosis_time", datetime.now().isoformat())
    
    response = QuickDiagnosisResponse(
        namespace=request.namespace,
        status=diagnosis_result.get("status", "completed"),
        critical_issues=diagnosis_result.get("critical_issues", []),
        warnings=diagnosis_result.get("warnings", []),
        recommendations=diagnosis_result.get("recommendations", []),
        timestamp=diagnosis_timestamp,
        analysis_duration=analysis_duration,
    )

    return ResponseWrapper.success(data=response.model_dump(), message="快速诊断完成")


@router.post("/event-patterns", summary="AI-CloudOps事件模式分析")
@api_response("AI-CloudOps事件模式分析")
async def get_event_patterns(
    request: RCAEventPatternsRequest
) -> Dict[str, Any]:
    """分析事件模式"""
    await rca_service.initialize()

    patterns_result = await rca_service.get_event_patterns(
        namespace=request.namespace, hours=request.hours
    )

    response = EventPatternsResponse(
        namespace=request.namespace,
        time_range_hours=request.hours,
        patterns=patterns_result.get("patterns", []),
        trending_events=patterns_result.get("trending_events", []),
        anomalous_events=patterns_result.get("anomalous_events", []),
        timestamp=datetime.now().isoformat(),
    )

    return ResponseWrapper.success(data=response.model_dump(), message="事件模式分析完成")


@router.post("/error-summary", summary="AI-CloudOps错误摘要")
@api_response("AI-CloudOps错误摘要")
async def get_error_summary(
    request: RCAErrorSummaryRequest
) -> Dict[str, Any]:
    """汇总错误信息"""
    await rca_service.initialize()

    summary_result = await rca_service.get_error_summary(
        namespace=request.namespace, hours=request.hours
    )

    response = ErrorSummaryResponse(
        namespace=request.namespace,
        time_range_hours=request.hours,
        total_errors=summary_result.get("total_errors", 0),
        error_categories=summary_result.get("error_categories", {}),
        top_errors=summary_result.get("top_errors", []),
        error_timeline=summary_result.get("error_timeline", []),
        timestamp=datetime.now().isoformat(),
    )

    return ResponseWrapper.success(data=response.model_dump(), message="错误摘要分析完成")


# 独立的数据查询接口
@router.post("/data/metrics", summary="AI-CloudOps查询指标数据")
@api_response("AI-CloudOps查询指标数据")
@log_api_call(log_request=True)
async def query_metrics_data(
    request: RCAMetricsDataRequest
) -> Dict[str, Any]:
    """直接查询指标数据"""
    await rca_service.initialize()

    # 使用service层的时间解析方法
    parsed_start_time = rca_service.parse_iso_timestamp(request.start_time, "开始时间")
    parsed_end_time = rca_service.parse_iso_timestamp(request.end_time, "结束时间")

    try:
        metrics_result = await rca_service.get_metrics(
            namespace=request.namespace,
            start_time=parsed_start_time,
            end_time=parsed_end_time,
            metrics=request.metrics,
        )

        response = RCADataResponse(
            namespace=request.namespace,
            items=metrics_result.get("items", []),
            total=metrics_result.get("total", 0),
            start_time=request.start_time,
            end_time=request.end_time,
            query_params={"metrics": request.metrics},
            timestamp=datetime.now().isoformat(),
        )

        return ResponseWrapper.success(
            data=response.model_dump(),
            message="指标数据查询成功"
        )
    except Exception as e:
        rca_service.handle_service_error("查询指标数据", e)


@router.post("/data/events", summary="AI-CloudOps查询事件数据")
@api_response("AI-CloudOps查询事件数据")
@log_api_call(log_request=True)
async def query_events_data(
    request: RCAEventsDataRequest
) -> Dict[str, Any]:
    """直接查询事件数据"""
    await rca_service.initialize()

    # 使用service层的时间解析方法
    parsed_start_time = rca_service.parse_iso_timestamp(request.start_time, "开始时间")
    parsed_end_time = rca_service.parse_iso_timestamp(request.end_time, "结束时间")

    try:
        events_result = await rca_service.get_events(
            namespace=request.namespace,
            start_time=parsed_start_time,
            end_time=parsed_end_time,
            severity=request.severity,
        )

        response = RCADataResponse(
            namespace=request.namespace,
            items=events_result.get("items", []),
            total=events_result.get("total", 0),
            start_time=request.start_time,
            end_time=request.end_time,
            query_params={"severity": request.severity},
            timestamp=datetime.now().isoformat(),
        )

        return ResponseWrapper.success(
            data=response.model_dump(),
            message="事件数据查询成功"
        )
    except Exception as e:
        rca_service.handle_service_error("查询事件数据", e)


@router.post("/data/logs", summary="AI-CloudOps查询日志数据")
@api_response("AI-CloudOps查询日志数据")
@log_api_call(log_request=True)
async def query_logs_data(
    request: RCALogsDataRequest
) -> Dict[str, Any]:
    """直接查询日志数据"""
    await rca_service.initialize()

    # 使用service层的时间解析方法
    parsed_start_time = rca_service.parse_iso_timestamp(request.start_time, "开始时间")
    parsed_end_time = rca_service.parse_iso_timestamp(request.end_time, "结束时间")

    try:
        logs_result = await rca_service.get_logs(
            namespace=request.namespace,
            start_time=parsed_start_time,
            end_time=parsed_end_time,
            pod_name=request.pod_name,
            error_only=request.error_only,
            max_lines=request.max_lines,
        )

        response = RCADataResponse(
            namespace=request.namespace,
            items=logs_result.get("items", []),
            total=logs_result.get("total", 0),
            start_time=request.start_time,
            end_time=request.end_time,
            query_params={
                "pod_name": request.pod_name,
                "error_only": request.error_only,
                "max_lines": request.max_lines,
            },
            timestamp=datetime.now().isoformat(),
        )

        return ResponseWrapper.success(
            data=response.model_dump(),
            message="日志数据查询成功"
        )
    except Exception as e:
        rca_service.handle_service_error("查询日志数据", e)


# 缓存管理接口
@router.get("/cache/stats", summary="AI-CloudOps获取RCA缓存统计")
@api_response("AI-CloudOps获取RCA缓存统计")
async def get_cache_stats() -> Dict[str, Any]:
    """获取RCA缓存统计信息"""
    try:
        await rca_service.initialize()
        cache_stats = await rca_service.get_cache_stats()
        
        response = RCACacheStatsResponse(
            available=cache_stats.get("available", False),
            healthy=cache_stats.get("healthy"),
            cache_prefix=cache_stats.get("cache_prefix"),
            default_ttl=cache_stats.get("default_ttl"),
            hit_rate=cache_stats.get("hit_rate"),
            total_keys=cache_stats.get("total_keys"),
            memory_usage=cache_stats.get("memory_usage"),
            timestamp=cache_stats.get("timestamp", datetime.now().isoformat()),
            message=cache_stats.get("message"),
        )
        
        return ResponseWrapper.success(
            data=response.model_dump(),
            message="缓存统计获取成功",
        )
    except Exception as e:
        logger.error(f"获取缓存统计失败: {str(e)}")
        return ResponseWrapper.error(message=f"获取缓存统计失败: {str(e)}")


@router.delete("/cache/clear", summary="AI-CloudOps清理所有RCA缓存")
@api_response("AI-CloudOps清理所有RCA缓存")
async def clear_all_cache() -> Dict[str, Any]:
    """清理所有RCA缓存"""
    try:
        await rca_service.initialize()
        result = await rca_service.clear_all_cache()
        
        response = RCAClearCacheResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            cleared_count=result.get("cleared_count", 0),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
        )
        
        return ResponseWrapper.success(
            data=response.model_dump(),
            message="缓存清理完成",
        )
    except Exception as e:
        logger.error(f"清理缓存失败: {str(e)}")
        return ResponseWrapper.error(message=f"清理缓存失败: {str(e)}")


@router.post("/cache/clear/namespace", summary="AI-CloudOps清理指定命名空间缓存")
@api_response("AI-CloudOps清理指定命名空间缓存")
async def clear_namespace_cache(request: RCAClearNamespaceCacheRequest) -> Dict[str, Any]:
    """清理指定命名空间的缓存"""
    try:
        await rca_service.initialize()
        result = await rca_service.clear_namespace_cache(request.namespace)
        
        response = RCAClearCacheResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            cleared_count=result.get("cleared_count", 0),
            namespace=result.get("namespace"),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
        )
        
        return ResponseWrapper.success(
            data=response.model_dump(),
            message=f"命名空间 {request.namespace} 缓存清理完成",
        )
    except Exception as e:
        logger.error(f"清理命名空间缓存失败: {str(e)}")
        return ResponseWrapper.error(message=f"清理命名空间缓存失败: {str(e)}")


@router.post("/cache/clear/operation", summary="AI-CloudOps清理指定操作缓存")
@api_response("AI-CloudOps清理指定操作缓存")
async def clear_operation_cache(request: RCAClearOperationCacheRequest) -> Dict[str, Any]:
    """清理指定操作类型的缓存"""
    try:
        await rca_service.initialize()
        result = await rca_service.clear_operation_cache(request.operation)
        
        response = RCAClearCacheResponse(
            success=result.get("success", False),
            message=result.get("message", ""),
            cleared_count=result.get("cleared_count", 0),
            operation=result.get("operation"),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
        )
        
        return ResponseWrapper.success(
            data=response.model_dump(),
            message=f"操作 {request.operation} 缓存清理完成",
        )
    except Exception as e:
        logger.error(f"清理操作缓存失败: {str(e)}")
        return ResponseWrapper.error(message=f"清理操作缓存失败: {str(e)}")


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
            "data_metrics": "/rca/data/metrics",
            "data_events": "/rca/data/events",
            "data_logs": "/rca/data/logs",
            "cache_stats": "/rca/cache/stats",
            "cache_clear_all": "/rca/cache/clear",
            "cache_clear_namespace": "/rca/cache/clear/namespace",
            "cache_clear_operation": "/rca/cache/clear/operation",
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
        status=info["status"],
    )

    return ResponseWrapper.success(data=response.model_dump(), message="服务信息获取成功")


__all__ = ["router"]
