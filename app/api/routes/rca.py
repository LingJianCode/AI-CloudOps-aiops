#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析FastAPI路由 - 提供RCA接口的统一访问
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, BackgroundTasks, Query

from app.api.decorators import api_response, log_api_call
from app.common.constants import ServiceConstants, AppConstants, ApiEndpoints
from app.common.response import ResponseWrapper
from app.models.rca_models import (
    RCAAnalyzeRequest, RCAMetricsRequest, RCAEventsRequest, 
    RCALogsRequest, RCAQuickDiagnosisRequest, RCAEventPatternsRequest,
    RCAErrorSummaryRequest
)
from app.services.rca_service import RCAService

logger = logging.getLogger("aiops.api.rca")

router = APIRouter(tags=["rca"])
rca_service = RCAService()


@router.post("/analyze", summary="执行根因分析")
@api_response("执行根因分析")
@log_api_call(log_request=True)
async def analyze_root_cause(request: RCAAnalyzeRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """执行根因分析，整合指标、事件和日志三种数据源"""
    await rca_service.initialize()

    analysis_result = await rca_service.analyze_root_cause(
        namespace=request.namespace,
        time_window_hours=request.time_window_hours,
        metrics=request.metrics
    )

    # 后台任务：缓存分析结果
    background_tasks.add_task(rca_service.cache_analysis_result, analysis_result)

    return ResponseWrapper.success(
        data=analysis_result,
        message="success"
    )


@router.get("/metrics", summary="获取指标数据")
@api_response("获取指标数据")
@log_api_call(log_request=True)
async def get_metrics(
    namespace: str,
    start_time: Optional[str] = Query(None, description="开始时间(ISO格式)"),
    end_time: Optional[str] = Query(None, description="结束时间(ISO格式)"),
    metrics: Optional[str] = Query(None, description="逗号分隔的指标名称")
) -> Dict[str, Any]:
    """获取Prometheus指标数据"""
    await rca_service.initialize()

    # 转换时间格式
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if start_time else None
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else None

    metrics_result = await rca_service.get_metrics(
        namespace=namespace,
        start_time=start_dt,
        end_time=end_dt,
        metrics=metrics
    )

    return ResponseWrapper.success(
        data=metrics_result,
        message="success"
    )


@router.get("/events", summary="获取事件数据")
@api_response("获取事件数据")
async def get_events(
    namespace: str,
    start_time: Optional[str] = Query(None, description="开始时间(ISO格式)"),
    end_time: Optional[str] = Query(None, description="结束时间(ISO格式)"),
    severity: Optional[str] = Query(None, description="严重程度过滤")
) -> Dict[str, Any]:
    """获取Kubernetes事件数据"""
    await rca_service.initialize()

    # 转换时间格式
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if start_time else None
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else None

    events_result = await rca_service.get_events(
        namespace=namespace,
        start_time=start_dt,
        end_time=end_dt,
        severity=severity
    )

    return ResponseWrapper.success(
        data=events_result,
        message="success"
    )


@router.get("/logs", summary="获取日志数据")
@api_response("获取日志数据")
async def get_logs(
    namespace: str,
    start_time: Optional[str] = Query(None, description="开始时间(ISO格式)"),
    end_time: Optional[str] = Query(None, description="结束时间(ISO格式)"),
    pod_name: Optional[str] = Query(None, description="Pod名称"),
    error_only: bool = Query(True, description="只返回错误日志"),
    max_lines: int = Query(100, le=1000, description="最大日志行数")
) -> Dict[str, Any]:
    """获取Pod日志数据"""
    await rca_service.initialize()

    # 转换时间格式
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00')) if start_time else None
    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else None

    logs_result = await rca_service.get_logs(
        namespace=namespace,
        start_time=start_dt,
        end_time=end_dt,
        pod_name=pod_name,
        error_only=error_only,
        max_lines=max_lines
    )

    return ResponseWrapper.success(
        data=logs_result,
        message="success"
    )


@router.get("/health", summary="健康检查")
@api_response("健康检查")
async def health_check() -> Dict[str, Any]:
    """检查RCA服务及其依赖的健康状态"""
    await rca_service.initialize()

    health_status = await rca_service.get_health_status()

    return ResponseWrapper.success(
        data=health_status,
        message="success"
    )


@router.get("/quick-diagnosis", summary="快速诊断")
@api_response("快速诊断")
async def quick_diagnosis(
    namespace: str
) -> Dict[str, Any]:
    """快速问题诊断，返回最近1小时内的关键问题"""
    await rca_service.initialize()

    diagnosis_result = await rca_service.quick_diagnosis(
        namespace=namespace
    )

    return ResponseWrapper.success(
        data=diagnosis_result,
        message="success"
    )


@router.get("/event-patterns", summary="事件模式分析")
@api_response("事件模式分析")
async def get_event_patterns(
    namespace: str,
    hours: float = Query(1.0, ge=0.1, le=24, description="分析时间范围（小时）")
) -> Dict[str, Any]:
    """分析事件模式和趋势"""
    await rca_service.initialize()

    patterns_result = await rca_service.get_event_patterns(
        namespace=namespace,
        hours=hours
    )

    return ResponseWrapper.success(
        data=patterns_result,
        message="success"
    )


@router.get("/error-summary", summary="错误摘要")
@api_response("错误摘要")
async def get_error_summary(
    namespace: str,
    hours: float = Query(1.0, ge=0.1, le=24, description="分析时间范围（小时）")
) -> Dict[str, Any]:
    """汇总错误信息"""
    await rca_service.initialize()

    summary_result = await rca_service.get_error_summary(
        namespace=namespace,
        hours=hours
    )

    return ResponseWrapper.success(
        data=summary_result,
        message="success"
    )


@router.get("/debug/data-collection", summary="调试数据收集")
@api_response("调试数据收集")
@log_api_call(log_request=True)
async def debug_data_collection(
    namespace: str,
    time_window_hours: float = Query(1.0, ge=0.1, le=24, description="时间窗口（小时）"),
    metrics: Optional[str] = Query(None, description="逗号分隔的指标名称")
) -> Dict[str, Any]:
    """调试端点：查看原始数据收集结果"""
    from datetime import datetime, timedelta, timezone
    
    await rca_service.initialize()
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=time_window_hours)
    
    # 准备指标列表
    metric_list = metrics.split(",") if metrics else ["apiserver_request_total", "node_cpu_seconds_total"]
    
    # 直接测试数据收集器
    debug_result = {
        "test_info": {
            "namespace": namespace,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "requested_metrics": metric_list,
            "time_window_hours": time_window_hours
        },
        "collection_results": {}
    }
    
    try:
        # 测试指标收集
        metrics_data = await rca_service._metrics_collector.collect(
            namespace=namespace,
            start_time=start_time,
            end_time=end_time,
            metrics=metric_list
        )
        debug_result["collection_results"]["metrics"] = {
            "success": True,
            "count": len(metrics_data),
            "data": [{"name": m.name, "values_count": len(m.values), "anomaly_score": m.anomaly_score} for m in metrics_data[:5]]
        }
    except Exception as e:
        debug_result["collection_results"]["metrics"] = {
            "success": False,
            "error": str(e)
        }
    
    try:
        # 测试事件收集
        events_data = await rca_service._events_collector.collect(
            namespace=namespace,
            start_time=start_time,
            end_time=end_time
        )
        debug_result["collection_results"]["events"] = {
            "success": True,
            "count": len(events_data),
            "critical_count": len([e for e in events_data if e.severity.value in ["critical", "high"]])
        }
    except Exception as e:
        debug_result["collection_results"]["events"] = {
            "success": False,
            "error": str(e)
        }
    
    try:
        # 测试日志收集
        logs_data = await rca_service._logs_collector.collect(
            namespace=namespace,
            start_time=start_time,
            end_time=end_time,
            error_only=True,
            max_lines=100
        )
        debug_result["collection_results"]["logs"] = {
            "success": True,
            "count": len(logs_data),
            "error_count": len([l for l in logs_data if l.level in ["ERROR", "FATAL"]])
        }
    except Exception as e:
        debug_result["collection_results"]["logs"] = {
            "success": False,
            "error": str(e)
        }
    
    # 健康检查
    try:
        health_checks = await rca_service._gather_health_checks()
        debug_result["health_checks"] = health_checks
    except Exception as e:
        debug_result["health_checks"] = {"error": str(e)}
    
    return ResponseWrapper.success(
        data=debug_result,
        message="调试数据收集完成"
    )


@router.get("/info", summary="RCA服务信息")
@api_response("RCA服务信息")
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
            "智能建议"
        ],
        "endpoints": {
            "analyze": "/rca/analyze",
            "metrics": "/rca/metrics",
            "events": "/rca/events",
            "logs": "/rca/logs",
            "health": "/rca/health",
            "quick_diagnosis": "/rca/quick-diagnosis",
            "event_patterns": "/rca/event-patterns",
            "error_summary": "/rca/error-summary",
            "info": "/rca/info"
        },
        "data_sources": [
            "Prometheus指标",
            "Kubernetes事件",
            "Pod日志"
        ],
        "analysis_methods": [
            "统计异常检测",
            "时间序列分析",
            "模式匹配",
            "关联分析",
            "因果推理"
        ],
        "constraints": {
            "max_time_window_hours": 24,
            "min_time_window_hours": 0.1,
            "max_log_lines": 1000,
            "default_log_lines": 100,
            "timeout": ServiceConstants.RCA_TIMEOUT if hasattr(ServiceConstants, 'RCA_TIMEOUT') else 300
        },
        "status": "available" if rca_service else "unavailable"
    }

    return ResponseWrapper.success(
        data=info,
        message="success"
    )


__all__ = ["router"]