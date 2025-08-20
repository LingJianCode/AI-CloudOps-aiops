#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析FastAPI路由 - 提供异常检测、相关性分析和根本原因识别功能
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.decorators import api_response, log_api_call
from app.common.constants import ServiceConstants, AppConstants, ApiEndpoints
from app.common.response import ResponseWrapper
from app.services.rca_service import RCAService
from app.models.rca_models import RCARequest, RCAResponse

logger = logging.getLogger("aiops.api.rca")

router = APIRouter(tags=["rca"])
rca_service = RCAService()


class RCAQueryRequest(BaseModel):
    """新的RCA请求模型 - 支持多数据源"""

    namespace: str = Field(default="default", description="Kubernetes命名空间")
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    metrics: Optional[List[str]] = Field(None, description="监控指标列表")
    service_name: Optional[str] = Field(None, description="服务名称")
    include_events: bool = Field(default=True, description="是否包含K8s事件分析")
    include_logs: bool = Field(default=True, description="是否包含Pod日志分析")
    severity_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="严重性阈值")
    correlation_window: int = Field(
        default=300, ge=60, le=3600, description="关联分析时间窗口（秒）"
    )
    max_candidates: int = Field(default=10, ge=1, le=20, description="最大根因候选数量")


class MetricsQueryRequest(BaseModel):
    metric_name: str = Field(..., description="指标名称")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    service_name: Optional[str] = Field(None, description="服务名称")


@router.post("/rca", summary="根因分析 - 多数据源")
@api_response("根因分析")
@log_api_call(log_request=True)
async def root_cause_analysis(request: RCAQueryRequest) -> Dict[str, Any]:
    """
    根因分析接口 - 支持多数据源的智能分析

    整合Prometheus指标、K8s事件和Pod日志进行关联分析，
    提供准确的根因推理和可执行的修复建议。
    """
    await rca_service.initialize()

    # 转换为内部请求格式
    rca_request = RCARequest(
        namespace=request.namespace,
        start_time=request.start_time,
        end_time=request.end_time,
        metrics=request.metrics or [],
        service_name=request.service_name,
        include_events=request.include_events,
        include_logs=request.include_logs,
        severity_threshold=request.severity_threshold,
        correlation_window=request.correlation_window,
        max_candidates=request.max_candidates,
    )

    # 执行根因分析
    rca_response = await rca_service.analyze_root_cause(rca_request)

    if rca_response.status == "error":
        raise HTTPException(status_code=500, detail=rca_response.error_message or "根因分析失败")

    # 转换响应格式以兼容API标准
    response_data = {
        "request_id": rca_response.result.request_id,
        "analysis": {
            "status": rca_response.status,
            "time_range": {
                "start": rca_response.result.time_range["start"].isoformat(),
                "end": rca_response.result.time_range["end"].isoformat(),
            },
            "namespace": rca_response.result.namespace,
            "data_sources": [ds.value for ds in rca_response.result.data_sources_analyzed],
            "metrics_analyzed": len(rca_response.result.metrics_data),
            "events_analyzed": len(rca_response.result.events_data),
            "logs_analyzed": len(rca_response.result.logs_data),
            "correlations_found": len(rca_response.result.correlations),
            "root_causes": [
                {
                    "identifier": rc.identifier,
                    "data_source": rc.data_source.value,
                    "confidence": rc.confidence,
                    "severity": rc.severity.value,
                    "first_occurrence": rc.first_occurrence.isoformat(),
                    "last_occurrence": rc.last_occurrence.isoformat(),
                    "description": rc.description,
                    "impact_scope": rc.impact_scope,
                    "recommendations": rc.recommendations,
                    "correlations_count": len(rc.correlations),
                }
                for rc in rca_response.result.root_causes
            ],
            "correlations": [
                {
                    "source_type": corr.source_type.value,
                    "target_type": corr.target_type.value,
                    "source_identifier": corr.source_identifier,
                    "target_identifier": corr.target_identifier,
                    "correlation_score": corr.correlation_score,
                    "temporal_offset": corr.temporal_offset,
                    "evidence": corr.evidence,
                }
                for corr in rca_response.result.correlations
            ],
            "summary": rca_response.result.analysis_summary,
            "confidence_score": rca_response.result.confidence_score,
            "processing_time": rca_response.result.processing_time,
        },
        "metadata": rca_response.metadata,
        "warnings": rca_response.warnings,
    }

    return ResponseWrapper.success(data=response_data, message="多数据源根因分析完成")


@router.get("/rca/metrics", summary="获取可用指标")
@api_response("获取可用指标")
async def get_available_metrics(
    service_name: Optional[str] = Query(None, description="服务名称"),
    category: Optional[str] = Query(None, description="指标类别"),
) -> Dict[str, Any]:

    await rca_service.initialize()

    # 获取可用指标
    metrics = await rca_service.get_available_metrics(service_name=service_name, category=category)

    return ResponseWrapper.success_list(items=metrics, message="success")


@router.post("/rca/metrics/query", summary="查询指标数据")
@api_response("查询指标数据")
async def query_metric_data(request: MetricsQueryRequest) -> Dict[str, Any]:

    await rca_service.initialize()

    # 设置默认时间范围（最近1小时）
    end_time = request.end_time or datetime.now()
    start_time = request.start_time or (end_time - timedelta(hours=1))

    # 查询指标数据
    metric_data = await rca_service.query_metric_data(
        metric_name=request.metric_name,
        start_time=start_time,
        end_time=end_time,
        service_name=request.service_name,
    )

    return ResponseWrapper.success_list(items=metric_data, message="success")


@router.get("/rca/config", summary="获取RCA配置")
@api_response("获取RCA配置")
async def get_rca_config() -> Dict[str, Any]:

    await rca_service.initialize()

    config_info = await rca_service.get_rca_config()

    return ResponseWrapper.success(data=config_info, message="success")


@router.get("/rca/health", summary="RCA服务健康检查")
@api_response("RCA服务健康检查")
async def rca_health() -> Dict[str, Any]:

    await rca_service.initialize()

    health_status = await rca_service.get_service_health_info()

    return ResponseWrapper.success(data=health_status, message="success")


@router.get("/rca/ready", summary="RCA服务就绪检查")
@api_response("RCA服务就绪检查")
async def rca_ready() -> Dict[str, Any]:

    await rca_service.initialize()

    is_ready = rca_service.is_initialized() and await rca_service.health_check()

    if not is_ready:
        raise HTTPException(status_code=503, detail="RCA服务未就绪")

    return ResponseWrapper.success(
        data={
            "ready": True,
            "initialized": rca_service.is_initialized(),
            "healthy": await rca_service.health_check(),
            "timestamp": datetime.now().isoformat(),
        },
        message="success",
    )


@router.get("/rca/info", summary="RCA服务信息")
@api_response("RCA服务信息")
async def rca_info() -> Dict[str, Any]:

    info = {
        "service": "根因分析",
        "version": AppConstants.APP_VERSION,
        "description": "基于机器学习和相关性分析的智能根因分析服务",
        "capabilities": ["异常检测", "相关性分析", "根因推理", "AI摘要生成"],
        "endpoints": {
            "analyze": ApiEndpoints.RCA,
            "metrics": ApiEndpoints.RCA_METRICS,
            "config": ApiEndpoints.RCA_CONFIG,
            "health": ApiEndpoints.RCA_HEALTH,
            "ready": ApiEndpoints.RCA_READY,
            "info": ApiEndpoints.RCA_INFO,
        },
        "algorithms": {
            "anomaly_detection": ["Z-Score", "IQR", "Isolation Forest"],
            "correlation": ["Pearson", "Spearman", "Cross-correlation"],
            "clustering": ["DBSCAN", "K-means"],
        },
        "constraints": {
            "min_metrics": ServiceConstants.RCA_MIN_METRICS,
            "max_metrics": ServiceConstants.RCA_MAX_METRICS,
            "default_severity_threshold": ServiceConstants.RCA_DEFAULT_SEVERITY_THRESHOLD,
            "timeout": ServiceConstants.RCA_TIMEOUT,
        },
        "supported_data_sources": ["Prometheus指标", "Kubernetes事件", "Pod日志"],
        "new_features": [
            "多数据源关联分析",
            "智能根因推理",
            "时间序列异常检测",
            "事件严重程度分级",
            "日志错误模式识别",
            "跨数据源关联性分析",
        ],
        "api_version": "/rca - 多数据源智能根因分析",
        "status": "available" if rca_service else "unavailable",
    }

    return ResponseWrapper.success(data=info, message="success")


__all__ = ["router"]
