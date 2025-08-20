#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析FastAPI路由 - 提供异常检测、相关性分析和根本原因识别功能
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config.settings import config
from app.core.rca.analyzer import RCAAnalyzer
from app.models.request_models import RCARequest
from app.models.response_models import APIResponse
from app.utils.validators import validate_metric_list, validate_time_range

logger = logging.getLogger("aiops.rca")

# 创建路由器
router = APIRouter(tags=["rca"])

# 初始化分析器
rca_analyzer = RCAAnalyzer()

# 请求模型
class RCAQueryRequest(BaseModel):
    metrics: List[str] = Field(..., description="监控指标列表", min_items=1)
    start_time: datetime = Field(..., description="开始时间")
    end_time: datetime = Field(..., description="结束时间")
    service_name: Optional[str] = Field(None, description="服务名称")
    namespace: Optional[str] = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(False, description="是否包含日志分析")
    severity_threshold: float = Field(0.7, description="严重性阈值", ge=0.0, le=1.0)

class MetricsQueryRequest(BaseModel):
    metric_name: str = Field(..., description="指标名称")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    service_name: Optional[str] = Field(None, description="服务名称")

# 响应模型
class RCAResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


@router.post("/rca", response_model=RCAResponse, summary="根因分析")
async def root_cause_analysis(request: RCAQueryRequest) -> RCAResponse:
    """根因分析接口"""
    try:
        logger.info(f"收到RCA请求: {request.dict()}")

        # 验证时间范围
        if not validate_time_range(request.start_time, request.end_time):
            raise HTTPException(status_code=400, detail="无效的时间范围")

        # 验证指标列表
        if not validate_metric_list(request.metrics):
            raise HTTPException(status_code=400, detail="无效的指标列表")

        # 创建RCARequest对象
        rca_request = RCARequest(
            metrics=request.metrics,
            start_time=request.start_time,
            end_time=request.end_time,
            service_name=request.service_name,
            namespace=request.namespace,
            include_logs=request.include_logs,
            severity_threshold=request.severity_threshold
        )

        # 执行根因分析
        try:
            analysis_result = await asyncio.to_thread(
                rca_analyzer.analyze,
                rca_request.metrics,
                rca_request.start_time,
                rca_request.end_time,
                rca_request.service_name,
                rca_request.namespace,
                rca_request.include_logs
            )
        except Exception as e:
            logger.error(f"RCA分析执行失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"根因分析失败: {str(e)}")

        return RCAResponse(
            code=0,
            message="根因分析完成",
            data={
                "analysis": analysis_result,
                "request_info": {
                    "metrics": rca_request.metrics,
                    "time_range": {
                        "start": rca_request.start_time.isoformat(),
                        "end": rca_request.end_time.isoformat()
                    },
                    "service_name": rca_request.service_name,
                    "namespace": rca_request.namespace
                },
                "timestamp": datetime.now().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RCA请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RCA请求处理失败: {str(e)}")


@router.get("/rca/metrics", response_model=RCAResponse, summary="获取可用指标")
async def get_available_metrics(
    service_name: Optional[str] = Query(None, description="服务名称"),
    category: Optional[str] = Query(None, description="指标类别")
) -> RCAResponse:
    """获取可用的监控指标列表"""
    try:
        # 获取可用指标
        try:
            metrics = await asyncio.to_thread(
                rca_analyzer.get_available_metrics,
                service_name=service_name,
                category=category
            )
        except Exception as e:
            logger.error(f"获取指标列表失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取指标列表失败: {str(e)}")

        return RCAResponse(
            code=0,
            message="获取指标列表成功",
            data={
                "metrics": metrics,
                "service_name": service_name,
                "category": category,
                "count": len(metrics) if metrics else 0,
                "timestamp": datetime.now().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取指标列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标列表失败: {str(e)}")


@router.post("/rca/metrics/query", response_model=RCAResponse, summary="查询指标数据")
async def query_metric_data(request: MetricsQueryRequest) -> RCAResponse:
    """查询指定指标的数据"""
    try:
        # 设置默认时间范围（最近1小时）
        end_time = request.end_time or datetime.now()
        start_time = request.start_time or (end_time - timedelta(hours=1))

        # 验证时间范围
        if not validate_time_range(start_time, end_time):
            raise HTTPException(status_code=400, detail="无效的时间范围")

        # 查询指标数据
        try:
            metric_data = await asyncio.to_thread(
                rca_analyzer.query_metric_data,
                request.metric_name,
                start_time,
                end_time,
                request.service_name
            )
        except Exception as e:
            logger.error(f"查询指标数据失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"查询指标数据失败: {str(e)}")

        return RCAResponse(
            code=0,
            message="查询指标数据成功",
            data={
                "metric_name": request.metric_name,
                "data": metric_data,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "service_name": request.service_name,
                "data_points": len(metric_data) if metric_data else 0,
                "timestamp": datetime.now().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询指标数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询指标数据失败: {str(e)}")


@router.get("/rca/config", response_model=RCAResponse, summary="获取RCA配置")
async def get_rca_config() -> RCAResponse:
    """获取RCA分析配置信息"""
    try:
        config_info = {
            "anomaly_detection": {
                "algorithm": "statistical",
                "threshold_factor": getattr(config, 'rca_threshold_factor', 2.0),
                "window_size": getattr(config, 'rca_window_size', 60)
            },
            "correlation_analysis": {
                "method": "pearson",
                "min_correlation": getattr(config, 'min_correlation', 0.7),
                "max_lag": getattr(config, 'max_correlation_lag', 300)
            },
            "supported_metrics": [
                "cpu_usage",
                "memory_usage", 
                "network_io",
                "disk_io",
                "request_rate",
                "error_rate",
                "response_time"
            ],
            "max_analysis_duration": getattr(config, 'max_rca_duration', 3600),
            "prometheus_config": {
                "endpoint": getattr(config.prometheus, 'url', 'unknown'),
                "timeout": getattr(config.prometheus, 'timeout', 30)
            }
        }

        return RCAResponse(
            code=0,
            message="获取配置成功",
            data=config_info
        )

    except Exception as e:
        logger.error(f"获取RCA配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取RCA配置失败: {str(e)}")


@router.get("/rca/health", response_model=RCAResponse, summary="RCA服务健康检查")
async def rca_health() -> RCAResponse:
    """RCA服务健康检查"""
    try:
        health_status = {
            "service": "rca",
            "status": "healthy" if rca_analyzer else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "anomaly_detector": "unknown",
                "correlator": "unknown", 
                "prometheus": "unknown",
                "llm_service": "unknown"
            }
        }

        # 检查各组件状态
        if rca_analyzer:
            # 检查异常检测器
            try:
                if hasattr(rca_analyzer, 'detector') and rca_analyzer.detector:
                    health_status["components"]["anomaly_detector"] = "healthy"
            except Exception:
                health_status["components"]["anomaly_detector"] = "unhealthy"

            # 检查相关性分析器
            try:
                if hasattr(rca_analyzer, 'correlator') and rca_analyzer.correlator:
                    health_status["components"]["correlator"] = "healthy"
            except Exception:
                health_status["components"]["correlator"] = "unhealthy"

            # 检查Prometheus连接
            try:
                if hasattr(rca_analyzer, 'prometheus') and rca_analyzer.prometheus:
                    prometheus_health = await asyncio.to_thread(rca_analyzer.prometheus.health_check)
                    health_status["components"]["prometheus"] = "healthy" if prometheus_health else "unhealthy"
            except Exception:
                health_status["components"]["prometheus"] = "unhealthy"

            # 检查LLM服务
            try:
                if hasattr(rca_analyzer, 'llm') and rca_analyzer.llm:
                    llm_health = rca_analyzer.llm.health_check()
                    health_status["components"]["llm_service"] = "healthy" if llm_health else "unhealthy"
            except Exception:
                health_status["components"]["llm_service"] = "unhealthy"

        return RCAResponse(
            code=0,
            message="健康检查完成",
            data=health_status
        )

    except Exception as e:
        logger.error(f"RCA健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RCA健康检查失败: {str(e)}")


@router.get("/rca/ready", response_model=RCAResponse, summary="RCA服务就绪检查")
async def rca_ready() -> RCAResponse:
    """RCA服务就绪检查"""
    try:
        is_ready = (
            rca_analyzer and
            hasattr(rca_analyzer, 'detector') and rca_analyzer.detector and
            hasattr(rca_analyzer, 'correlator') and rca_analyzer.correlator and
            hasattr(rca_analyzer, 'prometheus') and rca_analyzer.prometheus
        )

        if not is_ready:
            raise HTTPException(status_code=503, detail="RCA服务未就绪")

        return RCAResponse(
            code=0,
            message="RCA服务已就绪",
            data={
                "ready": True,
                "timestamp": datetime.now().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RCA就绪检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RCA就绪检查失败: {str(e)}")


@router.get("/rca/info", response_model=RCAResponse, summary="RCA服务信息")
async def rca_info() -> RCAResponse:
    """获取RCA服务信息"""
    try:
        info = {
            "service": "根因分析",
            "version": "1.0.0",
            "description": "基于机器学习和相关性分析的智能根因分析服务",
            "capabilities": [
                "异常检测",
                "相关性分析", 
                "根因推理",
                "AI摘要生成"
            ],
            "endpoints": {
                "analyze": "/api/v1/rca",
                "metrics": "/api/v1/rca/metrics",
                "config": "/api/v1/rca/config",
                "health": "/api/v1/rca/health",
                "ready": "/api/v1/rca/ready",
                "info": "/api/v1/rca/info"
            },
            "algorithms": {
                "anomaly_detection": ["Z-Score", "IQR", "Isolation Forest"],
                "correlation": ["Pearson", "Spearman", "Cross-correlation"],
                "clustering": ["DBSCAN", "K-means"]
            },
            "supported_data_sources": [
                "Prometheus",
                "Kubernetes Metrics",
                "Application Logs"
            ],
            "status": "available" if rca_analyzer else "unavailable",
            "timestamp": datetime.now().isoformat()
        }

        return RCAResponse(
            code=0,
            message="获取信息成功",
            data=info
        )

    except Exception as e:
        logger.error(f"获取RCA服务信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取RCA服务信息失败: {str(e)}")


# 导出
__all__ = ["router"]