#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI-CloudOps智能预测服务API接口
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.decorators import api_response, log_api_call
from app.common.constants import ErrorMessages, HttpStatusCodes
from app.common.response import ResponseWrapper
from app.models import (
    CpuPredictionRequest,
    DiskPredictionRequest,
    MemoryPredictionRequest,
    ModelInfoResponse,
    PredictionResponse,
    PredictionServiceHealthResponse,
    QpsPredictionRequest,
    ServiceInfoResponse,
    ServiceReadyResponse,
)
from app.services.prediction_service import PredictionService

logger = logging.getLogger("aiops.api.predict")

router = APIRouter(tags=["prediction"])
prediction_service = PredictionService()


@router.post("/qps", summary="AI-CloudOps QPS负载预测与实例建议")
@api_response("AI-CloudOps QPS负载预测与实例建议")
@log_api_call(log_request=True)
async def predict_qps(request: QpsPredictionRequest) -> Dict[str, Any]:
    """QPS负载预测"""
    await prediction_service.initialize()

    try:

        if request.enable_ai_insights:
            prediction_result = await prediction_service.predict_with_ai_analysis(
                prediction_type="qps",
                current_value=request.current_qps,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                enable_ai_insights=request.enable_ai_insights,
                report_style=request.ai_report_style,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )
        else:
            prediction_result = await prediction_service.predict_qps(
                current_qps=request.current_qps,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                include_confidence=request.include_confidence,
                include_anomaly_detection=request.include_anomaly_detection,
                consider_historical_pattern=request.consider_historical_pattern,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )

        response = PredictionResponse(**prediction_result)
        ai_status = "AI增强" if request.enable_ai_insights else "基础"
        return ResponseWrapper.success(
            data=response.dict(), message=f"QPS预测完成 ({ai_status}模式)"
        )

    except Exception as e:
        logger.error(f"QPS预测失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PREDICTION_SERVICE_ERROR,
        )


@router.post("/cpu", summary="AI-CloudOps CPU使用率预测与资源建议")
@api_response("AI-CloudOps CPU使用率预测与资源建议")
@log_api_call(log_request=True)
async def predict_cpu(request: CpuPredictionRequest) -> Dict[str, Any]:
    """CPU使用率预测"""
    await prediction_service.initialize()

    try:

        if request.enable_ai_insights:
            prediction_result = await prediction_service.predict_with_ai_analysis(
                prediction_type="cpu",
                current_value=request.current_cpu_percent,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                enable_ai_insights=request.enable_ai_insights,
                report_style=request.ai_report_style,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )
        else:
            prediction_result = await prediction_service.predict_cpu_utilization(
                current_cpu_percent=request.current_cpu_percent,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                include_confidence=request.include_confidence,
                include_anomaly_detection=request.include_anomaly_detection,
                consider_historical_pattern=request.consider_historical_pattern,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )

        response = PredictionResponse(**prediction_result)
        ai_status = "AI增强" if request.enable_ai_insights else "基础"
        return ResponseWrapper.success(
            data=response.dict(), message=f"CPU预测完成 ({ai_status}模式)"
        )

    except Exception as e:
        logger.error(f"CPU预测失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PREDICTION_SERVICE_ERROR,
        )


@router.post("/memory", summary="AI-CloudOps 内存使用率预测与资源建议")
@api_response("AI-CloudOps 内存使用率预测与资源建议")
@log_api_call(log_request=True)
async def predict_memory(request: MemoryPredictionRequest) -> Dict[str, Any]:
    """内存使用率预测"""
    await prediction_service.initialize()

    try:

        if request.enable_ai_insights:
            prediction_result = await prediction_service.predict_with_ai_analysis(
                prediction_type="memory",
                current_value=request.current_memory_percent,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                enable_ai_insights=request.enable_ai_insights,
                report_style=request.ai_report_style,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )
        else:
            prediction_result = await prediction_service.predict_memory_utilization(
                current_memory_percent=request.current_memory_percent,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                include_confidence=request.include_confidence,
                include_anomaly_detection=request.include_anomaly_detection,
                consider_historical_pattern=request.consider_historical_pattern,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )

        response = PredictionResponse(**prediction_result)
        ai_status = "AI增强" if request.enable_ai_insights else "基础"
        return ResponseWrapper.success(
            data=response.dict(), message=f"内存预测完成 ({ai_status}模式)"
        )

    except Exception as e:
        logger.error(f"内存预测失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PREDICTION_SERVICE_ERROR,
        )


@router.post("/disk", summary="AI-CloudOps 磁盘使用率预测与存储建议")
@api_response("AI-CloudOps 磁盘使用率预测与存储建议")
@log_api_call(log_request=True)
async def predict_disk(request: DiskPredictionRequest) -> Dict[str, Any]:
    """磁盘使用率预测"""
    await prediction_service.initialize()

    try:

        if request.enable_ai_insights:
            prediction_result = await prediction_service.predict_with_ai_analysis(
                prediction_type="disk",
                current_value=request.current_disk_percent,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                enable_ai_insights=request.enable_ai_insights,
                report_style=request.ai_report_style,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )
        else:
            prediction_result = await prediction_service.predict_disk_utilization(
                current_disk_percent=request.current_disk_percent,
                metric_query=request.metric_query,
                prediction_hours=request.prediction_hours,
                granularity=request.granularity.value,
                resource_constraints=(
                    request.resource_constraints.dict()
                    if request.resource_constraints
                    else None
                ),
                include_confidence=request.include_confidence,
                include_anomaly_detection=request.include_anomaly_detection,
                consider_historical_pattern=request.consider_historical_pattern,
                target_utilization=request.target_utilization,
                sensitivity=request.sensitivity,
            )

        response = PredictionResponse(**prediction_result)
        ai_status = "AI增强" if request.enable_ai_insights else "基础"
        return ResponseWrapper.success(
            data=response.dict(), message=f"磁盘预测完成 ({ai_status}模式)"
        )

    except Exception as e:
        logger.error(f"磁盘预测失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            detail=ErrorMessages.PREDICTION_SERVICE_ERROR,
        )


@router.get("/health", summary="AI-CloudOps 预测服务健康检查")
@api_response("AI-CloudOps 预测服务健康检查")
async def prediction_health() -> Dict[str, Any]:
    """获取预测服务的健康状态信息"""
    try:
        await prediction_service.initialize()
        health_info = await prediction_service.get_service_health_info()
        response = PredictionServiceHealthResponse(**health_info)
        return ResponseWrapper.success(data=response.dict(), message="健康检查完成")
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")

        return ResponseWrapper.success(
            data={
                "service_status": "error",
                "model_status": "unknown",
                "error_message": str(e),
            },
            message="健康检查部分失败",
        )


@router.get("/ready", summary="AI-CloudOps 预测服务就绪检查")
@api_response("AI-CloudOps 预测服务就绪检查")
async def prediction_ready() -> Dict[str, Any]:
    """检查服务就绪状态"""
    try:
        await prediction_service.initialize()

        is_initialized = prediction_service.is_initialized()
        is_healthy = await prediction_service.health_check()
        is_ready = is_initialized and is_healthy

        if not is_ready:
            raise HTTPException(
                status_code=HttpStatusCodes.SERVICE_UNAVAILABLE,
                detail=ErrorMessages.SERVICE_UNAVAILABLE,
            )

        response = ServiceReadyResponse(
            ready=True,
            service="prediction",
            timestamp=datetime.now().isoformat(),
            message="服务就绪",
        )
        return ResponseWrapper.success(
            data=response.dict(),
            message="服务就绪",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.SERVICE_UNAVAILABLE,
            detail=ErrorMessages.SERVICE_UNAVAILABLE,
        )


@router.get("/info", summary="AI-CloudOps 预测服务信息")
@api_response("AI-CloudOps 预测服务信息")
async def prediction_info() -> Dict[str, Any]:
    from app.common.constants import AppConstants
    from app.config.settings import config

    info = {
        "service": "智能预测服务",
        "version": AppConstants.APP_VERSION,
        "description": "基于机器学习的多维度资源预测和智能扩缩容建议服务",
        "supported_prediction_types": [
            {
                "type": "qps",
                "name": "QPS负载预测",
                "description": "基于QPS预测负载趋势并提供实例数建议",
                "endpoint": "/qps",
            },
            {
                "type": "cpu",
                "name": "CPU使用率预测",
                "description": "基于CPU使用率预测资源需求并提供扩容建议",
                "endpoint": "/cpu",
            },
            {
                "type": "memory",
                "name": "内存使用率预测",
                "description": "基于内存使用率预测资源需求并提供扩容建议",
                "endpoint": "/memory",
            },
            {
                "type": "disk",
                "name": "磁盘使用率预测",
                "description": "基于磁盘使用率预测存储需求并提供扩容建议",
                "endpoint": "/disk",
            },
        ],
        "capabilities": [
            "多类型资源预测",
            "自定义指标支持",
            "智能扩缩容建议",
            "异常预测与检测",
            "成本优化分析",
            "置信区间计算",
            "历史模式识别",
            "AI增强预测分析",
            "智能洞察生成",
            "专业报告生成",
        ],
        "supported_granularity": ["minute", "hour", "day"],
        "endpoints": {
            "qps_predict": "/qps",
            "cpu_predict": "/cpu",
            "memory_predict": "/memory",
            "disk_predict": "/disk",
            "health": "/predict/health",
            "ready": "/predict/ready",
            "info": "/predict/info",
            "models": "/predict/models",
        },
        "prediction_features": {
            "algorithms": ["时间序列分析", "机器学习回归", "历史模式识别"],
            "confidence_calculation": "支持置信区间计算",
            "anomaly_detection": "内置异常检测能力",
            "resource_optimization": "智能资源配置建议",
            "cost_analysis": "成本优化分析",
            "custom_metrics": "支持自定义Prometheus指标查询",
            "ai_enhancement": {
                "description": "可通过enable_ai_insights字段启用AI增强功能",
                "features": [
                    "历史数据上下文分析",
                    "预测结果智能解读",
                    "AI洞察生成",
                    "专业报告生成",
                ],
                "report_styles": ["professional", "executive", "technical", "concise"],
                "fallback_support": "AI功能不可用时自动降级到基础预测",
            },
        },
        "constraints": {
            "min_hours": config.prediction.min_prediction_hours,
            "max_hours": config.prediction.max_prediction_hours,
            "qps_range": "0.1-10000.0",
            "utilization_range": "0-100%",
            "timeout_seconds": 120,
            "default_prediction_hours": config.prediction.default_prediction_hours,
            "default_granularity": config.prediction.default_granularity,
            "default_target_utilization": config.prediction.default_target_utilization,
            "default_sensitivity": config.prediction.default_sensitivity,
        },
        "service_status": "available" if prediction_service else "unavailable",
    }

    response = ServiceInfoResponse(
        service=info["service"],
        version=info["version"],
        description=info["description"],
        capabilities=info["capabilities"],
        endpoints=info["endpoints"],
        constraints=info["constraints"],
        status=info["service_status"],
    )

    return ResponseWrapper.success(data=response.dict(), message="服务信息获取成功")


@router.get("/models", summary="AI-CloudOps 模型信息")
@api_response("AI-CloudOps 模型信息")
async def model_info() -> Dict[str, Any]:
    """获取预测服务中加载的模型详细信息"""
    try:
        await prediction_service.initialize()
        model_details = await prediction_service.get_model_info()

        response = ModelInfoResponse(
            models=model_details.get("models", []),
            total_models=model_details.get("total_models", 0),
            loaded_models=model_details.get("loaded_models", 0),
            status=model_details.get("status", "healthy"),
            timestamp=datetime.now().isoformat(),
        )
        return ResponseWrapper.success(data=response.dict(), message="模型信息获取成功")
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")

        error_response = ModelInfoResponse(
            models=[],
            total_models=0,
            loaded_models=0,
            status="error",
            timestamp=datetime.now().isoformat(),
        )
        return ResponseWrapper.success(
            data=error_response.dict(),
            message="模型信息获取失败",
        )


__all__ = ["router"]
