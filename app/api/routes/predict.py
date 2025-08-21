#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 预测服务API接口
"""

import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query
from app.api.decorators import api_response, log_api_call
from app.common.constants import ServiceConstants
from app.common.response import ResponseWrapper
from app.models import (
    PredictionRequest,
    PredictionResponse,
    PredictTrendRequest,
    PredictTrendResponse,
)
from app.services.prediction_service import PredictionService

logger = logging.getLogger("aiops.api.predict")

router = APIRouter(tags=["prediction"])
prediction_service = PredictionService()


@router.post("/predict", summary="QPS预测")
@api_response("QPS预测")
@log_api_call(log_request=True)
async def predict_instances(request: PredictionRequest) -> Dict[str, Any]:
    await prediction_service.initialize()

    prediction_result = await prediction_service.predict_instances(
        service_name=request.service_name,
        current_qps=request.current_qps,
        hours=request.hours,
        instance_cpu=request.instance_cpu,
        instance_memory=request.instance_memory,
    )

    response = PredictionResponse(
        service_name=request.service_name,
        prediction_hours=request.hours,
        **prediction_result,
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/predict/trend", summary="负载趋势分析")
@api_response("负载趋势分析")
async def predict_trend(
    service_name: Optional[str] = Query(None, description="服务名称"),
    hours: int = Query(ServiceConstants.PREDICTION_MIN_HOURS, description="预测小时数"),
) -> Dict[str, Any]:
    await prediction_service.initialize()

    # 创建请求对象
    request = PredictTrendRequest(service_name=service_name, hours=hours)

    trend_result = await prediction_service.predict_trend(
        service_name=request.service_name, hours=request.hours
    )

    # 使用统一的响应模型
    from datetime import datetime

    response = PredictTrendResponse(
        service_name=request.service_name,
        prediction_hours=request.hours,
        trend_data=trend_result.get("trend_data", []),
        insights=trend_result.get("insights", []),
        timestamp=datetime.now().isoformat(),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/predict/health", summary="预测服务健康检查")
@api_response("预测服务健康检查")
async def prediction_health() -> Dict[str, Any]:
    await prediction_service.initialize()

    health_status = await prediction_service.get_service_health_info()

    return ResponseWrapper.success(data=health_status, message="success")


@router.get("/predict/ready", summary="预测服务就绪检查")
@api_response("预测服务就绪检查")
async def prediction_ready() -> Dict[str, Any]:
    await prediction_service.initialize()

    is_ready = (
        prediction_service.is_initialized() and await prediction_service.health_check()
    )

    if not is_ready:
        raise HTTPException(status_code=503, detail="预测服务未就绪")

    return ResponseWrapper.success(
        data={
            "ready": True,
            "initialized": prediction_service.is_initialized(),
            "healthy": await prediction_service.health_check(),
            "timestamp": prediction_service.get_service_info(),
        },
        message="success",
    )


@router.get("/predict/info", summary="预测服务信息")
@api_response("预测服务信息")
async def prediction_info() -> Dict[str, Any]:
    from app.common.constants import ApiEndpoints, AppConstants

    info = {
        "service": "负载预测",
        "version": AppConstants.APP_VERSION,
        "description": "基于机器学习的QPS预测和实例数建议服务",
        "capabilities": ["QPS预测", "实例数建议", "负载趋势分析", "资源优化建议"],
        "endpoints": {
            "predict": ApiEndpoints.PREDICT,
            "trend": ApiEndpoints.PREDICT_TREND,
            "health": ApiEndpoints.PREDICT_HEALTH,
            "ready": ApiEndpoints.PREDICT_READY,
            "info": ApiEndpoints.PREDICT_INFO,
            "models": ApiEndpoints.PREDICT_MODELS,
        },
        "model_info": {
            "type": "机器学习模型",
            "features": ["时间模式", "历史QPS", "系统指标"],
            "prediction_range": f"{ServiceConstants.PREDICTION_MIN_HOURS}-{ServiceConstants.PREDICTION_MAX_HOURS}小时",
        },
        "constraints": {
            "min_qps": ServiceConstants.PREDICTION_MIN_QPS,
            "max_qps": ServiceConstants.PREDICTION_MAX_QPS,
            "min_hours": ServiceConstants.PREDICTION_MIN_HOURS,
            "max_hours": ServiceConstants.PREDICTION_MAX_HOURS,
            "timeout": ServiceConstants.PREDICTION_TIMEOUT,
        },
        "status": "available" if prediction_service else "unavailable",
    }

    return ResponseWrapper.success(data=info, message="success")


@router.get("/predict/models", summary="模型信息")
@api_response("模型信息")
async def model_info() -> Dict[str, Any]:
    await prediction_service.initialize()

    model_details = await prediction_service.get_model_info()

    return ResponseWrapper.success(data=model_details, message="success")


__all__ = ["router"]
