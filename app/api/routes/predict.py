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
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from app.api.decorators import api_response, log_api_call
from app.common.constants import ServiceConstants
from app.common.response import ResponseWrapper
from app.models import (
    PredictionRequest,
    PredictionResponse,
    PredictionServiceHealthResponse,
    PredictionType,
)
from app.services.prediction_service import PredictionService

logger = logging.getLogger("aiops.api.predict")

router = APIRouter(tags=["prediction"])
prediction_service = PredictionService()


@router.post("/predict/qps", summary="QPS负载预测与实例建议")
@api_response("QPS负载预测与实例建议")
@log_api_call(log_request=True)
async def predict_qps(request: PredictionRequest) -> Dict[str, Any]:
    """基于QPS进行负载预测和实例数建议"""
    # 验证预测类型
    if request.prediction_type != PredictionType.QPS:
        raise HTTPException(status_code=400, detail="此端点仅支持QPS预测类型")
    
    await prediction_service.initialize()
    
    try:
        prediction_result = await prediction_service.predict_qps(
            current_qps=request.current_value,
            metric_query=request.metric_query,
            prediction_hours=request.prediction_hours,
            granularity=request.granularity.value,
            resource_constraints=request.resource_constraints.dict() if request.resource_constraints else None,
            include_confidence=request.include_confidence,
            include_anomaly_detection=request.include_anomaly_detection,
            consider_historical_pattern=request.consider_historical_pattern,
            target_utilization=request.target_utilization,
            sensitivity=request.sensitivity,
        )
        
        response = PredictionResponse(**prediction_result)
        return ResponseWrapper.success(data=response.dict(), message="QPS预测完成")
    
    except Exception as e:
        logger.error(f"QPS预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail="QPS预测服务暂时不可用，请稍后重试")


@router.post("/predict/cpu", summary="CPU使用率预测与资源建议")
@api_response("CPU使用率预测与资源建议")
@log_api_call(log_request=True)
async def predict_cpu(request: PredictionRequest) -> Dict[str, Any]:
    """基于CPU使用率进行资源预测和扩容建议"""
    # 验证预测类型
    if request.prediction_type != PredictionType.CPU:
        raise HTTPException(status_code=400, detail="此端点仅支持CPU预测类型")
    
    await prediction_service.initialize()
    
    try:
        prediction_result = await prediction_service.predict_cpu_utilization(
            current_cpu_percent=request.current_value,
            metric_query=request.metric_query,
            prediction_hours=request.prediction_hours,
            granularity=request.granularity.value,
            resource_constraints=request.resource_constraints.dict() if request.resource_constraints else None,
            include_confidence=request.include_confidence,
            include_anomaly_detection=request.include_anomaly_detection,
            consider_historical_pattern=request.consider_historical_pattern,
            target_utilization=request.target_utilization,
            sensitivity=request.sensitivity,
        )
        
        response = PredictionResponse(**prediction_result)
        return ResponseWrapper.success(data=response.dict(), message="CPU预测完成")
    
    except Exception as e:
        logger.error(f"CPU预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail="CPU预测服务暂时不可用，请稍后重试")


@router.post("/predict/memory", summary="内存使用率预测与资源建议")
@api_response("内存使用率预测与资源建议")
@log_api_call(log_request=True)
async def predict_memory(request: PredictionRequest) -> Dict[str, Any]:
    """基于内存使用率进行资源预测和扩容建议"""
    # 验证预测类型
    if request.prediction_type != PredictionType.MEMORY:
        raise HTTPException(status_code=400, detail="此端点仅支持内存预测类型")
    
    await prediction_service.initialize()
    
    try:
        prediction_result = await prediction_service.predict_memory_utilization(
            current_memory_percent=request.current_value,
            metric_query=request.metric_query,
            prediction_hours=request.prediction_hours,
            granularity=request.granularity.value,
            resource_constraints=request.resource_constraints.dict() if request.resource_constraints else None,
            include_confidence=request.include_confidence,
            include_anomaly_detection=request.include_anomaly_detection,
            consider_historical_pattern=request.consider_historical_pattern,
            target_utilization=request.target_utilization,
            sensitivity=request.sensitivity,
        )
        
        response = PredictionResponse(**prediction_result)
        return ResponseWrapper.success(data=response.dict(), message="内存预测完成")
    
    except Exception as e:
        logger.error(f"内存预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail="内存预测服务暂时不可用，请稍后重试")


@router.post("/predict/disk", summary="磁盘使用率预测与存储建议")
@api_response("磁盘使用率预测与存储建议")  
@log_api_call(log_request=True)
async def predict_disk(request: PredictionRequest) -> Dict[str, Any]:
    """基于磁盘使用率进行存储预测和扩容建议"""
    # 验证预测类型
    if request.prediction_type != PredictionType.DISK:
        raise HTTPException(status_code=400, detail="此端点仅支持磁盘预测类型")
    
    await prediction_service.initialize()
    
    try:
        prediction_result = await prediction_service.predict_disk_utilization(
            current_disk_percent=request.current_value,
            metric_query=request.metric_query,
            prediction_hours=request.prediction_hours,
            granularity=request.granularity.value,
            resource_constraints=request.resource_constraints.dict() if request.resource_constraints else None,
            include_confidence=request.include_confidence,
            include_anomaly_detection=request.include_anomaly_detection,
            consider_historical_pattern=request.consider_historical_pattern,
            target_utilization=request.target_utilization,
            sensitivity=request.sensitivity,
        )
        
        response = PredictionResponse(**prediction_result)
        return ResponseWrapper.success(data=response.dict(), message="磁盘预测完成")
    
    except Exception as e:
        logger.error(f"磁盘预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail="磁盘预测服务暂时不可用，请稍后重试")


@router.get("/predict/health", summary="预测服务健康检查")
@api_response("预测服务健康检查")
async def prediction_health() -> Dict[str, Any]:
    """获取预测服务的健康状态信息"""
    try:
        await prediction_service.initialize()
        health_info = await prediction_service.get_service_health_info()
        response = PredictionServiceHealthResponse(**health_info)
        return ResponseWrapper.success(data=response.dict(), message="健康检查完成")
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        # 即使出错也要返回基本的健康信息
        return ResponseWrapper.success(
            data={
                "service_status": "error",
                "model_status": "unknown",
                "error_message": str(e)
            },
            message="健康检查部分失败"
        )


@router.get("/predict/ready", summary="预测服务就绪检查")
@api_response("预测服务就绪检查")
async def prediction_ready() -> Dict[str, Any]:
    """检查预测服务是否已就绪可以提供服务"""
    try:
        await prediction_service.initialize()
        
        is_initialized = prediction_service.is_initialized()
        is_healthy = await prediction_service.health_check()
        is_ready = is_initialized and is_healthy

        if not is_ready:
            raise HTTPException(status_code=503, detail="预测服务未就绪")

        from datetime import datetime
        return ResponseWrapper.success(
            data={
                "ready": True,
                "initialized": is_initialized,
                "healthy": is_healthy,
                "timestamp": datetime.now().isoformat(),
            },
            message="服务就绪",
        )
    except HTTPException:
        raise  # 重新抛出HTTP异常
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail="预测服务就绪检查失败，服务可能不可用"
        )


@router.get("/predict/info", summary="预测服务信息")
@api_response("预测服务信息")
async def prediction_info() -> Dict[str, Any]:
    from app.common.constants import AppConstants

    info = {
        "service": "智能预测服务",
        "version": AppConstants.APP_VERSION,
        "description": "基于机器学习的多维度资源预测和智能扩缩容建议服务",
        "supported_prediction_types": [
            {
                "type": "qps",
                "name": "QPS负载预测",
                "description": "基于QPS预测负载趋势并提供实例数建议",
                "endpoint": "/predict/qps"
            },
            {
                "type": "cpu", 
                "name": "CPU使用率预测",
                "description": "基于CPU使用率预测资源需求并提供扩容建议",
                "endpoint": "/predict/cpu"
            },
            {
                "type": "memory",
                "name": "内存使用率预测", 
                "description": "基于内存使用率预测资源需求并提供扩容建议",
                "endpoint": "/predict/memory"
            },
            {
                "type": "disk",
                "name": "磁盘使用率预测",
                "description": "基于磁盘使用率预测存储需求并提供扩容建议", 
                "endpoint": "/predict/disk"
            }
        ],
        "capabilities": [
            "多类型资源预测",
            "自定义指标支持",
            "智能扩缩容建议",
            "异常预测与检测",
            "成本优化分析", 
            "置信区间计算",
            "历史模式识别"
        ],
        "supported_granularity": ["minute", "hour", "day"],
        "endpoints": {
            "qps_predict": "/predict/qps",
            "cpu_predict": "/predict/cpu", 
            "memory_predict": "/predict/memory",
            "disk_predict": "/predict/disk",
            "health": "/predict/health",
            "ready": "/predict/ready",
            "info": "/predict/info",
            "models": "/predict/models"
        },
        "prediction_features": {
            "algorithms": ["时间序列分析", "机器学习回归", "历史模式识别"],
            "confidence_calculation": "支持置信区间计算",
            "anomaly_detection": "内置异常检测能力",
            "resource_optimization": "智能资源配置建议",
            "cost_analysis": "成本优化分析",
            "custom_metrics": "支持自定义Prometheus指标查询"
        },
        "constraints": {
            "min_hours": ServiceConstants.PREDICTION_MIN_HOURS if hasattr(ServiceConstants, 'PREDICTION_MIN_HOURS') else 1,
            "max_hours": ServiceConstants.PREDICTION_MAX_HOURS if hasattr(ServiceConstants, 'PREDICTION_MAX_HOURS') else 168,
            "qps_range": "1-100000", 
            "utilization_range": "0-100%",
            "timeout_seconds": ServiceConstants.PREDICTION_TIMEOUT if hasattr(ServiceConstants, 'PREDICTION_TIMEOUT') else 30,
        },
        "service_status": "available" if prediction_service else "unavailable",
    }

    return ResponseWrapper.success(data=info, message="服务信息获取成功")


@router.get("/predict/models", summary="模型信息")
@api_response("模型信息")
async def model_info() -> Dict[str, Any]:
    """获取预测服务中加载的模型详细信息"""
    try:
        await prediction_service.initialize()
        model_details = await prediction_service.get_model_info()
        return ResponseWrapper.success(data=model_details, message="模型信息获取成功")
    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        # 返回基本的错误信息而不是抛出异常
        return ResponseWrapper.success(
            data={
                "models": [],
                "status": "error",
                "error_message": "无法获取模型信息"
            },
            message="模型信息获取失败"
        )


__all__ = ["router"]
