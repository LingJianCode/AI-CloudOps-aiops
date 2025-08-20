#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 负载预测FastAPI路由 - 提供QPS预测、实例数建议和负载趋势分析接口
"""

import asyncio
import datetime
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.prediction.predictor import PredictionService
from app.models.request_models import PredictionRequest
from app.models.response_models import APIResponse, PredictionResponse
from app.utils.validators import validate_qps

logger = logging.getLogger("aiops.predict")

# 创建路由器
router = APIRouter(tags=["prediction"])

# 初始化预测服务
prediction_service = PredictionService()

# 请求模型
class PredictQueryParams(BaseModel):
    service_name: Optional[str] = Field(None, description="服务名称")
    current_qps: Optional[float] = Field(None, description="当前QPS", gt=0)
    hours: Optional[int] = Field(24, description="预测小时数", ge=1, le=168)
    instance_cpu: Optional[int] = Field(None, description="实例CPU数", gt=0)
    instance_memory: Optional[int] = Field(None, description="实例内存(GB)", gt=0)

# 响应模型
class PredictResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


@router.get("/predict", response_model=PredictResponse, summary="QPS预测 (GET)")
@router.post("/predict", response_model=PredictResponse, summary="QPS预测 (POST)")
async def predict_instances(
    # GET参数
    service_name: Optional[str] = Query(None, description="服务名称"),
    current_qps: Optional[float] = Query(None, description="当前QPS", gt=0),
    hours: Optional[int] = Query(24, description="预测小时数", ge=1, le=168),
    instance_cpu: Optional[int] = Query(None, description="实例CPU数", gt=0),
    instance_memory: Optional[int] = Query(None, description="实例内存(GB)", gt=0),
    # POST请求体
    predict_request: Optional[PredictionRequest] = None
) -> PredictResponse:
    """预测实例数接口"""
    try:
        # 如果是POST请求，使用请求体参数
        if predict_request:
            request_data = predict_request
        else:
            # GET请求使用查询参数或默认值
            request_data = PredictionRequest(
                service_name=service_name,
                current_qps=current_qps,
                hours=hours,
                instance_cpu=instance_cpu,
                instance_memory=instance_memory
            )

        logger.info(f"收到预测请求: {request_data.dict()}")

        # 验证QPS
        if request_data.current_qps:
            if not validate_qps(request_data.current_qps):
                raise HTTPException(status_code=400, detail="QPS值无效")

        # 调用预测服务
        prediction_result = await asyncio.to_thread(
            prediction_service.predict_instances,
            service_name=request_data.service_name,
            current_qps=request_data.current_qps,
            hours=request_data.hours,
            instance_cpu=request_data.instance_cpu,
            instance_memory=request_data.instance_memory,
        )

        # 包装响应
        response = PredictionResponse(
            service_name=request_data.service_name or "unknown",
            prediction_hours=request_data.hours,
            **prediction_result
        )

        return PredictResponse(
            code=0,
            message="预测完成",
            data=response.dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/predict/trend", response_model=PredictResponse, summary="负载趋势分析")
async def predict_trend(
    service_name: Optional[str] = Query(None, description="服务名称"),
    hours: int = Query(24, description="预测小时数", ge=1, le=168)
) -> PredictResponse:
    """负载趋势分析接口"""
    try:
        logger.info(f"收到趋势分析请求: service={service_name}, hours={hours}")

        # 调用趋势分析服务
        trend_result = await asyncio.to_thread(
            prediction_service.analyze_trend,
            service_name=service_name,
            hours=hours
        )

        return PredictResponse(
            code=0,
            message="趋势分析完成",
            data={
                "service_name": service_name,
                "analysis_hours": hours,
                "trend": trend_result,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"趋势分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"趋势分析失败: {str(e)}")


@router.get("/predict/health", response_model=PredictResponse, summary="预测服务健康检查")
async def prediction_health() -> PredictResponse:
    """预测服务健康检查"""
    try:
        # 检查预测服务状态
        health_status = {
            "service": "prediction",
            "status": "healthy" if prediction_service else "unhealthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "model_loaded": False,
            "scaler_loaded": False
        }

        # 检查模型状态
        if prediction_service:
            health_status["model_loaded"] = getattr(prediction_service, 'model_loaded', False)
            health_status["scaler_loaded"] = getattr(prediction_service, 'scaler_loaded', False)

        # 执行服务健康检查
        try:
            service_health = await asyncio.to_thread(prediction_service.health_check)
            health_status.update(service_health)
        except Exception as e:
            health_status["health_check_error"] = str(e)
            health_status["status"] = "unhealthy"

        return PredictResponse(
            code=0,
            message="健康检查完成",
            data=health_status
        )

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/predict/ready", response_model=PredictResponse, summary="预测服务就绪检查")
async def prediction_ready() -> PredictResponse:
    """预测服务就绪检查"""
    try:
        is_ready = (
            prediction_service and 
            getattr(prediction_service, 'model_loaded', False) and
            getattr(prediction_service, 'scaler_loaded', False)
        )

        if not is_ready:
            raise HTTPException(status_code=503, detail="预测服务未就绪")

        return PredictResponse(
            code=0,
            message="预测服务已就绪",
            data={
                "ready": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"就绪检查失败: {str(e)}")


@router.get("/predict/info", response_model=PredictResponse, summary="预测服务信息")
async def prediction_info() -> PredictResponse:
    """获取预测服务信息"""
    try:
        info = {
            "service": "负载预测",
            "version": "1.0.0",
            "description": "基于机器学习的QPS预测和实例数建议服务",
            "capabilities": [
                "QPS预测",
                "实例数建议",
                "负载趋势分析",
                "资源优化建议"
            ],
            "endpoints": {
                "predict": "/api/v1/predict",
                "trend": "/api/v1/predict/trend", 
                "health": "/api/v1/predict/health",
                "ready": "/api/v1/predict/ready",
                "info": "/api/v1/predict/info"
            },
            "model_info": {
                "type": "机器学习模型",
                "features": ["时间模式", "历史QPS", "系统指标"],
                "prediction_range": "1-168小时"
            },
            "status": "available" if prediction_service else "unavailable",
            "timestamp": datetime.datetime.now().isoformat()
        }

        return PredictResponse(
            code=0,
            message="获取信息成功",
            data=info
        )

    except Exception as e:
        logger.error(f"获取预测服务信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取预测服务信息失败: {str(e)}")


@router.get("/predict/models", response_model=PredictResponse, summary="模型信息")
async def model_info() -> PredictResponse:
    """获取预测模型详细信息"""
    try:
        model_details = {
            "models": [],
            "current_model": None,
            "last_trained": None,
            "performance_metrics": {}
        }

        # 获取模型信息
        if prediction_service and hasattr(prediction_service, 'model_loader'):
            try:
                model_info = await asyncio.to_thread(
                    getattr(prediction_service.model_loader, 'get_model_info', lambda: {})
                )
                model_details.update(model_info)
            except Exception as e:
                model_details["error"] = f"获取模型信息失败: {str(e)}"

        return PredictResponse(
            code=0,
            message="获取模型信息成功",
            data=model_details
        )

    except Exception as e:
        logger.error(f"获取模型信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


# 导出
__all__ = ["router"]