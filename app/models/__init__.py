#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 模块初始化文件
"""

from .data_models import AgentState, AnomalyResult, CorrelationResult, MetricData
from .request_models import (
    AssistantRequest,
    AutoFixRequest,
    DiagnoseRequest,
    RCARequest,
    SessionRequest,
)
from .response_models import (
    AssistantResponse,
    AutoFixResponse,
    DiagnoseResponse,
    HealthResponse,
    ListResponse,
    RCAResponse,
    SessionInfoResponse,
)
from .predict_models import (
    # 枚举类型
    PredictionType,
    PredictionGranularity,
    ScalingAction,
    # 基础数据模型
    ResourceConstraints,
    PredictionDataPoint,
    ResourceUtilization,
    ScalingRecommendation,
    CostAnalysis,
    AnomalyPrediction,
    ModelInfo,
    # 请求响应模型
    PredictionRequest,
    PredictionResponse,
    PredictionServiceHealthResponse,
)

__all__ = [
    # 请求模型
    "RCARequest",
    "AutoFixRequest",
    "AssistantRequest",
    "SessionRequest",
    "DiagnoseRequest",
    # 响应模型
    "RCAResponse",
    "AutoFixResponse",
    "HealthResponse",
    "AssistantResponse",
    "SessionInfoResponse",
    "DiagnoseResponse",
    "ListResponse",
    # 数据模型
    "MetricData",
    "AnomalyResult",
    "CorrelationResult",
    "AgentState",
    # 预测相关枚举
    "PredictionType",
    "PredictionGranularity",
    "ScalingAction",
    # 预测基础模型
    "ResourceConstraints",
    "PredictionDataPoint",
    "ResourceUtilization",
    "ScalingRecommendation",
    "CostAnalysis",
    "AnomalyPrediction",
    "ModelInfo",
    # 预测请求响应模型
    "PredictionRequest",
    "PredictionResponse",
    "PredictionServiceHealthResponse",
]
