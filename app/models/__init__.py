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
from .predict_models import (  # 枚举类型; 基础数据模型; 请求模型; 基础响应模型; AI增强响应模型
    AIAnalysisContext,
    AICapabilitiesResponse,
    AIEnhancedPredictionResponse,
    AIPredictionInterpretation,
    AIReport,
    AnomalyPrediction,
    BasePredictionRequest,
    CostAnalysis,
    CpuPredictionRequest,
    DiskPredictionRequest,
    MemoryPredictionRequest,
    ModelInfo,
    MultiDimensionPredictionResponse,
    PredictionDataPoint,
    PredictionGranularity,
    PredictionReportResponse,
    PredictionResponse,
    PredictionServiceHealthResponse,
    PredictionType,
    QpsPredictionRequest,
    ResourceConstraints,
    ResourceUtilization,
    ScalingAction,
    ScalingRecommendation,
)
from .request_models import (
    AddDocumentRequest,
    AssistantRequest,
    AutoFixRequest,
    CreateSessionRequest,
    DiagnoseRequest,
    UploadKnowledgeRequest,
)
from .response_models import (
    AddDocumentResponse,
    AssistantResponse,
    AutoFixResponse,
    ClearCacheResponse,
    CreateSessionResponse,
    DiagnoseResponse,
    ErrorSummaryResponse,
    EventPatternsResponse,
    ListResponse,
    ModelInfoResponse,
    QuickDiagnosisResponse,
    RefreshKnowledgeResponse,
    ServiceConfigResponse,
    ServiceHealthResponse,
    ServiceInfoResponse,
    ServiceReadyResponse,
    SessionInfoResponse,
    UploadKnowledgeResponse,
)

__all__ = [
    # 请求模型
    "AddDocumentRequest",
    "AutoFixRequest",
    "AssistantRequest",
    "CreateSessionRequest",
    "DiagnoseRequest",
    "UploadKnowledgeRequest",
    # 响应模型
    "AddDocumentResponse",
    "AutoFixResponse",
    "AssistantResponse",
    "ClearCacheResponse",
    "CreateSessionResponse",
    "DiagnoseResponse",
    "ErrorSummaryResponse",
    "EventPatternsResponse",
    "ListResponse",
    "ModelInfoResponse",
    "QuickDiagnosisResponse",
    "RefreshKnowledgeResponse",
    "ServiceConfigResponse",
    "ServiceHealthResponse",
    "ServiceInfoResponse",
    "ServiceReadyResponse",
    "SessionInfoResponse",
    "UploadKnowledgeResponse",
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
    # 预测请求模型
    "BasePredictionRequest",
    "QpsPredictionRequest",
    "CpuPredictionRequest",
    "MemoryPredictionRequest",
    "DiskPredictionRequest",
    # 基础预测响应模型
    "PredictionResponse",
    "PredictionServiceHealthResponse",
    # AI增强响应模型
    "AIAnalysisContext",
    "AIPredictionInterpretation",
    "AIReport",
    "AIEnhancedPredictionResponse",
    "MultiDimensionPredictionResponse",
    "PredictionReportResponse",
    "AICapabilitiesResponse",
]
