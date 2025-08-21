#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 响应模型定义
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """
    统一API响应格式 - 所有API响应的基础模型

    Attributes:
        code: 响应状态码，0表示成功
        message: 响应消息
        data: 响应数据，类型由泛型参数T决定
    """

    code: int = 0
    message: str = ""
    data: Optional[T] = None


class AnomalyInfo(BaseModel):
    """
    异常信息模型 - 描述检测到的单个指标异常

    Attributes:
        count: 异常点数量
        first_occurrence: 首次出现时间
        last_occurrence: 最后出现时间
        max_score: 最大异常分数
        avg_score: 平均异常分数
        detection_methods: 使用的检测方法及其参数
    """

    count: int
    first_occurrence: str
    last_occurrence: str
    max_score: float
    avg_score: float
    detection_methods: Dict[str, Any]


class RootCauseCandidate(BaseModel):
    """
    根因候选模型 - 描述可能的故障根因

    Attributes:
        metric: 根因指标名称
        confidence: 置信度分数
        first_occurrence: 首次出现时间
        anomaly_count: 异常点数量
        related_metrics: 相关联的指标列表
        description: 根因描述（可选）
    """

    metric: str
    confidence: float
    first_occurrence: str
    anomaly_count: int
    related_metrics: List[tuple]
    description: Optional[str] = None


class RCAResponse(BaseModel):
    """
    根因分析响应模型 - 包含完整的分析结果

    Attributes:
        status: 分析状态
        anomalies: 检测到的异常信息
        correlations: 指标间的相关性
        root_cause_candidates: 根因候选列表
        analysis_time: 分析完成时间
        time_range: 分析的时间范围
        metrics_analyzed: 分析的指标列表
        summary: 分析摘要（可选）
    """

    status: str
    anomalies: Dict[str, AnomalyInfo]
    correlations: Dict[str, List[tuple]]
    root_cause_candidates: List[RootCauseCandidate]
    analysis_time: str
    time_range: Dict[str, str]
    metrics_analyzed: List[str]
    summary: Optional[str] = None


class PredictionResponse(BaseModel):
    """
    负载预测响应模型 - 包含预测结果和相关信息

    Attributes:
        service_name: 服务名称
        prediction_hours: 预测小时数
        instances: 预测的实例数量
        current_qps: 当前QPS
        timestamp: 预测时间
        confidence: 预测置信度（可选）
        model_version: 使用的模型版本（可选）
        prediction_type: 预测类型（可选）
        features: 预测使用的特征（可选）
    """

    service_name: str
    prediction_hours: int
    instances: int
    current_qps: Optional[float] = None
    timestamp: str
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    prediction_type: Optional[str] = None
    features: Optional[Dict[str, float]] = None


class AutoFixResponse(BaseModel):
    """
    自动修复响应模型 - 包含修复操作的结果信息

    Attributes:
        status: 修复状态
        result: 修复结果描述
        deployment: 部署名称
        namespace: 命名空间
        event: 问题事件描述
        actions_taken: 执行的修复操作列表
        timestamp: 修复完成时间
        execution_time: 执行耗时（秒）
        success: 修复是否成功
        error_message: 错误信息（如果有）
    """

    status: str = "completed"
    result: str = ""
    deployment: str
    namespace: str
    event: str
    actions_taken: List[str] = []
    timestamp: str
    execution_time: float
    success: bool = True
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """
    健康检查响应模型 - 包含系统各组件的健康状态

    Attributes:
        status: 整体状态
        components: 各组件的健康状态
        timestamp: 检查时间
        version: 系统版本（可选）
        uptime: 系统运行时间（可选）
    """

    status: str
    components: Dict[str, bool]
    timestamp: str
    version: Optional[str] = None
    uptime: Optional[float] = None


class AssistantResponse(BaseModel):
    """
    智能小助手响应模型 - 包含助手的回答和相关信息

    Attributes:
        answer: 助手回答内容
        source_documents: 用于生成回答的参考文档（可选）
        relevance_score: 回答相关性分数（可选）
        recall_rate: 文档召回率（可选）
        follow_up_questions: 建议的后续问题（可选）
        session_id: 会话ID（可选）
    """

    answer: str
    source_documents: Optional[List[Dict[str, Any]]] = None
    relevance_score: Optional[float] = None
    recall_rate: Optional[float] = None  # 文档召回率
    follow_up_questions: Optional[List[str]] = None
    session_id: Optional[str] = None


class SessionInfoResponse(BaseModel):
    """会话信息响应模型"""

    session_id: str
    created_time: str
    last_activity: str
    message_count: int
    mode: int
    status: str


class ListResponse(BaseModel, Generic[T]):
    """统一的列表响应格式"""

    items: List[T]
    total: int


class DiagnoseResponse(BaseModel):
    """诊断响应模型"""

    deployment: Optional[str] = None
    namespace: str
    status: str
    issues_found: List[str] = []
    recommendations: List[str] = []
    pods_status: Optional[Dict[str, Any]] = None
    logs_summary: Optional[Dict[str, Any]] = None
    events_summary: Optional[Dict[str, Any]] = None
    timestamp: str


class PredictTrendResponse(BaseModel):
    """趋势预测响应模型"""

    service_name: Optional[str] = None
    prediction_hours: int
    trend_data: List[Dict[str, Any]]
    insights: List[str] = []
    timestamp: str
