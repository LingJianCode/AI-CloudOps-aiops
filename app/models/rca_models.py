#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: RCA根因分析数据模型 - 定义所有与根因分析相关的请求和响应数据结构
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class DataSourceType(str, Enum):
    """数据源类型枚举"""

    METRICS = "metrics"
    EVENTS = "events"
    LOGS = "logs"


class SeverityLevel(str, Enum):
    """严重程度级别"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RCARequest(BaseModel):
    """
    根因分析请求模型 - 支持多数据源的综合分析

    Attributes:
        namespace: Kubernetes命名空间
        start_time: 分析开始时间
        end_time: 分析结束时间
        metrics: 要分析的Prometheus指标列表
        service_name: 可选的服务名称过滤
        include_events: 是否包含K8s事件分析
        include_logs: 是否包含Pod日志分析
        severity_threshold: 严重性阈值
        correlation_window: 关联分析时间窗口（秒）
        max_candidates: 最大根因候选数量
    """

    namespace: str = Field(default="default", description="Kubernetes命名空间")
    start_time: datetime = Field(..., description="分析开始时间")
    end_time: datetime = Field(..., description="分析结束时间")
    metrics: List[str] = Field(default_factory=list, description="Prometheus指标列表")
    service_name: Optional[str] = Field(None, description="服务名称过滤")
    include_events: bool = Field(default=True, description="是否包含K8s事件分析")
    include_logs: bool = Field(default=True, description="是否包含Pod日志分析")
    severity_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="严重性阈值")
    correlation_window: int = Field(
        default=300, ge=60, le=3600, description="关联分析时间窗口（秒）"
    )
    max_candidates: int = Field(default=10, ge=1, le=20, description="最大根因候选数量")

    @validator("end_time")
    def validate_time_range(cls, v, values):
        """验证时间范围有效性"""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("结束时间必须晚于开始时间")
        return v


class MetricData(BaseModel):
    """
    指标数据模型

    Attributes:
        name: 指标名称
        values: 时间序列数据点
        labels: 指标标签
        anomaly_score: 异常分数
        trend: 趋势方向
    """

    name: str = Field(..., description="指标名称")
    values: List[Dict[str, Any]] = Field(default_factory=list, description="时间序列数据")
    labels: Dict[str, str] = Field(default_factory=dict, description="指标标签")
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0, description="异常分数")
    trend: str = Field(default="stable", description="趋势方向")


class EventData(BaseModel):
    """
    K8s事件数据模型

    Attributes:
        timestamp: 事件时间戳
        type: 事件类型
        reason: 事件原因
        message: 事件消息
        involved_object: 涉及的对象信息
        severity: 严重程度
        count: 事件计数
    """

    timestamp: datetime = Field(..., description="事件时间戳")
    type: str = Field(..., description="事件类型")
    reason: str = Field(..., description="事件原因")
    message: str = Field(..., description="事件消息")
    involved_object: Dict[str, str] = Field(default_factory=dict, description="涉及的对象")
    severity: SeverityLevel = Field(default=SeverityLevel.LOW, description="严重程度")
    count: int = Field(default=1, ge=1, description="事件计数")


class LogData(BaseModel):
    """
    Pod日志数据模型

    Attributes:
        timestamp: 日志时间戳
        pod_name: Pod名称
        container_name: 容器名称
        level: 日志级别
        message: 日志消息
        error_type: 错误类型（如果是错误日志）
        stack_trace: 堆栈跟踪（如果有）
    """

    timestamp: datetime = Field(..., description="日志时间戳")
    pod_name: str = Field(..., description="Pod名称")
    container_name: str = Field(..., description="容器名称")
    level: str = Field(..., description="日志级别")
    message: str = Field(..., description="日志消息")
    error_type: Optional[str] = Field(None, description="错误类型")
    stack_trace: Optional[str] = Field(None, description="堆栈跟踪")


class CorrelationResult(BaseModel):
    """
    关联分析结果模型

    Attributes:
        source_type: 源数据类型
        target_type: 目标数据类型
        source_identifier: 源数据标识
        target_identifier: 目标数据标识
        correlation_score: 关联分数
        temporal_offset: 时间偏移量（秒）
        evidence: 支撑证据
    """

    source_type: DataSourceType = Field(..., description="源数据类型")
    target_type: DataSourceType = Field(..., description="目标数据类型")
    source_identifier: str = Field(..., description="源数据标识")
    target_identifier: str = Field(..., description="目标数据标识")
    correlation_score: float = Field(..., ge=0.0, le=1.0, description="关联分数")
    temporal_offset: int = Field(default=0, description="时间偏移量（秒）")
    evidence: List[str] = Field(default_factory=list, description="支撑证据")


class RootCause(BaseModel):
    """
    根因候选模型

    Attributes:
        identifier: 根因标识
        data_source: 数据源类型
        confidence: 置信度
        severity: 严重程度
        first_occurrence: 首次出现时间
        last_occurrence: 最后出现时间
        description: 描述信息
        impact_scope: 影响范围
        correlations: 关联的其他数据
        recommendations: 修复建议
    """

    identifier: str = Field(..., description="根因标识")
    data_source: DataSourceType = Field(..., description="数据源类型")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    severity: SeverityLevel = Field(..., description="严重程度")
    first_occurrence: datetime = Field(..., description="首次出现时间")
    last_occurrence: datetime = Field(..., description="最后出现时间")
    description: str = Field(..., description="描述信息")
    impact_scope: List[str] = Field(default_factory=list, description="影响范围")
    correlations: List[CorrelationResult] = Field(default_factory=list, description="关联数据")
    recommendations: List[str] = Field(default_factory=list, description="修复建议")


class RCAAnalysisResult(BaseModel):
    """
    根因分析结果模型

    Attributes:
        request_id: 请求ID
        analysis_time: 分析时间
        time_range: 分析时间范围
        namespace: 分析的命名空间
        data_sources_analyzed: 分析的数据源
        metrics_data: 指标数据
        events_data: 事件数据
        logs_data: 日志数据
        correlations: 关联分析结果
        root_causes: 根因候选列表
        analysis_summary: 分析摘要
        confidence_score: 整体置信度
        processing_time: 处理耗时（秒）
    """

    request_id: str = Field(..., description="请求ID")
    analysis_time: datetime = Field(..., description="分析时间")
    time_range: Dict[str, datetime] = Field(..., description="分析时间范围")
    namespace: str = Field(..., description="分析的命名空间")
    data_sources_analyzed: List[DataSourceType] = Field(..., description="分析的数据源")
    metrics_data: List[MetricData] = Field(default_factory=list, description="指标数据")
    events_data: List[EventData] = Field(default_factory=list, description="事件数据")
    logs_data: List[LogData] = Field(default_factory=list, description="日志数据")
    correlations: List[CorrelationResult] = Field(default_factory=list, description="关联分析结果")
    root_causes: List[RootCause] = Field(default_factory=list, description="根因候选列表")
    analysis_summary: str = Field(default="", description="分析摘要")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="整体置信度")
    processing_time: float = Field(default=0.0, ge=0.0, description="处理耗时（秒）")


class RCAResponse(BaseModel):
    """
    根因分析响应模型

    Attributes:
        status: 分析状态
        result: 分析结果（成功时）
        error_message: 错误信息（失败时）
        warnings: 警告信息
        metadata: 元数据信息
    """

    status: str = Field(..., description="分析状态")
    result: Optional[RCAAnalysisResult] = Field(None, description="分析结果")
    error_message: Optional[str] = Field(None, description="错误信息")
    warnings: List[str] = Field(default_factory=list, description="警告信息")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据信息")
