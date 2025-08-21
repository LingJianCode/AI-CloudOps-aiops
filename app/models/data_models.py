#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 数据模型定义
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class MetricData:
    """
    指标数据模型 - 表示从监控系统收集的时序指标数据

    Attributes:
        name: 指标名称
        values: 时间序列数据帧，包含时间戳和指标值
        labels: 指标标签，如维度信息
        unit: 指标单位（可选）
        source: 数据来源（可选）
    """

    name: str
    values: pd.DataFrame
    labels: Dict[str, str]
    unit: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AnomalyResult:
    """
    异常检测结果模型 - 表示异常检测算法的输出结果

    Attributes:
        metric: 指标名称
        anomaly_points: 异常点的时间戳列表
        anomaly_scores: 对应的异常分数列表
        detection_methods: 使用的检测方法及其参数
        severity: 异常严重程度（低、中、高）
    """

    metric: str
    anomaly_points: List[datetime]
    anomaly_scores: List[float]
    detection_methods: Dict[str, Any]
    severity: str = "medium"  # low, medium, high


@dataclass
class CorrelationResult:
    """
    相关性分析结果模型 - 表示指标之间的相关性分析结果

    Attributes:
        metric_pairs: 分析的指标对列表
        correlation_matrix: 相关性矩阵
        significant_correlations: 显著相关的指标对及其相关系数
        method: 使用的相关性分析方法
    """

    metric_pairs: List[tuple]
    correlation_matrix: pd.DataFrame
    significant_correlations: Dict[str, List[tuple]]
    method: str = "pearson"


@dataclass
class AgentState:
    """
    智能体状态模型 - 表示AI代理的内部状态

    Attributes:
        messages: 对话历史消息列表
        current_step: 当前执行步骤
        context: 上下文信息
        next_action: 下一步行动（可选）
        iteration_count: 当前迭代次数
        max_iterations: 最大迭代次数限制
    """

    messages: List[Dict[str, Any]]
    current_step: str
    context: Dict[str, Any]
    next_action: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 10


@dataclass
class PredictionFeatures:
    """
    预测特征模型 - 表示用于负载预测的输入特征

    Attributes:
        qps: 每秒查询数
        sin_time: 时间的正弦变换（周期性特征）
        cos_time: 时间的余弦变换（周期性特征）
        hour: 小时（0-23）
        day_of_week: 星期几（0-6）
        timestamp: 时间戳
    """

    qps: float
    sin_time: float
    cos_time: float
    hour: int
    day_of_week: int
    timestamp: datetime
