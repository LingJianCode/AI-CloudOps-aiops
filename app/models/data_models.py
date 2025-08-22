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
    """指标数据模型"""

    name: str
    values: pd.DataFrame
    labels: Dict[str, str]
    unit: Optional[str] = None
    source: Optional[str] = None


@dataclass
class AnomalyResult:
    """异常检测结果模型"""

    metric: str
    anomaly_points: List[datetime]
    anomaly_scores: List[float]
    detection_methods: Dict[str, Any]
    severity: str = "medium"  # low, medium, high


@dataclass
class CorrelationResult:
    """相关性分析结果模型"""

    metric_pairs: List[tuple]
    correlation_matrix: pd.DataFrame
    significant_correlations: Dict[str, List[tuple]]
    method: str = "pearson"


@dataclass
class AgentState:
    """智能体状态模型"""

    messages: List[Dict[str, Any]]
    current_step: str
    context: Dict[str, Any]
    next_action: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 10


@dataclass
class PredictionFeatures:
    """预测特征模型"""

    qps: float
    sin_time: float
    cos_time: float
    hour: int
    day_of_week: int
    timestamp: datetime
