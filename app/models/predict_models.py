#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 预测服务专用数据模型
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class PredictionType(str, Enum):
    """预测类型枚举"""
    QPS = "qps"                # QPS负载预测
    CPU = "cpu"                # CPU使用率预测  
    MEMORY = "memory"          # 内存使用率预测
    DISK = "disk"              # 磁盘使用率预测


class PredictionGranularity(str, Enum):
    """预测时间粒度枚举"""
    MINUTE = "minute"          # 分钟级预测
    HOUR = "hour"              # 小时级预测  
    DAY = "day"                # 天级预测


class ScalingAction(str, Enum):
    """扩缩容建议动作枚举"""
    SCALE_UP = "scale_up"      # 扩容
    SCALE_DOWN = "scale_down"  # 缩容
    MAINTAIN = "maintain"      # 保持现状


class ResourceConstraints(BaseModel):
    """资源约束配置模型"""
    
    cpu_cores: Optional[float] = Field(None, description="CPU核数限制", gt=0)
    memory_gb: Optional[float] = Field(None, description="内存大小限制(GB)", gt=0)
    disk_gb: Optional[float] = Field(None, description="磁盘大小限制(GB)", gt=0)
    max_instances: Optional[int] = Field(None, description="最大实例数限制", gt=0, le=1000)
    min_instances: Optional[int] = Field(None, description="最小实例数限制", gt=0, le=100)
    cost_per_hour: Optional[float] = Field(None, description="每小时成本限制", ge=0)


class PredictionRequest(BaseModel):
    """统一预测请求模型"""
    
    prediction_type: PredictionType = Field(..., description="预测类型")
    current_value: float = Field(..., description="当前指标值")
    metric_query: Optional[str] = Field(None, description="自定义Prometheus指标查询，为空则使用默认指标")
    prediction_hours: int = Field(default=24, description="预测小时数", ge=1, le=168)
    granularity: PredictionGranularity = Field(default=PredictionGranularity.HOUR, description="预测时间粒度")
    resource_constraints: Optional[ResourceConstraints] = Field(None, description="资源约束配置")
    include_confidence: bool = Field(default=True, description="是否包含置信区间")
    include_anomaly_detection: bool = Field(default=True, description="是否包含异常检测")
    consider_historical_pattern: bool = Field(default=True, description="是否考虑历史模式")
    target_utilization: float = Field(default=0.7, description="目标资源利用率", gt=0, le=1)
    sensitivity: float = Field(default=0.8, description="异常检测敏感度", ge=0.1, le=1.0)
    
    @validator("current_value", always=True)
    def validate_current_value(cls, v, values):
        """根据预测类型验证当前值的合理性"""
        prediction_type = values.get("prediction_type")
        
        if prediction_type == PredictionType.QPS:
            if v <= 0 or v > 100000:
                raise ValueError("QPS值应在1-100000之间")
        elif prediction_type in [PredictionType.CPU, PredictionType.MEMORY, PredictionType.DISK]:
            if v < 0 or v > 100:
                raise ValueError("资源利用率应在0-100%之间")
        
        return v
    
    @validator("target_utilization")
    def validate_target_utilization(cls, v):
        if v < 0.1 or v > 0.9:
            raise ValueError("目标利用率应在0.1-0.9之间")
        return v
    
    @validator("metric_query")
    def validate_metric_query(cls, v):
        """验证Prometheus指标查询格式"""
        if v is not None and v.strip() == "":
            raise ValueError("指标查询不能为空字符串")
        return v


class PredictionDataPoint(BaseModel):
    """预测数据点模型"""
    
    timestamp: datetime = Field(..., description="预测时间点")
    predicted_value: float = Field(..., description="预测值")
    confidence_lower: Optional[float] = Field(None, description="置信区间下限")
    confidence_upper: Optional[float] = Field(None, description="置信区间上限")
    confidence_level: Optional[float] = Field(None, description="预测置信度", ge=0, le=1)


class ResourceUtilization(BaseModel):
    """资源利用率预测模型"""
    
    timestamp: datetime = Field(..., description="时间点")
    cpu_utilization: Optional[float] = Field(None, description="CPU利用率(%)", ge=0, le=100)
    memory_utilization: Optional[float] = Field(None, description="内存利用率(%)", ge=0, le=100)
    disk_utilization: Optional[float] = Field(None, description="磁盘利用率(%)", ge=0, le=100)
    predicted_load: Optional[float] = Field(None, description="预测负载值")


class ScalingRecommendation(BaseModel):
    """扩缩容建议模型"""
    
    action: ScalingAction = Field(..., description="推荐动作")
    trigger_time: datetime = Field(..., description="建议执行时间")
    confidence: float = Field(..., description="建议置信度", ge=0, le=1)
    reason: str = Field(..., description="建议原因描述")
    target_instances: Optional[int] = Field(None, description="目标实例数", gt=0)
    target_cpu_cores: Optional[float] = Field(None, description="目标CPU核数", gt=0)
    target_memory_gb: Optional[float] = Field(None, description="目标内存大小(GB)", gt=0)
    target_disk_gb: Optional[float] = Field(None, description="目标磁盘大小(GB)", gt=0)
    estimated_cost_change: Optional[float] = Field(None, description="预估成本变化(%)")


class CostAnalysis(BaseModel):
    """成本分析模型"""
    
    current_hourly_cost: Optional[float] = Field(None, description="当前每小时成本", ge=0)
    predicted_hourly_cost: Optional[float] = Field(None, description="预测每小时成本", ge=0)
    cost_savings_potential: Optional[float] = Field(None, description="潜在节省成本(%)")
    cost_trend_analysis: Dict[str, Any] = Field(default_factory=dict, description="成本趋势分析")


class AnomalyPrediction(BaseModel):
    """异常预测模型"""
    
    timestamp: datetime = Field(..., description="预测异常发生时间")
    anomaly_score: float = Field(..., description="异常分数", ge=0, le=1)
    anomaly_type: str = Field(..., description="异常类型")
    impact_level: str = Field(..., description="影响等级", pattern="^(low|medium|high|critical)$")
    predicted_value: float = Field(..., description="异常时预测值")
    expected_value: float = Field(..., description="正常情况下的预期值")


class PredictionResponse(BaseModel):
    """统一预测响应模型"""
    
    prediction_type: PredictionType = Field(..., description="预测类型")
    prediction_hours: int = Field(..., description="预测小时数")
    granularity: PredictionGranularity = Field(..., description="预测时间粒度")
    current_value: float = Field(..., description="当前指标值")
    predicted_data: List[PredictionDataPoint] = Field(..., description="预测数据点列表")
    resource_utilization: List[ResourceUtilization] = Field(default_factory=list, description="资源利用率预测")
    scaling_recommendations: List[ScalingRecommendation] = Field(default_factory=list, description="扩缩容建议列表")
    anomaly_predictions: List[AnomalyPrediction] = Field(default_factory=list, description="异常预测列表")
    cost_analysis: Optional[CostAnalysis] = Field(None, description="成本分析结果")
    pattern_analysis: Dict[str, Any] = Field(default_factory=dict, description="模式分析结果")
    trend_insights: List[str] = Field(default_factory=list, description="趋势洞察")
    model_accuracy: Optional[float] = Field(None, description="模型准确率", ge=0, le=1)
    prediction_summary: Dict[str, Any] = Field(default_factory=dict, description="预测摘要信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="预测生成时间")


class ModelInfo(BaseModel):
    """预测模型信息"""
    
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(..., description="模型版本号")
    model_type: str = Field(..., description="模型类型")
    supported_prediction_types: List[PredictionType] = Field(default_factory=list, description="支持的预测类型")
    training_data_size: Optional[int] = Field(None, description="训练数据样本数")
    last_trained: Optional[datetime] = Field(None, description="最后训练时间")
    accuracy_metrics: Dict[str, float] = Field(default_factory=dict, description="模型准确性指标")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="特征重要性分析")


class PredictionServiceHealthResponse(BaseModel):
    """预测服务健康检查响应"""
    
    service_status: str = Field(..., description="服务整体状态")
    model_status: str = Field(..., description="模型加载状态") 
    models_loaded: List[ModelInfo] = Field(default_factory=list, description="已加载模型列表")
    supported_prediction_types: List[PredictionType] = Field(default_factory=list, description="支持的预测类型")
    last_prediction_time: Optional[datetime] = Field(None, description="最后一次预测时间")
    total_predictions: int = Field(default=0, description="累计预测次数")
    error_rate: float = Field(default=0.0, description="预测错误率", ge=0, le=1)
    average_response_time_ms: Optional[float] = Field(None, description="平均响应时间(毫秒)")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="服务资源使用情况")
    timestamp: datetime = Field(default_factory=datetime.now, description="健康检查时间")


# 导出所有模型
__all__ = [
    # 枚举类型
    "PredictionType",
    "PredictionGranularity", 
    "ScalingAction",
    # 基础数据模型
    "ResourceConstraints",
    "PredictionDataPoint",
    "ResourceUtilization",
    "ScalingRecommendation", 
    "CostAnalysis",
    "AnomalyPrediction",
    "ModelInfo",
    # 请求响应模型
    "PredictionRequest",
    "PredictionResponse",
    "PredictionServiceHealthResponse"
]
