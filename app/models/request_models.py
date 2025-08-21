#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 请求模型定义
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator
from app.config.settings import config


class RCARequest(BaseModel):
    """
    根因分析请求模型 - 用于分析系统异常的根本原因

    Attributes:
        start_time: 分析开始时间
        end_time: 分析结束时间
        metrics: 要分析的指标列表
        time_range_minutes: 时间范围（分钟）
    """

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Optional[List[str]] = None
    time_range_minutes: Optional[int] = Field(None, ge=1, le=config.rca.max_time_range)

    @validator("start_time", "end_time", pre=True, allow_reuse=True)
    def parse_datetime(cls, v):
        """验证并解析日期时间字符串"""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return datetime.fromisoformat(v)
        return v

    def __init__(self, **data):
        super().__init__(**data)

        # 如果没有提供时间范围，使用默认值
        if not self.start_time or not self.end_time:
            tz = timezone.utc
            now = datetime.now(tz)
            if self.time_range_minutes:
                self.end_time = now
                self.start_time = self.end_time - timedelta(
                    minutes=self.time_range_minutes
                )
            else:
                self.end_time = now
                self.start_time = self.end_time - timedelta(
                    minutes=config.rca.default_time_range
                )

        # 如果没有提供指标，使用默认指标
        if not self.metrics:
            self.metrics = config.rca.default_metrics


class AutoFixRequest(BaseModel):
    """
    自动修复请求模型 - 用于自动修复Kubernetes集群中的问题

    Attributes:
        deployment: 部署名称
        namespace: 命名空间名称
        event: 触发事件描述
        force: 是否强制执行修复
        auto_restart: 是否在需要时自动重启Pod
    """

    deployment: str = Field(..., min_length=1)
    namespace: str = Field(default="default", min_length=1)
    event: str = Field(..., min_length=1)
    force: bool = Field(default=False)
    auto_restart: bool = Field(default=True)


class PredictionRequest(BaseModel):
    """
    负载预测请求模型 - 用于预测未来系统负载

    Attributes:
        service_name: 服务名称
        current_qps: 当前每秒查询数
        hours: 预测小时数
        instance_cpu: 实例CPU数
        instance_memory: 实例内存(GB)
        include_confidence: 是否包含置信区间
    """

    service_name: str = Field(default="unknown", description="服务名称")
    current_qps: float = Field(..., description="当前QPS", gt=0)
    hours: int = Field(default=24, description="预测小时数", ge=1, le=168)
    instance_cpu: Optional[int] = Field(None, description="实例CPU数", gt=0)
    instance_memory: Optional[int] = Field(None, description="实例内存(GB)", gt=0)
    include_confidence: bool = Field(default=True, description="是否包含置信区间")

    @validator("current_qps", allow_reuse=True)
    def validate_qps(cls, v):
        """验证QPS值非负"""
        if v < 0:
            raise ValueError("QPS不能为负数")
        return v


class AssistantRequest(BaseModel):
    """
    智能小助手请求模型 - 用于与AI助手交互

    Attributes:
        question: 用户提问内容
        mode: 助手模式，1为RAG模式，2为MCP模式
        chat_history: 历史对话记录
        use_web_search: 是否使用网络搜索增强回答
        max_context_docs: 最大上下文文档数量
        session_id: 会话ID，为空则创建新会话
    """

    question: str = Field(..., min_length=1, description="用户提问")
    mode: int = Field(
        default=1, description="助手模式：1=RAG模式，2=MCP模式", ge=1, le=2
    )
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="对话历史记录"
    )
    use_web_search: bool = Field(default=False, description="是否使用网络搜索增强回答")
    max_context_docs: int = Field(
        default=4, ge=1, le=10, description="最大上下文文档数量"
    )
    session_id: Optional[str] = Field(
        default=None, description="会话ID，为空则创建新会话"
    )


class SessionRequest(BaseModel):
    """会话请求模型"""

    session_id: str = Field(..., description="会话ID")


class DiagnoseRequest(BaseModel):
    """Kubernetes问题诊断请求模型"""

    deployment: Optional[str] = Field(None, description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(True, description="是否包含日志分析")
    include_pods: bool = Field(True, description="是否包含Pod信息")
    include_events: bool = Field(True, description="是否包含事件信息")


class PredictTrendRequest(BaseModel):
    """负载趋势预测请求模型"""

    service_name: Optional[str] = Field(None, description="服务名称")
    hours: int = Field(24, description="预测小时数", ge=1, le=168)
