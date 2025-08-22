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

from pydantic import BaseModel, Field, field_validator

from app.config.settings import config


class RCARequest(BaseModel):
    """根因分析请求模型"""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Optional[List[str]] = None
    time_range_minutes: Optional[int] = Field(None, ge=1, le=config.rca.max_time_range)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return datetime.fromisoformat(v)
        return v

    def __init__(self, **data):
        super().__init__(**data)

        # 设置默认时间范围
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

        # 设置默认指标
        if not self.metrics:
            self.metrics = config.rca.default_metrics


class AutoFixRequest(BaseModel):
    """自动修复请求模型"""

    deployment: str = Field(..., min_length=1)
    namespace: str = Field(default="default", min_length=1)
    event: str = Field(..., min_length=1)
    force: bool = Field(default=False)
    auto_restart: bool = Field(default=True)


class AssistantRequest(BaseModel):
    """智能小助手请求模型"""

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
    """K8s问题诊断请求模型"""

    deployment: Optional[str] = Field(None, description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(True, description="是否包含日志分析")
    include_pods: bool = Field(True, description="是否包含Pod信息")
    include_events: bool = Field(True, description="是否包含事件信息")


class PredictTrendRequest(BaseModel):
    """负载趋势预测请求模型"""

    service_name: Optional[str] = Field(None, description="服务名称")
    hours: int = Field(24, description="预测小时数", ge=1, le=168)
