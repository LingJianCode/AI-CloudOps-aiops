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
    """统一API响应格式"""

    code: int = 0
    message: str = ""
    data: Optional[T] = None


class AutoFixResponse(BaseModel):
    """自动修复响应模型"""

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
    """健康检查响应模型"""

    status: str
    components: Dict[str, bool]
    timestamp: str
    version: Optional[str] = None
    uptime: Optional[float] = None


class AssistantResponse(BaseModel):
    """智能小助手响应模型"""

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
