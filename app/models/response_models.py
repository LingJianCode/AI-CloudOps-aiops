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


class APIResponse(BaseModel):
    """通用API响应模型"""
    
    code: int
    message: str
    data: Optional[Any] = None



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


# 通用响应模型
class ServiceInfoResponse(BaseModel):
    """服务信息响应模型"""
    
    service: str
    version: str
    description: str
    capabilities: List[str]
    endpoints: Dict[str, str]
    constraints: Optional[Dict[str, Any]] = None
    status: str
    
    
class ServiceReadyResponse(BaseModel):
    """服务就绪响应模型"""
    
    ready: bool
    service: str
    timestamp: str
    message: Optional[str] = None


class ServiceHealthResponse(BaseModel):
    """服务健康检查响应模型"""
    
    status: str
    service: str
    version: Optional[str] = None
    dependencies: Optional[Dict[str, bool]] = None
    last_check_time: str
    uptime: Optional[float] = None


class ServiceConfigResponse(BaseModel):
    """服务配置响应模型"""
    
    service: str
    config: Dict[str, Any]
    version: Optional[str] = None
    timestamp: str


# Assistant API 专用响应模型
class RefreshKnowledgeResponse(BaseModel):
    """刷新知识库响应模型"""
    
    refreshed: bool
    documents_count: int
    vector_count: int
    timestamp: str
    message: str


class ClearCacheResponse(BaseModel):
    """清除缓存响应模型"""
    
    cleared: bool
    cache_keys_cleared: int
    timestamp: str
    message: str


class CreateSessionResponse(BaseModel):
    """创建会话响应模型"""
    
    session_id: str
    mode: int
    created_time: str
    status: str


class UploadKnowledgeResponse(BaseModel):
    """上传知识库响应模型"""
    
    uploaded: bool
    document_id: Optional[str] = None
    filename: Optional[str] = None
    file_size: Optional[int] = None
    message: str
    timestamp: str


class AddDocumentResponse(BaseModel):
    """添加文档响应模型"""
    
    added: bool
    document_id: str
    message: str
    timestamp: str


# Predict API 专用响应模型
class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    
    models: List[Dict[str, Any]]
    total_models: int
    loaded_models: int
    status: str
    timestamp: str


# RCA API 专用响应模型
class QuickDiagnosisResponse(BaseModel):
    """快速诊断响应模型"""
    
    namespace: str
    status: str
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    timestamp: str
    analysis_duration: float


class EventPatternsResponse(BaseModel):
    """事件模式响应模型"""
    
    namespace: str
    time_range_hours: float
    patterns: List[Dict[str, Any]]
    trending_events: List[str]
    anomalous_events: List[str]
    timestamp: str


class ErrorSummaryResponse(BaseModel):
    """错误摘要响应模型"""
    
    namespace: str
    time_range_hours: float
    total_errors: int
    error_categories: Dict[str, int]
    top_errors: List[Dict[str, Any]]
    error_timeline: List[Dict[str, Any]]
    timestamp: str
