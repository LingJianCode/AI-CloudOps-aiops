#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 请求模型定义
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    session_id: Optional[str] = Field(
        default=None, description="会话ID，为空则创建新会话"
    )


class DiagnoseRequest(BaseModel):
    """K8s问题诊断请求模型"""

    deployment: Optional[str] = Field(None, description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(True, description="是否包含日志分析")
    include_pods: bool = Field(True, description="是否包含Pod信息")
    include_events: bool = Field(True, description="是否包含事件信息")


class UploadKnowledgeRequest(BaseModel):
    """上传知识库请求模型"""

    title: str = Field(..., min_length=1, description="知识库文档标题")
    content: str = Field(..., min_length=1, description="知识库文档内容")
    source: Optional[str] = Field(None, description="文档来源")
    category: Optional[str] = Field(None, description="文档分类")
    tags: Optional[List[str]] = Field(default=None, description="文档标签")
    metadata: Optional[Dict[str, str]] = Field(default=None, description="额外元数据")


class AddDocumentRequest(BaseModel):
    """添加文档请求模型"""

    title: str = Field(..., min_length=1, description="文档标题")
    content: str = Field(..., min_length=1, description="文档内容")
    source: Optional[str] = Field(default="api", description="文档来源")
    category: Optional[str] = Field(default="general", description="文档分类")
    tags: Optional[List[str]] = Field(default=None, description="文档标签")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="额外元数据")
