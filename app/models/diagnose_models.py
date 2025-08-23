#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 诊断模块模型定义
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    """K8s问题诊断请求模型"""

    deployment: Optional[str] = Field(None, description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(True, description="是否包含日志分析")
    include_pods: bool = Field(True, description="是否包含Pod信息")
    include_events: bool = Field(True, description="是否包含事件信息")


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
