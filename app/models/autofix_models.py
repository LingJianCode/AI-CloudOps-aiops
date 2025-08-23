#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 自动修复模块模型定义
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class AutoFixRequest(BaseModel):
    """自动修复请求模型"""

    deployment: str = Field(..., min_length=1)
    namespace: str = Field(default="default", min_length=1)
    event: str = Field(..., min_length=1)
    force: bool = Field(default=False)
    auto_restart: bool = Field(default=True)


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
