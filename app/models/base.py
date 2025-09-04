#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 基础响应模型
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """统一的基础响应模型"""

    code: int = Field(default=0, description="业务状态码，0 表示成功")
    message: str = Field(default="success", description="业务消息")
    data: Optional[Any] = Field(default=None, description="数据载荷")


T = TypeVar("T")


class ListResponse(BaseModel, Generic[T]):
    """统一的列表响应结构"""

    items: List[T]
    total: int


class ErrorResponse(BaseModel):
    """统一错误响应模型"""

    code: int
    message: str
    detail: Optional[str] = None


