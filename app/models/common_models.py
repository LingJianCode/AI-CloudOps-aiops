#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 通用模型定义
"""

from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel):
    """通用API响应模型"""

    code: int
    message: str
    data: Optional[Any] = None


class ListResponse(BaseModel, Generic[T]):
    """统一的列表响应格式"""

    items: List[T]
    total: int
