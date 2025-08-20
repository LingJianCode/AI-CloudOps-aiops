#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 统一响应处理器 - 提供标准化的API响应包装和错误处理
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .exceptions import AIOpsException

logger = logging.getLogger("aiops.response")


class ResponseWrapper:
    """
    统一响应包装器
    
    提供标准化的成功和错误响应格式
    """
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "success",
        code: int = 0
    ) -> Dict[str, Any]:
        """
        创建成功响应
        
        Args:
            data: 响应数据
            message: 响应消息
            code: 业务状态码，0表示成功
            
        Returns:
            标准化的成功响应字典
        """
        return {
            "code": code,
            "message": message,
            "data": data
        }
    
    @staticmethod
    def success_list(
        items: list,
        message: str = "success", 
        code: int = 0,
        total: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        创建列表数据成功响应
        
        Args:
            items: 列表数据
            message: 响应消息
            code: 业务状态码，0表示成功
            total: 总数量，如果不提供则使用items的长度
            
        Returns:
            标准化的列表响应字典
        """
        if total is None:
            total = len(items) if items else 0
            
        return {
            "code": code,
            "message": message,
            "data": {
                "items": items,
                "total": total
            }
        }
    
    @staticmethod
    def error(
        message: str,
        code: int = 1,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        创建错误响应
        
        Args:
            message: 错误消息
            code: 业务错误码
            details: 错误详情
            error_code: 错误类型码
            
        Returns:
            标准化的错误响应字典
        """
        response_data = {
            "code": code,
            "message": message
        }
        
        if details:
            response_data["details"] = details
            
        if error_code:
            response_data["error_code"] = error_code
            
        return response_data


async def handle_aiops_exception(request: Request, exc: AIOpsException) -> JSONResponse:
    """
    处理业务异常
    
    Args:
        request: FastAPI请求对象
        exc: AIOps业务异常
        
    Returns:
        JSON错误响应
    """
    logger.error(f"业务异常: {exc.message}, 错误码: {exc.error_code}, 详情: {exc.details}")
    
    response_data = ResponseWrapper.error(
        message=exc.message,
        code=1,
        details=exc.details,
        error_code=exc.error_code
    )
    
    # 添加请求上下文信息
    if response_data.get("details"):
        response_data["details"].update({
            "path": str(request.url.path),
            "method": request.method
        })
    else:
        response_data["details"] = {
            "path": str(request.url.path),
            "method": request.method
        }
    
    # 根据异常类型返回合适的HTTP状态码
    http_status = _get_http_status_for_exception(exc)
    
    return JSONResponse(
        status_code=http_status,
        content=response_data
    )


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """
    处理HTTP异常
    
    Args:
        request: FastAPI请求对象
        exc: HTTP异常
        
    Returns:
        JSON错误响应
    """
    logger.error(f"HTTP异常: {exc.detail}, 状态码: {exc.status_code}")
    
    response_data = ResponseWrapper.error(
        message=str(exc.detail),
        code=exc.status_code,
        details={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exc.status_code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )


async def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
    """
    处理通用异常
    
    Args:
        request: FastAPI请求对象
        exc: 通用异常
        
    Returns:
        JSON错误响应
    """
    logger.error(f"未处理异常: {str(exc)}", exc_info=True)
    
    response_data = ResponseWrapper.error(
        message="服务器内部错误",
        code=500,
        details={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": 500
        }
    )
    
    return JSONResponse(
        status_code=500,
        content=response_data
    )


def _get_http_status_for_exception(exc: AIOpsException) -> int:
    """
    根据业务异常类型返回合适的HTTP状态码
    
    Args:
        exc: AIOps业务异常
        
    Returns:
        HTTP状态码
    """
    from .exceptions import (
        ServiceUnavailableError,
        ValidationError,
        ConfigurationError,
        ExternalServiceError
    )
    
    if isinstance(exc, ServiceUnavailableError):
        return 503
    elif isinstance(exc, ValidationError):
        return 400
    elif isinstance(exc, ConfigurationError):
        return 500
    elif isinstance(exc, ExternalServiceError):
        return 502
    else:
        return 500
