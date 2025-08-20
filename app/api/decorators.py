#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: API装饰器 - 提供统一的API行为装饰器，包括响应包装、错误处理等
"""

import functools
import logging
from typing import Any, Callable, Dict

from fastapi import HTTPException, Request

from ..common.exceptions import AIOpsException
from ..common.response import ResponseWrapper

logger = logging.getLogger("aiops.api.decorators")


def api_response(operation_name: str = "操作"):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                result = await func(*args, **kwargs)
                
                if isinstance(result, dict) and "code" in result:
                    return result
                
                return ResponseWrapper.success(
                    data=result,
                    message=f"{operation_name}成功"
                )
                
            except HTTPException:
                raise
            except AIOpsException as e:
                logger.error(f"{operation_name}失败: {e.message}")
                raise HTTPException(
                    status_code=_get_http_status_for_aiops_exception(e),
                    detail=e.message
                )
            except Exception as e:
                logger.error(f"{operation_name}发生未处理异常: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"{operation_name}失败: 服务器内部错误"
                )
        
        return wrapper
    return decorator


def validate_request(validator_func: Callable = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if validator_func:
                validation_result = validator_func(*args, **kwargs)
                if validation_result is not True:
                    raise HTTPException(
                        status_code=400,
                        detail=f"请求验证失败: {validation_result}"
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_api_call(log_request: bool = True, log_response: bool = False):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if log_request and request:
                logger.info(
                    f"API调用: {request.method} {request.url.path} "
                    f"- 客户端: {request.client.host if request.client else 'unknown'}"
                )
            
            result = await func(*args, **kwargs)
            
            if log_response:
                if isinstance(result, dict):
                    status = result.get("code", "unknown")
                    message = result.get("message", "")
                    logger.info(f"API响应: {func.__name__} - 状态: {status}, 消息: {message}")
            
            return result
        
        return wrapper
    return decorator


def _get_http_status_for_aiops_exception(exc: AIOpsException) -> int:
    from ..common.exceptions import (
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
