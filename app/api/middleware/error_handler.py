#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 错误处理中间件
"""

import logging
import traceback
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.common.constants import HttpStatusCodes, ServiceConstants
from app.models.response_models import APIResponse

logger = logging.getLogger("aiops.error_handler")


async def custom_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """自定义HTTP异常处理器"""
    try:
        # 4xx 作为可预期客户端错误，使用 WARNING；其他仍为 ERROR
        is_client_error = (
            HttpStatusCodes.BAD_REQUEST
            <= exc.status_code
            < HttpStatusCodes.INTERNAL_SERVER_ERROR
        )

        log_method = logger.warning if is_client_error else logger.error
        log_method(
            f"HTTP异常处理器被触发 - 状态码: {exc.status_code}, 详情: {exc.detail}"
        )
        log_method(f"请求信息 - URL: {request.url}, Method: {request.method}")

        # 构建错误响应
        error_response = APIResponse(
            code=exc.status_code,
            message=str(exc.detail),
            data=_build_error_data(request, exc.status_code, exc.detail),
        )

        return JSONResponse(status_code=exc.status_code, content=error_response.dict())

    except Exception as e:
        logger.error(f"错误处理器本身出错: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")

        return _create_fallback_response()


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """验证异常处理器"""
    try:
        # 使用 WARNING 记录可预期的 400 校验错误
        details = _extract_validation_details(exc)

        logger.warning(f"验证异常: {details}")
        logger.warning(f"请求信息 - URL: {request.url}, Method: {request.method}")

        error_response = APIResponse(
            code=HttpStatusCodes.BAD_REQUEST,
            message=ServiceConstants.VALIDATION_ERROR_MESSAGE,
            data=_build_error_data(request, HttpStatusCodes.BAD_REQUEST, details),
        )

        return JSONResponse(
            status_code=HttpStatusCodes.BAD_REQUEST, content=error_response.dict()
        )

    except Exception as e:
        logger.error(f"验证异常处理器出错: {str(e)}")
        return _create_fallback_response(message="验证异常处理器错误", error=str(e))


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    try:
        logger.error(f"未捕获的异常: {str(exc)}")
        logger.error(f"异常类型: {type(exc).__name__}")
        logger.error(f"请求信息 - URL: {request.url}, Method: {request.method}")
        logger.error(f"异常堆栈: {traceback.format_exc()}")

        error_data = _build_error_data(
            request, HttpStatusCodes.INTERNAL_SERVER_ERROR, str(exc)
        )
        error_data["type"] = type(exc).__name__

        error_response = APIResponse(
            code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            message=ServiceConstants.INTERNAL_SERVER_ERROR_MESSAGE,
            data=error_data,
        )

        return JSONResponse(
            status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
            content=error_response.dict(),
        )

    except Exception as handler_error:
        logger.critical(f"异常处理器本身出现严重错误: {str(handler_error)}")
        logger.critical(f"原异常: {str(exc)}")

        return _create_critical_fallback_response()


def setup_error_handlers(app: FastAPI) -> None:
    """设置错误处理器"""
    try:
        # 注册HTTP异常处理器
        app.add_exception_handler(HTTPException, custom_http_exception_handler)
        app.add_exception_handler(StarletteHTTPException, custom_http_exception_handler)

        # 注册验证异常处理器
        app.add_exception_handler(ValidationError, validation_exception_handler)
        app.add_exception_handler(RequestValidationError, validation_exception_handler)

        # 注册通用异常处理器
        app.add_exception_handler(Exception, general_exception_handler)

        logger.info("错误处理器设置完成")

    except Exception as e:
        logger.error(f"错误处理器设置失败: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")


def _extract_validation_details(exc: Exception) -> Any:
    """提取验证异常详情"""
    try:
        if isinstance(exc, RequestValidationError):
            errors = exc.errors()
            # 处理可能包含bytes的错误信息
            return _sanitize_error_details(errors)
        elif isinstance(exc, ValidationError):
            errors = exc.errors()
            return _sanitize_error_details(errors)
        else:
            return str(exc)
    except Exception as e:
        logger.warning(f"处理验证异常详情时出错: {e}")
        return "验证异常详情处理失败"


def _sanitize_error_details(errors):
    """清理错误详情，处理不可JSON序列化的对象"""
    try:
        import json

        def sanitize_value(obj):
            """递归清理对象中不可序列化的值"""
            if isinstance(obj, bytes):
                # 对于bytes对象，尝试解码或返回类型信息
                try:
                    if len(obj) > 200:  # 限制长度避免日志过长
                        return f"<bytes: {len(obj)} bytes, starts with: {obj[:100].decode('utf-8', errors='ignore')}...>"
                    return f"<bytes: {obj.decode('utf-8', errors='ignore')}>"
                except Exception:
                    return f"<bytes: {len(obj)} bytes>"
            elif isinstance(obj, dict):
                return {k: sanitize_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_value(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(sanitize_value(item) for item in obj)
            else:
                # 测试是否可以JSON序列化
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        return sanitize_value(errors)

    except Exception as e:
        logger.warning(f"清理错误详情时出错: {e}")
        return "错误详情清理失败"


def _build_error_data(
    request: Request, status_code: int, detail: Any
) -> Dict[str, Any]:
    """构建错误数据"""
    return {
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url.path),
        "method": request.method,
        "status_code": status_code,
    }


def _create_fallback_response(
    message: str = ServiceConstants.INTERNAL_SERVER_ERROR_MESSAGE,
    error: str = "错误处理器异常",
) -> JSONResponse:
    """创建回退错误响应"""
    fallback_response = {
        "code": HttpStatusCodes.INTERNAL_SERVER_ERROR,
        "message": message,
        "data": {
            "timestamp": datetime.now().isoformat(),
            "error": error,
        },
    }
    return JSONResponse(
        status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
        content=fallback_response,
    )


def _create_critical_fallback_response() -> JSONResponse:
    """创建严重错误的回退响应"""
    return JSONResponse(
        status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
        content={
            "code": HttpStatusCodes.INTERNAL_SERVER_ERROR,
            "message": ServiceConstants.CRITICAL_ERROR_MESSAGE,
            "data": {
                "timestamp": datetime.now().isoformat(),
                "error": "异常处理器失败",
            },
        },
    )
