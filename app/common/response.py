#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 响应处理模块
"""

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .constants import HttpStatusCodes
from .exceptions import AIOpsException

logger = logging.getLogger("aiops.response")


class ResponseWrapper:
    """统一响应包装器"""

    @staticmethod
    def success(
        data: Any = None, message: str = "success", code: int = 0
    ) -> Dict[str, Any]:
        return {"code": code, "message": message, "data": data}

    @staticmethod
    def success_list(
        items: list,
        message: str = "success",
        code: int = 0,
        total: Optional[int] = None,
    ) -> Dict[str, Any]:
        """创建列表数据成功响应"""
        if total is None:
            total = len(items) if items else 0

        return {
            "code": code,
            "message": message,
            "data": {"items": items, "total": total},
        }

    @staticmethod
    def error(
        message: str,
        code: int = 1,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建错误响应"""
        response_data = {"code": code, "message": message}

        if details:
            response_data["details"] = details

        if error_code:
            response_data["error_code"] = error_code

        return response_data


async def handle_aiops_exception(request: Request, exc: AIOpsException) -> JSONResponse:
    """处理业务异常"""
    logger.error(
        f"业务异常: {exc.message}, 错误码: {exc.error_code}, 详情: {exc.details}"
    )

    response_data = ResponseWrapper.error(
        message=exc.message, code=1, details=exc.details, error_code=exc.error_code
    )

    # 添加请求上下文信息
    if response_data.get("details"):
        response_data["details"].update(
            {"path": str(request.url.path), "method": request.method}
        )
    else:
        response_data["details"] = {
            "path": str(request.url.path),
            "method": request.method,
        }

    # 根据异常类型返回合适的HTTP状态码
    http_status = _get_http_status_for_exception(exc)

    return JSONResponse(status_code=http_status, content=response_data)


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """处理HTTP异常"""
    logger.error(f"HTTP异常: {exc.detail}, 状态码: {exc.status_code}")

    response_data = ResponseWrapper.error(
        message=str(exc.detail),
        code=exc.status_code,
        details={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": exc.status_code,
        },
    )

    return JSONResponse(status_code=exc.status_code, content=response_data)


async def handle_general_exception(request: Request, exc: Exception) -> JSONResponse:
    """处理通用异常"""
    logger.error(f"未处理异常: {str(exc)}", exc_info=True)

    response_data = ResponseWrapper.error(
        message="服务器内部错误",
        code=HttpStatusCodes.INTERNAL_SERVER_ERROR,
        details={
            "path": str(request.url.path),
            "method": request.method,
            "status_code": HttpStatusCodes.INTERNAL_SERVER_ERROR,
        },
    )

    return JSONResponse(
        status_code=HttpStatusCodes.INTERNAL_SERVER_ERROR, content=response_data
    )


def _get_http_status_for_exception(exc: AIOpsException) -> int:
    """根据业务异常类型返回HTTP状态码"""
    from .exceptions import (
        ConfigurationError,
        ExternalServiceError,
        ServiceUnavailableError,
        ValidationError,
    )

    if isinstance(exc, ServiceUnavailableError):
        return HttpStatusCodes.SERVICE_UNAVAILABLE
    elif isinstance(exc, ValidationError):
        return HttpStatusCodes.BAD_REQUEST
    elif isinstance(exc, ConfigurationError):
        return HttpStatusCodes.INTERNAL_SERVER_ERROR
    elif isinstance(exc, ExternalServiceError):
        return HttpStatusCodes.BAD_GATEWAY
    else:
        return HttpStatusCodes.INTERNAL_SERVER_ERROR
