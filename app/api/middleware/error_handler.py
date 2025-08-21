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

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.response_models import APIResponse

logger = logging.getLogger("aiops.error_handler")


async def custom_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    try:
        logger.error(
            f"HTTP异常处理器被触发 - 状态码: {exc.status_code}, 详情: {exc.detail}"
        )
        logger.error(f"请求信息 - URL: {request.url}, Method: {request.method}")

        # 构建错误响应
        error_response = APIResponse(
            code=exc.status_code,
            message=str(exc.detail),
            data={
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "method": request.method,
                "status_code": exc.status_code,
            },
        )

        return JSONResponse(status_code=exc.status_code, content=error_response.dict())

    except Exception as e:
        logger.error(f"错误处理器本身出错: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")

        # 回退到基本错误响应
        fallback_response = {
            "code": 500,
            "message": "内部服务器错误",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "error": "错误处理器异常",
            },
        }

        return JSONResponse(status_code=500, content=fallback_response)


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    try:
        logger.error(f"验证异常: {str(exc)}")
        logger.error(f"请求信息 - URL: {request.url}, Method: {request.method}")

        error_response = APIResponse(
            code=400,
            message="请求参数验证失败",
            data={
                "detail": str(exc),
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "method": request.method,
            },
        )

        return JSONResponse(status_code=400, content=error_response.dict())

    except Exception as e:
        logger.error(f"验证异常处理器出错: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": "验证异常处理器错误",
                "data": {"error": str(e)},
            },
        )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    try:
        logger.error(f"未捕获的异常: {str(exc)}")
        logger.error(f"异常类型: {type(exc).__name__}")
        logger.error(f"请求信息 - URL: {request.url}, Method: {request.method}")
        logger.error(f"异常堆栈: {traceback.format_exc()}")

        error_response = APIResponse(
            code=500,
            message="内部服务器错误",
            data={
                "error": str(exc),
                "type": type(exc).__name__,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url.path),
                "method": request.method,
            },
        )

        return JSONResponse(status_code=500, content=error_response.dict())

    except Exception as handler_error:
        logger.critical(f"异常处理器本身出现严重错误: {str(handler_error)}")
        logger.critical(f"原异常: {str(exc)}")

        # 最基本的错误响应
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": "严重内部错误",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "error": "异常处理器失败",
                },
            },
        )


def setup_error_handlers(app: FastAPI):
    try:
        # 注册HTTP异常处理器
        app.add_exception_handler(HTTPException, custom_http_exception_handler)
        app.add_exception_handler(StarletteHTTPException, custom_http_exception_handler)

        # 注册验证异常处理器
        from pydantic import ValidationError

        app.add_exception_handler(ValidationError, validation_exception_handler)
        app.add_exception_handler(RequestValidationError, validation_exception_handler)

        # 注册通用异常处理器
        app.add_exception_handler(Exception, general_exception_handler)

        logger.info("错误处理器设置完成")

    except Exception as e:
        logger.error(f"错误处理器设置失败: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
