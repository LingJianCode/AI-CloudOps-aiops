#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 请求上下文中间件 - 注入 request_id 并记录请求开始/结束
"""

import logging
from uuid import uuid4

from fastapi import FastAPI, Request

logger = logging.getLogger("aiops.api.middleware.request_context")


def _derive_request_id(request: Request) -> str:
    # 优先使用上游传入的 X-Request-ID，否则生成新的
    return request.headers.get("x-request-id") or uuid4().hex


async def _before_request(request: Request, request_id: str) -> None:
    # 将 request_id 放入 state，便于后续访问
    request.state.request_id = request_id
    try:
        # 也放入全局contextvars，配合通用logger使用
        from app.common.logger import request_id_ctx

        request_id_ctx.set(request_id)
    except Exception:
        pass
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        f"request.start {request.method} {request.url.path} client={client_host} request_id={request_id}"
    )


async def _after_request(request: Request, request_id: str) -> None:
    client_host = request.client.host if request.client else "unknown"
    logger.info(
        f"request.end {request.method} {request.url.path} client={client_host} request_id={request_id}"
    )
    try:
        # 清理contextvars中的request_id
        from app.common.logger import request_id_ctx

        request_id_ctx.set(None)
    except Exception:
        pass


def setup_request_context(app: FastAPI) -> None:
    """注册请求上下文中间件"""

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):  # type: ignore[misc]
        request_id = _derive_request_id(request)
        await _before_request(request, request_id)
        try:
            response = await call_next(request)
            return response
        finally:
            await _after_request(request, request_id)
