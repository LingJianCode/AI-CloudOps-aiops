#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: CORS中间件配置 - 处理跨域资源共享，支持浏览器端API访问
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("aiops.cors")


def setup_cors(app: FastAPI):
    try:
        # 配置CORS中间件
        # 在允许凭据的情况下不能使用通配 *，改为常见本地来源，避免浏览器报错
        allowed_origins = [
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

        logger.info("CORS中间件设置完成")

    except Exception as e:
        logger.error(f"CORS中间件设置失败: {str(e)}")