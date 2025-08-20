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
    """设置CORS中间件"""
    try:
        # 配置CORS中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 允许所有源
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],  # 允许所有头部
        )

        logger.info("CORS中间件设置完成")

    except Exception as e:
        logger.error(f"CORS中间件设置失败: {str(e)}")