#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 主应用程序入口
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.api.middleware import register_middleware
from app.api.routes import register_routes
from app.common.constants import AppConstants
from app.config.logging import setup_logging
from app.config.settings import config
from app.services.startup import StartupService

# 全局启动服务实例
startup_service = StartupService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用生命周期管理
    """
    # 启动时执行
    logger = logging.getLogger("aiops")
    logger.info("=" * 50)
    logger.info(f"{AppConstants.APP_NAME} 启动中...")
    logger.info(f"版本: {AppConstants.APP_VERSION}")
    logger.info(f"调试模式: {config.debug}")
    logger.info(f"日志级别: {config.log_level}")
    logger.info("=" * 50)

    # 初始化启动服务
    await startup_service.initialize()

    # 启动预热机制
    warmup_results = await startup_service.warmup_services()

    # 记录启动信息
    startup_time = startup_service.get_uptime()
    logger.info(f"{AppConstants.APP_NAME} 启动完成，耗时: {startup_time:.2f}秒")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info("主要API端点:")
    logger.info(f"  - GET  {AppConstants.API_VERSION_V1}/health        - 健康检查")
    logger.info(f"  - GET  {AppConstants.API_VERSION_V1}/predict       - 负载预测")
    logger.info(f"  - POST {AppConstants.API_VERSION_V1}/rca           - 根因分析")
    logger.info(f"  - POST {AppConstants.API_VERSION_V1}/autofix       - 自动修复")
    logger.info(f"  - POST {AppConstants.API_VERSION_V1}/assistant/query - 智能小助手")

    if not warmup_results["success"]:
        logger.warning("部分服务预热失败，系统仍可正常使用")

    yield

    # 关闭时执行
    total_time = startup_service.get_uptime()
    logger.info(f"{AppConstants.APP_NAME} 运行总时长: {total_time:.2f}秒")
    logger.info(f"{AppConstants.APP_NAME} 已关闭")


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例

    Returns:
        FastAPI: 配置好的FastAPI应用实例
    """
    # 创建FastAPI应用
    app = FastAPI(
        title=AppConstants.APP_NAME,
        description=AppConstants.APP_DESCRIPTION,
        version=AppConstants.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 设置日志系统（用临时app对象）
    setup_logging(app)

    # 注册中间件
    try:
        register_middleware(app)
        logger = logging.getLogger("aiops")
        logger.info("中间件注册完成")
    except Exception as e:
        logger = logging.getLogger("aiops")
        logger.error(f"中间件注册失败: {str(e)}")
        logger.warning("将继续启动，但部分中间件功能可能不可用")

    # 注册路由
    try:
        register_routes(app)
        logger = logging.getLogger("aiops")
        logger.info("路由注册完成")
    except Exception as e:
        logger = logging.getLogger("aiops")
        logger.error(f"路由注册失败: {str(e)}")
        logger.warning("将继续启动，但部分路由功能可能不可用")

    return app


# 创建应用实例
app = create_app()

if __name__ == "__main__":
    """直接运行时的主入口"""
    import uvicorn

    logger = logging.getLogger("aiops")

    try:
        logger.info(f"在 {config.host}:{config.port} 启动FastAPI服务器")
        uvicorn.run(
            "app.main:app",
            host=config.host,
            port=config.port,
            reload=config.debug,
            reload_dirs=["app", "config"] if config.debug else None,  # 指定监控的目录
            reload_excludes=(
                ["logs", "data", "__pycache__", "*.pyc"] if config.debug else None
            ),  # 排除不需要监控的目录
            log_level="info" if not config.debug else "debug",
            access_log=True,
            reload_delay=0.25 if config.debug else None,  # 减少重载延迟
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise
