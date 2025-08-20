#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 主应用模块 - 提供FastAPI应用的创建和初始化功能
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.api.middleware import register_middleware
from app.api.routes import register_routes
from app.config.logging import setup_logging
from app.config.settings import config

# 预热机制导入
from app.core.agents.assistant_manager import init_assistant_in_background

start_time = time.time()


async def warm_up_services():
    """
    预热关键服务，减少首次请求延迟
    """
    logger = logging.getLogger("aiops")
    logger.info("开始预热关键服务...")
    
    try:
        # 同步初始化智能助手，确保在启动完成前完成初始化
        from app.core.agents.assistant_manager import get_assistant_agent
        logger.info("正在初始化智能助手...")
        
        # 设置一个合理的超时时间
        start_time = time.time()
        max_wait_time = 60  # 最多等待60秒
        
        agent = None
        while agent is None and (time.time() - start_time) < max_wait_time:
            agent = get_assistant_agent()
            if agent is None:
                logger.info("智能助手初始化中，等待2秒后重试...")
                await asyncio.sleep(2)
        
        if agent is not None:
            init_time = time.time() - start_time
            logger.info(f"智能助手预热完成，耗时: {init_time:.2f}秒")
        else:
            logger.warning(f"智能助手预热超时({max_wait_time}秒)，将在首次使用时初始化")
            
    except Exception as e:
        logger.warning(f"智能助手预热失败: {str(e)}，将在首次使用时初始化")
    
    logger.info("服务预热完成")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用生命周期管理
    """
    # 启动时执行
    logger = logging.getLogger("aiops")
    logger.info("=" * 50)
    logger.info("AIOps平台启动中...")
    logger.info(f"调试模式: {config.debug}")
    logger.info(f"日志级别: {config.log_level}")
    logger.info("=" * 50)
    
    # 启动预热机制
    await warm_up_services()
    
    # 记录启动信息
    startup_time = time.time() - start_time
    logger.info(f"AIOps平台启动完成，耗时: {startup_time:.2f}秒")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info("可用的API端点:")
    logger.info("  - GET  /api/v1/health        - 健康检查")
    logger.info("  - GET  /api/v1/predict       - 负载预测")
    logger.info("  - POST /api/v1/rca           - 根因分析")
    logger.info("  - POST /api/v1/autofix       - 自动修复")
    logger.info("  - POST /api/v1/assistant/query - 智能小助手")
    
    yield
    
    # 关闭时执行
    total_time = time.time() - start_time
    logger.info(f"AIOps平台运行总时长: {total_time:.2f}秒")
    logger.info("AIOps平台已关闭")


def create_app() -> FastAPI:
    """
    创建FastAPI应用实例

    Returns:
        FastAPI: 配置好的FastAPI应用实例
    """
    # 创建FastAPI应用
    app = FastAPI(
        title="AIOps Platform",
        description="智能云原生运维平台",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
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
            log_level="info" if not config.debug else "debug",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise