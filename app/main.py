#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 主应用模块 - 提供Flask应用的创建和初始化功能
"""

import os
import sys
import logging
import time
from flask import Flask

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.config.settings import config
from app.config.logging import setup_logging
from app.api.routes import register_routes
from app.api.middleware import register_middleware

start_time = time.time()

def create_app():
    """创建Flask应用实例"""
    app = Flask(__name__)

    # 设置日志系统
    setup_logging(app)
    log = logging.getLogger("aiops")
    log.info("=" * 50)
    log.info("AIOps平台启动中...")
    log.info(f"调试模式: {config.debug}")
    log.info(f"日志级别: {config.log_level}")
    log.info("=" * 50)

    # 注册中间件
    try:
        register_middleware(app)
        log.info("中间件注册完成")
    except Exception as e:
        log.error(f"中间件注册失败: {str(e)}")
        log.warning("将继续启动，但部分中间件功能可能不可用")

    # 注册路由
    try:
        register_routes(app)
        log.info("路由注册完成")
    except Exception as e:
        log.error(f"路由注册失败: {str(e)}")
        log.warning("将继续启动，但部分路由功能可能不可用")


    def log_startup_info():
        """记录服务启动信息"""
        startup_time = time.time() - start_time
        log.info(f"AIOps平台启动完成，耗时: {startup_time:.2f}秒")
        log.info(f"服务地址: http://{config.host}:{config.port}")

    # Flask 2.2+ 兼容性处理
    app_started = False

    @app.before_request
    def _log_startup_wrapper():
        """首次请求前记录启动信息"""
        nonlocal app_started
        if not app_started:
            log_startup_info()
            app_started = True

    @app.teardown_appcontext
    def cleanup(error):
        """应用上下文清理错误处理"""
        if error:
            log.error(f"应用上下文清理时发生错误: {str(error)}")

    return app

server = create_app()

if __name__ == "__main__":
    """直接运行时的主入口"""
    logger = logging.getLogger("aiops")

    try:
        logger.info(f"在 {config.host}:{config.port} 启动Flask服务器")
        server.run(
            host=config.host,
            port=config.port,
            debug=config.debug,
            threaded=True # 开启多线程
        )
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise
    finally:
        total_time = time.time() - start_time
        logger.info(f"AIOps平台运行总时长: {total_time:.2f}秒")
        logger.info("AIOps平台已关闭")
