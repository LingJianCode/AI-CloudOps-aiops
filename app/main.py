"""
AIOps平台主应用入口

这是整个AI-CloudOps平台的主入口文件，负责：
1. 初始化Flask应用实例
2. 配置日志系统
3. 注册中间件和路由
4. 启动WebSocket服务
5. 处理应用生命周期事件

架构说明：
- 使用Flask作为Web框架
- 集成WebSocket支持实时通信
- 提供RESTful API接口
- 支持多种运维功能：健康检查、负载预测、根因分析、自动修复等
- 内置智能小助手提供运维支持

主要功能模块：
- 健康检查：监控系统状态
- 负载预测：基于历史数据预测系统负载
- 根因分析：自动分析系统故障原因
- 自动修复：智能化的故障自动修复
- 智能小助手：基于RAG的运维知识问答
"""

import os
import sys
import logging
import time
from flask import Flask

# 添加项目根目录到系统路径
# 这确保了无论从哪个目录启动程序，都能正确导入项目模块
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入应用配置和组件
from app.config.settings import config  # 全局配置管理
from app.config.logging import setup_logging  # 日志配置
from app.api.routes import register_routes  # API路由注册
from app.api.middleware import register_middleware  # 中间件注册

# 记录启动时间，用于计算启动耗时
start_time = time.time()

def create_app():
    """
    创建Flask应用实例
    
    这个函数负责创建和配置Flask应用，包括：
    1. 设置日志系统
    2. 注册中间件（CORS、错误处理等）
    3. 注册API路由
    4. 初始化WebSocket服务
    5. 设置应用生命周期回调
    
    Returns:
        Flask: 配置完成的Flask应用实例
    """
    # 创建Flask应用实例
    app = Flask(__name__)
    
    # 设置日志系统
    # 根据配置文件设置日志级别、格式和输出方式
    setup_logging(app)
    
    # 获取应用日志器，用于记录应用运行日志
    logger = logging.getLogger("aiops")
    logger.info("=" * 50)
    logger.info("AIOps平台启动中...")
    logger.info(f"调试模式: {config.debug}")
    logger.info(f"日志级别: {config.log_level}")
    logger.info("=" * 50)
    
    # 注册中间件
    # 包括CORS处理、错误处理、请求日志等
    try:
        register_middleware(app)
        logger.info("中间件注册完成")
    except Exception as e:
        logger.error(f"中间件注册失败: {str(e)}")
        logger.warning("将继续启动，但部分中间件功能可能不可用")
    
    # 注册路由
    # 注册所有API端点，包括健康检查、预测、分析等功能
    try:
        register_routes(app)
        logger.info("路由注册完成")
    except Exception as e:
        logger.error(f"路由注册失败: {str(e)}")
        logger.warning("将继续启动，但部分路由功能可能不可用")
        
    # 初始化WebSocket
    # 为智能小助手提供实时通信支持
    try:
        from app.api.routes.assistant import init_websocket
        init_websocket(app)
        logger.info("WebSocket初始化完成")
    except ImportError as e:
        logger.warning(f"WebSocket模块导入失败: {str(e)}")
        logger.warning("将继续启动，但WebSocket功能不可用")
    except Exception as e:
        logger.error(f"WebSocket初始化失败: {str(e)}")
        logger.warning("将继续启动，但WebSocket功能不可用")
    
    # 定义启动信息函数
    def log_startup_info():
        """
        记录服务启动信息和可用的API端点
        """
        startup_time = time.time() - start_time
        logger.info(f"AIOps平台启动完成，耗时: {startup_time:.2f}秒")
        logger.info(f"服务地址: http://{config.host}:{config.port}")
        logger.info("可用的API端点:")
        logger.info("  - GET  /api/v1/health        - 健康检查")
        logger.info("  - GET  /api/v1/predict       - 负载预测")
        logger.info("  - POST /api/v1/rca           - 根因分析")
        logger.info("  - POST /api/v1/autofix       - 自动修复")
        logger.info("  - POST /api/v1/assistant/query - 智能小助手")
        logger.info("  - WS   /api/v1/assistant/stream - 流式智能小助手")
    
    # 替代 before_first_request 的解决方案
    # Flask 2.2+ 版本移除了 before_first_request 装饰器
    # 使用 before_request 和全局变量来实现相同功能
    app_started = False
    
    @app.before_request
    def _log_startup_wrapper():
        """
        在第一个请求之前记录启动信息
        """
        nonlocal app_started
        if not app_started:
            log_startup_info()
            app_started = True
    
    # 添加关闭处理
    @app.teardown_appcontext
    def cleanup(error):
        """
        应用上下文清理时的错误处理
        
        Args:
            error: 如果有错误发生，这里会包含错误信息
        """
        if error:
            logger = logging.getLogger("aiops")
            logger.error(f"应用上下文清理时发生错误: {str(error)}")
    
    return app

# 创建应用实例
# 这个实例将被WSGI服务器使用
app = create_app()

if __name__ == "__main__":
    """
    直接运行时的主入口
    
    当直接运行此文件时（而不是通过WSGI服务器），
    将使用Flask的内置开发服务器启动应用
    """
    logger = logging.getLogger("aiops")
    
    try:
        logger.info(f"在 {config.host}:{config.port} 启动Flask服务器")
        # 启动Flask开发服务器
        app.run(
            host=config.host,        # 绑定地址
            port=config.port,        # 端口号
            debug=config.debug,      # 是否开启调试模式
            threaded=True           # 启用多线程处理请求
        )
    except KeyboardInterrupt:
        # 处理Ctrl+C中断信号
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        # 处理其他启动异常
        logger.error(f"服务启动失败: {str(e)}")
        raise
    finally:
        # 记录运行时长和关闭信息
        total_time = time.time() - start_time
        logger.info(f"AIOps平台运行总时长: {total_time:.2f}秒")
        logger.info("AIOps平台已关闭")