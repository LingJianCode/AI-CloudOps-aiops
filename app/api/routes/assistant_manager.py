#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能小助手单例管理
"""

import threading
import logging
from app.api.routes.assistant_utils import safe_async_run

# 创建日志器
logger = logging.getLogger("aiops.api.assistant.manager")

# 创建助手代理全局实例
_assistant_agent = None
_init_lock = threading.Lock()
_is_initializing = False

def get_assistant_agent():
    """获取助手代理单例实例，采用懒加载+锁机制优化初始化性能"""
    global _assistant_agent, _is_initializing
    
    if _assistant_agent is not None:
        return _assistant_agent
        
    # 使用锁避免多线程重复初始化
    with _init_lock:
        if _is_initializing:
            # 如果正在初始化中，等待一小段时间后再检查
            logger.info("另一个线程正在初始化小助手，等待...")
            import time
            for _ in range(10):  # 最多等待5秒
                time.sleep(0.5)
                if _assistant_agent is not None:
                    return _assistant_agent
        
        # 标记为正在初始化
        _is_initializing = True
        
        try:
            logger.info("初始化智能小助手代理...")
            from app.core.agents.assistant import AssistantAgent
            _assistant_agent = AssistantAgent()
            logger.info("智能小助手代理初始化完成")
        except Exception as e:
            logger.error(f"初始化智能小助手代理失败: {str(e)}")
            _is_initializing = False
            return None
            
        _is_initializing = False
        return _assistant_agent

# 提前在后台线程初始化小助手
def init_assistant_in_background():
    """在后台线程中初始化小助手，避免首次调用时的延迟"""
    def _init_thread():
        try:
            logger.info("开始在后台预初始化智能小助手...")
            get_assistant_agent()
            logger.info("小助手后台预初始化完成")
        except Exception as e:
            logger.error(f"后台初始化小助手失败: {str(e)}")
    
    threading.Thread(target=_init_thread, daemon=True).start()

# 应用启动时自动初始化
init_assistant_in_background()

def reinitialize_assistant():
    """完全重置助手实例，重建向量数据库，重新载入知识库"""
    global _assistant_agent, _is_initializing
    
    logger.info("开始执行助手完全重置...")
    
    # 使用锁保证线程安全
    with _init_lock:
        # 关闭当前实例（如果存在）
        if _assistant_agent is not None:
            try:
                logger.info("关闭当前小助手实例...")
                safe_async_run(_assistant_agent.shutdown())
                logger.info("当前小助手实例已关闭")
            except Exception as e:
                logger.error(f"关闭当前小助手实例时出错: {str(e)}")
        
        # 重置实例
        _assistant_agent = None
        _is_initializing = False
        
        # 清理缓存和向量存储（防止旧数据影响）
        try:
            logger.info("清理残留的缓存和向量数据...")
            from app.core.cache.redis_cache_manager import RedisCacheManager
            from app.config.settings import config
            
            # 创建临时缓存管理器来清理数据
            redis_config = {
                'host': config.redis.host,
                'port': config.redis.port,
                'db': config.redis.db + 1,
                'password': config.redis.password,
                'connection_timeout': config.redis.connection_timeout,
                'socket_timeout': config.redis.socket_timeout,
                'max_connections': config.redis.max_connections,
                'decode_responses': config.redis.decode_responses
            }
            
            temp_cache_manager = RedisCacheManager(
                redis_config=redis_config,
                cache_prefix="aiops_assistant_cache:",
                default_ttl=3600,
                max_cache_size=1000,
                enable_compression=True
            )
            temp_cache_manager.clear_all()
            temp_cache_manager.shutdown()
            logger.info("缓存数据清理完成")
            
        except Exception as e:
            logger.warning(f"清理缓存数据时出现警告: {str(e)}")
        
        # 重新初始化小助手
        return get_assistant_agent() 