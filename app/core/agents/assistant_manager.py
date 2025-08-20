#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能小助手单例管理 - 提供全局访问点和初始化控制
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional, Union

from app.constants import (
    ASSISTANT_MANAGER_WAIT_CYCLES,
    ASSISTANT_MANAGER_SLEEP_INTERVAL,
    ASSISTANT_CACHE_DB_OFFSET,
    ASSISTANT_CACHE_DEFAULT_TTL,
    ASSISTANT_CACHE_MAX_SIZE
)

# 创建日志器
logger = logging.getLogger("aiops.core.assistant.manager")

# 创建助手代理全局实例
_assistant_agent = None
_init_lock = threading.Lock()
_is_initializing = False


def get_assistant_agent() -> Optional["AssistantAgent"]:
    """
    获取助手代理单例实例，采用懒加载+锁机制优化初始化性能

    Returns:
        Optional[AssistantAgent]: 助手代理实例或None（如果初始化失败）
    """
    global _assistant_agent, _is_initializing

    # 双重检查锁定模式，提高性能
    if _assistant_agent is not None:
        return _assistant_agent

    # 使用锁避免多线程重复初始化
    with _init_lock:
        # 再次检查，避免等待期间其他线程已经初始化
        if _assistant_agent is not None:
            return _assistant_agent

        if _is_initializing:
            # 如果正在初始化中，等待一小段时间后再检查
            logger.info("另一个线程正在初始化小助手，等待...")
            for _ in range(ASSISTANT_MANAGER_WAIT_CYCLES):  # 最多等待一定时间
                time.sleep(ASSISTANT_MANAGER_SLEEP_INTERVAL)
                if _assistant_agent is not None:
                    return _assistant_agent

        # 标记为正在初始化
        _is_initializing = True

        try:
            start_time = time.time()
            logger.info("初始化智能小助手代理...")
            from app.core.agents.assistant_optimized import AssistantAgent

            _assistant_agent = AssistantAgent()
            
            # 异步初始化（避免阻塞）
            from app.core.agents.assistant_utils import safe_sync_run
            init_success = safe_sync_run(_assistant_agent.initialize())
            
            if init_success:
                init_time = time.time() - start_time
                logger.info(f"智能小助手代理初始化完成，耗时: {init_time:.2f}秒")
            else:
                logger.error("智能小助手代理初始化失败")
                _assistant_agent = None
                
        except Exception as e:
            logger.error(f"初始化智能小助手代理失败: {str(e)}")
            _assistant_agent = None
        finally:
            _is_initializing = False

        return _assistant_agent


def reset_assistant_agent() -> None:
    """
    重置助手代理全局实例，用于重新初始化
    """
    global _assistant_agent, _is_initializing
    _assistant_agent = None
    _is_initializing = False
    logger.info("已重置智能小助手代理全局实例")


# 提前在后台线程初始化小助手
def init_assistant_in_background() -> None:
    """
    在后台线程中初始化小助手，避免首次调用时的延迟
    """

    def _init_thread() -> None:
        try:
            logger.info("开始在后台预初始化智能小助手...")
            get_assistant_agent()
            logger.info("小助手后台预初始化完成")
        except Exception as e:
            logger.error(f"后台初始化小助手失败: {str(e)}")

    threading.Thread(target=_init_thread, daemon=True).start()


def reinitialize_assistant() -> Optional["AssistantAgent"]:
    """
    完全重置助手实例，重建向量数据库，重新载入知识库

    Returns:
        Optional[AssistantAgent]: 重新初始化的助手代理实例或None（如果初始化失败）
    """
    global _assistant_agent, _is_initializing

    logger.info("开始执行助手完全重置...")

    # 使用锁保证线程安全
    with _init_lock:
        # 关闭当前实例（如果存在）
        if _assistant_agent is not None:
            try:
                logger.info("关闭当前小助手实例...")
                # 异步函数需要安全执行
                from app.core.agents.assistant_utils import safe_sync_run

                safe_sync_run(_assistant_agent.shutdown())
                logger.info("当前小助手实例已关闭")
            except Exception as e:
                logger.error(f"关闭当前小助手实例时出错: {str(e)}")

        # 重置实例
        _assistant_agent = None
        _is_initializing = False

        # 清理缓存和向量存储（防止旧数据影响）
        try:
            logger.info("清理残留的缓存和向量数据...")
            from app.config.settings import config
            from app.core.cache.redis_cache_manager import RedisCacheManager

            # 创建临时缓存管理器来清理数据
            redis_config = {
                "host": config.redis.host,
                "port": config.redis.port,
                "db": config.redis.db + ASSISTANT_CACHE_DB_OFFSET,
                "password": config.redis.password,
                "connection_timeout": config.redis.connection_timeout,
                "socket_timeout": config.redis.socket_timeout,
                "max_connections": config.redis.max_connections,
                "decode_responses": config.redis.decode_responses,
            }

            temp_cache_manager = RedisCacheManager(
                redis_config=redis_config,
                cache_prefix="aiops_assistant_cache:",
                default_ttl=ASSISTANT_CACHE_DEFAULT_TTL,
                max_cache_size=ASSISTANT_CACHE_MAX_SIZE,
                enable_compression=True,
            )
            temp_cache_manager.clear_all()
            temp_cache_manager.shutdown()
            logger.info("缓存数据清理完成")

        except Exception as e:
            logger.warning(f"清理缓存数据时出现警告: {str(e)}")

        # 重新初始化小助手
        return get_assistant_agent()
