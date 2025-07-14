#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI助手初始化模块
"""

import threading
import logging
import time

# 创建日志器
logger = logging.getLogger("aiops.api.assistant.init")

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

def reset_assistant_agent():
    """重置助手代理全局实例，用于重新初始化"""
    global _assistant_agent, _is_initializing
    _assistant_agent = None
    _is_initializing = False
    logger.info("已重置智能小助手代理全局实例")

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

