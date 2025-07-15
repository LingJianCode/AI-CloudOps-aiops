#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI助手工具函数模块 - 核心层
"""

import asyncio
import logging
import re
import threading
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# 创建日志器
logger = logging.getLogger("aiops.core.assistant.utils")


def sanitize_for_json(text: Any) -> Any:
    """
    清理文本中的控制字符，确保JSON安全

    Args:
        text: 需要清理的文本或其他类型的数据

    Returns:
        Any: 清理后的文本或原始数据（如果不是字符串）
    """
    if not isinstance(text, str):
        return text

    # 替换换行符为空格，而不是转义序列，避免在JSON响应中出现真实换行符
    text = text.replace("\n", " ").replace("\r", " ")
    # 替换多个连续空格为单个空格
    text = re.sub(r"\s+", " ", text)
    # 移除其他控制字符
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text.strip()


def sanitize_result_data(data: Any) -> Any:
    """
    递归清理结果数据中的所有字符串字段

    Args:
        data: 需要清理的数据，可以是字典、列表或基本类型

    Returns:
        Any: 清理后的数据结构
    """
    if isinstance(data, dict):
        return {k: sanitize_result_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_result_data(item) for item in data]
    elif isinstance(data, str):
        return sanitize_for_json(data)
    else:
        return data


def safe_async_run(coroutine: Union[Coroutine, Any]) -> Any:
    """
    安全地运行异步函数，处理不同环境下的运行方式

    Args:
        coroutine: 异步协程对象或其他值

    Returns:
        Any: 协程执行的结果或原始值（如果不是协程）

    Raises:
        Exception: 执行过程中的任何异常
    """
    try:
        # 检查是否已有事件循环在运行
        try:
            loop = asyncio.get_running_loop()
            # 如果有运行中的事件循环，使用 asyncio.create_task 或直接同步调用
            # 对于Flask同步环境，我们需要在新线程中运行事件循环
            import concurrent.futures
            import threading

            result = None
            exception = None

            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # 确保所有协程都被等待
                        if asyncio.iscoroutine(coroutine):
                            result = new_loop.run_until_complete(coroutine)
                        else:
                            # 如果不是协程对象，直接返回
                            result = coroutine
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()

            if exception:
                raise exception
            return result

        except RuntimeError:
            # 没有运行中的事件循环，可以安全创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 确保所有协程都被等待
                if asyncio.iscoroutine(coroutine):
                    return loop.run_until_complete(coroutine)
                else:
                    # 如果不是协程对象，直接返回
                    return coroutine
            finally:
                loop.close()

    except Exception as e:
        logger.error(f"执行异步函数失败: {str(e)}")
        raise e


def build_context_with_history(session: Optional[Any]) -> str:
    """
    构建包含对话历史的上下文

    Args:
        session: 会话对象，包含历史记录

    Returns:
        str: 格式化的历史对话上下文字符串
    """
    if not session or not session.history:
        return ""

    history_text = "### 对话历史:\n"
    for msg in session.history[-3:]:  # 只包含最近3条
        if msg.get("role") == "user":
            history_text += f"用户: {msg.get('content')}\n"
        elif msg.get("role") == "assistant":
            history_text += f"助手: {msg.get('content')}\n"

    return history_text
