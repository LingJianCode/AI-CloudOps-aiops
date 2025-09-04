#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Core层LLM客户端接口定义与空实现
"""

from typing import Any, Dict, List, Optional, Protocol, Union


class LLMClient(Protocol):
    """LLM客户端接口，供Core层依赖注入使用"""

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        use_task_model: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        ...

    # 扩展能力：部分Core模块会直接调用这些高阶方法
    async def analyze_k8s_problem(
        self,
        deployment_yaml: str,
        error_event: str,
        additional_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    async def generate_fix_explanation(
        self, deployment: str, actions_taken: List[str], success: bool
    ) -> Optional[str]:
        ...

    async def is_healthy(self) -> bool:
        ...

    # 属性：用于日志输出
    provider: str
    model: str


class NullLLMClient:
    """空实现：返回空字符串，触发Core层的fallback路径"""

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        use_task_model: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        return ""

    async def analyze_k8s_problem(
        self,
        deployment_yaml: str,
        error_event: str,
        additional_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return None

    async def generate_fix_explanation(
        self, deployment: str, actions_taken: List[str], success: bool
    ) -> Optional[str]:
        return None

    async def is_healthy(self) -> bool:
        return False

    provider: str = "null"
    model: str = "null"


