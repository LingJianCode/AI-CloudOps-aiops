#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 基础服务抽象类
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from app.common.constants import ServiceConstants
from app.common.exceptions import ServiceUnavailableError


class BaseService(ABC):
    """
    业务服务基类

    定义所有业务服务必须实现的接口和提供公共功能
    """

    def __init__(self, service_name: str) -> None:
        """
        初始化服务

        Args:
            service_name: 服务名称
        """
        self.service_name = service_name
        self.logger = logging.getLogger(f"aiops.services.{service_name}")
        self._initialized = False
        self._last_health_check = None
        self._health_status = False

    async def initialize(self) -> None:
        """
        异步初始化服务

        子类可以重写此方法来实现具体的初始化逻辑
        """
        self.logger.info(f"正在初始化服务: {self.service_name}")

        try:
            await self._do_initialize()
            self._initialized = True
            self.logger.info(f"服务 {self.service_name} 初始化完成")
        except Exception as e:
            self.logger.error(f"服务 {self.service_name} 初始化失败: {str(e)}")
            raise ServiceUnavailableError(
                service_name=self.service_name,
                details={"error": str(e), "phase": "initialization"},
            )

    @abstractmethod
    async def _do_initialize(self) -> None:
        """
        子类实现具体的初始化逻辑
        """
        pass

    async def health_check(self) -> bool:
        """
        异步健康检查

        Returns:
            健康状态，True表示健康
        """
        try:
            # 缓存健康检查结果，避免频繁检查
            now = datetime.now()
            if self._last_health_check and (now - self._last_health_check).seconds < 30:
                return self._health_status

            self._health_status = await self._do_health_check()
            self._last_health_check = now

            return self._health_status

        except Exception as e:
            self.logger.warning(f"服务 {self.service_name} 健康检查失败: {str(e)}")
            self._health_status = False
            return False

    @abstractmethod
    async def _do_health_check(self) -> bool:
        """
        子类实现具体的健康检查逻辑

        Returns:
            健康状态
        """
        pass

    def is_healthy(self) -> bool:
        """
        同步获取健康状态（使用缓存的结果）

        Returns:
            健康状态
        """
        return self._health_status

    def is_initialized(self) -> bool:
        """
        检查服务是否已初始化

        Returns:
            初始化状态
        """
        return self._initialized

    async def execute_with_timeout(
        self,
        operation,
        timeout: float = ServiceConstants.DEFAULT_SERVICE_TIMEOUT,
        operation_name: str = "operation",
    ) -> Any:
        """
        执行带超时保护的操作

        Args:
            operation: 要执行的操作（可以是协程或可调用对象）
            timeout: 超时时间（秒）
            operation_name: 操作名称，用于日志

        Returns:
            操作结果

        Raises:
            ServiceUnavailableError: 操作超时或失败
        """
        try:
            if asyncio.iscoroutine(operation):
                result = await asyncio.wait_for(operation, timeout=timeout)
            elif callable(operation):
                result = await asyncio.wait_for(
                    asyncio.to_thread(operation), timeout=timeout
                )
            else:
                raise ValueError("operation must be a coroutine or callable")

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"服务 {self.service_name} 执行 {operation_name} 超时")
            raise ServiceUnavailableError(
                service_name=self.service_name,
                details={"error": f"操作 {operation_name} 超时", "timeout": timeout},
            )
        except Exception as e:
            self.logger.error(
                f"服务 {self.service_name} 执行 {operation_name} 失败: {str(e)}"
            )
            raise ServiceUnavailableError(
                service_name=self.service_name,
                details={"error": str(e), "operation": operation_name},
            )

    def get_service_info(self) -> Dict[str, Any]:
        """
        获取服务信息

        Returns:
            服务信息字典
        """
        return {
            "name": self.service_name,
            "initialized": self._initialized,
            "healthy": self._health_status,
            "last_health_check": (
                self._last_health_check.isoformat() if self._last_health_check else None
            ),
        }

    def _ensure_initialized(self) -> None:
        """
        确保服务已初始化

        Raises:
            ServiceUnavailableError: 服务未初始化
        """
        if not self._initialized:
            raise ServiceUnavailableError(
                service_name=self.service_name, details={"error": "服务未初始化"}
            )


class HealthCheckMixin:
    """
    健康检查混合类

    为服务提供标准的健康检查功能
    """

    @staticmethod
    def create_health_response(
        service_name: str,
        is_healthy: bool,
        components: Optional[Dict[str, bool]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        创建标准的健康检查响应

        Args:
            service_name: 服务名称
            is_healthy: 健康状态
            components: 组件健康状态
            details: 详细信息

        Returns:
            健康检查响应字典
        """
        response = {
            "service": service_name,
            "status": (
                ServiceConstants.STATUS_HEALTHY
                if is_healthy
                else ServiceConstants.STATUS_UNHEALTHY
            ),
            "healthy": is_healthy,
            "timestamp": datetime.now().isoformat(),
        }

        if components:
            response["components"] = components

        if details:
            response.update(details)

        return response
