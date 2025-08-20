#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查服务 - 提供系统健康监控和状态检查的业务逻辑
"""

import logging
from typing import Dict, Any

from .base import BaseService, HealthCheckMixin
from ..api.routes.health_manager import health_manager


logger = logging.getLogger("aiops.services.health")


class HealthService(BaseService, HealthCheckMixin):
    """
    健康检查服务 - 管理系统整体健康状态
    """
    
    def __init__(self) -> None:
        super().__init__("health")
    
    async def _do_initialize(self) -> None:
        """初始化健康检查服务"""
        self.logger.info("健康检查服务初始化完成")
    
    async def _do_health_check(self) -> bool:
        """健康检查服务自身的健康检查"""
        return True
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """
        获取系统整体健康状态
        
        Returns:
            系统健康状态字典
        """
        self._ensure_initialized()
        
        try:
            health_data = health_manager.get_overall_health()
            return health_data
        except Exception as e:
            self.logger.error(f"获取整体健康状态失败: {str(e)}")
            raise
    
    async def get_components_health(self) -> Dict[str, Any]:
        """
        获取各组件健康状态
        
        Returns:
            组件健康状态字典
        """
        self._ensure_initialized()
        
        try:
            components_data = health_manager.get_components_health()
            return components_data
        except Exception as e:
            self.logger.error(f"获取组件健康状态失败: {str(e)}")
            raise
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        获取系统指标
        
        Returns:
            系统指标字典
        """
        self._ensure_initialized()
        
        try:
            metrics_data = health_manager.get_system_metrics()
            return metrics_data
        except Exception as e:
            self.logger.error(f"获取系统指标失败: {str(e)}")
            raise
    
    async def check_readiness(self) -> Dict[str, Any]:
        """
        检查系统就绪状态
        
        Returns:
            就绪状态字典
        """
        self._ensure_initialized()
        
        try:
            ready_status = health_manager.check_readiness()
            return ready_status
        except Exception as e:
            self.logger.error(f"检查就绪状态失败: {str(e)}")
            raise
    
    async def check_liveness(self) -> Dict[str, Any]:
        """
        检查系统存活状态
        
        Returns:
            存活状态字典
        """
        self._ensure_initialized()
        
        try:
            live_status = health_manager.check_liveness()
            return live_status
        except Exception as e:
            self.logger.error(f"检查存活状态失败: {str(e)}")
            raise
    
    async def check_startup(self) -> Dict[str, Any]:
        """
        检查系统启动状态
        
        Returns:
            启动状态字典
        """
        self._ensure_initialized()
        
        try:
            startup_status = health_manager.check_startup()
            return startup_status
        except Exception as e:
            self.logger.error(f"检查启动状态失败: {str(e)}")
            raise
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """
        检查依赖服务状态
        
        Returns:
            依赖服务状态字典
        """
        self._ensure_initialized()
        
        try:
            deps_status = health_manager.check_dependencies()
            return deps_status
        except Exception as e:
            self.logger.error(f"检查依赖服务状态失败: {str(e)}")
            raise
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """
        获取详细健康检查信息
        
        Returns:
            详细健康状态字典
        """
        self._ensure_initialized()
        
        try:
            detailed_data = {
                "overall": await self.get_overall_health(),
                "components": await self.get_components_health(), 
                "metrics": await self.get_system_metrics(),
                "dependencies": await self.check_dependencies(),
                "readiness": await self.check_readiness(),
                "liveness": await self.check_liveness(),
                "startup": await self.check_startup()
            }
            
            return detailed_data
        except Exception as e:
            self.logger.error(f"获取详细健康信息失败: {str(e)}")
            raise
