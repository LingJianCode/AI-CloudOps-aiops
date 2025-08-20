#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 启动服务 - 负责应用启动时的服务预热和初始化逻辑
"""

import asyncio
import time
from typing import Dict, Any

from .base import BaseService
from ..common.constants import ServiceConstants



class StartupService(BaseService):
    """
    启动服务 - 管理应用启动流程和服务预热
    """
    
    def __init__(self) -> None:
        super().__init__("startup")
        self.start_time = time.time()
    
    async def _do_initialize(self) -> None:
        """初始化启动服务"""
        self.logger.info("启动服务初始化完成")
    
    async def _do_health_check(self) -> bool:
        """启动服务健康检查"""
        return True
    
    async def warmup_services(self) -> Dict[str, Any]:
        """
        预热关键服务
        
        Returns:
            预热结果报告
        """
        self.logger.info("开始预热关键服务...")
        
        warmup_results = {
            "started_at": time.time(),
            "services": {},
            "total_time": 0,
            "success": True
        }
        
        # 预热智能助手服务
        assistant_result = await self._warmup_assistant()
        warmup_results["services"]["assistant"] = assistant_result
        
        # 可以添加其他服务的预热逻辑
        # warmup_results["services"]["prediction"] = await self._warmup_prediction()
        # warmup_results["services"]["rca"] = await self._warmup_rca()
        
        warmup_results["total_time"] = time.time() - warmup_results["started_at"]
        warmup_results["success"] = all(
            service.get("success", False) 
            for service in warmup_results["services"].values()
        )
        
        self.logger.info(f"服务预热完成，总耗时: {warmup_results['total_time']:.2f}秒")
        return warmup_results
    
    async def _warmup_assistant(self) -> Dict[str, Any]:
        """
        预热智能助手服务
        
        Returns:
            预热结果
        """
        result = {
            "service": "assistant",
            "success": False,
            "error": None,
            "duration": 0
        }
        
        try:
            start_time = time.time()
            
            # 动态导入避免循环依赖
            from ..core.agents.enterprise_assistant import get_enterprise_assistant
            
            self.logger.info("正在初始化企业级智能助手...")
            
            # 使用超时机制
            agent = None
            max_wait_time = getattr(ServiceConstants, 'DEFAULT_WARMUP_TIMEOUT', 60)
            retry_count = 0
            max_retries = max_wait_time // ServiceConstants.DEFAULT_RETRY_DELAY
            
            while agent is None and retry_count < max_retries:
                try:
                    agent = await get_enterprise_assistant()
                    break
                except Exception as e:
                    self.logger.debug(f"企业级智能助手初始化中，第{retry_count + 1}次重试: {str(e)}")
                    await asyncio.sleep(ServiceConstants.DEFAULT_RETRY_DELAY)
                    retry_count += 1
            
            result["duration"] = time.time() - start_time
            
            if agent is not None:
                result["success"] = True
                self.logger.info(f"智能助手预热完成，耗时: {result['duration']:.2f}秒")
            else:
                result["error"] = f"预热超时({max_wait_time}秒)"
                self.logger.warning(f"智能助手预热超时，将在首次使用时初始化")
                
        except Exception as e:
            result["duration"] = time.time() - start_time if 'start_time' in locals() else 0
            result["error"] = str(e)
            self.logger.warning(f"智能助手预热失败: {str(e)}，将在首次使用时初始化")
        
        return result
    
    def get_uptime(self) -> float:
        """
        获取应用运行时间
        
        Returns:
            运行时间（秒）
        """
        return time.time() - self.start_time
    
    def get_startup_info(self) -> Dict[str, Any]:
        """
        获取启动信息
        
        Returns:
            启动信息字典
        """
        return {
            "start_time": self.start_time,
            "uptime": self.get_uptime(),
            "initialized": self.is_initialized(),
            "healthy": self.is_healthy()
        }
