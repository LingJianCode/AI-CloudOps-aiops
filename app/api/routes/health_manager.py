#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo  
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查管理器 - 统一健康检查逻辑和组件状态管理
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any

import psutil

from app.config.settings import config

from app.core.prediction.predictor import PredictionService
from app.services.kubernetes import KubernetesService
from app.services.llm import LLMService
from app.services.notification import NotificationService
from app.services.prometheus import PrometheusService

logger = logging.getLogger("aiops.health_manager")


class HealthManager:
    """健康检查管理器 - 统一管理所有组件的健康状态"""
    
    def __init__(self):
        self.start_time = time.time()
        self._service_cache = {}
        self._cache_ttl = 10  # 缓存10秒，加快响应速度
        self._last_check = {}
    
    def get_service(self, service_name: str):
        """获取服务实例"""
        services = {
            'prometheus': PrometheusService,
            'kubernetes': KubernetesService,
            'llm': LLMService,
            'notification': NotificationService,
            'prediction': PredictionService
        }
        
        if service_name not in self._service_cache:
            if service_name not in services:
                return None
            try:
                self._service_cache[service_name] = services[service_name]()
            except Exception as e:
                logger.error(f"创建服务实例失败 {service_name}: {str(e)}")
                return None
        
        return self._service_cache[service_name]
    
    def check_component_health(self, component: str) -> Dict[str, Any]:
        """检查单个组件健康状态"""
        current_time = time.time()
        
        # 检查缓存 - 优化缓存查询
        last_check = self._last_check.get(component)
        if last_check and current_time - last_check['time'] < self._cache_ttl:
            return last_check['result']
        
        try:
            service = self.get_service(component)
            if not service:
                result = {
                    "healthy": False,
                    "error": f"未知的组件: {component}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # 特殊处理 LLM 服务的异步健康检查
                if component == 'llm' and hasattr(service, 'is_healthy'):
                    try:
                        # 检查是否有运行中的事件循环
                        try:
                            asyncio.get_running_loop()
                            # 如果在事件循环中，使用 asyncio.create_task 但需要在线程中执行
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                # 在新线程中创建新的事件循环运行异步方法
                                future = executor.submit(
                                    lambda: asyncio.run(service.is_healthy())
                                )
                                is_healthy = future.result(timeout=config.rag.timeout)
                        except RuntimeError:
                            # 没有运行中的事件循环，可以直接使用 asyncio.run
                            is_healthy = asyncio.run(service.is_healthy())
                    except Exception as e:
                        logger.error(f"LLM健康检查失败: {str(e)}")
                        is_healthy = False
                else:
                    # 其他服务的同步健康检查
                    is_healthy = service.is_healthy()
                
                result = {
                    "healthy": is_healthy,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # 添加组件特定信息 - 优化异常处理
                if hasattr(service, 'get_service_info') and callable(service.get_service_info):
                    try:
                        service_info = service.get_service_info()
                        if isinstance(service_info, dict):
                            result.update(service_info)
                    except Exception as e:
                        logger.debug(f"获取组件信息失败: {str(e)}")
        
        except Exception as e:
            result = {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 更新缓存
        self._last_check[component] = {
            'time': current_time,
            'result': result
        }
        
        return result
    
    def check_all_components(self) -> Dict[str, Dict[str, Any]]:
        """检查所有组件健康状态"""
        components = ['prometheus', 'kubernetes', 'llm', 'notification', 'prediction']
        return {comp: self.check_component_health(comp) for comp in components}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统资源指标 - 优化性能"""
        try:
            # 减少CPU采样时间以提高响应速度
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # 只获取主要的磁盘信息
            try:
                disk = psutil.disk_usage("/")
                disk_info = {
                    "usage_percent": round((disk.used / disk.total) * 100, 2),
                    "free_gb": round(disk.free / (1024 * 1024 * 1024), 2),
                    "total_gb": round(disk.total / (1024 * 1024 * 1024), 2)
                }
            except Exception:
                disk_info = {"error": "无法获取磁盘信息"}
            
            # 获取网络信息
            try:
                network = psutil.net_io_counters()
                network_info = {
                    "bytes_sent_mb": round(network.bytes_sent / (1024 * 1024), 2),
                    "bytes_recv_mb": round(network.bytes_recv / (1024 * 1024), 2)
                }
            except Exception:
                network_info = {"error": "无法获取网络信息"}
            
            # 获取进程信息
            try:
                process = psutil.Process()
                process_memory = process.memory_info()
                process_info = {
                    "memory_mb": round(process_memory.rss / (1024 * 1024), 2),
                    "cpu_percent": process.cpu_percent(interval=0.1),
                    "threads": process.num_threads()
                }
            except Exception:
                process_info = {"error": "无法获取进程信息"}
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "available_gb": round(memory.available / (1024 * 1024 * 1024), 2),
                    "total_gb": round(memory.total / (1024 * 1024 * 1024), 2)
                },
                "disk": disk_info,
                "network": network_info,
                "process": process_info,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {str(e)}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    def get_uptime(self) -> float:
        """获取系统运行时间"""
        return time.time() - self.start_time
    
    def get_overall_health(self) -> Dict[str, Any]:
        """获取整体健康状态"""
        components = self.check_all_components()
        system_metrics = self.get_system_metrics()
        
        # 判断整体健康状态
        all_healthy = all(comp.get("healthy", False) for comp in components.values())
        
        # 核心组件健康检查
        critical_components = ['prometheus', 'prediction']
        critical_healthy = all(
            components.get(comp, {}).get("healthy", False) 
            for comp in critical_components
        )
        
        if all_healthy:
            status = "healthy"
        elif critical_healthy:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "healthy": all_healthy,
            "uptime": self.get_uptime(),
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {name: comp.get("healthy", False) for name, comp in components.items()},
            "system": system_metrics,
            "details": components
        }


# 全局健康检查管理器实例
health_manager = HealthManager()