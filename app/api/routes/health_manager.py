#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo  
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查管理器 - 统一健康检查逻辑和组件状态管理
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

import psutil

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
        
        # 检查缓存
        if (component in self._last_check and 
            current_time - self._last_check[component]['time'] < self._cache_ttl):
            return self._last_check[component]['result']
        
        try:
            service = self.get_service(component)
            if not service:
                result = {
                    "healthy": False,
                    "error": f"未知的组件: {component}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                is_healthy = service.is_healthy()
                result = {
                    "healthy": is_healthy,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # 添加组件特定信息
                if hasattr(service, 'get_service_info'):
                    try:
                        result.update(service.get_service_info())
                    except:
                        pass
        
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
        """获取系统资源指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()
            
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "available_bytes": memory.available,
                    "total_bytes": memory.total,
                    "used_bytes": memory.used
                },
                "disk": {
                    "usage_percent": round((disk.used / disk.total) * 100, 2),
                    "free_bytes": disk.free,
                    "total_bytes": disk.total,
                    "used_bytes": disk.used
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "process": {
                    "memory_rss": process_memory.rss,
                    "memory_vms": process_memory.vms,
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "created": process.create_time()
                },
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