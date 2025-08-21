#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查管理器
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict

import psutil

from app.core.prediction.predictor import PredictionService
from app.services.kubernetes import KubernetesService
from app.services.llm import LLMService
from app.services.notification import NotificationService
from app.services.prometheus import PrometheusService
from app.services.rca_service import RCAService

logger = logging.getLogger("aiops.health_manager")


class HealthManager:

    def __init__(self):
        self.start_time = time.time()
        self._service_cache = {}
        self._cache_ttl = 10  # 缓存10秒，加快响应速度
        self._last_check = {}

    def get_service(self, service_name: str):

        services = {
            "prometheus": PrometheusService,
            "kubernetes": KubernetesService,
            "llm": LLMService,
            "notification": NotificationService,
            "prediction": PredictionService,
            "rca": RCAService,
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

    def _check_fallback_capabilities(self) -> Dict[str, bool]:
        """检查备用实现的可用性"""
        fallback_status = {
            "fallback_chat_model": False,
            "fallback_embeddings": False,
            "session_manager": False,
            "response_templates": False,
        }

        try:
            # 检查备用聊天模型
            from app.core.agents.fallback_models import FallbackChatModel

            FallbackChatModel()
            fallback_status["fallback_chat_model"] = True
        except Exception as e:
            logger.debug(f"备用聊天模型不可用: {e}")

        try:
            # 检查备用嵌入模型
            from app.core.agents.fallback_models import FallbackEmbeddings

            FallbackEmbeddings()
            fallback_status["fallback_embeddings"] = True
        except Exception as e:
            logger.debug(f"备用嵌入模型不可用: {e}")

        try:
            # 检查会话管理器
            from app.core.agents.fallback_models import SessionManager

            SessionManager()
            fallback_status["session_manager"] = True
        except Exception as e:
            logger.debug(f"会话管理器不可用: {e}")

        try:
            # 检查响应模板管理器
            from app.core.agents.fallback_models import ResponseTemplateManager

            ResponseTemplateManager()
            fallback_status["response_templates"] = True
        except Exception as e:
            logger.debug(f"响应模板管理器不可用: {e}")

        return fallback_status

    def check_component_health(self, component: str) -> Dict[str, Any]:

        current_time = time.time()

        # 检查缓存 - 优化缓存查询
        last_check = self._last_check.get(component)
        if last_check and current_time - last_check["time"] < self._cache_ttl:
            return last_check["result"]

        try:
            service = self.get_service(component)
            if not service:
                result = {
                    "healthy": False,
                    "error": f"未知的组件: {component}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            else:
                # 特殊处理 LLM 服务的健康检查
                if component == "llm" and hasattr(service, "is_healthy"):
                    try:
                        # 避免在异步环境中的死锁问题，使用同步检查
                        if hasattr(service, "health_check_sync"):
                            # 如果有同步健康检查方法，优先使用
                            is_healthy = service.health_check_sync()
                        elif hasattr(service, "is_available"):
                            # 使用简单的可用性检查
                            is_healthy = service.is_available()
                        else:
                            # 基础连接检查
                            is_healthy = bool(
                                getattr(service, "client", None)
                                or getattr(service, "_client", None)
                            )
                    except Exception as e:
                        logger.error(f"LLM健康检查失败: {str(e)}")
                        is_healthy = False
                else:
                    # 其他服务的同步健康检查
                    is_healthy = service.is_healthy()

                result = {
                    "healthy": is_healthy,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # 添加组件特定信息 - 优化异常处理
                if hasattr(service, "get_service_info") and callable(
                    service.get_service_info
                ):
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
                "timestamp": datetime.utcnow().isoformat(),
            }

        # 更新缓存
        self._last_check[component] = {"time": current_time, "result": result}

        return result

    def check_all_components(self) -> Dict[str, Dict[str, Any]]:

        components = [
            "prometheus",
            "kubernetes",
            "llm",
            "notification",
            "prediction",
            "rca",
        ]
        return {comp: self.check_component_health(comp) for comp in components}

    # 兼容路由层调用：返回包含 components 字段的结构
    def get_components_health(self) -> Dict[str, Any]:

        comps = self.check_all_components()
        return {"components": comps, "timestamp": datetime.utcnow().isoformat()}

    def get_system_metrics(self) -> Dict[str, Any]:

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
                    "total_gb": round(disk.total / (1024 * 1024 * 1024), 2),
                }
            except Exception:
                disk_info = {"error": "无法获取磁盘信息"}

            # 获取网络信息
            try:
                network = psutil.net_io_counters()
                network_info = {
                    "bytes_sent_mb": round(network.bytes_sent / (1024 * 1024), 2),
                    "bytes_recv_mb": round(network.bytes_recv / (1024 * 1024), 2),
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
                    "threads": process.num_threads(),
                }
            except Exception:
                process_info = {"error": "无法获取进程信息"}

            return {
                "cpu": {"usage_percent": cpu_percent, "count": psutil.cpu_count()},
                "memory": {
                    "usage_percent": memory.percent,
                    "available_bytes": int(memory.available),
                    "total_bytes": int(memory.total),
                },
                "disk": disk_info,
                "network": network_info,
                "process": process_info,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {str(e)}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

    def get_uptime(self) -> float:

        return time.time() - self.start_time

    def get_overall_health(self) -> Dict[str, Any]:

        components = self.check_all_components()
        system_metrics = self.get_system_metrics()

        # 检查备用实现的可用性
        fallback_status = self._check_fallback_capabilities()

        # 判断整体健康状态
        all_healthy = all(comp.get("healthy", False) for comp in components.values())

        # 核心组件健康检查
        critical_components = ["prometheus", "prediction"]
        critical_healthy = all(
            components.get(comp, {}).get("healthy", False)
            for comp in critical_components
        )

        # 考虑备用实现的状态
        fallback_available = any(fallback_status.values())

        if all_healthy:
            status = "healthy"
        elif critical_healthy and fallback_available:
            status = "degraded"  # 关键组件正常且有备用实现
        elif fallback_available:
            status = "degraded"  # 至少有备用实现可用
        else:
            status = "unhealthy"

        return {
            "status": status,
            "healthy": all_healthy,
            "uptime": self.get_uptime(),
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                name: comp.get("healthy", False) for name, comp in components.items()
            },
            "fallback_capabilities": fallback_status,
            "system": system_metrics,
            "details": components,
        }

    def check_readiness(self) -> Dict[str, Any]:

        comps = self.check_all_components()
        # 将 Prometheus 与 Prediction 视为关键依赖
        critical_ready = all(
            comps.get(name, {}).get("healthy", False)
            for name in ["prometheus", "prediction"]
        )
        status = "ready" if critical_ready else "not ready"
        return {
            "status": status,
            "ready": critical_ready,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def check_liveness(self) -> Dict[str, Any]:

        return {
            "status": "alive",
            "uptime": self.get_uptime(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def check_startup(self) -> Dict[str, Any]:

        started = self.get_uptime() >= 0.5
        return {"started": started, "timestamp": datetime.utcnow().isoformat()}

    def check_dependencies(self) -> Dict[str, Any]:

        comps = self.check_all_components()
        return {
            "dependencies": {
                name: info.get("healthy", False) for name, info in comps.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


# 全局健康检查管理器实例
health_manager = HealthManager()
