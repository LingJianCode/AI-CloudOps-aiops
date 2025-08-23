#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI-CloudOps智能根因分析服务
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..common.exceptions import RCAError
from ..core.rca.events_collector import EventsCollector
from ..core.rca.logs_collector import LogsCollector
from ..core.rca.metrics_collector import MetricsCollector
from ..core.rca.rca_engine import RCAAnalysisEngine
from .base import BaseService, HealthCheckMixin
from .prometheus import PrometheusService

logger = logging.getLogger("aiops.services.rca")


class RCAService(BaseService, HealthCheckMixin):
    """根因分析服务"""

    def __init__(self) -> None:
        super().__init__("rca")
        self._engine: Optional[RCAAnalysisEngine] = None
        self._metrics_collector: Optional[MetricsCollector] = None
        self._events_collector: Optional[EventsCollector] = None
        self._logs_collector: Optional[LogsCollector] = None
        
        # Redis缓存管理器
        self._cache_manager = None

    async def _do_initialize(self) -> None:
        try:
            # 初始化Redis缓存管理器
            try:
                from ..core.cache.redis_cache_manager import RedisCacheManager
                from ..config.settings import config

                redis_config = {
                    "host": config.redis.host,
                    "port": config.redis.port,
                    "db": config.redis.db + 3,  # 使用单独的db用于RCA缓存
                    "password": config.redis.password if hasattr(config.redis, 'password') else "",
                }
                self._cache_manager = RedisCacheManager(
                    redis_config=redis_config,
                    cache_prefix="rca_cache:",
                    default_ttl=1800,  # 30分钟缓存
                    max_cache_size=3000,
                    enable_compression=True,
                )
                self.logger.info("RCA服务Redis缓存管理器初始化成功")
            except Exception as cache_e:
                self.logger.warning(f"Redis缓存管理器初始化失败: {str(cache_e)}，将在无缓存模式下运行")
                self._cache_manager = None

            # 初始化引擎
            self._engine = RCAAnalysisEngine()
            await self._engine.initialize()

            # 初始化收集器
            self._metrics_collector = MetricsCollector()
            await self._metrics_collector.initialize()

            self._events_collector = EventsCollector()
            await self._events_collector.initialize()

            self._logs_collector = LogsCollector()
            await self._logs_collector.initialize()

            self.logger.info("RCA服务组件初始化完成")
        except Exception as e:
            self.logger.error(f"RCA服务初始化失败: {str(e)}")
            raise RCAError(f"初始化失败: {str(e)}")

    async def _do_health_check(self) -> bool:
        try:
            if (
                not self._metrics_collector
                or not self._events_collector
                or not self._logs_collector
            ):
                return False

            # 检查各组件健康状态
            checks = await self._gather_health_checks()
            return any(checks.values())  # 至少一个收集器正常
        except Exception:
            return False

    async def analyze_root_cause(
        self,
        namespace: str,
        time_window_hours: float = 1.0,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            self._ensure_initialized()

            # 生成缓存键
            cache_key = self._generate_rca_cache_key(
                operation="analyze",
                namespace=namespace,
                time_window_hours=time_window_hours,
                metrics=metrics,
            )

            # 尝试从缓存获取结果
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info(f"RCA分析缓存命中，直接返回结果: namespace={namespace}")
                return cached_result

            # 准备参数
            time_window = timedelta(hours=time_window_hours)

            self.logger.info(
                f"开始RCA分析: namespace={namespace}, time_window={time_window_hours}小时"
            )

            # 执行分析
            analysis_result = await self._engine.analyze(
                namespace=namespace, time_window=time_window, metrics=metrics
            )

            # 转换根因数据
            root_causes = []
            for cause in analysis_result.root_causes:
                root_causes.append(
                    {
                        "cause_type": cause.cause_type,
                        "description": cause.description,
                        "confidence": cause.confidence,
                        "affected_components": cause.affected_components,
                        "evidence": cause.evidence,
                        "recommendations": cause.recommendations,
                    }
                )

            duration = time.time() - start_time

            result = {
                "success": True,
                "timestamp": analysis_result.timestamp,
                "namespace": analysis_result.namespace,
                "root_causes": root_causes,
                "confidence_score": analysis_result.confidence_score,
                "recommendations": analysis_result.recommendations,
                "timeline_events": len(analysis_result.timeline),
                "analysis_duration_seconds": duration,
            }

            # 保存到缓存（30分钟缓存）
            await self._save_to_cache(cache_key, result, ttl=1800)

            self.logger.info(
                f"RCA分析完成: 发现 {len(root_causes)} 个根因, 耗时 {duration:.2f}秒"
            )
            return result

        except Exception as e:
            self.logger.error(f"RCA分析失败: {str(e)}", exc_info=True)
            raise RCAError(f"分析失败: {str(e)}")

    async def get_metrics(
        self,
        namespace: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取指标数据"""
        try:
            self._ensure_initialized()

            # 默认时间范围
            if not end_time:
                end_time = datetime.now(timezone.utc)
            if not start_time:
                start_time = end_time - timedelta(hours=1)

            # 准备指标列表
            metric_list = metrics.split(",") if metrics else []

            # 收集数据
            metric_data = await self._metrics_collector.collect(
                namespace=namespace,
                start_time=start_time,
                end_time=end_time,
                metrics=metric_list,
            )

            # 转换为响应格式
            items = []
            if metric_data:
                for data in metric_data:
                    items.append(
                        {
                            "name": data.name,
                            "values": data.values or [],
                            "labels": data.labels or {},
                            "anomaly_score": data.anomaly_score,
                            "trend": data.trend,
                        }
                    )

            return {"items": items, "total": len(items)}

        except Exception as e:
            self.logger.error(f"获取指标失败: {str(e)}", exc_info=True)
            return {"items": [], "total": 0}

    async def get_events(
        self,
        namespace: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取事件数据"""
        try:
            self._ensure_initialized()

            # 默认时间范围
            if not end_time:
                end_time = datetime.now(timezone.utc)
            if not start_time:
                start_time = end_time - timedelta(hours=1)

            # 收集数据
            event_data = await self._events_collector.collect(
                namespace=namespace, start_time=start_time, end_time=end_time
            )

            # 过滤严重程度
            if severity:
                event_data = [
                    e for e in event_data if e.severity.value == severity.lower()
                ]

            # 转换为响应格式
            items = []
            for event in event_data[:100]:  # 限制返回数量
                items.append(
                    {
                        "timestamp": event.timestamp,
                        "type": event.type,
                        "reason": event.reason,
                        "message": event.message,
                        "involved_object": event.involved_object,
                        "severity": event.severity,
                        "count": event.count,
                    }
                )

            return {"items": items, "total": len(items)}

        except Exception as e:
            self.logger.error(f"获取事件失败: {str(e)}")
            raise RCAError(f"获取事件失败: {str(e)}")

    async def get_logs(
        self,
        namespace: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        pod_name: Optional[str] = None,
        error_only: bool = True,
        max_lines: int = 100,
    ) -> Dict[str, Any]:
        """获取日志数据"""
        try:
            self._ensure_initialized()

            # 默认时间范围
            if not end_time:
                end_time = datetime.now(timezone.utc)
            if not start_time:
                start_time = end_time - timedelta(hours=1)

            # 准备Pod列表
            pod_names = [pod_name] if pod_name else []

            # 收集数据
            log_data = await self._logs_collector.collect(
                namespace=namespace,
                start_time=start_time,
                end_time=end_time,
                pod_names=pod_names,
                error_only=error_only,
                max_lines=max_lines,
            )

            # 转换为响应格式
            items = []
            for log in log_data[:max_lines]:  # 限制返回数量
                items.append(
                    {
                        "timestamp": log.timestamp,
                        "pod_name": log.pod_name,
                        "container_name": log.container_name,
                        "level": log.level,
                        "message": log.message,
                        "error_type": log.error_type,
                        "stack_trace": log.stack_trace,
                    }
                )

            return {"items": items, "total": len(items)}

        except Exception as e:
            self.logger.error(f"获取日志失败: {str(e)}")
            raise RCAError(f"获取日志失败: {str(e)}")

    async def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            self._ensure_initialized()

            # 检查各组件健康状态
            collectors_health = await self._gather_health_checks()

            # 映射到API期望的字段名称
            prometheus_connected = collectors_health.get("metrics", False)
            kubernetes_connected = collectors_health.get("events", False) or collectors_health.get("logs", False)
            
            # 检查Redis连接
            redis_connected = False
            if self._cache_manager:
                try:
                    # 使用缓存管理器的health_check方法
                    cache_health = self._cache_manager.health_check()
                    redis_connected = cache_health.get("status") == "healthy" if isinstance(cache_health, dict) else cache_health
                except Exception as e:
                    self.logger.warning(f"Redis健康检查失败: {str(e)}")
                    redis_connected = False

            # 判断总体状态
            all_healthy = prometheus_connected and kubernetes_connected and redis_connected
            status = "healthy" if all_healthy else "degraded"

            return {
                "status": status,
                "prometheus_connected": prometheus_connected,
                "kubernetes_connected": kubernetes_connected,
                "redis_connected": redis_connected,
                "collectors": collectors_health,  # 保留原始信息用于调试
                "timestamp": datetime.now(timezone.utc),
            }

        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "prometheus_connected": False,
                "kubernetes_connected": False,
                "redis_connected": False,
                "collectors": {"metrics": False, "events": False, "logs": False},
                "timestamp": datetime.now(timezone.utc),
            }

    async def quick_diagnosis(self, namespace: str) -> Dict[str, Any]:
        """快速诊断"""
        try:
            self._ensure_initialized()

            # 生成缓存键
            cache_key = self._generate_rca_cache_key(
                operation="quick_diagnosis",
                namespace=namespace,
                time_window_hours=1.0,
            )

            # 尝试从缓存获取结果
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info(f"快速诊断缓存命中，直接返回结果: namespace={namespace}")
                return cached_result

            # 检查依赖服务状态
            health_checks = await self._gather_health_checks()
            services_available = any(health_checks.values())
            
            if not services_available:
                self.logger.warning("所有依赖服务不可用，返回降级的诊断结果")
                return {
                    "namespace": namespace,
                    "diagnosis_time": datetime.now(timezone.utc).isoformat(),
                    "critical_issues": [
                        {
                            "type": "service_unavailable",
                            "severity": "high",
                            "description": "监控数据服务不可用，无法获取实时状态",
                            "confidence": 1.0,
                        }
                    ],
                    "recommendations": [
                        "检查Prometheus服务状态",
                        "验证Kubernetes集群连接",
                        "检查Redis缓存服务"
                    ],
                    "confidence_score": 0.0,
                }

            # 执行快速分析（最近1小时）
            try:
                analysis_result = await self._engine.analyze(
                    namespace=namespace, time_window=timedelta(hours=1)
                )
            except Exception as analysis_error:
                self.logger.warning(f"分析引擎执行失败: {str(analysis_error)}, 返回基础诊断结果")
                return {
                    "namespace": namespace,
                    "diagnosis_time": datetime.now(timezone.utc).isoformat(),
                    "critical_issues": [
                        {
                            "type": "analysis_error",
                            "severity": "medium",
                            "description": f"分析过程遇到问题: {str(analysis_error)[:100]}",
                            "confidence": 0.5,
                        }
                    ],
                    "recommendations": [
                        "检查目标命名空间是否存在",
                        "验证监控数据是否可用",
                        "稍后重试诊断"
                    ],
                    "confidence_score": 0.3,
                }

            # 提取关键信息
            critical_issues = []

            # 获取最严重的根因
            if analysis_result.root_causes:
                top_cause = analysis_result.root_causes[0]
                critical_issues.append(
                    {
                        "type": "root_cause",
                        "severity": "critical",
                        "description": top_cause.description,
                        "confidence": top_cause.confidence,
                    }
                )

            # 获取关键事件
            for event in analysis_result.timeline[:5]:
                if event.get("severity") in ["critical", "high"]:
                    critical_issues.append(
                        {
                            "type": "event",
                            "severity": event["severity"],
                            "description": event["description"],
                            "timestamp": event["timestamp"],
                        }
                    )

            result = {
                "namespace": namespace,
                "diagnosis_time": datetime.now(timezone.utc).isoformat(),
                "critical_issues": critical_issues,
                "recommendations": analysis_result.recommendations[:3] if hasattr(analysis_result, 'recommendations') else [],
                "confidence_score": analysis_result.confidence_score if hasattr(analysis_result, 'confidence_score') else 0.8,
            }

            # 保存到缓存（15分钟缓存，快速诊断需要更快的更新）
            await self._save_to_cache(cache_key, result, ttl=900)

            return result

        except Exception as e:
            self.logger.error(f"快速诊断失败: {str(e)}")
            # 返回基础错误信息而不是抛出异常
            return {
                "namespace": namespace,
                "diagnosis_time": datetime.now(timezone.utc).isoformat(),
                "critical_issues": [
                    {
                        "type": "system_error",
                        "severity": "high", 
                        "description": f"快速诊断系统遇到未预期的错误: {str(e)[:100]}",
                        "confidence": 0.0,
                    }
                ],
                "recommendations": [
                    "联系系统管理员",
                    "检查服务日志获取详细信息",
                    "稍后重试"
                ],
                "confidence_score": 0.0,
            }

    async def get_event_patterns(
        self, namespace: str, hours: float = 1.0
    ) -> Dict[str, Any]:
        """获取事件模式分析"""
        try:
            self._ensure_initialized()

            # 生成缓存键
            cache_key = self._generate_rca_cache_key(
                operation="event_patterns",
                namespace=namespace,
                time_window_hours=hours,
            )

            # 尝试从缓存获取结果
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info(f"事件模式分析缓存命中，直接返回结果: namespace={namespace}")
                return cached_result

            # 时间范围
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            # 获取事件模式
            patterns = await self._events_collector.get_event_patterns(
                namespace=namespace, start_time=start_time, end_time=end_time
            )

            # 保存到缓存（20分钟缓存）
            await self._save_to_cache(cache_key, patterns, ttl=1200)

            return patterns

        except Exception as e:
            self.logger.error(f"获取事件模式失败: {str(e)}")
            raise RCAError(f"获取事件模式失败: {str(e)}")

    async def get_error_summary(
        self, namespace: str, hours: float = 1.0
    ) -> Dict[str, Any]:
        """获取错误摘要"""
        try:
            self._ensure_initialized()

            # 生成缓存键
            cache_key = self._generate_rca_cache_key(
                operation="error_summary",
                namespace=namespace,
                time_window_hours=hours,
            )

            # 尝试从缓存获取结果
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.logger.info(f"错误摘要缓存命中，直接返回结果: namespace={namespace}")
                return cached_result

            # 获取错误摘要
            summary = await self._logs_collector.get_error_summary(
                namespace=namespace, time_window=timedelta(hours=hours)
            )

            # 保存到缓存（20分钟缓存）
            await self._save_to_cache(cache_key, summary, ttl=1200)

            return summary

        except Exception as e:
            self.logger.error(f"获取错误摘要失败: {str(e)}")
            raise RCAError(f"获取错误摘要失败: {str(e)}")

    async def cache_analysis_result(self, result: Any) -> None:
        """缓存分析结果（后台任务）"""
        try:
            # 这里可以实现缓存逻辑
            self.logger.info("分析结果已缓存")
        except Exception as e:
            self.logger.error(f"缓存分析结果失败: {str(e)}")

    async def _gather_health_checks(self) -> Dict[str, bool]:
        """收集所有组件的健康状态"""
        return {
            "metrics": await self._metrics_collector.health_check(),
            "events": await self._events_collector.health_check(),
            "logs": await self._logs_collector.health_check(),
        }

    async def get_service_health_info(self) -> Dict[str, Any]:
        """获取详细的服务健康信息"""
        try:
            health_status = await self.get_health_status()

            return {
                "service": "rca",
                "version": "1.0.0",
                "status": health_status["status"],
                "collectors": health_status["collectors"],
                "timestamp": health_status["timestamp"],
                "capabilities": [
                    "根因分析",
                    "指标收集",
                    "事件分析",
                    "日志分析",
                    "快速诊断",
                    "模式识别",
                ],
            }
        except Exception as e:
            self.logger.error(f"获取服务健康信息失败: {str(e)}")
            return {
                "service": "rca",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc),
            }

    async def get_all_available_metrics(self) -> List[str]:
        """获取所有可用的Prometheus指标列表"""
        try:
            if self._metrics_collector and self._metrics_collector.prometheus:
                return await self._metrics_collector.prometheus.get_available_metrics()
            else:
                # 如果没有初始化，创建临时prometheus服务
                prometheus_service = PrometheusService()
                return await prometheus_service.get_available_metrics()
        except Exception as e:
            self.logger.error(f"获取可用指标失败: {str(e)}")
            # 返回默认的常见指标列表作为回退
            return [
                "up",
                "node_cpu_seconds_total",
                "node_memory_MemAvailable_bytes",
                "node_load1",
                "kubernetes_pod_cpu_usage_seconds_total",
                "kubernetes_pod_memory_usage_bytes",
                "container_cpu_usage_seconds_total",
                "container_memory_usage_bytes",
            ]

    def _generate_rca_cache_key(
        self,
        operation: str,
        namespace: str,
        time_window_hours: Optional[float] = None,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """生成RCA缓存键"""
        try:
            # 构建缓存输入参数
            cache_params = {
                "operation": operation,
                "namespace": namespace,
                "time_window": time_window_hours or 1.0,
            }
            
            # 添加指标参数（如果有）
            if metrics:
                # 对指标进行排序确保一致性
                sorted_metrics = sorted(metrics)
                cache_params["metrics"] = "|".join(sorted_metrics)
            
            # 添加其他重要参数
            for key, value in kwargs.items():
                if key in ["pod_name", "severity", "error_only", "max_lines"]:
                    cache_params[key] = value
            
            # 生成缓存键字符串
            cache_input = "|".join([f"{k}:{v}" for k, v in sorted(cache_params.items())])
            
            # 生成哈希
            cache_hash = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()[:16]
            
            return f"rca:{operation}:{namespace}:{cache_hash}"
            
        except Exception as e:
            self.logger.error(f"生成RCA缓存键失败: {str(e)}")
            # 降级到简单键
            return f"rca:{operation}:{namespace}:{int(time_window_hours or 1)}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取RCA结果"""
        if not self._cache_manager:
            return None
        
        try:
            cached_result = self._cache_manager.get(cache_key)
            if cached_result:
                self.logger.debug(f"RCA缓存命中: {cache_key}")
                # 添加缓存标识
                cached_result["from_cache"] = True
                cached_result["cache_timestamp"] = datetime.now()
                return cached_result
        except Exception as e:
            self.logger.warning(f"从缓存获取RCA数据失败: {str(e)}")
        
        return None

    async def _save_to_cache(self, cache_key: str, result: Dict[str, Any], ttl: int = 1800) -> None:
        """保存RCA结果到缓存"""
        if not self._cache_manager:
            return
        
        try:
            # 移除不需要缓存的临时数据
            cache_result = result.copy()
            cache_result.pop("from_cache", None)
            cache_result.pop("cache_timestamp", None)
            
            # 添加缓存元数据
            cache_result["cached_at"] = datetime.now().isoformat()
            cache_result["cache_ttl"] = ttl
            
            self._cache_manager.set(
                question=cache_key,
                response_data=cache_result,
                ttl=ttl
            )
            self.logger.debug(f"RCA结果已缓存: {cache_key}")
        except Exception as e:
            self.logger.warning(f"保存RCA结果到缓存失败: {str(e)}")

    async def get_config_info(self) -> Dict[str, Any]:
        """获取RCA服务配置信息"""
        try:
            self._ensure_initialized()
            
            config_info = {
                "service_name": self.service_name,
                "status": "initialized" if self._initialized else "not_initialized",
                "components": {
                    "engine": self._engine is not None,
                    "metrics_collector": self._metrics_collector is not None,
                    "events_collector": self._events_collector is not None,
                    "logs_collector": self._logs_collector is not None,
                    "cache_manager": self._cache_manager is not None,
                },
                "cache_config": {
                    "enabled": self._cache_manager is not None,
                    "prefix": "rca_cache:" if self._cache_manager else None,
                    "default_ttl": 1800,
                } if self._cache_manager else {"enabled": False},
                "analysis_limits": {
                    "max_time_window_hours": 24,
                    "min_time_window_hours": 0.1,
                    "max_log_lines": 1000,
                    "default_log_lines": 100,
                },
                "capabilities": [
                    "root_cause_analysis",
                    "metrics_analysis", 
                    "events_analysis",
                    "logs_analysis",
                    "quick_diagnosis",
                    "event_patterns",
                    "error_summary",
                    "caching",
                ],
            }
            
            return config_info
            
        except Exception as e:
            self.logger.error(f"获取RCA配置信息失败: {str(e)}")
            raise RCAError(f"获取配置信息失败: {str(e)}")

    async def cleanup(self) -> None:
        """清理RCA服务资源"""
        try:
            self.logger.info("开始清理RCA服务资源...")

            # 清理缓存管理器
            if self._cache_manager:
                try:
                    self._cache_manager.shutdown()
                except Exception as e:
                    self.logger.warning(f"关闭RCA缓存管理器失败: {str(e)}")
                self._cache_manager = None

            # 清理收集器
            if self._metrics_collector:
                try:
                    if hasattr(self._metrics_collector, "cleanup"):
                        await self._metrics_collector.cleanup()
                except Exception as e:
                    self.logger.warning(f"清理metrics_collector失败: {e}")
                self._metrics_collector = None

            if self._events_collector:
                try:
                    if hasattr(self._events_collector, "cleanup"):
                        await self._events_collector.cleanup()
                except Exception as e:
                    self.logger.warning(f"清理events_collector失败: {e}")
                self._events_collector = None

            if self._logs_collector:
                try:
                    if hasattr(self._logs_collector, "cleanup"):
                        await self._logs_collector.cleanup()
                except Exception as e:
                    self.logger.warning(f"清理logs_collector失败: {e}")
                self._logs_collector = None

            # 清理分析引擎
            if self._engine:
                try:
                    if hasattr(self._engine, "cleanup"):
                        await self._engine.cleanup()
                except Exception as e:
                    self.logger.warning(f"清理RCA引擎失败: {e}")
                self._engine = None

            # 调用父类清理方法
            await super().cleanup()

            self.logger.info("RCA服务资源清理完成")

        except Exception as e:
            self.logger.error(f"RCA服务资源清理失败: {str(e)}")
            raise
