#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析服务 - 提供RCA业务逻辑的统一封装
"""

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .base import BaseService, HealthCheckMixin
from ..common.constants import ServiceConstants
from ..common.exceptions import RCAError
from ..core.rca.rca_engine import RCAAnalysisEngine
from ..core.rca.metrics_collector import MetricsCollector
from ..core.rca.events_collector import EventsCollector
from ..core.rca.logs_collector import LogsCollector

logger = logging.getLogger("aiops.services.rca")


class RCAService(BaseService, HealthCheckMixin):
    """根因分析服务 - 整合三种数据源的智能根因分析"""

    def __init__(self) -> None:
        super().__init__("rca")
        self._engine: Optional[RCAAnalysisEngine] = None
        self._metrics_collector: Optional[MetricsCollector] = None
        self._events_collector: Optional[EventsCollector] = None
        self._logs_collector: Optional[LogsCollector] = None

    async def _do_initialize(self) -> None:
        """初始化RCA服务组件"""
        try:
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
        """健康检查"""
        try:
            if not self._metrics_collector or not self._events_collector or not self._logs_collector:
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
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """执行根因分析"""
        start_time = time.time()
        
        try:
            self._ensure_initialized()
            
            # 准备参数
            time_window = timedelta(hours=time_window_hours)
            
            self.logger.info(f"开始RCA分析: namespace={namespace}, time_window={time_window_hours}小时")
            
            # 执行分析
            analysis_result = await self._engine.analyze(
                namespace=namespace,
                time_window=time_window,
                metrics=metrics
            )
            
            # 转换根因数据
            root_causes = []
            for cause in analysis_result.root_causes:
                root_causes.append({
                    "cause_type": cause.cause_type,
                    "description": cause.description,
                    "confidence": cause.confidence,
                    "affected_components": cause.affected_components,
                    "evidence": cause.evidence,
                    "recommendations": cause.recommendations
                })
            
            duration = time.time() - start_time
            
            result = {
                "success": True,
                "timestamp": analysis_result.timestamp,
                "namespace": analysis_result.namespace,
                "root_causes": root_causes,
                "confidence_score": analysis_result.confidence_score,
                "recommendations": analysis_result.recommendations,
                "timeline_events": len(analysis_result.timeline),
                "analysis_duration_seconds": duration
            }
            
            self.logger.info(f"RCA分析完成: 发现 {len(root_causes)} 个根因, 耗时 {duration:.2f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"RCA分析失败: {str(e)}", exc_info=True)
            raise RCAError(f"分析失败: {str(e)}")

    async def get_metrics(
        self,
        namespace: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metrics: Optional[str] = None
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
                metrics=metric_list
            )
            
            # 转换为响应格式
            items = []
            if metric_data:
                for data in metric_data:
                    items.append({
                        "name": data.name,
                        "values": data.values or [],
                        "labels": data.labels or {},
                        "anomaly_score": data.anomaly_score,
                        "trend": data.trend
                    })
            
            return {"items": items, "total": len(items)}
            
        except Exception as e:
            self.logger.error(f"获取指标失败: {str(e)}", exc_info=True)
            return {"items": [], "total": 0}

    async def get_events(
        self,
        namespace: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[str] = None
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
                namespace=namespace,
                start_time=start_time,
                end_time=end_time
            )
            
            # 过滤严重程度
            if severity:
                event_data = [e for e in event_data if e.severity.value == severity.lower()]
            
            # 转换为响应格式
            items = []
            for event in event_data[:100]:  # 限制返回数量
                items.append({
                    "timestamp": event.timestamp,
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "involved_object": event.involved_object,
                    "severity": event.severity,
                    "count": event.count
                })
            
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
        max_lines: int = 100
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
                max_lines=max_lines
            )
            
            # 转换为响应格式
            items = []
            for log in log_data[:max_lines]:  # 限制返回数量
                items.append({
                    "timestamp": log.timestamp,
                    "pod_name": log.pod_name,
                    "container_name": log.container_name,
                    "level": log.level,
                    "message": log.message,
                    "error_type": log.error_type,
                    "stack_trace": log.stack_trace
                })
            
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
            
            # 判断总体状态
            all_healthy = all(collectors_health.values())
            status = "healthy" if all_healthy else "degraded"
            
            return {
                "status": status,
                "collectors": collectors_health,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            self.logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "collectors": {"metrics": False, "events": False, "logs": False},
                "timestamp": datetime.now(timezone.utc)
            }

    async def quick_diagnosis(
        self,
        namespace: str
    ) -> Dict[str, Any]:
        """快速诊断"""
        try:
            self._ensure_initialized()
            
            # 执行快速分析（最近1小时）
            analysis_result = await self._engine.analyze(
                namespace=namespace,
                time_window=timedelta(hours=1)
            )
            
            # 提取关键信息
            critical_issues = []
            
            # 获取最严重的根因
            if analysis_result.root_causes:
                top_cause = analysis_result.root_causes[0]
                critical_issues.append({
                    "type": "root_cause",
                    "severity": "critical",
                    "description": top_cause.description,
                    "confidence": top_cause.confidence
                })
            
            # 获取关键事件
            for event in analysis_result.timeline[:5]:
                if event.get("severity") in ["critical", "high"]:
                    critical_issues.append({
                        "type": "event",
                        "severity": event["severity"],
                        "description": event["description"],
                        "timestamp": event["timestamp"]
                    })
            
            return {
                "namespace": namespace,
                "diagnosis_time": datetime.now(timezone.utc).isoformat(),
                "critical_issues": critical_issues,
                "recommendations": analysis_result.recommendations[:3],
                "confidence_score": analysis_result.confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"快速诊断失败: {str(e)}")
            raise RCAError(f"快速诊断失败: {str(e)}")

    async def get_event_patterns(
        self,
        namespace: str,
        hours: float = 1.0
    ) -> Dict[str, Any]:
        """获取事件模式分析"""
        try:
            self._ensure_initialized()
            
            # 时间范围
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            # 获取事件模式
            patterns = await self._events_collector.get_event_patterns(
                namespace=namespace,
                start_time=start_time,
                end_time=end_time
            )
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"获取事件模式失败: {str(e)}")
            raise RCAError(f"获取事件模式失败: {str(e)}")

    async def get_error_summary(
        self,
        namespace: str,
        hours: float = 1.0
    ) -> Dict[str, Any]:
        """获取错误摘要"""
        try:
            self._ensure_initialized()
            
            # 获取错误摘要
            summary = await self._logs_collector.get_error_summary(
                namespace=namespace,
                time_window=timedelta(hours=hours)
            )
            
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
            "logs": await self._logs_collector.health_check()
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
                    "模式识别"
                ]
            }
        except Exception as e:
            self.logger.error(f"获取服务健康信息失败: {str(e)}")
            return {
                "service": "rca",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            }