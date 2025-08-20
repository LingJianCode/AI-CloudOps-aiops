#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 核心根因分析引擎 - 整合三种数据源进行智能分析
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from app.models.rca_models import (
    MetricData, EventData, LogData, 
    SeverityLevel, RootCauseAnalysis, CorrelationResult, RootCause
)
from .metrics_collector import MetricsCollector
from .events_collector import EventsCollector
from .logs_collector import LogsCollector





class RCAAnalysisEngine:
    """根因分析引擎 - 整合指标异常检测、事件关联分析和日志模式识别"""
    
    # 根因模式定义
    ROOT_CAUSE_PATTERNS = {
        "OOM": {
            "metrics": ["container_memory_usage_bytes", "container_memory_working_set_bytes"],
            "events": ["OOMKilled", "Killing"],
            "logs": ["out of memory", "oom", "memory exhausted"],
            "confidence_weight": 0.9
        },
        "CPU_THROTTLING": {
            "metrics": ["container_cpu_cfs_throttled_periods_total", "container_cpu_usage_seconds_total"],
            "events": ["CPUThrottling", "HighCPU"],
            "logs": ["cpu throttled", "high cpu usage"],
            "confidence_weight": 0.85
        },
        "CRASH_LOOP": {
            "metrics": ["kube_pod_container_status_restarts_total"],
            "events": ["CrashLoopBackOff", "BackOff"],
            "logs": ["panic", "fatal error", "segmentation fault"],
            "confidence_weight": 0.95
        },
        "NETWORK_ISSUE": {
            "metrics": ["container_network_receive_errors_total", "container_network_transmit_errors_total"],
            "events": ["NetworkNotReady", "NetworkPluginNotReady"],
            "logs": ["connection refused", "timeout", "network unreachable"],
            "confidence_weight": 0.8
        },
        "IMAGE_PULL": {
            "metrics": [],
            "events": ["ImagePullBackOff", "ErrImagePull"],
            "logs": ["pull access denied", "image not found"],
            "confidence_weight": 0.95
        },
        "RESOURCE_QUOTA": {
            "metrics": ["kube_resourcequota"],
            "events": ["FailedScheduling", "InsufficientCPU", "InsufficientMemory"],
            "logs": ["exceeded quota", "insufficient resources"],
            "confidence_weight": 0.9
        },
        "DISK_PRESSURE": {
            "metrics": ["node_filesystem_avail_bytes", "node_filesystem_size_bytes"],
            "events": ["DiskPressure", "EvictedByNodeCondition"],
            "logs": ["no space left", "disk full"],
            "confidence_weight": 0.85
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("aiops.rca.engine")
        
        # 初始化收集器
        self.metrics_collector = MetricsCollector(config)
        self.events_collector = EventsCollector(config)
        self.logs_collector = LogsCollector(config)
        
        # 分析缓存
        self._analysis_cache = {}
    
    async def initialize(self) -> None:
        """初始化所有收集器"""
        await asyncio.gather(
            self.metrics_collector.initialize(),
            self.events_collector.initialize(),
            self.logs_collector.initialize()
        )
        self.logger.info("RCA分析引擎初始化完成")
    
    async def analyze(
        self,
        namespace: str,
        service_name: Optional[str] = None,
        time_window: timedelta = timedelta(hours=1),
        incident_time: Optional[datetime] = None
    ) -> RootCauseAnalysis:
        """执行根因分析"""
        # 确定时间范围
        end_time = incident_time or datetime.now(timezone.utc)
        start_time = end_time - time_window
        
        self.logger.info(f"开始根因分析: namespace={namespace}, service={service_name}, "
                        f"time_range={start_time} to {end_time}")
        
        # 并行收集三种数据
        metrics, events, logs = await self._collect_all_data(
            namespace, service_name, start_time, end_time
        )
        
        # 执行多维度分析
        analysis_results = await asyncio.gather(
            self._analyze_metrics_anomalies(metrics),
            self._analyze_event_patterns(events),
            self._analyze_log_errors(logs),
            return_exceptions=True
        )
        
        # 处理分析结果
        metric_anomalies = analysis_results[0] if not isinstance(analysis_results[0], Exception) else {}
        event_patterns = analysis_results[1] if not isinstance(analysis_results[1], Exception) else {}
        log_patterns = analysis_results[2] if not isinstance(analysis_results[2], Exception) else {}
        
        # 关联分析
        correlations = self._correlate_data(
            metric_anomalies, event_patterns, log_patterns, start_time, end_time
        )
        
        # 识别根因
        root_causes = self._identify_root_causes(
            metric_anomalies, event_patterns, log_patterns, correlations
        )
        
        # 生成时间线
        timeline = self._build_timeline(metrics, events, logs)
        
        # 生成建议
        recommendations = self._generate_recommendations(root_causes)
        
        return RootCauseAnalysis(
            timestamp=datetime.now(timezone.utc),
            namespace=namespace,
            service_name=service_name,
            root_causes=root_causes,
            correlations=correlations,
            timeline=timeline,
            recommendations=recommendations,
            confidence_score=self._calculate_confidence(root_causes),
            analysis_metadata={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "metrics_analyzed": len(metrics),
                "events_analyzed": len(events),
                "logs_analyzed": len(logs)
            }
        )
    
    async def _collect_all_data(
        self,
        namespace: str,
        service_name: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[List[MetricData], List[EventData], List[LogData]]:
        """并行收集所有数据"""
        collect_tasks = [
            self.metrics_collector.collect(
                namespace, start_time, end_time,
                service_name=service_name
            ),
            self.events_collector.collect(
                namespace, start_time, end_time
            ),
            self.logs_collector.collect(
                namespace, start_time, end_time,
                error_only=True,
                max_lines=500
            )
        ]
        
        results = await asyncio.gather(*collect_tasks, return_exceptions=True)
        
        # 处理异常
        metrics = results[0] if not isinstance(results[0], Exception) else []
        events = results[1] if not isinstance(results[1], Exception) else []
        logs = results[2] if not isinstance(results[2], Exception) else []
        
        return metrics, events, logs
    
    async def _analyze_metrics_anomalies(self, metrics: List[MetricData]) -> Dict[str, Any]:
        """分析指标异常"""
        anomalies = {
            "high_anomaly_metrics": [],
            "trending_metrics": [],
            "threshold_violations": []
        }
        
        for metric in metrics:
            # 高异常分数的指标
            if metric.anomaly_score > 0.7:
                anomalies["high_anomaly_metrics"].append({
                    "name": metric.name,
                    "score": metric.anomaly_score,
                    "trend": metric.trend,
                    "labels": metric.labels
                })
            
            # 趋势异常
            if metric.trend in ["increasing", "decreasing"]:
                anomalies["trending_metrics"].append({
                    "name": metric.name,
                    "trend": metric.trend,
                    "labels": metric.labels
                })
            
            # 检查阈值违规
            violations = self._check_threshold_violations(metric)
            if violations:
                anomalies["threshold_violations"].extend(violations)
        
        return anomalies
    
    async def _analyze_event_patterns(self, events: List[EventData]) -> Dict[str, Any]:
        """分析事件模式"""
        patterns = {
            "critical_events": [],
            "event_clusters": defaultdict(list),
            "frequency_spikes": []
        }
        
        # 关键事件
        for event in events:
            if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                patterns["critical_events"].append({
                    "timestamp": event.timestamp.isoformat(),
                    "reason": event.reason,
                    "message": event.message,
                    "object": event.involved_object,
                    "count": event.count
                })
            
            # 事件聚类
            cluster_key = f"{event.reason}_{event.involved_object.get('kind')}"
            patterns["event_clusters"][cluster_key].append(event)
        
        # 检测频率异常
        patterns["frequency_spikes"] = self._detect_frequency_spikes(
            patterns["event_clusters"]
        )
        
        return patterns
    
    async def _analyze_log_errors(self, logs: List[LogData]) -> Dict[str, Any]:
        """分析日志错误"""
        patterns = {
            "error_types": defaultdict(list),
            "stack_traces": [],
            "error_frequency": defaultdict(int)
        }
        
        for log in logs:
            if log.level in ["ERROR", "FATAL"]:
                # 错误类型分组
                if log.error_type:
                    patterns["error_types"][log.error_type].append({
                        "timestamp": log.timestamp.isoformat(),
                        "pod": log.pod_name,
                        "message": log.message[:200]  # 截断消息
                    })
                
                # 收集堆栈跟踪
                if log.stack_trace:
                    patterns["stack_traces"].append({
                        "timestamp": log.timestamp.isoformat(),
                        "pod": log.pod_name,
                        "trace": log.stack_trace[:500]  # 截断堆栈
                    })
                
                # 错误频率统计
                error_key = f"{log.pod_name}_{log.error_type or 'unknown'}"
                patterns["error_frequency"][error_key] += 1
        
        return patterns
    
    def _correlate_data(
        self,
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ) -> List[CorrelationResult]:
        """关联三种数据源"""
        correlations = []
        
        # 时间关联 - 查找同一时间窗口内的异常
        time_correlated = self._find_temporal_correlations(
            metric_anomalies, event_patterns, log_patterns
        )
        if time_correlated:
            correlations.append(time_correlated)
        
        # 组件关联 - 查找影响同一组件的问题
        component_correlated = self._find_component_correlations(
            metric_anomalies, event_patterns, log_patterns
        )
        if component_correlated:
            correlations.append(component_correlated)
        
        # 因果关联 - 识别因果链
        causal_chain = self._find_causal_chain(
            metric_anomalies, event_patterns, log_patterns
        )
        if causal_chain:
            correlations.append(causal_chain)
        
        return correlations
    
    def _identify_root_causes(
        self,
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any],
        correlations: List[CorrelationResult]
    ) -> List[RootCause]:
        """识别根因"""
        root_causes = []
        
        # 基于模式匹配的根因识别
        for cause_type, pattern in self.ROOT_CAUSE_PATTERNS.items():
            confidence = self._match_pattern(
                pattern, metric_anomalies, event_patterns, log_patterns
            )
            
            if confidence > 0.5:  # 置信度阈值
                root_cause = self._create_root_cause(
                    cause_type, confidence, 
                    metric_anomalies, event_patterns, log_patterns
                )
                root_causes.append(root_cause)
        
        # 基于关联分析的根因识别
        for correlation in correlations:
            if correlation.confidence > 0.7:
                root_cause = self._derive_root_cause_from_correlation(correlation)
                if root_cause and root_cause not in root_causes:
                    root_causes.append(root_cause)
        
        # 按置信度排序
        root_causes.sort(key=lambda x: x.confidence, reverse=True)
        
        # 只返回前3个最可能的根因
        return root_causes[:3]
    
    def _match_pattern(
        self,
        pattern: Dict[str, Any],
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any]
    ) -> float:
        """匹配根因模式"""
        matches = []
        
        # 检查指标匹配
        if pattern["metrics"]:
            metric_match = any(
                any(m in anomaly["name"] for m in pattern["metrics"])
                for anomaly in metric_anomalies.get("high_anomaly_metrics", [])
            )
            matches.append(metric_match)
        
        # 检查事件匹配
        if pattern["events"]:
            event_match = any(
                any(e in event["reason"] for e in pattern["events"])
                for event in event_patterns.get("critical_events", [])
            )
            matches.append(event_match)
        
        # 检查日志匹配
        if pattern["logs"]:
            log_match = any(
                any(keyword in log_type.lower() for keyword in pattern["logs"])
                for log_type in log_patterns.get("error_types", {}).keys()
            )
            matches.append(log_match)
        
        # 计算置信度
        if not matches:
            return 0.0
        
        match_ratio = sum(matches) / len(matches)
        return match_ratio * pattern["confidence_weight"]
    
    def _create_root_cause(
        self,
        cause_type: str,
        confidence: float,
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any]
    ) -> RootCause:
        """创建根因对象"""
        descriptions = {
            "OOM": "内存不足导致容器被终止",
            "CPU_THROTTLING": "CPU限制导致性能下降",
            "CRASH_LOOP": "应用程序持续崩溃重启",
            "NETWORK_ISSUE": "网络连接问题",
            "IMAGE_PULL": "镜像拉取失败",
            "RESOURCE_QUOTA": "资源配额不足",
            "DISK_PRESSURE": "磁盘空间不足"
        }
        
        recommendations_map = {
            "OOM": [
                "增加容器内存限制",
                "优化应用程序内存使用",
                "检查内存泄漏"
            ],
            "CPU_THROTTLING": [
                "增加CPU限制",
                "优化应用程序CPU使用",
                "考虑水平扩展"
            ],
            "CRASH_LOOP": [
                "检查应用程序日志定位崩溃原因",
                "回滚到稳定版本",
                "增加健康检查和优雅关闭"
            ],
            "NETWORK_ISSUE": [
                "检查网络策略配置",
                "验证服务端点可达性",
                "检查DNS配置"
            ],
            "IMAGE_PULL": [
                "验证镜像仓库凭证",
                "检查镜像标签是否存在",
                "确认网络访问权限"
            ],
            "RESOURCE_QUOTA": [
                "申请增加资源配额",
                "优化资源使用",
                "清理未使用的资源"
            ],
            "DISK_PRESSURE": [
                "清理磁盘空间",
                "增加持久卷大小",
                "配置日志轮转"
            ]
        }
        
        # 收集受影响的组件
        affected_components = set()
        for event in event_patterns.get("critical_events", []):
            obj = event.get("object", {})
            if obj.get("name"):
                affected_components.add(f"{obj.get('kind', 'unknown')}/{obj.get('name')}")
        
        return RootCause(
            cause_type=cause_type,
            description=descriptions.get(cause_type, "未知问题"),
            confidence=confidence,
            affected_components=list(affected_components)[:5],
            evidence={
                "metrics": metric_anomalies.get("high_anomaly_metrics", [])[:3],
                "events": event_patterns.get("critical_events", [])[:3],
                "logs": list(log_patterns.get("error_types", {}).keys())[:3]
            },
            recommendations=recommendations_map.get(cause_type, ["请联系技术支持"])
        )
    
    def _build_timeline(
        self,
        metrics: List[MetricData],
        events: List[EventData],
        logs: List[LogData]
    ) -> List[Dict[str, Any]]:
        """构建时间线"""
        timeline = []
        
        # 添加关键事件
        for event in events[:20]:  # 限制数量
            if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                timeline.append({
                    "timestamp": event.timestamp.isoformat(),
                    "type": "event",
                    "severity": event.severity.value,
                    "description": f"{event.reason}: {event.message[:100]}"
                })
        
        # 添加错误日志
        for log in logs[:10]:  # 限制数量
            if log.level in ["ERROR", "FATAL"]:
                timeline.append({
                    "timestamp": log.timestamp.isoformat(),
                    "type": "log",
                    "severity": "high",
                    "description": f"[{log.pod_name}] {log.message[:100]}"
                })
        
        # 添加指标异常
        for metric in metrics:
            if metric.anomaly_score > 0.8:
                # 取最新的值
                if metric.values:
                    latest_value = metric.values[-1]
                    timeline.append({
                        "timestamp": latest_value["timestamp"],
                        "type": "metric",
                        "severity": "medium",
                        "description": f"指标异常 {metric.name}: score={metric.anomaly_score:.2f}"
                    })
        
        # 按时间排序
        timeline.sort(key=lambda x: x["timestamp"])
        
        return timeline[:50]  # 返回最多50个事件
    
    def _generate_recommendations(self, root_causes: List[RootCause]) -> List[str]:
        """生成综合建议"""
        recommendations = []
        
        # 添加每个根因的建议
        for cause in root_causes:
            recommendations.extend(cause.recommendations)
        
        # 添加通用建议
        if not root_causes:
            recommendations.extend([
                "启用更详细的日志记录",
                "增加监控覆盖范围",
                "检查最近的配置变更"
            ])
        
        # 去重并限制数量
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]
    
    def _calculate_confidence(self, root_causes: List[RootCause]) -> float:
        """计算总体置信度"""
        if not root_causes:
            return 0.0
        
        # 使用最高置信度
        return max(cause.confidence for cause in root_causes)
    
    def _check_threshold_violations(self, metric: MetricData) -> List[Dict[str, Any]]:
        """检查阈值违规"""
        violations = []
        
        # 定义常见指标的阈值
        thresholds = {
            "cpu": 0.8,  # 80% CPU使用率
            "memory": 0.9,  # 90% 内存使用率
            "disk": 0.85,  # 85% 磁盘使用率
            "error_rate": 0.01,  # 1% 错误率
        }
        
        metric_name_lower = metric.name.lower()
        
        for key, threshold in thresholds.items():
            if key in metric_name_lower:
                # 检查最新值
                if metric.values:
                    latest = metric.values[-1]
                    if latest["value"] > threshold:
                        violations.append({
                            "metric": metric.name,
                            "value": latest["value"],
                            "threshold": threshold,
                            "timestamp": latest["timestamp"]
                        })
        
        return violations
    
    def _detect_frequency_spikes(
        self, 
        event_clusters: Dict[str, List[EventData]]
    ) -> List[Dict[str, Any]]:
        """检测事件频率峰值"""
        spikes = []
        
        for cluster_key, events in event_clusters.items():
            if len(events) > 5:  # 事件数量阈值
                # 计算事件频率
                timestamps = [e.timestamp for e in events]
                if len(timestamps) > 1:
                    time_diffs = [
                        (timestamps[i+1] - timestamps[i]).total_seconds()
                        for i in range(len(timestamps)-1)
                    ]
                    avg_interval = np.mean(time_diffs)
                    
                    if avg_interval < 60:  # 平均间隔小于1分钟
                        spikes.append({
                            "cluster": cluster_key,
                            "count": len(events),
                            "avg_interval_seconds": avg_interval,
                            "first_event": timestamps[0].isoformat(),
                            "last_event": timestamps[-1].isoformat()
                        })
        
        return spikes
    
    def _find_temporal_correlations(
        self,
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any]
    ) -> Optional[CorrelationResult]:
        """查找时间相关性"""
        # 简化实现 - 实际应该使用更复杂的时间序列分析
        evidence = []
        
        if metric_anomalies.get("high_anomaly_metrics"):
            evidence.append("发现异常指标")
        if event_patterns.get("critical_events"):
            evidence.append("发现关键事件")
        if log_patterns.get("error_types"):
            evidence.append("发现错误日志")
        
        if len(evidence) >= 2:
            return CorrelationResult(
                confidence=0.7 + 0.1 * len(evidence),
                correlation_type="temporal",
                evidence=evidence,
                timeline=[]
            )
        
        return None
    
    def _find_component_correlations(
        self,
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any]
    ) -> Optional[CorrelationResult]:
        """查找组件相关性"""
        # 提取所有涉及的组件
        components = defaultdict(list)
        
        # 从事件中提取
        for event in event_patterns.get("critical_events", []):
            obj = event.get("object", {})
            if obj.get("name"):
                component_key = f"{obj.get('kind')}/{obj.get('name')}"
                components[component_key].append("event")
        
        # 从日志中提取
        for error_list in log_patterns.get("error_types", {}).values():
            for error in error_list:
                if error.get("pod"):
                    components[f"Pod/{error['pod']}"].append("log")
        
        # 查找多数据源都涉及的组件
        multi_source_components = [
            comp for comp, sources in components.items()
            if len(set(sources)) > 1
        ]
        
        if multi_source_components:
            return CorrelationResult(
                confidence=0.75,
                correlation_type="component",
                evidence=[f"组件 {comp} 在多个数据源中出现异常" for comp in multi_source_components[:3]],
                timeline=[]
            )
        
        return None
    
    def _find_causal_chain(
        self,
        metric_anomalies: Dict[str, Any],
        event_patterns: Dict[str, Any],
        log_patterns: Dict[str, Any]
    ) -> Optional[CorrelationResult]:
        """识别因果链"""
        # 简化的因果链检测
        chain = []
        
        # 检查常见的因果模式
        # 例如：资源不足 -> 调度失败 -> 服务不可用
        if any("InsufficientMemory" in e.get("reason", "") for e in event_patterns.get("critical_events", [])):
            chain.append("资源不足")
            
            if any("FailedScheduling" in e.get("reason", "") for e in event_patterns.get("critical_events", [])):
                chain.append("调度失败")
                
                if log_patterns.get("error_frequency"):
                    chain.append("服务异常")
        
        if len(chain) >= 2:
            return CorrelationResult(
                confidence=0.8,
                correlation_type="causal",
                evidence=[f"因果链: {' -> '.join(chain)}"],
                timeline=[]
            )
        
        return None
    
    def _derive_root_cause_from_correlation(
        self, 
        correlation: CorrelationResult
    ) -> Optional[RootCause]:
        """从关联结果推导根因"""
        if correlation.correlation_type == "causal" and correlation.confidence > 0.7:
            # 基于因果链创建根因
            return RootCause(
                cause_type="CORRELATION_BASED",
                description=f"基于关联分析: {correlation.evidence[0] if correlation.evidence else '复杂问题'}",
                confidence=correlation.confidence,
                affected_components=[],
                evidence={"correlation": correlation.evidence},
                recommendations=["基于关联分析，建议进一步调查相关组件"]
            )
        
        return None