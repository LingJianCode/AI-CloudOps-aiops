#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析引擎 - 整合多数据源进行智能根因分析
"""

import logging
import uuid
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional

from .collectors import MetricsCollector, EventsCollector, LogsCollector
from .correlation_analyzer import CorrelationAnalyzer
from app.models.rca_models import (
    RCARequest,
    RCAAnalysisResult,
    RootCause,
    MetricData,
    EventData,
    LogData,
    CorrelationResult,
    DataSourceType,
    SeverityLevel,
)
from app.services.llm import LLMService
from app.config.settings import config

logger = logging.getLogger("aiops.rca.engine")


class RCAEngine:
    """
    根因分析引擎

    这是RCA模块的核心组件，负责协调各个数据收集器和分析器，
    整合来自Prometheus指标、K8s事件和Pod日志的数据，
    通过关联分析和智能推理，生成根因分析结果。
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化根因分析引擎

        Args:
            config_dict: 引擎配置
        """
        self.config = config_dict or {}
        self.logger = logging.getLogger("aiops.rca.engine")

        # 初始化数据收集器
        self.metrics_collector = MetricsCollector(self.config.get("metrics", {}))
        self.events_collector = EventsCollector(self.config.get("events", {}))
        self.logs_collector = LogsCollector(self.config.get("logs", {}))

        # 初始化关联分析器
        self.correlation_analyzer = CorrelationAnalyzer(self.config.get("correlation", {}))

        # 初始化LLM服务用于摘要生成
        self.llm_service = LLMService()

        self._initialized = False

    async def initialize(self) -> None:
        """
        初始化引擎和所有组件

        允许部分数据收集器初始化失败，只要至少有一个可用即可
        """
        try:
            self.logger.info("开始初始化RCA引擎")

            # 记录初始化状态
            collector_status = {}

            # 尝试初始化指标收集器
            try:
                await self.metrics_collector.initialize()
                collector_status["metrics"] = True
                self.logger.info("指标收集器初始化成功")
            except Exception as e:
                collector_status["metrics"] = False
                self.logger.warning(f"指标收集器初始化失败: {str(e)}")

            # 尝试初始化事件收集器
            try:
                await self.events_collector.initialize()
                collector_status["events"] = True
                self.logger.info("事件收集器初始化成功")
            except Exception as e:
                collector_status["events"] = False
                self.logger.warning(f"事件收集器初始化失败: {str(e)}")

            # 尝试初始化日志收集器
            try:
                await self.logs_collector.initialize()
                collector_status["logs"] = True
                self.logger.info("日志收集器初始化成功")
            except Exception as e:
                collector_status["logs"] = False
                self.logger.warning(f"日志收集器初始化失败: {str(e)}")

            # 检查是否至少有一个收集器可用
            available_collectors = sum(collector_status.values())
            if available_collectors == 0:
                raise RuntimeError("所有数据收集器都初始化失败，无法提供RCA服务")

            self.collector_status = collector_status
            self._initialized = True

            available_list = [name for name, status in collector_status.items() if status]
            unavailable_list = [name for name, status in collector_status.items() if not status]

            self.logger.info(f"RCA引擎初始化完成")
            self.logger.info(f"可用收集器: {', '.join(available_list)}")
            if unavailable_list:
                self.logger.warning(f"不可用收集器: {', '.join(unavailable_list)}")

        except Exception as e:
            self.logger.error(f"RCA引擎初始化失败: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """
        检查RCA引擎健康状态

        Returns:
            bool: 引擎是否健康（至少有一个数据收集器可用）
        """
        if not self._initialized:
            return False

        # 检查是否有可用的收集器
        return any(self.collector_status.values())

    def get_status(self) -> Dict[str, Any]:
        """
        获取RCA引擎详细状态

        Returns:
            Dict[str, Any]: 引擎状态信息
        """
        if not self._initialized:
            return {"initialized": False, "collectors": {}, "healthy": False}

        return {
            "initialized": True,
            "collectors": self.collector_status.copy(),
            "healthy": any(self.collector_status.values()),
            "available_collectors": [
                name for name, status in self.collector_status.items() if status
            ],
            "unavailable_collectors": [
                name for name, status in self.collector_status.items() if not status
            ],
        }

    async def analyze(self, request: RCARequest) -> RCAAnalysisResult:
        """
        执行根因分析

        Args:
            request: 根因分析请求

        Returns:
            RCAAnalysisResult: 分析结果
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()
        request_id = str(uuid.uuid4())

        try:
            self.logger.info(
                f"开始根因分析 [{request_id}]: {request.namespace} "
                f"from {request.start_time} to {request.end_time}"
            )

            # 1. 数据收集阶段
            self.logger.info("开始数据收集阶段")
            collection_results = await self._collect_data(request)

            metrics_data = collection_results["metrics"]
            events_data = collection_results["events"]
            logs_data = collection_results["logs"]

            self.logger.info(
                f"数据收集完成: {len(metrics_data)}个指标, "
                f"{len(events_data)}个事件, {len(logs_data)}条日志"
            )

            # 2. 关联分析阶段
            self.logger.info("开始关联分析阶段")
            correlations = await self.correlation_analyzer.analyze_correlations(
                metrics_data, events_data, logs_data, request.correlation_window
            )

            self.logger.info(f"关联分析完成: 发现 {len(correlations)} 个关联关系")

            # 3. 根因推理阶段
            self.logger.info("开始根因推理阶段")
            root_causes = await self._generate_root_causes(
                metrics_data, events_data, logs_data, correlations, request
            )

            self.logger.info(f"根因推理完成: 生成 {len(root_causes)} 个根因候选")

            # 4. 生成分析摘要
            self.logger.info("开始生成分析摘要")
            analysis_summary = await self._generate_summary(
                metrics_data, events_data, logs_data, correlations, root_causes
            )

            # 5. 计算整体置信度
            confidence_score = self._calculate_overall_confidence(root_causes, correlations)

            # 构建分析结果
            processing_time = (datetime.now() - start_time).total_seconds()

            result = RCAAnalysisResult(
                request_id=request_id,
                analysis_time=datetime.now(),
                time_range={"start": request.start_time, "end": request.end_time},
                namespace=request.namespace,
                data_sources_analyzed=self._get_analyzed_data_sources(request),
                metrics_data=metrics_data,
                events_data=events_data,
                logs_data=logs_data,
                correlations=correlations,
                root_causes=root_causes,
                analysis_summary=analysis_summary,
                confidence_score=confidence_score,
                processing_time=processing_time,
            )

            self.logger.info(
                f"根因分析完成 [{request_id}]: "
                f"处理时间 {processing_time:.2f}秒, "
                f"置信度 {confidence_score:.3f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"根因分析失败 [{request_id}]: {str(e)}")
            raise

    async def _collect_data(self, request: RCARequest) -> Dict[str, List]:
        """
        收集所有数据源的数据

        Args:
            request: 分析请求

        Returns:
            Dict[str, List]: 收集到的数据
        """
        collection_kwargs = {
            "service_name": request.service_name,
            "metrics": request.metrics or config.rca.default_metrics,
        }

        # 初始化结果
        metrics_data = []
        events_data = []
        logs_data = []
        tasks = []
        task_names = []

        # 只收集可用收集器的数据
        if (
            self.collector_status.get("metrics", False)
            and request.metrics
            and len(request.metrics) > 0
        ):
            tasks.append(
                self.metrics_collector.collect_with_retry(
                    request.namespace, request.start_time, request.end_time, **collection_kwargs
                )
            )
            task_names.append("metrics")
        else:
            if not self.collector_status.get("metrics", False):
                self.logger.info("跳过指标收集 - 收集器不可用")
            elif not request.metrics:
                self.logger.info("跳过指标收集 - 未请求指标数据")

        if self.collector_status.get("events", False) and request.include_events:
            tasks.append(
                self.events_collector.collect_with_retry(
                    request.namespace, request.start_time, request.end_time
                )
            )
            task_names.append("events")
        else:
            if not self.collector_status.get("events", False):
                self.logger.info("跳过事件收集 - 收集器不可用")
            elif not request.include_events:
                self.logger.info("跳过事件收集 - 未请求事件数据")

        if self.collector_status.get("logs", False) and request.include_logs:
            tasks.append(
                self.logs_collector.collect_with_retry(
                    request.namespace,
                    request.start_time,
                    request.end_time,
                    error_only=True,
                    max_lines=500,
                )
            )
            task_names.append("logs")
        else:
            if not self.collector_status.get("logs", False):
                self.logger.info("跳过日志收集 - 收集器不可用")
            elif not request.include_logs:
                self.logger.info("跳过日志收集 - 未请求日志数据")

        # 并行执行所有可用的收集任务
        if tasks:
            try:
                import asyncio

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 按任务名称分配结果
                for i, (task_name, result) in enumerate(zip(task_names, results)):
                    if isinstance(result, Exception):
                        self.logger.warning(f"{task_name}收集失败: {result}")
                        result = []

                    if task_name == "metrics":
                        metrics_data = result
                    elif task_name == "events":
                        events_data = result
                    elif task_name == "logs":
                        logs_data = result

            except Exception as e:
                self.logger.error(f"数据收集过程中发生异常: {str(e)}")
        else:
            self.logger.warning("没有可用的数据收集器，返回空数据")

        return {"metrics": metrics_data or [], "events": events_data or [], "logs": logs_data or []}

    async def _generate_root_causes(
        self,
        metrics: List[MetricData],
        events: List[EventData],
        logs: List[LogData],
        correlations: List[CorrelationResult],
        request: RCARequest,
    ) -> List[RootCause]:
        """
        生成根因候选

        Args:
            metrics: 指标数据
            events: 事件数据
            logs: 日志数据
            correlations: 关联关系
            request: 分析请求

        Returns:
            List[RootCause]: 根因候选列表
        """
        root_causes = []

        try:
            # 1. 基于异常指标生成根因
            metric_root_causes = self._generate_metric_root_causes(
                metrics, correlations, request.severity_threshold
            )
            root_causes.extend(metric_root_causes)

            # 2. 基于严重事件生成根因
            event_root_causes = self._generate_event_root_causes(events, correlations)
            root_causes.extend(event_root_causes)

            # 3. 基于错误日志生成根因
            log_root_causes = self._generate_log_root_causes(logs, correlations)
            root_causes.extend(log_root_causes)

            # 4. 去重和排序
            root_causes = self._deduplicate_and_rank_root_causes(
                root_causes, request.max_candidates
            )

        except Exception as e:
            self.logger.error(f"生成根因候选失败: {str(e)}")

        return root_causes

    def _generate_metric_root_causes(
        self, metrics: List[MetricData], correlations: List[CorrelationResult], threshold: float
    ) -> List[RootCause]:
        """
        基于异常指标生成根因候选

        Args:
            metrics: 指标数据
            correlations: 关联关系
            threshold: 异常阈值

        Returns:
            List[RootCause]: 基于指标的根因候选
        """
        root_causes = []

        for metric in metrics:
            if metric.anomaly_score >= threshold:
                # 查找相关的关联关系
                related_correlations = [
                    corr
                    for corr in correlations
                    if corr.source_identifier == metric.name
                    or corr.target_identifier == metric.name
                ]

                # 计算置信度
                confidence = self._calculate_metric_confidence(metric, related_correlations)

                # 确定严重程度
                severity = self._determine_metric_severity(metric)

                # 生成描述和建议
                description = self._generate_metric_description(metric)
                recommendations = self._generate_metric_recommendations(metric)

                # 获取时间信息
                timestamps = [v["timestamp"] for v in metric.values if v.get("timestamp")]
                first_occurrence = min(timestamps) if timestamps else datetime.now()
                last_occurrence = max(timestamps) if timestamps else datetime.now()

                root_cause = RootCause(
                    identifier=metric.name,
                    data_source=DataSourceType.METRICS,
                    confidence=confidence,
                    severity=severity,
                    first_occurrence=pd.to_datetime(first_occurrence),
                    last_occurrence=pd.to_datetime(last_occurrence),
                    description=description,
                    impact_scope=self._determine_metric_impact_scope(metric),
                    correlations=related_correlations,
                    recommendations=recommendations,
                )

                root_causes.append(root_cause)

        return root_causes

    def _generate_event_root_causes(
        self, events: List[EventData], correlations: List[CorrelationResult]
    ) -> List[RootCause]:
        """
        基于严重事件生成根因候选

        Args:
            events: 事件数据
            correlations: 关联关系

        Returns:
            List[RootCause]: 基于事件的根因候选
        """
        root_causes = []

        # 只考虑高严重程度的事件
        critical_events = [
            e for e in events if e.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        ]

        for event in critical_events:
            event_id = f"{event.reason}:{event.involved_object.get('name', '')}"

            # 查找相关的关联关系
            related_correlations = [
                corr
                for corr in correlations
                if corr.source_identifier == event_id or corr.target_identifier == event_id
            ]

            # 计算置信度
            confidence = self._calculate_event_confidence(event, related_correlations)

            # 生成描述和建议
            description = self._generate_event_description(event)
            recommendations = self._generate_event_recommendations(event)

            root_cause = RootCause(
                identifier=event_id,
                data_source=DataSourceType.EVENTS,
                confidence=confidence,
                severity=event.severity,
                first_occurrence=event.timestamp,
                last_occurrence=event.timestamp,
                description=description,
                impact_scope=self._determine_event_impact_scope(event),
                correlations=related_correlations,
                recommendations=recommendations,
            )

            root_causes.append(root_cause)

        return root_causes

    def _generate_log_root_causes(
        self, logs: List[LogData], correlations: List[CorrelationResult]
    ) -> List[RootCause]:
        """
        基于错误日志生成根因候选

        Args:
            logs: 日志数据
            correlations: 关联关系

        Returns:
            List[RootCause]: 基于日志的根因候选
        """
        root_causes = []

        # 按错误类型分组日志
        from collections import defaultdict

        error_groups = defaultdict(list)

        error_logs = [log for log in logs if log.level in ["ERROR", "FATAL"] and log.error_type]

        for log in error_logs:
            key = f"{log.pod_name}:{log.error_type}"
            error_groups[key].append(log)

        # 为每个错误组生成根因候选
        for group_key, group_logs in error_groups.items():
            if len(group_logs) < 3:  # 至少需要3个相同错误才考虑
                continue

            # 查找相关的关联关系
            related_correlations = [
                corr
                for corr in correlations
                if corr.source_identifier == group_key or corr.target_identifier == group_key
            ]

            # 计算置信度
            confidence = self._calculate_log_confidence(group_logs, related_correlations)

            # 确定严重程度
            severity = self._determine_log_severity(group_logs)

            # 生成描述和建议
            description = self._generate_log_description(group_logs)
            recommendations = self._generate_log_recommendations(group_logs)

            # 获取时间信息
            timestamps = [log.timestamp for log in group_logs]
            first_occurrence = min(timestamps)
            last_occurrence = max(timestamps)

            root_cause = RootCause(
                identifier=group_key,
                data_source=DataSourceType.LOGS,
                confidence=confidence,
                severity=severity,
                first_occurrence=first_occurrence,
                last_occurrence=last_occurrence,
                description=description,
                impact_scope=self._determine_log_impact_scope(group_logs),
                correlations=related_correlations,
                recommendations=recommendations,
            )

            root_causes.append(root_cause)

        return root_causes

    def _deduplicate_and_rank_root_causes(
        self, root_causes: List[RootCause], max_candidates: int
    ) -> List[RootCause]:
        """
        去重和排序根因候选

        Args:
            root_causes: 原始根因候选列表
            max_candidates: 最大候选数量

        Returns:
            List[RootCause]: 去重并排序后的根因候选
        """

        # 按置信度和严重程度排序
        def sort_key(rc):
            severity_weight = {
                SeverityLevel.CRITICAL: 4,
                SeverityLevel.HIGH: 3,
                SeverityLevel.MEDIUM: 2,
                SeverityLevel.LOW: 1,
            }
            return (rc.confidence * 0.7 + severity_weight.get(rc.severity, 0) * 0.1, rc.confidence)

        root_causes.sort(key=sort_key, reverse=True)

        # 简单去重（基于标识符）
        seen_identifiers = set()
        unique_root_causes = []

        for rc in root_causes:
            if rc.identifier not in seen_identifiers:
                seen_identifiers.add(rc.identifier)
                unique_root_causes.append(rc)

        return unique_root_causes[:max_candidates]

    async def _generate_summary(
        self,
        metrics: List[MetricData],
        events: List[EventData],
        logs: List[LogData],
        correlations: List[CorrelationResult],
        root_causes: List[RootCause],
    ) -> str:
        """
        生成分析摘要

        Args:
            metrics: 指标数据
            events: 事件数据
            logs: 日志数据
            correlations: 关联关系
            root_causes: 根因候选

        Returns:
            str: 分析摘要
        """
        try:
            # 构建摘要数据 - 按LLM服务期望的格式
            anomalies_data = {
                "metrics_count": len(metrics),
                "events_count": len(events), 
                "logs_count": len(logs),
                "anomalous_metrics": [
                    {
                        "name": m.name,
                        "anomaly_score": m.anomaly_score,
                        "trend": m.trend
                    }
                    for m in metrics if m.anomaly_score > 0.7
                ]
            }
            
            correlations_data = {
                "correlation_count": len(correlations),
                "correlations": [
                    {
                        "source": c.source_identifier,
                        "target": c.target_identifier,
                        "score": c.correlation_score,
                        "type": f"{c.source_type.value}->{c.target_type.value}"
                    }
                    for c in correlations[:10]  # 只包含前10个
                ]
            }
            
            candidates_data = [
                {
                    "identifier": rc.identifier,
                    "data_source": rc.data_source.value,
                    "confidence": rc.confidence,
                    "severity": rc.severity.value,
                    "description": rc.description,
                    "impact_scope": rc.impact_scope,
                    "recommendations": rc.recommendations[:3]  # 只包含前3个建议
                }
                for rc in root_causes[:5]  # 只包含前5个根因
            ]

            # 使用LLM生成摘要
            summary = await self.llm_service.generate_rca_summary(
                anomalies_data, correlations_data, candidates_data
            )

            # 清理摘要中的控制字符以确保JSON兼容性
            if summary:
                import re
                import json
                # 移除所有控制字符（除了换行符、制表符、回车符）
                summary = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', summary)
                # 替换不可见字符
                summary = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]', '', summary)
                # 确保摘要不为空，并且可以被JSON序列化
                summary = summary.strip()
                if summary:
                    try:
                        # 测试JSON序列化
                        json.dumps(summary)
                    except (TypeError, ValueError):
                        # 如果仍然无法序列化，使用ASCII编码清理
                        summary = summary.encode('ascii', 'ignore').decode('ascii')

            return summary or self._generate_fallback_summary(metrics, events, logs, root_causes)

        except Exception as e:
            self.logger.warning(f"生成AI摘要失败: {str(e)}")
            return self._generate_fallback_summary(metrics, events, logs, root_causes)

    def _generate_fallback_summary(
        self,
        metrics: List[MetricData],
        events: List[EventData],
        logs: List[LogData],
        root_causes: List[RootCause],
    ) -> str:
        """
        生成备用摘要（不使用LLM）

        Args:
            metrics: 指标数据
            events: 事件数据
            logs: 日志数据
            root_causes: 根因候选

        Returns:
            str: 备用摘要
        """
        if not root_causes:
            return "未发现明显的异常模式，系统运行正常。"

        top_cause = root_causes[0]

        summary_parts = [
            f"根因分析完成，共分析了 {len(metrics)} 个指标、{len(events)} 个事件和 {len(logs)} 条日志。"
        ]

        if root_causes:
            summary_parts.append(
                f"最可能的根因是 {top_cause.identifier}（{top_cause.data_source.value}），"
                f"置信度 {top_cause.confidence:.2f}，严重程度 {top_cause.severity.value}。"
            )

            if len(root_causes) > 1:
                summary_parts.append(f"共发现 {len(root_causes)} 个根因候选。")

        # 生成并清理摘要
        summary = " ".join(summary_parts)
        # 清理控制字符
        import re
        summary = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', summary)
        summary = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]', '', summary)
        return summary.strip()

    def _calculate_overall_confidence(
        self, root_causes: List[RootCause], correlations: List[CorrelationResult]
    ) -> float:
        """
        计算整体置信度

        Args:
            root_causes: 根因候选
            correlations: 关联关系

        Returns:
            float: 整体置信度
        """
        if not root_causes:
            return 0.0

        # 基于最高置信度的根因
        max_confidence = max(rc.confidence for rc in root_causes)

        # 考虑关联关系的数量和质量
        if correlations:
            avg_correlation_score = sum(c.correlation_score for c in correlations) / len(
                correlations
            )
            correlation_factor = min(len(correlations) / 10, 1.0)  # 最多贡献1.0

            overall_confidence = (
                max_confidence * 0.7 + avg_correlation_score * 0.2 + correlation_factor * 0.1
            )
        else:
            overall_confidence = max_confidence * 0.8  # 没有关联关系时降低置信度

        return min(overall_confidence, 1.0)

    def _get_analyzed_data_sources(self, request: RCARequest) -> List[DataSourceType]:
        """
        获取分析的数据源类型

        Args:
            request: 分析请求

        Returns:
            List[DataSourceType]: 数据源类型列表
        """
        sources = []

        if request.metrics:
            sources.append(DataSourceType.METRICS)
        if request.include_events:
            sources.append(DataSourceType.EVENTS)
        if request.include_logs:
            sources.append(DataSourceType.LOGS)

        return sources

    # 辅助方法的实现（简化版本）
    def _calculate_metric_confidence(
        self, metric: MetricData, correlations: List[CorrelationResult]
    ) -> float:
        """计算指标根因的置信度"""
        base_confidence = metric.anomaly_score
        correlation_boost = min(len(correlations) * 0.1, 0.3)
        return min(base_confidence + correlation_boost, 1.0)

    def _determine_metric_severity(self, metric: MetricData) -> SeverityLevel:
        """确定指标的严重程度"""
        if metric.anomaly_score >= 0.9:
            return SeverityLevel.CRITICAL
        elif metric.anomaly_score >= 0.7:
            return SeverityLevel.HIGH
        elif metric.anomaly_score >= 0.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _generate_metric_description(self, metric: MetricData) -> str:
        """生成指标描述"""
        return (
            f"指标 {metric.name} 出现异常，异常分数 {metric.anomaly_score:.3f}，趋势 {metric.trend}"
        )

    def _generate_metric_recommendations(self, metric: MetricData) -> List[str]:
        """生成指标相关的建议"""
        recommendations = []
        metric_lower = metric.name.lower()

        if "cpu" in metric_lower:
            recommendations.append("检查CPU使用率，考虑扩容或优化应用性能")
        elif "memory" in metric_lower:
            recommendations.append("检查内存使用情况，考虑增加内存限制或优化内存使用")
        elif "restart" in metric_lower:
            recommendations.append("检查容器重启原因，查看相关日志和健康检查配置")
        else:
            recommendations.append("监控指标变化趋势，分析可能的影响因素")

        return recommendations

    def _determine_metric_impact_scope(self, metric: MetricData) -> List[str]:
        """确定指标影响范围"""
        scope = []
        if "pod" in metric.labels:
            scope.append(f"Pod: {metric.labels['pod']}")
        if "container" in metric.labels:
            scope.append(f"Container: {metric.labels['container']}")
        return scope

    def _calculate_event_confidence(
        self, event: EventData, correlations: List[CorrelationResult]
    ) -> float:
        """计算事件根因的置信度"""
        severity_scores = {
            SeverityLevel.CRITICAL: 0.9,
            SeverityLevel.HIGH: 0.7,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.LOW: 0.3,
        }
        base_confidence = severity_scores.get(event.severity, 0.3)
        correlation_boost = min(len(correlations) * 0.1, 0.2)
        return min(base_confidence + correlation_boost, 1.0)

    def _generate_event_description(self, event: EventData) -> str:
        """生成事件描述"""
        return f"事件 {event.reason}: {event.message}"

    def _generate_event_recommendations(self, event: EventData) -> List[str]:
        """生成事件相关的建议"""
        recommendations = []
        reason_lower = event.reason.lower()

        if "oom" in reason_lower:
            recommendations.append("容器内存不足，需要增加内存限制或优化内存使用")
        elif "failed" in reason_lower:
            recommendations.append("检查失败原因，查看详细错误信息和日志")
        elif "unhealthy" in reason_lower:
            recommendations.append("检查健康检查配置和应用状态")
        else:
            recommendations.append("查看事件详细信息，分析可能的解决方案")

        return recommendations

    def _determine_event_impact_scope(self, event: EventData) -> List[str]:
        """确定事件影响范围"""
        scope = []
        obj = event.involved_object
        if obj.get("kind") and obj.get("name"):
            scope.append(f"{obj['kind']}: {obj['name']}")
        return scope

    def _calculate_log_confidence(
        self, logs: List[LogData], correlations: List[CorrelationResult]
    ) -> float:
        """计算日志根因的置信度"""
        error_count = len([log for log in logs if log.level in ["ERROR", "FATAL"]])
        frequency_score = min(error_count / 10, 0.8)  # 错误频率分数
        correlation_boost = min(len(correlations) * 0.1, 0.2)
        return min(frequency_score + correlation_boost, 1.0)

    def _determine_log_severity(self, logs: List[LogData]) -> SeverityLevel:
        """确定日志的严重程度"""
        fatal_count = len([log for log in logs if log.level == "FATAL"])
        error_count = len([log for log in logs if log.level == "ERROR"])

        if fatal_count > 0:
            return SeverityLevel.CRITICAL
        elif error_count >= 5:
            return SeverityLevel.HIGH
        elif error_count >= 2:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _generate_log_description(self, logs: List[LogData]) -> str:
        """生成日志描述"""
        first_log = logs[0]
        return f"Pod {first_log.pod_name} 出现 {first_log.error_type} 错误，共 {len(logs)} 次"

    def _generate_log_recommendations(self, logs: List[LogData]) -> List[str]:
        """生成日志相关的建议"""
        recommendations = []
        error_type = logs[0].error_type or ""

        if "java exception" in error_type.lower():
            recommendations.append("检查Java应用堆栈跟踪，分析异常原因")
        elif "python exception" in error_type.lower():
            recommendations.append("检查Python应用错误，查看异常堆栈")
        elif "timeout" in error_type.lower():
            recommendations.append("检查网络连接和服务响应时间")
        else:
            recommendations.append("查看详细日志内容，分析错误模式和频率")

        return recommendations

    def _determine_log_impact_scope(self, logs: List[LogData]) -> List[str]:
        """确定日志影响范围"""
        pods = set(log.pod_name for log in logs)
        containers = set(log.container_name for log in logs)

        scope = []
        for pod in pods:
            scope.append(f"Pod: {pod}")
        for container in containers:
            scope.append(f"Container: {container}")

        return scope

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 引擎是否健康
        """
        try:
            if not self._initialized:
                return False

            # 检查所有收集器的健康状态
            metrics_health = await self.metrics_collector.health_check()
            events_health = await self.events_collector.health_check()
            logs_health = await self.logs_collector.health_check()

            return metrics_health and events_health and logs_health

        except Exception:
            return False
