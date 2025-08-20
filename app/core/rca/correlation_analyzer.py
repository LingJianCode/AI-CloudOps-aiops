#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 关联分析器 - 分析指标、事件、日志之间的关联关系
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
from collections import defaultdict

from app.models.rca_models import (
    MetricData,
    EventData,
    LogData,
    CorrelationResult,
    DataSourceType,
    SeverityLevel,
)

logger = logging.getLogger("aiops.rca.correlation")


class CorrelationAnalyzer:
    """
    关联分析器

    负责分析不同数据源（指标、事件、日志）之间的关联关系，
    包括时间相关性、统计相关性和语义相关性。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化关联分析器

        Args:
            config: 分析器配置
        """
        self.config = config or {}
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)
        self.time_window_seconds = self.config.get("time_window_seconds", 300)  # 5分钟
        self.max_time_offset = self.config.get("max_time_offset", 600)  # 10分钟

        self.logger = logging.getLogger("aiops.rca.correlation")

    async def analyze_correlations(
        self,
        metrics: List[MetricData],
        events: List[EventData],
        logs: List[LogData],
        time_window: int = 300,
    ) -> List[CorrelationResult]:
        """
        分析所有数据源之间的关联关系

        Args:
            metrics: 指标数据列表
            events: 事件数据列表
            logs: 日志数据列表
            time_window: 时间窗口（秒）

        Returns:
            List[CorrelationResult]: 关联分析结果列表
        """
        correlations = []

        try:
            self.logger.info(
                f"开始关联分析: {len(metrics)}个指标, {len(events)}个事件, {len(logs)}条日志"
            )

            # 1. 指标之间的关联分析
            metric_correlations = await self._analyze_metric_correlations(metrics)
            correlations.extend(metric_correlations)

            # 2. 指标与事件的关联分析
            metric_event_correlations = await self._analyze_metric_event_correlations(
                metrics, events, time_window
            )
            correlations.extend(metric_event_correlations)

            # 3. 指标与日志的关联分析
            metric_log_correlations = await self._analyze_metric_log_correlations(
                metrics, logs, time_window
            )
            correlations.extend(metric_log_correlations)

            # 4. 事件与日志的关联分析
            event_log_correlations = await self._analyze_event_log_correlations(
                events, logs, time_window
            )
            correlations.extend(event_log_correlations)

            # 5. 事件之间的关联分析
            event_correlations = await self._analyze_event_correlations(events, time_window)
            correlations.extend(event_correlations)

            # 按关联分数排序
            correlations.sort(key=lambda x: x.correlation_score, reverse=True)

            self.logger.info(f"完成关联分析，发现 {len(correlations)} 个关联关系")
            return correlations

        except Exception as e:
            self.logger.error(f"关联分析失败: {str(e)}")
            return []

    async def _analyze_metric_correlations(
        self, metrics: List[MetricData]
    ) -> List[CorrelationResult]:
        """
        分析指标之间的关联关系

        Args:
            metrics: 指标数据列表

        Returns:
            List[CorrelationResult]: 指标关联结果
        """
        correlations = []

        try:
            # 将指标数据转换为时间序列
            metric_series = {}
            for metric in metrics:
                if metric.values:
                    timestamps = [
                        pd.to_datetime(v["timestamp"]) for v in metric.values if v.get("timestamp")
                    ]
                    values = [v["value"] for v in metric.values]

                    if len(timestamps) > 10 and len(values) > 10:  # 需要足够的数据点
                        series = pd.Series(values, index=timestamps)
                        metric_series[metric.name] = series

            # 计算两两相关性
            metric_names = list(metric_series.keys())
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    metric1_name = metric_names[i]
                    metric2_name = metric_names[j]

                    correlation = await self._calculate_time_series_correlation(
                        metric_series[metric1_name], metric_series[metric2_name]
                    )

                    if correlation["score"] >= self.correlation_threshold:
                        correlations.append(
                            CorrelationResult(
                                source_type=DataSourceType.METRICS,
                                target_type=DataSourceType.METRICS,
                                source_identifier=metric1_name,
                                target_identifier=metric2_name,
                                correlation_score=correlation["score"],
                                temporal_offset=correlation["offset"],
                                evidence=[
                                    f"皮尔逊相关系数: {correlation.get('pearson', 0):.3f}",
                                    f"斯皮尔曼相关系数: {correlation.get('spearman', 0):.3f}",
                                    f"时间偏移: {correlation['offset']}秒",
                                ],
                            )
                        )

        except Exception as e:
            self.logger.warning(f"指标关联分析失败: {str(e)}")

        return correlations

    async def _analyze_metric_event_correlations(
        self, metrics: List[MetricData], events: List[EventData], time_window: int
    ) -> List[CorrelationResult]:
        """
        分析指标与事件的关联关系

        Args:
            metrics: 指标数据列表
            events: 事件数据列表
            time_window: 时间窗口

        Returns:
            List[CorrelationResult]: 指标-事件关联结果
        """
        correlations = []

        try:
            # 按严重程度筛选重要事件
            important_events = [
                e for e in events if e.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
            ]

            for metric in metrics:
                if metric.anomaly_score < 0.5:  # 只分析有异常的指标
                    continue

                for event in important_events:
                    correlation = await self._analyze_metric_event_temporal_correlation(
                        metric, event, time_window
                    )

                    if correlation["score"] >= 0.6:  # 指标-事件关联阈值较低
                        correlations.append(
                            CorrelationResult(
                                source_type=DataSourceType.METRICS,
                                target_type=DataSourceType.EVENTS,
                                source_identifier=metric.name,
                                target_identifier=f"{event.reason}:{event.involved_object.get('name', '')}",
                                correlation_score=correlation["score"],
                                temporal_offset=correlation["offset"],
                                evidence=correlation["evidence"],
                            )
                        )

        except Exception as e:
            self.logger.warning(f"指标-事件关联分析失败: {str(e)}")

        return correlations

    async def _analyze_metric_log_correlations(
        self, metrics: List[MetricData], logs: List[LogData], time_window: int
    ) -> List[CorrelationResult]:
        """
        分析指标与日志的关联关系

        Args:
            metrics: 指标数据列表
            logs: 日志数据列表
            time_window: 时间窗口

        Returns:
            List[CorrelationResult]: 指标-日志关联结果
        """
        correlations = []

        try:
            # 只分析错误日志
            error_logs = [log for log in logs if log.level in ["ERROR", "FATAL", "WARN", "WARNING"]]

            # 按Pod和错误类型分组日志
            log_groups = defaultdict(list)
            for log in error_logs:
                key = f"{log.pod_name}:{log.error_type or 'unknown'}"
                log_groups[key].append(log)

            for metric in metrics:
                if metric.anomaly_score < 0.5:
                    continue

                for log_group_key, group_logs in log_groups.items():
                    if len(group_logs) < 3:  # 至少需要3条日志
                        continue

                    correlation = await self._analyze_metric_log_temporal_correlation(
                        metric, group_logs, time_window
                    )

                    if correlation["score"] >= 0.6:
                        correlations.append(
                            CorrelationResult(
                                source_type=DataSourceType.METRICS,
                                target_type=DataSourceType.LOGS,
                                source_identifier=metric.name,
                                target_identifier=log_group_key,
                                correlation_score=correlation["score"],
                                temporal_offset=correlation["offset"],
                                evidence=correlation["evidence"],
                            )
                        )

        except Exception as e:
            self.logger.warning(f"指标-日志关联分析失败: {str(e)}")

        return correlations

    async def _analyze_event_log_correlations(
        self, events: List[EventData], logs: List[LogData], time_window: int
    ) -> List[CorrelationResult]:
        """
        分析事件与日志的关联关系

        Args:
            events: 事件数据列表
            logs: 日志数据列表
            time_window: 时间窗口

        Returns:
            List[CorrelationResult]: 事件-日志关联结果
        """
        correlations = []

        try:
            important_events = [
                e for e in events if e.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
            ]

            error_logs = [log for log in logs if log.level in ["ERROR", "FATAL", "WARN", "WARNING"]]

            for event in important_events:
                for log in error_logs:
                    correlation = await self._analyze_event_log_correlation(event, log, time_window)

                    if correlation["score"] >= 0.7:
                        correlations.append(
                            CorrelationResult(
                                source_type=DataSourceType.EVENTS,
                                target_type=DataSourceType.LOGS,
                                source_identifier=f"{event.reason}:{event.involved_object.get('name', '')}",
                                target_identifier=f"{log.pod_name}:{log.error_type or 'unknown'}",
                                correlation_score=correlation["score"],
                                temporal_offset=correlation["offset"],
                                evidence=correlation["evidence"],
                            )
                        )

        except Exception as e:
            self.logger.warning(f"事件-日志关联分析失败: {str(e)}")

        return correlations

    async def _analyze_event_correlations(
        self, events: List[EventData], time_window: int
    ) -> List[CorrelationResult]:
        """
        分析事件之间的关联关系

        Args:
            events: 事件数据列表
            time_window: 时间窗口

        Returns:
            List[CorrelationResult]: 事件关联结果
        """
        correlations = []

        try:
            important_events = [
                e for e in events if e.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
            ]

            for i in range(len(important_events)):
                for j in range(i + 1, len(important_events)):
                    event1 = important_events[i]
                    event2 = important_events[j]

                    correlation = await self._analyze_event_event_correlation(
                        event1, event2, time_window
                    )

                    if correlation["score"] >= 0.7:
                        correlations.append(
                            CorrelationResult(
                                source_type=DataSourceType.EVENTS,
                                target_type=DataSourceType.EVENTS,
                                source_identifier=f"{event1.reason}:{event1.involved_object.get('name', '')}",
                                target_identifier=f"{event2.reason}:{event2.involved_object.get('name', '')}",
                                correlation_score=correlation["score"],
                                temporal_offset=correlation["offset"],
                                evidence=correlation["evidence"],
                            )
                        )

        except Exception as e:
            self.logger.warning(f"事件关联分析失败: {str(e)}")

        return correlations

    async def _calculate_time_series_correlation(
        self, series1: pd.Series, series2: pd.Series
    ) -> Dict[str, Any]:
        """
        计算时间序列相关性

        Args:
            series1: 时间序列1
            series2: 时间序列2

        Returns:
            Dict[str, Any]: 相关性结果
        """
        try:
            # 对齐时间序列
            aligned = pd.concat([series1, series2], axis=1, join="inner")
            if len(aligned) < 10:
                return {"score": 0.0, "offset": 0}

            values1 = aligned.iloc[:, 0].values
            values2 = aligned.iloc[:, 1].values

            # 计算皮尔逊相关系数
            pearson_corr, _ = pearsonr(values1, values2)

            # 计算斯皮尔曼相关系数
            spearman_corr, _ = spearmanr(values1, values2)

            # 计算互相关以找到最佳时间偏移
            cross_corr = correlate(values1, values2, mode="full")
            max_corr_index = np.argmax(np.abs(cross_corr))
            offset = max_corr_index - len(values2) + 1

            # 综合评分
            score = (abs(pearson_corr) + abs(spearman_corr)) / 2

            return {
                "score": score,
                "offset": offset,
                "pearson": pearson_corr,
                "spearman": spearman_corr,
            }

        except Exception as e:
            self.logger.warning(f"时间序列相关性计算失败: {str(e)}")
            return {"score": 0.0, "offset": 0}

    async def _analyze_metric_event_temporal_correlation(
        self, metric: MetricData, event: EventData, time_window: int
    ) -> Dict[str, Any]:
        """
        分析指标与事件的时间相关性

        Args:
            metric: 指标数据
            event: 事件数据
            time_window: 时间窗口

        Returns:
            Dict[str, Any]: 相关性结果
        """
        try:
            # 检查时间接近性
            event_time = event.timestamp
            metric_times = [
                pd.to_datetime(v["timestamp"]) for v in metric.values if v.get("timestamp")
            ]

            if not metric_times:
                return {"score": 0.0, "offset": 0, "evidence": []}

            # 找到最接近事件时间的指标点
            time_diffs = [abs((t - event_time).total_seconds()) for t in metric_times]
            min_diff = min(time_diffs)

            if min_diff > time_window:
                return {"score": 0.0, "offset": 0, "evidence": []}

            # 计算时间相关性分数
            time_score = max(0, 1 - min_diff / time_window)

            # 考虑指标异常分数
            anomaly_score = metric.anomaly_score

            # 考虑事件严重程度
            severity_score = self._get_severity_score(event.severity)

            # 语义相关性（基于名称匹配）
            semantic_score = self._calculate_semantic_correlation(metric.name, event)

            # 综合评分
            final_score = (
                time_score * 0.4 + anomaly_score * 0.3 + severity_score * 0.2 + semantic_score * 0.1
            )

            evidence = [
                f"时间接近性: {min_diff:.1f}秒",
                f"指标异常分数: {anomaly_score:.3f}",
                f"事件严重程度: {event.severity.value}",
                f"语义相关性: {semantic_score:.3f}",
            ]

            return {"score": final_score, "offset": int(min_diff), "evidence": evidence}

        except Exception as e:
            self.logger.warning(f"指标-事件时间相关性分析失败: {str(e)}")
            return {"score": 0.0, "offset": 0, "evidence": []}

    async def _analyze_metric_log_temporal_correlation(
        self, metric: MetricData, logs: List[LogData], time_window: int
    ) -> Dict[str, Any]:
        """
        分析指标与日志组的时间相关性

        Args:
            metric: 指标数据
            logs: 日志数据列表
            time_window: 时间窗口

        Returns:
            Dict[str, Any]: 相关性结果
        """
        try:
            metric_times = [
                pd.to_datetime(v["timestamp"]) for v in metric.values if v.get("timestamp")
            ]
            if not metric_times:
                return {"score": 0.0, "offset": 0, "evidence": []}

            # 计算日志与指标的时间关联
            close_logs = 0
            total_offset = 0

            for log in logs:
                log_time = log.timestamp
                time_diffs = [abs((t - log_time).total_seconds()) for t in metric_times]
                min_diff = min(time_diffs)

                if min_diff <= time_window:
                    close_logs += 1
                    total_offset += min_diff

            if close_logs == 0:
                return {"score": 0.0, "offset": 0, "evidence": []}

            # 计算平均时间偏移
            avg_offset = total_offset / close_logs

            # 时间相关性分数
            time_score = close_logs / len(logs)  # 时间接近的日志比例

            # 异常分数
            anomaly_score = metric.anomaly_score

            # 综合评分
            final_score = time_score * 0.6 + anomaly_score * 0.4

            evidence = [
                f"时间接近的日志数: {close_logs}/{len(logs)}",
                f"平均时间偏移: {avg_offset:.1f}秒",
                f"指标异常分数: {anomaly_score:.3f}",
            ]

            return {"score": final_score, "offset": int(avg_offset), "evidence": evidence}

        except Exception as e:
            self.logger.warning(f"指标-日志时间相关性分析失败: {str(e)}")
            return {"score": 0.0, "offset": 0, "evidence": []}

    async def _analyze_event_log_correlation(
        self, event: EventData, log: LogData, time_window: int
    ) -> Dict[str, Any]:
        """
        分析事件与日志的关联性

        Args:
            event: 事件数据
            log: 日志数据
            time_window: 时间窗口

        Returns:
            Dict[str, Any]: 相关性结果
        """
        try:
            # 时间相关性
            time_diff = abs((event.timestamp - log.timestamp).total_seconds())
            if time_diff > time_window:
                return {"score": 0.0, "offset": 0, "evidence": []}

            time_score = max(0, 1 - time_diff / time_window)

            # 对象相关性（同一个Pod）
            object_score = 0.0
            event_pod = event.involved_object.get("name", "")
            if event_pod and event_pod == log.pod_name:
                object_score = 1.0
            elif event.involved_object.get("kind") == "Pod":
                object_score = 0.5

            # 语义相关性
            semantic_score = self._calculate_event_log_semantic_correlation(event, log)

            # 综合评分
            final_score = time_score * 0.4 + object_score * 0.4 + semantic_score * 0.2

            evidence = [
                f"时间差: {time_diff:.1f}秒",
                f"对象关联: {object_score:.1f}",
                f"语义相关性: {semantic_score:.3f}",
            ]

            return {"score": final_score, "offset": int(time_diff), "evidence": evidence}

        except Exception as e:
            self.logger.warning(f"事件-日志相关性分析失败: {str(e)}")
            return {"score": 0.0, "offset": 0, "evidence": []}

    async def _analyze_event_event_correlation(
        self, event1: EventData, event2: EventData, time_window: int
    ) -> Dict[str, Any]:
        """
        分析事件之间的关联性

        Args:
            event1: 事件1
            event2: 事件2
            time_window: 时间窗口

        Returns:
            Dict[str, Any]: 相关性结果
        """
        try:
            # 时间相关性
            time_diff = abs((event1.timestamp - event2.timestamp).total_seconds())
            if time_diff > time_window:
                return {"score": 0.0, "offset": 0, "evidence": []}

            time_score = max(0, 1 - time_diff / time_window)

            # 对象相关性
            object_score = 0.0
            if event1.involved_object.get("name") == event2.involved_object.get("name"):
                object_score = 1.0
            elif event1.involved_object.get("kind") == event2.involved_object.get("kind"):
                object_score = 0.5

            # 因果关系分析（某些事件类型通常相关）
            causal_score = self._calculate_event_causal_correlation(event1, event2)

            # 综合评分
            final_score = time_score * 0.4 + object_score * 0.4 + causal_score * 0.2

            evidence = [
                f"时间差: {time_diff:.1f}秒",
                f"对象关联: {object_score:.1f}",
                f"因果关联: {causal_score:.3f}",
            ]

            return {"score": final_score, "offset": int(time_diff), "evidence": evidence}

        except Exception as e:
            self.logger.warning(f"事件-事件相关性分析失败: {str(e)}")
            return {"score": 0.0, "offset": 0, "evidence": []}

    def _get_severity_score(self, severity: SeverityLevel) -> float:
        """
        获取严重程度分数

        Args:
            severity: 严重程度

        Returns:
            float: 分数
        """
        severity_scores = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.LOW: 0.2,
        }
        return severity_scores.get(severity, 0.0)

    def _calculate_semantic_correlation(self, metric_name: str, event: EventData) -> float:
        """
        计算指标名称与事件的语义相关性

        Args:
            metric_name: 指标名称
            event: 事件数据

        Returns:
            float: 语义相关性分数
        """
        metric_lower = metric_name.lower()
        event_reason_lower = event.reason.lower()
        event_message_lower = event.message.lower()

        # 关键词匹配
        if "cpu" in metric_lower and any(
            keyword in event_reason_lower or keyword in event_message_lower
            for keyword in ["cpu", "throttle", "limit"]
        ):
            return 0.9
        elif "memory" in metric_lower and any(
            keyword in event_reason_lower or keyword in event_message_lower
            for keyword in ["memory", "oom", "limit"]
        ):
            return 0.9
        elif "restart" in metric_lower and any(
            keyword in event_reason_lower for keyword in ["killing", "killed", "failed"]
        ):
            return 0.8
        elif "network" in metric_lower and any(
            keyword in event_reason_lower or keyword in event_message_lower
            for keyword in ["network", "connection", "timeout"]
        ):
            return 0.7

        return 0.0

    def _calculate_event_log_semantic_correlation(self, event: EventData, log: LogData) -> float:
        """
        计算事件与日志的语义相关性

        Args:
            event: 事件数据
            log: 日志数据

        Returns:
            float: 语义相关性分数
        """
        event_reason_lower = event.reason.lower()
        log_message_lower = log.message.lower()

        # 关键词匹配
        if "oom" in event_reason_lower and "memory" in log_message_lower:
            return 0.9
        elif "failed" in event_reason_lower and (
            "error" in log_message_lower or "exception" in log_message_lower
        ):
            return 0.8
        elif "unhealthy" in event_reason_lower and "health" in log_message_lower:
            return 0.8
        elif "backoff" in event_reason_lower and "error" in log_message_lower:
            return 0.7

        # 通用错误匹配
        if event.type == "Warning" and log.level in ["ERROR", "FATAL"]:
            return 0.5

        return 0.0

    def _calculate_event_causal_correlation(self, event1: EventData, event2: EventData) -> float:
        """
        计算事件之间的因果关联性

        Args:
            event1: 事件1
            event2: 事件2

        Returns:
            float: 因果关联分数
        """
        reason1 = event1.reason.lower()
        reason2 = event2.reason.lower()

        # 定义因果关系模式
        causal_patterns = [
            (["failedscheduling"], ["pending", "backoff"]),
            (["oomkilled"], ["killing", "failed"]),
            (["unhealthy"], ["killing", "failed"]),
            (["imagepullbackoff"], ["failed", "backoff"]),
            (["failedmount"], ["containercannotrun", "failed"]),
        ]

        for causes, effects in causal_patterns:
            if any(cause in reason1 for cause in causes) and any(
                effect in reason2 for effect in effects
            ):
                return 0.9
            elif any(cause in reason2 for cause in causes) and any(
                effect in reason1 for effect in effects
            ):
                return 0.9

        return 0.0
