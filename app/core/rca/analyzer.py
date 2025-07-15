#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 系统根因分析
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from app.config.settings import config  # 系统配置
from app.core.rca.correlator import CorrelationAnalyzer  # 相关性分析器

# ==================== 内部组件导入 ====================
from app.core.rca.detector import AnomalyDetector  # 异常检测器
from app.models.response_models import (  # 响应模型
    AnomalyInfo,
    RCAResponse,
    RootCauseCandidate,
)
from app.services.llm import LLMService  # 大语言模型服务
from app.services.prometheus import PrometheusService  # Prometheus数据服务

logger = logging.getLogger("aiops.rca")


class RCAAnalyzer:
    """
    根因分析器 - AI-CloudOps系统的核心故障诊断引擎

    这个类是整个根因分析系统的主入口，集成了异常检测、相关性分析、
    根因推理和AI摘要生成等功能。它能够从纷繁复杂的系统指标中提取出
    有意义的模式，并生成可操作的分析结果。

    工作流程：
    1. 数据收集 - 从监控系统获取指标数据
    2. 异常检测 - 识别异常指标和异常时间点
    3. 相关性分析 - 分析指标间的因果关系
    4. 根因推理 - 生成和排序根因候选
    5. 结果整合 - 生成综合分析报告

    核心组件：
    - AnomalyDetector: 异常检测引擎
    - CorrelationAnalyzer: 相关性分析引擎
    - PrometheusService: 数据获取服务
    - LLMService: AI摘要生成服务

    Attributes:
        prometheus (PrometheusService): Prometheus数据服务实例
        detector (AnomalyDetector): 异常检测器实例
        correlator (CorrelationAnalyzer): 相关性分析器实例
        llm (LLMService): 大语言模型服务实例
    """

    def __init__(self):
        """
        初始化根因分析器

        初始化所有必要的组件和服务，包括数据源、分析引擎和配置参数。
        使用配置文件中的参数来设置各个组件的阈值和参数。
        """
        # 初始化Prometheus数据服务，用于获取监控指标
        self.prometheus = PrometheusService()

        # 初始化异常检测器，使用配置中的阈值
        self.detector = AnomalyDetector(config.rca.anomaly_threshold)

        # 初始化相关性分析器，使用配置中的相关性阈值
        self.correlator = CorrelationAnalyzer(config.rca.correlation_threshold)

        # 初始化大语言模型服务，用于生成AI摘要
        self.llm = LLMService()

        logger.info("根因分析器初始化完成")

    async def analyze(
        self, start_time: datetime, end_time: datetime, metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        执行全面的根因分析

        这是根因分析的主入口方法，它统筹整个分析流程，从数据收集到
        最终的结果输出。方法支持自定义时间范围和指标列表，并提供
        完整的错误处理和日志记录。

        Args:
            start_time (datetime): 分析起始时间
            end_time (datetime): 分析结束时间
            metrics (Optional[List[str]]): 要分析的指标列表，为None时使用默认指标

        Returns:
            Dict: 分析结果，包含：
                - status: 分析状态（success/error）
                - anomalies: 检测到的异常指标和详细信息
                - correlations: 指标间的相关性分析结果
                - root_cause_candidates: 排序后的根因候选列表
                - analysis_time: 分析执行时间
                - time_range: 分析的时间范围
                - metrics_analyzed: 实际分析的指标列表
                - summary: AI生成的分析摘要
                - statistics: 分析过程的统计信息

        分析流程：
        1. 参数验证和初始化
        2. 数据收集和预处理
        3. 异常检测和标记
        4. 相关性分析和计算
        5. 根因候选生成和排序
        6. AI摘要生成和结果整合
        """
        try:
            logger.info(f"开始根因分析: {start_time} - {end_time}")

            # 准备分析指标列表 - 使用默认指标如果未提供
            if not metrics:
                metrics = config.rca.default_metrics

            # 收集监控指标数据，这是整个分析的数据基础
            metrics_data = await self._collect_metrics_data(start_time, end_time, metrics)

            # 数据有效性检查 - 确保有足够的数据进行分析
            if not metrics_data:
                return {"error": "未获取到有效的监控数据"}

            logger.info(f"收集到 {len(metrics_data)} 个指标的数据")

            # 异常检测阶段 - 识别所有异常指标和时间点
            anomalies = await self.detector.detect_anomalies(metrics_data)
            logger.info(f"检测到 {len(anomalies)} 个指标存在异常")

            # 相关性分析阶段 - 分析指标间的因果关系
            correlations = await self.correlator.analyze_correlations(metrics_data)
            logger.info(f"分析了 {len(correlations)} 个指标的相关性")

            # 根因候选生成阶段 - 综合异常和相关性信息生成可能的根因
            root_causes = self._generate_root_cause_candidates(anomalies, correlations)

            # AI摘要生成阶段 - 使用LLM生成人类可读的分析报告
            summary = await self._generate_summary(anomalies, correlations, root_causes)

            # 构建综合分析结果响应
            response = {
                "status": "success",
                # 异常信息：将内部数据结构转换为响应格式
                "anomalies": {
                    metric: AnomalyInfo(**info).__dict__ for metric, info in anomalies.items()
                },
                # 相关性分析结果
                "correlations": correlations,
                # 根因候选列表：按置信度排序
                "root_cause_candidates": [
                    RootCauseCandidate(**candidate).__dict__ for candidate in root_causes
                ],
                # 分析时间和统计信息
                "analysis_time": datetime.utcnow().isoformat(),
                "time_range": {"start": start_time.isoformat(), "end": end_time.isoformat()},
                "metrics_analyzed": list(metrics_data.keys()),
                "summary": summary,
                # 详细的分析统计信息
                "statistics": {
                    "total_metrics": len(metrics_data),
                    "anomalous_metrics": len(anomalies),
                    "correlation_pairs": sum(len(corrs) for corrs in correlations.values()),
                    "analysis_duration": (datetime.utcnow() - start_time).total_seconds(),
                },
            }

            logger.info("根因分析完成")
            return response

        except Exception as e:
            logger.error(f"根因分析失败: {str(e)}")
            return {"error": f"分析失败: {str(e)}"}

    async def _collect_metrics_data(
        self, start_time: datetime, end_time: datetime, metrics: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """收集指标数据"""
        metrics_data = {}

        for metric in metrics:
            try:
                data = await self.prometheus.query_range(metric, start_time, end_time, "1m")

                if data is not None and not data.empty:
                    # 处理多个时间序列
                    if len(data) > 0:
                        # 如果有多个系列，按标签分组
                        if "label_pod" in data.columns:
                            # 按pod分组
                            grouped_data = {}
                            for pod in data["label_pod"].unique():
                                if pd.notna(pod):
                                    pod_data = data[data["label_pod"] == pod]
                                    if not pod_data.empty:
                                        metric_name = f"{metric}|pod:{pod}"
                                        grouped_data[metric_name] = pod_data[["value"]]
                            metrics_data.update(grouped_data)
                        elif "label_container" in data.columns:
                            # 按容器分组
                            grouped_data = {}
                            for container in data["label_container"].unique():
                                if pd.notna(container):
                                    container_data = data[data["label_container"] == container]
                                    if not container_data.empty:
                                        metric_name = f"{metric}|container:{container}"
                                        grouped_data[metric_name] = container_data[["value"]]
                            metrics_data.update(grouped_data)
                        else:
                            # 单个序列或聚合数据
                            metrics_data[metric] = data[["value"]]

            except Exception as e:
                logger.warning(f"获取指标 {metric} 失败: {str(e)}")
                continue

        # 过滤掉空数据
        metrics_data = {k: v for k, v in metrics_data.items() if not v.empty}

        logger.info(f"成功收集 {len(metrics_data)} 个时间序列数据")
        return metrics_data

    def _generate_root_cause_candidates(self, anomalies: Dict, correlations: Dict) -> List[Dict]:
        """生成根因候选列表"""
        candidates = []

        try:
            # 基于异常分数生成候选
            for metric, anomaly_info in anomalies.items():
                if anomaly_info.get("count", 0) > 0:
                    # 计算置信度
                    confidence = self._calculate_confidence(
                        anomaly_info, correlations.get(metric, [])
                    )

                    # 生成描述
                    description = self._generate_description(metric, anomaly_info)

                    candidate = {
                        "metric": metric,
                        "confidence": confidence,
                        "first_occurrence": anomaly_info.get("first_occurrence"),
                        "anomaly_count": anomaly_info.get("count"),
                        "related_metrics": correlations.get(metric, []),
                        "description": description,
                    }
                    candidates.append(candidate)

            # 按置信度排序
            candidates.sort(key=lambda x: x["confidence"], reverse=True)

            # 返回前5个候选
            return candidates[:5]

        except Exception as e:
            logger.error(f"生成根因候选失败: {str(e)}")
            return []

    def _calculate_confidence(self, anomaly_info: Dict, related_metrics: List) -> float:
        """
        计算根因候选的置信度评分

        综合多个因素计算每个根因候选的可信度，包括异常严重程度、
        持续时间、相关性强度和检测一致性等。置信度越高，
        表示该指标越可能是故障的根本原因。

        Args:
            anomaly_info (Dict): 异常信息，包含异常分数、数量等
            related_metrics (List): 相关指标列表

        Returns:
            float: 置信度评分（0.0-1.0之间）

        计算公式：
        基础置信度 + 持续性加权 + 相关性加权 + 一致性加权
        所有加权都有上限，以防止过度拟合。
        """
        try:
            # 基础置信度：基于异常的最高分数，反映异常的严重程度
            base_confidence = min(anomaly_info.get("max_score", 0), 1.0)

            # 持续性加权：异常持续时间越长，越可能是根因
            count = anomaly_info.get("count", 0)
            count_factor = min(count / 20, 0.3)  # 最多加 0.3 分

            # 相关性加权：相关指标越多，越可能是根因
            correlation_factor = min(len(related_metrics) * 0.05, 0.2)  # 最多加 0.2 分

            # 检测方法一致性加权：多种检测方法都检测到异常
            detection_methods = anomaly_info.get("detection_methods", {})
            method_consistency = sum(
                1 for v in detection_methods.values() if isinstance(v, (int, float)) and v > 0
            )
            consistency_factor = min(method_consistency * 0.05, 0.15)  # 最多加 0.15 分

            # 综合置信度计算
            confidence = base_confidence + count_factor + correlation_factor + consistency_factor

            # 确保置信度不超过 1.0
            return min(confidence, 1.0)

        except Exception:
            # 异常情况下返回中等置信度
            return 0.0

    def _generate_description(self, metric: str, anomaly_info: Dict) -> str:
        """生成根因描述"""
        try:
            count = anomaly_info.get("count", 0)
            max_score = anomaly_info.get("max_score", 0)
            avg_score = anomaly_info.get("avg_score", 0)

            # 基于指标名称生成描述
            metric_lower = metric.lower()

            if "cpu" in metric_lower:
                return f"CPU使用率异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            elif "memory" in metric_lower:
                return f"内存使用异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            elif "restart" in metric_lower:
                return f"容器重启异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            elif any(keyword in metric_lower for keyword in ["network", "http", "request"]):
                return f"网络/HTTP请求异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            elif "disk" in metric_lower or "storage" in metric_lower:
                return f"磁盘/存储异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            elif "node" in metric_lower:
                return f"节点指标异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            elif "pod" in metric_lower:
                return f"Pod状态异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"
            else:
                return f"指标 {metric} 异常，检测到 {count} 个异常点，最高异常分数 {max_score:.2f}，平均异常分数 {avg_score:.2f}"

        except Exception:
            return f"指标 {metric} 存在异常"

    async def _generate_summary(
        self, anomalies: Dict, correlations: Dict, candidates: List[Dict]
    ) -> Optional[str]:
        """生成AI摘要"""
        try:
            if not candidates:
                return "未发现明显的异常模式，系统运行正常。"

            # 调用LLM生成摘要
            summary = await self.llm.generate_rca_summary(anomalies, correlations, candidates)

            return summary or "无法生成分析摘要，但检测到异常模式。"

        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            return None

    async def analyze_specific_incident(
        self,
        start_time: datetime,
        end_time: datetime,
        affected_services: List[str],
        symptoms: List[str],
    ) -> Dict:
        """
        分析特定事件的根因 - 针对具体故障事件的定制化分析

        这个方法专门用于分析已知的故障事件，它会根据受影响的服务和观察到的症状
        来选择最相关的指标进行分析，并提供针对性的修复建议。这比通用分析更精确，
        因为它利用了更多的上下文信息。

        Args:
            start_time (datetime): 事件开始时间
            end_time (datetime): 事件结束时间
            affected_services (List[str]): 受影响的服务列表
            symptoms (List[str]): 观察到的症状描述列表

        Returns:
            Dict: 事件分析结果，包含标准分析结果加上：
                - incident_analysis: 事件特定的分析信息
                  - affected_services: 受影响的服务
                  - reported_symptoms: 报告的症状
                  - relevant_metrics: 选择的相关指标
                  - recommendation: 针对性的修复建议

        分析流程：
        1. 根据服务和症状选择相关指标
        2. 执行针对性的根因分析
        3. 生成定制化的修复建议
        4. 整合事件特定的上下文信息
        """
        try:
            logger.info(f"分析特定事件: 服务={affected_services}, 症状={symptoms}")

            # 基于受影响的服务和症状，智能选择相关指标进行分析
            relevant_metrics = self._select_relevant_metrics(affected_services, symptoms)

            # 执行针对性的根因分析，使用筛选后的指标集
            result = await self.analyze(start_time, end_time, relevant_metrics)

            # 如果基础分析成功，添加事件特定的分析结果
            if "error" not in result:
                # 构建事件特定的分析信息
                result["incident_analysis"] = {
                    "affected_services": affected_services,
                    "reported_symptoms": symptoms,
                    "relevant_metrics": relevant_metrics,
                    "recommendation": self._generate_incident_recommendation(
                        result.get("root_cause_candidates", []), affected_services, symptoms
                    ),
                }

            return result

        except Exception as e:
            logger.error(f"特定事件分析失败: {str(e)}")
            return {"error": f"事件分析失败: {str(e)}"}

    def _select_relevant_metrics(self, services: List[str], symptoms: List[str]) -> List[str]:
        """
        基于服务和症状选择相关指标 - 智能指标选择算法

        根据受影响的服务类型和观察到的症状，智能选择最相关的监控指标。
        这个方法使用启发式规则来映射症状到具体的指标，提高分析的针对性和效率。

        Args:
            services (List[str]): 受影响的服务列表
            symptoms (List[str]): 观察到的症状描述列表

        Returns:
            List[str]: 选择的相关指标列表

        选择逻辑：
        1. 从默认指标集开始
        2. 根据症状关键词添加特定指标
        3. 根据服务类型调整指标优先级
        4. 确保指标覆盖性和相关性
        """
        # 从配置的默认指标开始，确保基础监控覆盖
        relevant_metrics = set(config.rca.default_metrics)

        # 基于症状关键词添加特定的监控指标
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if "slow" in symptom_lower or "latency" in symptom_lower:
                # 响应慢或延迟问题 - 添加延迟相关指标
                relevant_metrics.update(
                    [
                        "kubelet_http_requests_duration_seconds_sum",
                        "kubelet_http_requests_duration_seconds_count",
                    ]
                )
            elif "error" in symptom_lower or "fail" in symptom_lower:
                # 错误或失败问题 - 添加错误率和重启相关指标
                relevant_metrics.update(["kube_pod_container_status_restarts_total"])
            elif "cpu" in symptom_lower:
                # CPU相关问题 - 添加CPU使用率指标
                relevant_metrics.update(
                    ["container_cpu_usage_seconds_total", "node_cpu_seconds_total"]
                )
            elif "memory" in symptom_lower:
                # 内存相关问题 - 添加内存使用指标
                relevant_metrics.update(
                    ["container_memory_working_set_bytes", "node_memory_MemFree_bytes"]
                )

        # 返回去重后的指标列表
        return list(relevant_metrics)

    def _generate_incident_recommendation(
        self, root_causes: List[Dict], services: List[str], symptoms: List[str]
    ) -> str:
        """
        生成事件处理建议 - 基于根因分析结果的智能建议系统

        根据识别出的根因候选、受影响的服务和观察到的症状，生成具体的
        处理建议和行动计划。建议包括即时修复措施和长期预防措施。

        Args:
            root_causes (List[Dict]): 根因候选列表，按置信度排序
            services (List[str]): 受影响的服务列表
            symptoms (List[str]): 观察到的症状列表

        Returns:
            str: 详细的处理建议和行动计划

        建议生成逻辑：
        1. 分析最可能的根因（置信度最高）
        2. 根据根因类型生成特定建议
        3. 考虑置信度水平调整建议强度
        4. 提供备选方案和预防措施
        """
        # 如果没有识别出根因，提供通用建议
        if not root_causes:
            return "建议检查服务配置和资源分配，监控系统负载变化。"

        # 获取置信度最高的根因候选
        top_cause = root_causes[0]
        metric = top_cause.get("metric", "")
        confidence = top_cause.get("confidence", 0)

        recommendations = []

        # 根据根因类型生成具体的修复建议
        if "cpu" in metric.lower():
            recommendations.append("检查CPU使用率，考虑扩容或优化应用性能")
        elif "memory" in metric.lower():
            recommendations.append("检查内存使用情况，可能需要增加内存限制或优化内存使用")
        elif "restart" in metric.lower():
            recommendations.append("检查容器重启原因，查看相关日志和健康检查配置")
        elif "network" in metric.lower() or "http" in metric.lower():
            recommendations.append("检查网络连接和服务间通信，查看负载均衡配置")

        # 根据置信度水平调整建议的紧迫性
        if confidence > 0.8:
            recommendations.append(f"根因分析置信度较高({confidence:.2f})，建议优先处理该问题")
        elif confidence < 0.5:
            recommendations.append("根因分析置信度较低，建议进行更详细的调查")

        # 如果没有生成特定建议，提供通用建议
        return (
            "; ".join(recommendations) if recommendations else "建议进行详细的系统检查和日志分析。"
        )
