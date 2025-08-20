#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Prometheus指标数据收集器 - 收集和处理Prometheus监控指标数据
"""

import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional
from sklearn.ensemble import IsolationForest
from scipy import stats
import numpy as np

from .base_collector import BaseDataCollector
from app.models.rca_models import MetricData
from app.services.prometheus import PrometheusService
from app.config.settings import config


class MetricsCollector(BaseDataCollector):
    """
    Prometheus指标数据收集器

    负责从Prometheus收集指标数据，并进行初步的异常检测和趋势分析。
    支持多种异常检测算法，包括统计方法和机器学习方法。
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化指标收集器

        Args:
            config_dict: 收集器配置
        """
        super().__init__("metrics", config_dict)
        self.prometheus: Optional[PrometheusService] = None
        self.default_metrics = config.rca.default_metrics

    async def _do_initialize(self) -> None:
        """初始化Prometheus服务连接"""
        self.prometheus = PrometheusService()

        # 验证Prometheus连接
        health = await self.prometheus.health_check()
        if not health:
            raise RuntimeError("无法连接到Prometheus服务")

    async def collect(
        self, namespace: str, start_time: datetime, end_time: datetime, **kwargs
    ) -> List[MetricData]:
        """
        收集指标数据

        Args:
            namespace: Kubernetes命名空间
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数，可包含：
                - metrics: 指定要收集的指标列表
                - service_name: 服务名称过滤

        Returns:
            List[MetricData]: 指标数据列表
        """
        self._ensure_initialized()

        metrics_to_collect = kwargs.get("metrics", self.default_metrics)
        service_name = kwargs.get("service_name")

        collected_metrics = []

        for metric_name in metrics_to_collect:
            try:
                # 构建查询，添加命名空间过滤
                query = self._build_metric_query(metric_name, namespace, service_name)

                # 从Prometheus获取数据
                data = await self.prometheus.query_range(
                    query=query, start_time=start_time, end_time=end_time, step="1m"
                )

                if data is not None and not data.empty:
                    # 处理数据并创建MetricData对象
                    metric_data_list = self._process_metric_data(metric_name, data)
                    collected_metrics.extend(metric_data_list)

            except Exception as e:
                self.logger.warning(f"收集指标 {metric_name} 失败: {str(e)}")
                continue

        self.logger.info(f"成功收集 {len(collected_metrics)} 个指标数据")
        return collected_metrics

    def _build_metric_query(
        self, metric_name: str, namespace: str, service_name: Optional[str] = None
    ) -> str:
        """
        构建Prometheus查询语句

        Args:
            metric_name: 指标名称
            namespace: 命名空间
            service_name: 服务名称（可选）

        Returns:
            str: Prometheus查询语句
        """
        # 基础查询
        query = metric_name

        # 添加命名空间过滤
        filters = [f'namespace="{namespace}"']

        # 添加服务名称过滤（如果提供）
        if service_name:
            # 尝试多种标签名
            service_filters = [
                f'service="{service_name}"',
                f'app="{service_name}"',
                f'app_kubernetes_io_name="{service_name}"',
            ]
            filters.extend(service_filters)

        # 构建完整查询
        if filters:
            # 对于不同的指标，使用不同的过滤策略
            if "container_" in metric_name:
                # 容器级指标
                query = f'{metric_name}{{namespace="{namespace}"}}'
            elif "kube_pod" in metric_name:
                # Pod级指标
                query = f'{metric_name}{{namespace="{namespace}"}}'
            elif "node_" in metric_name:
                # 节点级指标，不需要命名空间过滤
                query = metric_name
            else:
                # 其他指标，使用通用过滤
                query = f'{metric_name}{{namespace="{namespace}"}}'

        return query

    def _process_metric_data(self, metric_name: str, data: pd.DataFrame) -> List[MetricData]:
        """
        处理指标数据

        Args:
            metric_name: 指标名称
            data: 原始数据

        Returns:
            List[MetricData]: 处理后的指标数据列表
        """
        metric_data_list = []

        try:
            # 根据标签对数据进行分组
            if "label_pod" in data.columns:
                # 按Pod分组
                for pod_name in data["label_pod"].unique():
                    if pd.notna(pod_name):
                        pod_data = data[data["label_pod"] == pod_name]
                        if not pod_data.empty:
                            processed_data = self._create_metric_data(
                                f"{metric_name}|pod:{pod_name}", pod_data, {"pod": pod_name}
                            )
                            metric_data_list.append(processed_data)

            elif "label_container" in data.columns:
                # 按容器分组
                for container_name in data["label_container"].unique():
                    if pd.notna(container_name):
                        container_data = data[data["label_container"] == container_name]
                        if not container_data.empty:
                            processed_data = self._create_metric_data(
                                f"{metric_name}|container:{container_name}",
                                container_data,
                                {"container": container_name},
                            )
                            metric_data_list.append(processed_data)
            else:
                # 聚合数据
                processed_data = self._create_metric_data(metric_name, data, {})
                metric_data_list.append(processed_data)

        except Exception as e:
            self.logger.error(f"处理指标数据失败: {str(e)}")

        return metric_data_list

    def _create_metric_data(
        self, name: str, data: pd.DataFrame, labels: Dict[str, str]
    ) -> MetricData:
        """
        创建MetricData对象

        Args:
            name: 指标名称
            data: 时间序列数据
            labels: 标签信息

        Returns:
            MetricData: 指标数据对象
        """
        # 提取时间序列值
        values = []
        if "timestamp" in data.columns and "value" in data.columns:
            for _, row in data.iterrows():
                values.append(
                    {
                        "timestamp": (
                            row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None
                        ),
                        "value": float(row["value"]) if pd.notna(row["value"]) else 0.0,
                    }
                )

        # 计算异常分数
        anomaly_score = self._calculate_anomaly_score(data)

        # 分析趋势
        trend = self._analyze_trend(data)

        return MetricData(
            name=name, values=values, labels=labels, anomaly_score=anomaly_score, trend=trend
        )

    def _calculate_anomaly_score(self, data: pd.DataFrame) -> float:
        """
        计算异常分数

        Args:
            data: 时间序列数据

        Returns:
            float: 异常分数 (0.0 - 1.0)
        """
        try:
            if "value" not in data.columns or data.empty:
                return 0.0

            values = data["value"].dropna()
            if len(values) < 10:  # 数据点太少，无法计算异常分数
                return 0.0

            # 使用多种方法计算异常分数
            scores = []

            # 1. Z-Score方法
            z_scores = np.abs(stats.zscore(values))
            z_anomaly_score = min(np.max(z_scores) / 3.0, 1.0)  # 标准化到0-1
            # 处理nan值
            if np.isnan(z_anomaly_score) or np.isinf(z_anomaly_score):
                z_anomaly_score = 0.0
            scores.append(z_anomaly_score)

            # 2. IQR方法
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                iqr_anomaly_score = min(len(outliers) / len(values), 1.0)
                scores.append(iqr_anomaly_score)

            # 3. Isolation Forest方法（如果数据足够）
            if len(values) >= 20:
                try:
                    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = isolation_forest.fit_predict(values.values.reshape(-1, 1))
                    isolation_score = np.sum(anomaly_labels == -1) / len(anomaly_labels)
                    scores.append(isolation_score)
                except Exception:
                    pass  # 如果Isolation Forest失败，忽略这个分数

            # 返回平均分数，确保处理nan值
            if scores:
                final_score = np.mean(scores)
                if np.isnan(final_score) or np.isinf(final_score):
                    return 0.0
                return min(max(final_score, 0.0), 1.0)  # 确保在0-1范围内
            return 0.0

        except Exception as e:
            self.logger.warning(f"计算异常分数失败: {str(e)}")
            return 0.0

    def _analyze_trend(self, data: pd.DataFrame) -> str:
        """
        分析趋势

        Args:
            data: 时间序列数据

        Returns:
            str: 趋势描述 ('increasing', 'decreasing', 'stable')
        """
        try:
            if "value" not in data.columns or len(data) < 5:
                return "stable"

            values = data["value"].dropna()
            if len(values) < 5:
                return "stable"

            # 使用线性回归检测趋势
            x = np.arange(len(values))
            slope, _, r_value, _, _ = stats.linregress(x, values)

            # 只有在相关性足够强时才认为存在趋势
            if abs(r_value) > 0.5:
                if slope > 0:
                    return "increasing"
                elif slope < 0:
                    return "decreasing"

            return "stable"

        except Exception as e:
            self.logger.warning(f"分析趋势失败: {str(e)}")
            return "stable"

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 收集器是否健康
        """
        try:
            if not self.prometheus:
                return False
            return await self.prometheus.health_check()
        except Exception:
            return False
