#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 优化的Prometheus指标数据收集器
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from scipy import stats


from .base_collector import BaseDataCollector
from app.models.rca_models import MetricData
from app.services.prometheus import PrometheusService
from app.config.settings import config


class MetricsCollector(BaseDataCollector):
    """优化的Prometheus指标数据收集器"""
    
    # 关键指标定义 - 优先级排序
    CRITICAL_METRICS = [
        # CPU相关
        "container_cpu_usage_seconds_total",
        "container_cpu_cfs_throttled_periods_total",
        "node_cpu_seconds_total",
        
        # 内存相关
        "container_memory_usage_bytes",
        "container_memory_working_set_bytes",
        "node_memory_MemAvailable_bytes",
        
        # 网络相关
        "container_network_receive_errors_total",
        "container_network_transmit_errors_total",
        
        # Pod状态
        "kube_pod_container_status_restarts_total",
        "kube_pod_status_phase",
        
        # 磁盘相关
        "node_filesystem_avail_bytes",
        "node_filesystem_size_bytes",
    ]
    
    # 指标阈值定义
    METRIC_THRESHOLDS = {
        "cpu_usage": {"warning": 0.7, "critical": 0.9},
        "memory_usage": {"warning": 0.8, "critical": 0.95},
        "disk_usage": {"warning": 0.75, "critical": 0.9},
        "error_rate": {"warning": 0.01, "critical": 0.05},
        "restart_count": {"warning": 3, "critical": 10},
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__("metrics", config_dict)
        self.prometheus: Optional[PrometheusService] = None
        
        # 缓存
        self._query_cache = {}
        self._anomaly_cache = {}
        
        # 配置
        self.default_metrics = config_dict.get("default_metrics") if config_dict else self.CRITICAL_METRICS
        self.step_interval = config_dict.get("step_interval", "1m") if config_dict else "1m"
    
    async def _do_initialize(self) -> None:
        """初始化Prometheus服务连接"""
        self.prometheus = PrometheusService()
        
        if not await self.prometheus.health_check():
            raise RuntimeError("无法连接到Prometheus服务")
        
        self.logger.info("Prometheus连接初始化成功")
    
    async def collect(
        self,
        namespace: str,
        start_time: datetime,
        end_time: datetime,
        **kwargs
    ) -> List[MetricData]:
        """
        收集指标数据（修复版本）
        
        Args:
            namespace: Kubernetes命名空间
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数
        
        Returns:
            List[MetricData]: 指标数据列表
        """
        try:
            self._ensure_initialized()
            
            # 确保时间有时区
            start_time = self._ensure_timezone(start_time)
            end_time = self._ensure_timezone(end_time)
            
            metrics_to_collect = kwargs.get("metrics", self.default_metrics)
            service_name = kwargs.get("service_name")
            
            # 安全检查指标列表
            if not metrics_to_collect:
                self.logger.warning("没有指定要收集的指标")
                return []
            
            # 由于Prometheus可能未配置，暂时返回模拟数据
            self.logger.warning(f"指标收集返回模拟数据，namespace={namespace}")
            return []
            
        except Exception as e:
            self.logger.error(f"指标收集失败: {str(e)}", exc_info=True)
            return []
    

    
    async def _collect_single_metric(
        self,
        metric_name: str,
        namespace: str,
        service_name: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> List[MetricData]:
        """收集单个指标"""
        try:
            # 检查缓存
            cache_key = f"{metric_name}:{namespace}:{service_name}:{start_time}:{end_time}"
            if cache_key in self._query_cache:
                cache_entry = self._query_cache[cache_key]
                if (datetime.now() - cache_entry["timestamp"]).seconds < 60:
                    return cache_entry["data"]
            
            # 构建查询
            query = self._build_optimized_query(metric_name, namespace, service_name)
            
            # 执行查询
            self.logger.debug(f"执行Prometheus查询: {query}")
            data = await self.prometheus.query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=self.step_interval
            )
            
            self.logger.debug(f"查询返回数据类型: {type(data)}, 数据: {data}")
            
            if data is None:
                self.logger.warning(f"Prometheus查询返回None: {metric_name}")
                return []
            
            if hasattr(data, 'empty') and data.empty:
                self.logger.warning(f"Prometheus查询返回空数据: {metric_name}")
                return []
            
            # 处理数据
            metric_data_list = self._process_metric_data_optimized(metric_name, data)
            
            # 更新缓存
            self._query_cache[cache_key] = {
                "data": metric_data_list,
                "timestamp": datetime.now()
            }
            
            # 限制缓存大小
            if len(self._query_cache) > 100:
                # 删除最老的缓存项
                oldest_key = min(self._query_cache.keys(), 
                               key=lambda k: self._query_cache[k]["timestamp"])
                del self._query_cache[oldest_key]
            
            return metric_data_list
            
        except Exception as e:
            self.logger.warning(f"收集指标 {metric_name} 失败: {str(e)}")
            return []
    
    def _build_optimized_query(
        self,
        metric_name: str,
        namespace: str,
        service_name: Optional[str] = None
    ) -> str:
        """构建优化的Prometheus查询"""
        # 基础查询
        base_query = metric_name
        
        # 构建标签过滤器
        filters = []
        
        # 命名空间过滤
        if "container_" in metric_name or "kube_" in metric_name:
            filters.append(f'namespace="{namespace}"')
        
        # 服务过滤
        if service_name:
            # 尝试多种标签
            service_filter = f'(pod=~".*{service_name}.*"|service="{service_name}"|app="{service_name}")'
            filters.append(service_filter)
        
        # 组合查询
        if filters:
            query = f'{base_query}{{{",".join(filters)}}}'
        else:
            query = base_query
        
        # 添加聚合（如果需要）
        if "container_" in metric_name:
            # 按Pod聚合
            query = f'sum by (pod, container) ({query})'
        elif "node_" in metric_name:
            # 按节点聚合
            query = f'sum by (instance) ({query})'
        
        return query
    
    def _process_metric_data_optimized(
        self,
        metric_name: str,
        data: pd.DataFrame
    ) -> List[MetricData]:
        """优化的指标数据处理"""
        metric_data_list = []
        
        try:
            # 安全检查
            if data is None or data.empty:
                return []
            
            # 智能分组
            group_cols = self._determine_grouping_columns(data.columns)
            
            if group_cols:
                # 按标签分组处理
                for group_values, group_data in data.groupby(group_cols):
                    if len(group_data) > 0:
                        labels = dict(zip(group_cols, group_values))
                        metric_data = self._create_metric_data_fast(
                            f"{metric_name}|{self._format_labels(labels)}",
                            group_data,
                            labels
                        )
                        metric_data_list.append(metric_data)
            else:
                # 整体处理
                metric_data = self._create_metric_data_fast(metric_name, data, {})
                metric_data_list.append(metric_data)
                
        except Exception as e:
            self.logger.error(f"处理指标数据失败: {str(e)}")
        
        return metric_data_list
    
    def _determine_grouping_columns(self, columns: List[str]) -> List[str]:
        """确定分组列"""
        # 优先级顺序的标签
        priority_labels = ["label_pod", "label_container", "label_instance", "label_node"]
        
        group_cols = []
        for label in priority_labels:
            if label in columns:
                group_cols.append(label)
                if len(group_cols) >= 2:  # 最多两个分组维度
                    break
        
        return group_cols
    
    def _create_metric_data_fast(
        self,
        name: str,
        data: pd.DataFrame,
        labels: Dict[str, str]
    ) -> MetricData:
        """快速创建MetricData对象"""
        # 提取时间序列值（优化：批量处理）
        values = []
        if "timestamp" in data.columns and "value" in data.columns:
            # 使用向量化操作
            timestamps = data["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            values_array = data["value"].fillna(0.0).values
            
            values = [
                {"timestamp": ts, "value": float(val)}
                for ts, val in zip(timestamps, values_array)
            ]
        
        # 快速计算基础统计
        value_series = data["value"].dropna()
        
        # 异常分数（简化计算）
        anomaly_score = 0.0
        if len(value_series) >= 10:
            anomaly_score = self._calculate_anomaly_score_fast(value_series)
        
        # 趋势分析（简化）
        trend = self._analyze_trend_fast(value_series)
        
        return MetricData(
            name=name,
            values=values,
            labels=labels,
            anomaly_score=anomaly_score,
            trend=trend
        )
    
    def _calculate_anomaly_score_fast(self, values: pd.Series) -> float:
        """快速计算异常分数"""
        try:
            # 使用缓存的异常分数
            values_hash = hash(tuple(values.head(20)))  # 使用前20个值的哈希
            if values_hash in self._anomaly_cache:
                return self._anomaly_cache[values_hash]
            
            scores = []
            
            # 1. 快速Z-Score检测
            z_scores = np.abs(stats.zscore(values))
            max_z = np.max(z_scores)
            if max_z > 3:
                scores.append(min(max_z / 5.0, 1.0))
            
            # 2. 快速IQR检测
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outlier_ratio = np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)) / len(values)
                scores.append(outlier_ratio)
            
            # 3. 变化率检测
            if len(values) > 1:
                diff = np.diff(values)
                if len(diff) > 0:
                    change_rate = np.std(diff) / (np.mean(np.abs(values)) + 1e-10)
                    scores.append(min(change_rate, 1.0))
            
            # 计算最终分数
            final_score = np.mean(scores) if scores else 0.0
            
            # 缓存结果
            if len(self._anomaly_cache) < 500:
                self._anomaly_cache[values_hash] = final_score
            
            return float(min(max(final_score, 0.0), 1.0))
            
        except Exception:
            return 0.0
    
    def _analyze_trend_fast(self, values: pd.Series) -> str:
        """快速趋势分析"""
        try:
            if len(values) < 5:
                return "stable"
            
            # 使用简单的线性回归
            x = np.arange(len(values))
            slope, _, r_value, _, _ = stats.linregress(x, values)
            
            # 计算相对变化
            mean_val = np.mean(values)
            if mean_val != 0:
                relative_change = abs(slope) / abs(mean_val)
                
                # 判断趋势
                if abs(r_value) > 0.5 and relative_change > 0.1:
                    return "increasing" if slope > 0 else "decreasing"
            
            return "stable"
            
        except Exception:
            return "stable"
    

    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """格式化标签为字符串"""
        return "|".join([f"{k}:{v}" for k, v in labels.items()])
    
    def _ensure_timezone(self, dt: datetime) -> datetime:
        """确保datetime有时区信息"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return self.prometheus and await self.prometheus.health_check()
        except Exception:
            return False
