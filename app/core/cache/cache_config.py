#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 缓存配置和策略管理
"""

from enum import Enum
from typing import Any, Dict, Tuple


class CacheType(Enum):
    """缓存类型枚举"""

    PREDICTION_QPS = "prediction_qps"
    PREDICTION_CPU = "prediction_cpu"
    PREDICTION_MEMORY = "prediction_memory"
    PREDICTION_DISK = "prediction_disk"
    PREDICTION_AI_ENHANCED = "prediction_ai_enhanced"

    RCA_ANALYSIS = "rca_analysis"
    RCA_QUICK_DIAGNOSIS = "rca_quick_diagnosis"
    RCA_EVENT_PATTERNS = "rca_event_patterns"
    RCA_ERROR_SUMMARY = "rca_error_summary"
    RCA_METRICS = "rca_metrics"


class CacheStrategy:
    """缓存策略配置"""

    # 缓存TTL配置（秒）
    CACHE_TTL_CONFIG = {
        # 预测服务缓存策略
        CacheType.PREDICTION_QPS: 3600,  # 1小时 - QPS预测相对稳定
        CacheType.PREDICTION_CPU: 3600,  # 1小时 - CPU预测
        CacheType.PREDICTION_MEMORY: 3600,  # 1小时 - 内存预测
        CacheType.PREDICTION_DISK: 7200,  # 2小时 - 磁盘变化较慢
        CacheType.PREDICTION_AI_ENHANCED: 7200,  # 2小时 - AI增强预测计算耗时
        # RCA服务缓存策略
        CacheType.RCA_ANALYSIS: 1800,  # 30分钟 - 根因分析结果
        CacheType.RCA_QUICK_DIAGNOSIS: 900,  # 15分钟 - 快速诊断需要更新
        CacheType.RCA_EVENT_PATTERNS: 1200,  # 20分钟 - 事件模式分析
        CacheType.RCA_ERROR_SUMMARY: 1200,  # 20分钟 - 错误摘要
        CacheType.RCA_METRICS: 600,  # 10分钟 - 指标数据变化较快
    }

    # 缓存优先级配置（数字越大优先级越高）
    CACHE_PRIORITY_CONFIG = {
        # 高优先级：AI增强预测和复杂分析
        CacheType.PREDICTION_AI_ENHANCED: 10,
        CacheType.RCA_ANALYSIS: 9,
        # 中等优先级：基础预测
        CacheType.PREDICTION_QPS: 7,
        CacheType.PREDICTION_CPU: 7,
        CacheType.PREDICTION_MEMORY: 7,
        CacheType.PREDICTION_DISK: 6,
        # 低优先级：快速查询
        CacheType.RCA_QUICK_DIAGNOSIS: 5,
        CacheType.RCA_EVENT_PATTERNS: 4,
        CacheType.RCA_ERROR_SUMMARY: 4,
        CacheType.RCA_METRICS: 3,
    }

    # 缓存键前缀配置
    CACHE_PREFIX_CONFIG = {
        CacheType.PREDICTION_QPS: "pred:qps:",
        CacheType.PREDICTION_CPU: "pred:cpu:",
        CacheType.PREDICTION_MEMORY: "pred:mem:",
        CacheType.PREDICTION_DISK: "pred:disk:",
        CacheType.PREDICTION_AI_ENHANCED: "pred:ai:",
        CacheType.RCA_ANALYSIS: "rca:analyze:",
        CacheType.RCA_QUICK_DIAGNOSIS: "rca:quick:",
        CacheType.RCA_EVENT_PATTERNS: "rca:events:",
        CacheType.RCA_ERROR_SUMMARY: "rca:errors:",
        CacheType.RCA_METRICS: "rca:metrics:",
    }

    @classmethod
    def get_cache_config(cls, cache_type: CacheType) -> Tuple[int, int, str]:
        """
        获取缓存配置

        Args:
            cache_type: 缓存类型

        Returns:
            Tuple[ttl, priority, prefix]: TTL、优先级、前缀
        """
        ttl = cls.CACHE_TTL_CONFIG.get(cache_type, 1800)  # 默认30分钟
        priority = cls.CACHE_PRIORITY_CONFIG.get(cache_type, 5)  # 默认中等优先级
        prefix = cls.CACHE_PREFIX_CONFIG.get(cache_type, "cache:")  # 默认前缀

        return ttl, priority, prefix

    @classmethod
    def should_cache_result(
        cls, cache_type: CacheType, result_size: int = 0, execution_time: float = 0
    ) -> bool:
        """
        判断是否应该缓存结果

        Args:
            cache_type: 缓存类型
            result_size: 结果大小（字节）
            execution_time: 执行时间（秒）

        Returns:
            bool: 是否应该缓存
        """
        # 基本规则
        if result_size > 10 * 1024 * 1024:  # 结果大于10MB不缓存
            return False

        # AI增强预测和复杂分析总是缓存
        if cache_type in [CacheType.PREDICTION_AI_ENHANCED, CacheType.RCA_ANALYSIS]:
            return True

        # 根据执行时间决定
        if execution_time > 5:  # 执行时间超过5秒的总是缓存
            return True
        elif execution_time > 1:  # 执行时间超过1秒的优先缓存
            return True

        # 其他情况默认缓存
        return True

    @classmethod
    def get_cache_compression_threshold(cls, cache_type: CacheType) -> int:
        """
        获取缓存压缩阈值

        Args:
            cache_type: 缓存类型

        Returns:
            int: 压缩阈值（字节）
        """
        # AI增强预测结果通常较大，使用较小的压缩阈值
        if cache_type == CacheType.PREDICTION_AI_ENHANCED:
            return 512  # 512字节

        # RCA分析结果也可能较大
        if cache_type == CacheType.RCA_ANALYSIS:
            return 1024  # 1KB

        # 其他情况使用默认阈值
        return 2048  # 2KB


class CacheKeyBuilder:
    """缓存键构建器"""

    @staticmethod
    def build_prediction_cache_key(cache_type: CacheType, **params) -> str:
        """构建预测服务缓存键"""
        _, _, prefix = CacheStrategy.get_cache_config(cache_type)

        # 提取关键参数
        key_parts = [
            f"val:{params.get('current_value', 0):.2f}",
            f"hours:{params.get('prediction_hours', 24)}",
            f"gran:{params.get('granularity', 'hour')}",
        ]

        # 添加查询参数
        if params.get("metric_query"):
            import hashlib

            query_hash = hashlib.md5(params["metric_query"].encode()).hexdigest()[:8]
            key_parts.append(f"query:{query_hash}")

        # 添加AI参数
        if params.get("ai_enhanced"):
            key_parts.append("ai:true")
            if params.get("report_style"):
                key_parts.append(f"style:{params['report_style']}")

        # 添加约束参数
        if params.get("resource_constraints"):
            constraints_str = str(sorted(params["resource_constraints"].items()))
            import hashlib

            constraints_hash = hashlib.md5(constraints_str.encode()).hexdigest()[:8]
            key_parts.append(f"const:{constraints_hash}")

        return prefix + "|".join(key_parts)

    @staticmethod
    def build_rca_cache_key(cache_type: CacheType, **params) -> str:
        """构建RCA服务缓存键"""
        _, _, prefix = CacheStrategy.get_cache_config(cache_type)

        # 提取关键参数
        key_parts = [
            f"ns:{params.get('namespace', 'default')}",
            f"hours:{params.get('time_window_hours', 1.0)}",
        ]

        # 添加指标参数
        if params.get("metrics"):
            metrics_str = "|".join(sorted(params["metrics"]))
            import hashlib

            metrics_hash = hashlib.md5(metrics_str.encode()).hexdigest()[:8]
            key_parts.append(f"metrics:{metrics_hash}")

        # 添加其他参数
        for param_name in ["pod_name", "severity", "error_only", "max_lines"]:
            if params.get(param_name) is not None:
                key_parts.append(f"{param_name}:{params[param_name]}")

        return prefix + "|".join(key_parts)


class CacheMonitor:
    """缓存监控和统计"""

    def __init__(self):
        self._cache_stats = {}

    def record_cache_access(
        self, cache_type: CacheType, hit: bool, execution_time: float = 0
    ):
        """记录缓存访问统计"""
        type_name = cache_type.value
        if type_name not in self._cache_stats:
            self._cache_stats[type_name] = {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "avg_execution_time": 0,
                "total_execution_time": 0,
            }

        stats = self._cache_stats[type_name]
        stats["total_requests"] += 1

        if hit:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1
            stats["total_execution_time"] += execution_time
            stats["avg_execution_time"] = (
                stats["total_execution_time"] / stats["cache_misses"]
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        result = {}
        for cache_type, stats in self._cache_stats.items():
            hit_rate = 0
            if stats["total_requests"] > 0:
                hit_rate = stats["cache_hits"] / stats["total_requests"] * 100

            result[cache_type] = {
                **stats,
                "hit_rate": round(hit_rate, 2),
                "miss_rate": round(100 - hit_rate, 2),
            }

        return result

    def get_performance_insights(self) -> Dict[str, Any]:
        """获取性能洞察"""
        insights = []
        total_saved_time = 0

        for cache_type, stats in self._cache_stats.items():
            if stats["cache_hits"] > 0 and stats["avg_execution_time"] > 0:
                saved_time = stats["cache_hits"] * stats["avg_execution_time"]
                total_saved_time += saved_time

                insights.append(
                    {
                        "cache_type": cache_type,
                        "saved_time_seconds": round(saved_time, 2),
                        "hit_rate": round(
                            stats["cache_hits"] / stats["total_requests"] * 100, 2
                        ),
                    }
                )

        return {
            "total_saved_time_seconds": round(total_saved_time, 2),
            "insights_by_type": insights,
            "cache_effectiveness": "high"
            if total_saved_time > 300
            else "medium"
            if total_saved_time > 60
            else "low",
        }


# 全局缓存监控实例
cache_monitor = CacheMonitor()
