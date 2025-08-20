#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析服务 - 提供多数据源根因分析的业务逻辑编排
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base import BaseService, HealthCheckMixin
from ..common.constants import ServiceConstants
from ..common.exceptions import RCAError, ValidationError
from ..core.rca import RCAEngine
from ..models.rca_models import RCARequest, RCAResponse, RCAAnalysisResult

logger = logging.getLogger("aiops.services.rca")


class RCAService(BaseService, HealthCheckMixin):
    """
    根因分析服务 - 整合多数据源的智能根因分析

    负责业务逻辑编排，包括请求验证、分析流程控制、
    结果处理和错误管理。使用新的RCA引擎进行分析。
    """

    def __init__(self) -> None:
        super().__init__("rca")
        self._engine: Optional[RCAEngine] = None

    async def _do_initialize(self) -> None:
        """初始化RCA服务"""
        try:
            # 初始化新的RCA引擎
            self._engine = RCAEngine()
            await self._engine.initialize()
            self.logger.info("RCA引擎初始化完成")
        except Exception as e:
            self.logger.error(f"RCA引擎初始化失败: {str(e)}")
            raise RCAError(f"初始化失败: {str(e)}")

    async def _do_health_check(self) -> bool:
        """
        RCA服务健康检查

        只要至少有一个数据收集器可用，就认为服务是健康的
        """
        try:
            if not self._engine:
                return False

            # 尝试初始化引擎（如果未初始化）
            if not self._engine._initialized:
                try:
                    await self._engine.initialize()
                except Exception as e:
                    self.logger.warning(f"RCA引擎初始化失败: {str(e)}")
                    return False

            # 检查引擎健康状态
            return await self._engine.health_check()

        except Exception as e:
            self.logger.warning(f"RCA服务健康检查失败: {str(e)}")
            return False

    async def analyze_root_cause(self, request: RCARequest) -> RCAResponse:
        """
        执行根因分析 - 新的统一接口

        Args:
            request: 根因分析请求对象

        Returns:
            RCAResponse: 分析响应结果

        Raises:
            ValidationError: 参数验证失败
            RCAError: 分析过程失败
        """
        self._ensure_initialized()

        # 验证请求参数
        self._validate_rca_request(request)

        try:
            self.logger.info(
                f"开始根因分析: namespace={request.namespace}, "
                f"时间范围={request.start_time} - {request.end_time}"
            )

            # 使用新的RCA引擎进行分析
            analysis_result = await self.execute_with_timeout(
                self._engine.analyze(request),
                timeout=ServiceConstants.RCA_TIMEOUT,
                operation_name="root_cause_analysis",
            )

            # 构建成功响应
            response = RCAResponse(
                status="success",
                result=analysis_result,
                metadata={
                    "service_version": "2.0",
                    "analysis_engine": "multi_source_rca",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            self.logger.info(
                f"根因分析完成: 请求ID={analysis_result.request_id}, "
                f"处理时间={analysis_result.processing_time:.2f}秒"
            )

            return response

        except Exception as e:
            self.logger.error(f"根因分析执行失败: {str(e)}")

            # 构建错误响应
            return RCAResponse(
                status="error",
                error_message=f"分析失败: {str(e)}",
                metadata={"service_version": "2.0", "timestamp": datetime.now().isoformat()},
            )



    async def get_available_metrics(
        self, service_name: Optional[str] = None, category: Optional[str] = None
    ) -> List[str]:
        """
        获取可用的监控指标列表

        Args:
            service_name: 服务名称
            category: 指标类别

        Returns:
            可用指标列表

        Raises:
            RCAError: 获取指标失败
        """
        self._ensure_initialized()

        try:
            # 返回配置中的默认指标列表
            from ..config.settings import config

            available_metrics = config.rca.default_metrics.copy()

            # 可以根据服务名称和类别进行过滤
            if category:
                category_lower = category.lower()
                if category_lower == "cpu":
                    available_metrics = [m for m in available_metrics if "cpu" in m.lower()]
                elif category_lower == "memory":
                    available_metrics = [m for m in available_metrics if "memory" in m.lower()]
                elif category_lower == "network":
                    available_metrics = [
                        m
                        for m in available_metrics
                        if "network" in m.lower() or "http" in m.lower()
                    ]
                elif category_lower == "pod":
                    available_metrics = [m for m in available_metrics if "pod" in m.lower()]
                elif category_lower == "node":
                    available_metrics = [m for m in available_metrics if "node" in m.lower()]

            return available_metrics

        except Exception as e:
            self.logger.error(f"获取可用指标失败: {str(e)}")
            raise RCAError(f"获取指标失败: {str(e)}")

    async def query_metric_data(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None,
        namespace: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        查询指标数据

        Args:
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            service_name: 服务名称
            namespace: 命名空间

        Returns:
            指标数据列表

        Raises:
            ValidationError: 参数验证失败
            RCAError: 查询失败
        """
        self._ensure_initialized()

        # 验证时间范围
        if start_time >= end_time:
            raise ValidationError("time_range", "开始时间必须早于结束时间")

        try:
            # 使用指标收集器查询数据
            metrics_collector = self._engine.metrics_collector

            metric_data_list = await metrics_collector.collect_with_retry(
                namespace=namespace,
                start_time=start_time,
                end_time=end_time,
                metrics=[metric_name],
                service_name=service_name,
            )

            # 转换为兼容的格式
            result = []
            for metric_data in metric_data_list:
                if metric_data.name == metric_name or metric_data.name.startswith(
                    f"{metric_name}|"
                ):
                    for value_point in metric_data.values:
                        result.append(
                            {
                                "timestamp": value_point["timestamp"],
                                "value": value_point["value"],
                                "metric": metric_data.name,
                                "labels": metric_data.labels,
                            }
                        )

            return result

        except Exception as e:
            self.logger.error(f"查询指标数据失败: {str(e)}")
            raise RCAError(f"查询失败: {str(e)}")

    async def get_rca_config(self) -> Dict[str, Any]:
        """
        获取RCA配置信息

        Returns:
            配置信息字典
        """
        from ..config.settings import config

        config_info = {
            "anomaly_detection": {
                "algorithm": "statistical",
                "threshold_factor": getattr(config, "rca_threshold_factor", 2.0),
                "window_size": getattr(config, "rca_window_size", 60),
            },
            "correlation_analysis": {
                "method": "pearson",
                "min_correlation": getattr(config, "min_correlation", 0.7),
                "max_lag": getattr(config, "max_correlation_lag", 300),
            },
            "supported_metrics": [
                "cpu_usage",
                "memory_usage",
                "network_io",
                "disk_io",
                "request_rate",
                "error_rate",
                "response_time",
            ],
            "constraints": {
                "min_metrics": ServiceConstants.RCA_MIN_METRICS,
                "max_metrics": ServiceConstants.RCA_MAX_METRICS,
                "default_severity_threshold": ServiceConstants.RCA_DEFAULT_SEVERITY_THRESHOLD,
                "timeout": ServiceConstants.RCA_TIMEOUT,
            },
            "prometheus_config": {
                "endpoint": getattr(config.prometheus, "url", "unknown"),
                "timeout": getattr(config.prometheus, "timeout", 30),
            },
        }

        return config_info

    async def get_service_health_info(self) -> Dict[str, Any]:
        """
        获取RCA服务详细健康信息

        Returns:
            健康信息字典
        """
        try:
            health_status = {
                "service": "rca",
                "status": (
                    ServiceConstants.STATUS_HEALTHY
                    if await self.health_check()
                    else ServiceConstants.STATUS_UNHEALTHY
                ),
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "rca_engine": "unknown",
                    "metrics_collector": "unknown",
                    "events_collector": "unknown",
                    "logs_collector": "unknown",
                    "correlation_analyzer": "unknown",
                    "llm_service": "unknown",
                },
            }

            # 检查RCA引擎状态
            if self._engine:
                try:
                    # 获取引擎状态
                    engine_status = self._engine.get_status()
                    health_status["components"]["rca_engine"] = (
                        ServiceConstants.STATUS_HEALTHY
                        if engine_status["healthy"]
                        else ServiceConstants.STATUS_UNHEALTHY
                    )

                    # 检查各数据收集器状态
                    if "collectors" in engine_status:
                        collectors = engine_status["collectors"]
                        health_status["components"]["metrics_collector"] = (
                            ServiceConstants.STATUS_HEALTHY
                            if collectors.get("metrics", False)
                            else ServiceConstants.STATUS_UNHEALTHY
                        )
                        health_status["components"]["events_collector"] = (
                            ServiceConstants.STATUS_HEALTHY
                            if collectors.get("events", False)
                            else ServiceConstants.STATUS_UNHEALTHY
                        )
                        health_status["components"]["logs_collector"] = (
                            ServiceConstants.STATUS_HEALTHY
                            if collectors.get("logs", False)
                            else ServiceConstants.STATUS_UNHEALTHY
                        )

                    # 检查关联分析器
                    try:
                        if (
                            hasattr(self._engine, "correlation_analyzer")
                            and self._engine.correlation_analyzer
                        ):
                            health_status["components"][
                                "correlation_analyzer"
                            ] = ServiceConstants.STATUS_HEALTHY
                    except Exception:
                        health_status["components"][
                            "correlation_analyzer"
                        ] = ServiceConstants.STATUS_UNHEALTHY

                    # 检查LLM服务
                    try:
                        if hasattr(self._engine, "llm_service") and self._engine.llm_service:
                            # 尝试同步健康检查
                            if hasattr(self._engine.llm_service, "health_check_sync"):
                                llm_health = self._engine.llm_service.health_check_sync()
                            elif hasattr(self._engine.llm_service, "is_available"):
                                llm_health = self._engine.llm_service.is_available()
                            else:
                                llm_health = bool(getattr(self._engine.llm_service, "client", None))
                            health_status["components"]["llm_service"] = (
                                ServiceConstants.STATUS_HEALTHY
                                if llm_health
                                else ServiceConstants.STATUS_UNHEALTHY
                            )
                    except Exception:
                        health_status["components"][
                            "llm_service"
                        ] = ServiceConstants.STATUS_UNHEALTHY

                except Exception as e:
                    self.logger.warning(f"检查引擎状态失败: {str(e)}")
                    health_status["components"]["rca_engine"] = ServiceConstants.STATUS_UNHEALTHY

            return health_status

        except Exception as e:
            self.logger.error(f"获取RCA服务健康信息失败: {str(e)}")
            return {
                "service": "rca",
                "status": ServiceConstants.STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _validate_rca_request(self, request: RCARequest) -> None:
        """
        验证RCA分析请求

        Args:
            request: RCA请求对象

        Raises:
            ValidationError: 参数验证失败
        """
        # 验证时间范围
        if request.start_time >= request.end_time:
            raise ValidationError("time_range", "开始时间必须早于结束时间")

        # 验证时间范围不能超过最大限制
        time_diff = (request.end_time - request.start_time).total_seconds()
        max_range_seconds = 24 * 3600  # 24小时
        if time_diff > max_range_seconds:
            raise ValidationError("time_range", f"时间范围不能超过{max_range_seconds/3600}小时")

        # 验证命名空间
        if not request.namespace or not request.namespace.strip():
            raise ValidationError("namespace", "命名空间不能为空")

        # 验证指标列表（如果提供）
        if request.metrics:
            if len(request.metrics) > ServiceConstants.RCA_MAX_METRICS:
                raise ValidationError(
                    "metrics", f"指标数量不能超过 {ServiceConstants.RCA_MAX_METRICS} 个"
                )

        # 验证严重性阈值
        if not (0.0 <= request.severity_threshold <= 1.0):
            raise ValidationError("severity_threshold", "严重性阈值必须在 0.0-1.0 之间")

        # 验证关联分析时间窗口
        if not (60 <= request.correlation_window <= 3600):
            raise ValidationError("correlation_window", "关联分析时间窗口必须在 60-3600 秒之间")

        # 验证最大候选数量
        if not (1 <= request.max_candidates <= 20):
            raise ValidationError("max_candidates", "最大根因候选数量必须在 1-20 之间")


