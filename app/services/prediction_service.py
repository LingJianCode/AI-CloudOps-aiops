#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 预测服务
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..common.constants import ServiceConstants
from ..common.exceptions import PredictionError, ValidationError
from ..core.prediction.predictor import PredictionService as CorePredictionService
from .base import BaseService, HealthCheckMixin

logger = logging.getLogger("aiops.services.prediction")


class PredictionService(BaseService, HealthCheckMixin):
    """
    负载预测服务 - 管理QPS预测和负载趋势分析
    """

    def __init__(self) -> None:
        super().__init__("prediction")
        self._core_service: Optional[CorePredictionService] = None

    async def _do_initialize(self) -> None:
        """初始化预测服务"""
        try:
            self._core_service = CorePredictionService()
            self.logger.info("预测服务核心组件初始化完成")
        except Exception as e:
            self.logger.error(f"预测服务核心组件初始化失败: {str(e)}")
            raise PredictionError(f"初始化失败: {str(e)}")

    async def _do_health_check(self) -> bool:
        """预测服务健康检查"""
        try:
            if not self._core_service:
                return False

            # 检查核心服务健康状态
            health_result = await self.execute_with_timeout(
                self._core_service.health_check,
                timeout=ServiceConstants.PREDICTION_TIMEOUT,
                operation_name="health_check",
            )

            return (
                health_result.get("healthy", False)
                if isinstance(health_result, dict)
                else bool(health_result)
            )

        except Exception as e:
            self.logger.warning(f"预测服务健康检查失败: {str(e)}")
            return False

    async def predict_instances(
        self,
        service_name: str,
        current_qps: Optional[float] = None,
        hours: int = ServiceConstants.PREDICTION_MIN_HOURS,
        instance_cpu: Optional[int] = None,
        instance_memory: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        预测实例数量

        Args:
            service_name: 服务名称
            current_qps: 当前QPS
            hours: 预测小时数
            instance_cpu: 实例CPU数
            instance_memory: 实例内存(GB)

        Returns:
            预测结果字典

        Raises:
            ValidationError: 参数验证失败
            PredictionError: 预测过程失败
        """
        self._ensure_initialized()

        # 验证输入参数
        self._validate_prediction_params(current_qps, hours)

        try:
            # 调用核心预测服务
            prediction_result = await self.execute_with_timeout(
                lambda: self._core_service.predict(
                    service_name=service_name,
                    current_qps=current_qps,
                    hours=hours,
                    instance_cpu=instance_cpu,
                    instance_memory=instance_memory,
                ),
                timeout=ServiceConstants.PREDICTION_TIMEOUT,
                operation_name="predict_instances",
            )

            # 包装预测结果
            return self._wrap_prediction_result(
                prediction_result, service_name, hours, current_qps
            )

        except Exception as e:
            self.logger.error(f"实例数预测失败: {str(e)}")
            raise PredictionError(f"预测失败: {str(e)}")

    async def predict_trend(
        self,
        service_name: Optional[str] = None,
        hours: int = ServiceConstants.PREDICTION_MIN_HOURS,
    ) -> Dict[str, Any]:
        """
        预测负载趋势

        Args:
            service_name: 服务名称
            hours: 预测小时数

        Returns:
            趋势分析结果

        Raises:
            ValidationError: 参数验证失败
            PredictionError: 趋势分析失败
        """
        self._ensure_initialized()

        # 验证小时数
        if not (
            ServiceConstants.PREDICTION_MIN_HOURS
            <= hours
            <= ServiceConstants.PREDICTION_MAX_HOURS
        ):
            raise ValidationError(
                "hours",
                f"预测小时数必须在 {ServiceConstants.PREDICTION_MIN_HOURS}-{ServiceConstants.PREDICTION_MAX_HOURS} 之间",
            )

        try:
            # 调用核心趋势分析服务
            trend_result = await self.execute_with_timeout(
                lambda: self._core_service.predict_trend(
                    service_name=service_name, hours=hours
                ),
                timeout=ServiceConstants.PREDICTION_TIMEOUT,
                operation_name="predict_trend",
            )

            return {
                "service_name": service_name or "unknown",
                "analysis_hours": hours,
                "trend": trend_result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"趋势分析失败: {str(e)}")
            raise PredictionError(f"趋势分析失败: {str(e)}")

    async def get_model_info(self) -> Dict[str, Any]:
        """
        获取预测模型信息

        Returns:
            模型信息字典

        Raises:
            PredictionError: 获取模型信息失败
        """
        self._ensure_initialized()

        try:
            model_details = {
                "models": [],
                "current_model": None,
                "last_trained": None,
                "performance_metrics": {},
            }

            # 获取模型信息
            if self._core_service and hasattr(self._core_service, "model_loader"):
                model_info = await self.execute_with_timeout(
                    lambda: getattr(
                        self._core_service.model_loader, "get_model_info", lambda: {}
                    )(),
                    timeout=30.0,
                    operation_name="get_model_info",
                )
                model_details.update(model_info)

            return model_details

        except Exception as e:
            self.logger.error(f"获取模型信息失败: {str(e)}")
            raise PredictionError(f"获取模型信息失败: {str(e)}")

    async def get_service_health_info(self) -> Dict[str, Any]:
        """
        获取预测服务详细健康信息

        Returns:
            健康信息字典
        """
        try:
            health_status = {
                "service": "prediction",
                "status": (
                    ServiceConstants.STATUS_HEALTHY
                    if await self.health_check()
                    else ServiceConstants.STATUS_UNHEALTHY
                ),
                "timestamp": datetime.now().isoformat(),
                "model_loaded": False,
                "scaler_loaded": False,
            }

            # 检查模型状态
            if self._core_service:
                health_status["model_loaded"] = getattr(
                    self._core_service, "model_loaded", False
                )
                health_status["scaler_loaded"] = getattr(
                    self._core_service, "scaler_loaded", False
                )

            # 执行核心服务健康检查
            try:
                service_health = await self.execute_with_timeout(
                    self._core_service.health_check,
                    timeout=30.0,
                    operation_name="service_health_check",
                )
                if isinstance(service_health, dict):
                    health_status.update(service_health)
            except Exception as e:
                health_status["health_check_error"] = str(e)
                health_status["status"] = ServiceConstants.STATUS_UNHEALTHY

            return health_status

        except Exception as e:
            self.logger.error(f"获取服务健康信息失败: {str(e)}")
            return {
                "service": "prediction",
                "status": ServiceConstants.STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _validate_prediction_params(
        self, current_qps: Optional[float], hours: int
    ) -> None:
        """
        验证预测参数

        Args:
            current_qps: 当前QPS
            hours: 预测小时数

        Raises:
            ValidationError: 参数验证失败
        """
        if current_qps is not None:
            if not (
                ServiceConstants.PREDICTION_MIN_QPS
                <= current_qps
                <= ServiceConstants.PREDICTION_MAX_QPS
            ):
                raise ValidationError(
                    "current_qps",
                    f"QPS值必须在 {ServiceConstants.PREDICTION_MIN_QPS}-{ServiceConstants.PREDICTION_MAX_QPS} 之间",
                )

        if not (
            ServiceConstants.PREDICTION_MIN_HOURS
            <= hours
            <= ServiceConstants.PREDICTION_MAX_HOURS
        ):
            raise ValidationError(
                "hours",
                f"预测小时数必须在 {ServiceConstants.PREDICTION_MIN_HOURS}-{ServiceConstants.PREDICTION_MAX_HOURS} 之间",
            )

    def _wrap_prediction_result(
        self,
        prediction_result: Any,
        service_name: str,
        hours: int,
        current_qps: Optional[float],
    ) -> Dict[str, Any]:
        """
        包装预测结果

        Args:
            prediction_result: 原始预测结果
            service_name: 服务名称
            hours: 预测小时数
            current_qps: 当前QPS

        Returns:
            标准化的预测结果
        """
        if isinstance(prediction_result, dict):
            # 如果已经是字典格式，直接使用
            wrapped_result = prediction_result.copy()
        else:
            # 否则创建标准格式
            wrapped_result = {
                "instances": (
                    prediction_result
                    if isinstance(prediction_result, (int, float))
                    else 1
                ),
                "timestamp": datetime.now().isoformat(),
            }

        # 添加请求参数信息
        wrapped_result.update(
            {
                "service_name": service_name,
                "prediction_hours": hours,
                "current_qps": current_qps,
            }
        )

        return wrapped_result
