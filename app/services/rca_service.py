#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 根因分析服务 - 提供异常检测、相关性分析和根本原因识别的业务逻辑
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base import BaseService, HealthCheckMixin
from ..common.constants import ServiceConstants
from ..common.exceptions import RCAError, ValidationError
from ..core.rca.analyzer import RCAAnalyzer

logger = logging.getLogger("aiops.services.rca")


class RCAService(BaseService, HealthCheckMixin):
    """
    根因分析服务 - 管理异常检测和根因分析流程
    """
    
    def __init__(self) -> None:
        super().__init__("rca")
        self._analyzer: Optional[RCAAnalyzer] = None
    
    async def _do_initialize(self) -> None:
        """初始化RCA服务"""
        try:
            self._analyzer = RCAAnalyzer()
            self.logger.info("RCA分析器初始化完成")
        except Exception as e:
            self.logger.error(f"RCA分析器初始化失败: {str(e)}")
            raise RCAError(f"初始化失败: {str(e)}")
    
    async def _do_health_check(self) -> bool:
        """RCA服务健康检查"""
        try:
            if not self._analyzer:
                return False
            
            # 检查各组件状态
            components_healthy = all([
                hasattr(self._analyzer, 'detector') and self._analyzer.detector,
                hasattr(self._analyzer, 'correlator') and self._analyzer.correlator,
                hasattr(self._analyzer, 'prometheus') and self._analyzer.prometheus
            ])
            
            return components_healthy
            
        except Exception as e:
            self.logger.warning(f"RCA服务健康检查失败: {str(e)}")
            return False
    
    async def analyze_root_cause(
        self,
        metrics: List[str],
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None,
        namespace: str = "default",
        include_logs: bool = False,
        severity_threshold: float = ServiceConstants.RCA_DEFAULT_SEVERITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        执行根因分析
        
        Args:
            metrics: 监控指标列表
            start_time: 开始时间
            end_time: 结束时间
            service_name: 服务名称
            namespace: Kubernetes命名空间
            include_logs: 是否包含日志分析
            severity_threshold: 严重性阈值
            
        Returns:
            根因分析结果字典
            
        Raises:
            ValidationError: 参数验证失败
            RCAError: 分析过程失败
        """
        self._ensure_initialized()
        
        # 验证输入参数
        self._validate_rca_params(metrics, start_time, end_time, severity_threshold)
        
        try:
            # 调用核心分析服务
            analysis_result = await self.execute_with_timeout(
                lambda: self._analyzer.analyze(
                    metrics,
                    start_time,
                    end_time,
                    service_name,
                    namespace,
                    include_logs
                ),
                timeout=ServiceConstants.RCA_TIMEOUT,
                operation_name="root_cause_analysis"
            )
            
            # 包装分析结果
            return {
                "analysis": analysis_result,
                "request_info": {
                    "metrics": metrics,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "service_name": service_name,
                    "namespace": namespace,
                    "include_logs": include_logs,
                    "severity_threshold": severity_threshold
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"根因分析执行失败: {str(e)}")
            raise RCAError(f"分析失败: {str(e)}")
    
    async def get_available_metrics(
        self,
        service_name: Optional[str] = None,
        category: Optional[str] = None
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
            # 调用分析器获取指标
            metrics = await self.execute_with_timeout(
                lambda: self._analyzer.get_available_metrics(
                    service_name=service_name,
                    category=category
                ),
                timeout=30.0,
                operation_name="get_available_metrics"
            )
            
            return metrics if metrics else []
            
        except Exception as e:
            self.logger.error(f"获取可用指标失败: {str(e)}")
            raise RCAError(f"获取指标失败: {str(e)}")
    
    async def query_metric_data(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        查询指标数据
        
        Args:
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间
            service_name: 服务名称
            
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
            # 调用分析器查询数据
            metric_data = await self.execute_with_timeout(
                lambda: self._analyzer.query_metric_data(
                    metric_name,
                    start_time,
                    end_time,
                    service_name
                ),
                timeout=60.0,
                operation_name="query_metric_data"
            )
            
            return metric_data if metric_data else []
            
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
                "threshold_factor": getattr(config, 'rca_threshold_factor', 2.0),
                "window_size": getattr(config, 'rca_window_size', 60)
            },
            "correlation_analysis": {
                "method": "pearson",
                "min_correlation": getattr(config, 'min_correlation', 0.7),
                "max_lag": getattr(config, 'max_correlation_lag', 300)
            },
            "supported_metrics": [
                "cpu_usage",
                "memory_usage", 
                "network_io",
                "disk_io",
                "request_rate",
                "error_rate",
                "response_time"
            ],
            "constraints": {
                "min_metrics": ServiceConstants.RCA_MIN_METRICS,
                "max_metrics": ServiceConstants.RCA_MAX_METRICS,
                "default_severity_threshold": ServiceConstants.RCA_DEFAULT_SEVERITY_THRESHOLD,
                "timeout": ServiceConstants.RCA_TIMEOUT
            },
            "prometheus_config": {
                "endpoint": getattr(config.prometheus, 'url', 'unknown'),
                "timeout": getattr(config.prometheus, 'timeout', 30)
            }
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
                "status": ServiceConstants.STATUS_HEALTHY if await self.health_check() else ServiceConstants.STATUS_UNHEALTHY,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "anomaly_detector": "unknown",
                    "correlator": "unknown", 
                    "prometheus": "unknown",
                    "llm_service": "unknown"
                }
            }

            # 检查各组件状态
            if self._analyzer:
                # 检查异常检测器
                try:
                    if hasattr(self._analyzer, 'detector') and self._analyzer.detector:
                        health_status["components"]["anomaly_detector"] = ServiceConstants.STATUS_HEALTHY
                except Exception:
                    health_status["components"]["anomaly_detector"] = ServiceConstants.STATUS_UNHEALTHY

                # 检查相关性分析器
                try:
                    if hasattr(self._analyzer, 'correlator') and self._analyzer.correlator:
                        health_status["components"]["correlator"] = ServiceConstants.STATUS_HEALTHY
                except Exception:
                    health_status["components"]["correlator"] = ServiceConstants.STATUS_UNHEALTHY

                # 检查Prometheus连接
                try:
                    if hasattr(self._analyzer, 'prometheus') and self._analyzer.prometheus:
                        if hasattr(self._analyzer.prometheus, 'is_healthy'):
                            prometheus_health = self._analyzer.prometheus.is_healthy()
                        else:
                            prometheus_health = bool(getattr(self._analyzer.prometheus, 'client', None))
                        health_status["components"]["prometheus"] = ServiceConstants.STATUS_HEALTHY if prometheus_health else ServiceConstants.STATUS_UNHEALTHY
                except Exception:
                    health_status["components"]["prometheus"] = ServiceConstants.STATUS_UNHEALTHY

                # 检查LLM服务
                try:
                    if hasattr(self._analyzer, 'llm') and self._analyzer.llm:
                        if hasattr(self._analyzer.llm, 'health_check_sync'):
                            llm_health = self._analyzer.llm.health_check_sync()
                        elif hasattr(self._analyzer.llm, 'is_available'):
                            llm_health = self._analyzer.llm.is_available()
                        else:
                            llm_health = bool(getattr(self._analyzer.llm, 'client', None))
                        health_status["components"]["llm_service"] = ServiceConstants.STATUS_HEALTHY if llm_health else ServiceConstants.STATUS_UNHEALTHY
                except Exception:
                    health_status["components"]["llm_service"] = ServiceConstants.STATUS_UNHEALTHY

            return health_status
            
        except Exception as e:
            self.logger.error(f"获取RCA服务健康信息失败: {str(e)}")
            return {
                "service": "rca",
                "status": ServiceConstants.STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_rca_params(
        self,
        metrics: List[str],
        start_time: datetime,
        end_time: datetime,
        severity_threshold: float
    ) -> None:
        """
        验证RCA分析参数
        
        Args:
            metrics: 指标列表
            start_time: 开始时间
            end_time: 结束时间
            severity_threshold: 严重性阈值
            
        Raises:
            ValidationError: 参数验证失败
        """
        # 验证指标数量
        if not (ServiceConstants.RCA_MIN_METRICS <= len(metrics) <= ServiceConstants.RCA_MAX_METRICS):
            raise ValidationError(
                "metrics",
                f"指标数量必须在 {ServiceConstants.RCA_MIN_METRICS}-{ServiceConstants.RCA_MAX_METRICS} 之间"
            )
        
        # 验证时间范围
        if start_time >= end_time:
            raise ValidationError("time_range", "开始时间必须早于结束时间")
        
        # 验证严重性阈值
        if not (0.0 <= severity_threshold <= 1.0):
            raise ValidationError("severity_threshold", "严重性阈值必须在 0.0-1.0 之间")
