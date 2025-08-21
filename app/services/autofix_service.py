#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 自动修复服务
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..common.constants import ServiceConstants
from ..common.exceptions import AutoFixError, ResourceNotFoundError, ValidationError
from ..core.agents.k8s_fixer import K8sFixerAgent
from ..core.agents.supervisor import SupervisorAgent
from .base import BaseService

logger = logging.getLogger("aiops.services.autofix")


class AutoFixService(BaseService):
    """
    自动修复服务 - 管理Kubernetes问题诊断和自动修复流程
    """

    def __init__(self) -> None:
        super().__init__("autofix")
        self._k8s_fixer: Optional[K8sFixerAgent] = None
        self._supervisor: Optional[SupervisorAgent] = None

    async def _do_initialize(self) -> None:
        """初始化自动修复服务"""
        try:
            # 初始化K8s修复代理
            self._k8s_fixer = K8sFixerAgent()
            self.logger.info("K8s修复代理初始化完成")

            # 初始化监督代理
            self._supervisor = SupervisorAgent()
            self.logger.info("监督代理初始化完成")

        except Exception as e:
            self.logger.error(f"自动修复服务组件初始化失败: {str(e)}")
            raise AutoFixError(f"初始化失败: {str(e)}")

    async def _do_health_check(self) -> bool:
        """自动修复服务健康检查"""
        try:
            if not self._k8s_fixer or not self._supervisor:
                return False

            # 检查K8s连接状态
            k8s_healthy = await self.execute_with_timeout(
                self._k8s_fixer.k8s_service.health_check,
                timeout=ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                operation_name="k8s_health_check",
            )

            return bool(k8s_healthy)

        except Exception as e:
            self.logger.warning(f"自动修复服务健康检查失败: {str(e)}")
            return False

    async def fix_kubernetes_deployment(
        self,
        deployment: str,
        namespace: str = "default",
        force_fix: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        修复Kubernetes部署问题

        Args:
            deployment: 部署名称
            namespace: 命名空间
            force_fix: 是否强制修复
            dry_run: 是否为演练模式

        Returns:
            修复结果字典

        Raises:
            ValidationError: 参数验证失败
            ResourceNotFoundError: 资源未找到
            AutoFixError: 修复过程失败
        """
        self._ensure_initialized()

        # 验证输入参数
        self._validate_k8s_params(deployment, namespace)

        try:
            # 获取部署信息
            deployment_info = await self.execute_with_timeout(
                lambda: self._k8s_fixer.k8s_service.get_deployment(
                    deployment, namespace
                ),
                timeout=ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                operation_name="get_deployment",
            )

            if not deployment_info:
                raise ResourceNotFoundError(
                    "Deployment", f"{deployment} (namespace: {namespace})"
                )

            # 执行修复操作
            if force_fix and not dry_run:
                # 使用监督代理执行修复工作流
                fix_result = await self.execute_with_timeout(
                    lambda: self._supervisor.execute_workflow(
                        workflow_type="k8s_deployment_fix",
                        params={
                            "deployment": deployment,
                            "namespace": namespace,
                            "deployment_info": deployment_info,
                        },
                    ),
                    timeout=ServiceConstants.AUTOFIX_WORKFLOW_TIMEOUT,
                    operation_name="supervisor_workflow",
                )

                return self._wrap_fix_result(
                    fix_result, deployment, namespace, "supervisor"
                )
            else:
                # 使用K8s修复代理分析和修复
                fix_result = await self.execute_with_timeout(
                    lambda: self._k8s_fixer.analyze_and_fix_deployment(
                        deployment_info, namespace, dry_run=dry_run
                    ),
                    timeout=ServiceConstants.AUTOFIX_ANALYSIS_TIMEOUT,
                    operation_name="k8s_fix_analysis",
                )

                return self._wrap_fix_result(
                    fix_result, deployment, namespace, "k8s_fixer", dry_run
                )

        except Exception as e:
            self.logger.error(f"修复Kubernetes部署失败: {str(e)}")
            if isinstance(e, (ValidationError, ResourceNotFoundError, AutoFixError)):
                raise e
            raise AutoFixError(f"修复失败: {str(e)}")

    async def diagnose_kubernetes_issues(
        self,
        deployment: Optional[str] = None,
        namespace: str = "default",
        include_pods: bool = True,
        include_logs: bool = False,
        include_events: bool = True,
    ) -> Dict[str, Any]:
        """
        诊断Kubernetes问题

        Args:
            deployment: 部署名称（可选）
            namespace: 命名空间
            include_pods: 是否包含Pod信息
            include_logs: 是否包含日志
            include_events: 是否包含事件

        Returns:
            诊断结果字典

        Raises:
            ValidationError: 参数验证失败
            AutoFixError: 诊断过程失败
        """
        self._ensure_initialized()

        # 验证命名空间
        if not namespace or not isinstance(namespace, str):
            raise ValidationError("namespace", "命名空间不能为空")

        try:
            diagnosis_result = {
                "namespace": namespace,
                "deployment": deployment,
                "timestamp": datetime.now().isoformat(),
                "diagnosis": {},
            }

            # 获取部署信息
            if deployment:
                deployment_info = await self.execute_with_timeout(
                    lambda: self._k8s_fixer.k8s_service.get_deployment(
                        deployment, namespace
                    ),
                    timeout=ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                    operation_name="get_deployment_for_diagnosis",
                )
                diagnosis_result["diagnosis"]["deployment"] = deployment_info

            # 获取Pod信息
            if include_pods:
                pods = await self.execute_with_timeout(
                    lambda: self._k8s_fixer.k8s_service.get_pods(namespace, deployment),
                    timeout=ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                    operation_name="get_pods",
                )
                diagnosis_result["diagnosis"]["pods"] = pods

                # 获取Pod日志
                if include_logs and pods:
                    logs = {}
                    for pod_info in pods[: ServiceConstants.AUTOFIX_MAX_PODS_FOR_LOGS]:
                        try:
                            pod_logs = await self.execute_with_timeout(
                                lambda: self._k8s_fixer.k8s_service.get_pod_logs(
                                    pod_info["name"], namespace
                                ),
                                timeout=ServiceConstants.AUTOFIX_LOGS_TIMEOUT,
                                operation_name="get_pod_logs",
                            )
                            logs[pod_info["name"]] = pod_logs
                        except Exception as e:
                            logs[pod_info["name"]] = f"获取日志失败: {str(e)}"

                    diagnosis_result["diagnosis"]["logs"] = logs

            # 获取事件信息
            if include_events:
                events = await self.execute_with_timeout(
                    lambda: self._k8s_fixer.k8s_service.get_events(
                        namespace, deployment
                    ),
                    timeout=ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                    operation_name="get_events",
                )
                diagnosis_result["diagnosis"]["events"] = events

            return diagnosis_result

        except Exception as e:
            self.logger.error(f"诊断Kubernetes问题失败: {str(e)}")
            if isinstance(e, (ValidationError, AutoFixError)):
                raise e
            raise AutoFixError(f"诊断失败: {str(e)}")

    async def get_autofix_config(self) -> Dict[str, Any]:
        """
        获取自动修复配置信息

        Returns:
            配置信息字典
        """
        from ..config.settings import config

        config_info = {
            "k8s_config": {
                "namespace_default": "default",
                "timeout": ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                "max_pods_for_logs": ServiceConstants.AUTOFIX_MAX_PODS_FOR_LOGS,
                "logs_timeout": ServiceConstants.AUTOFIX_LOGS_TIMEOUT,
            },
            "workflow_config": {
                "supervisor_timeout": ServiceConstants.AUTOFIX_WORKFLOW_TIMEOUT,
                "analysis_timeout": ServiceConstants.AUTOFIX_ANALYSIS_TIMEOUT,
                "retry_attempts": getattr(config, "autofix_retry_attempts", 3),
                "retry_delay": getattr(config, "autofix_retry_delay", 5),
            },
            "supported_resources": [
                "deployments",
                "pods",
                "services",
                "configmaps",
                "secrets",
            ],
            "fix_strategies": [
                "restart_deployment",
                "scale_deployment",
                "update_image",
                "fix_configuration",
                "resource_adjustment",
            ],
            "constraints": {
                "max_deployment_name_length": ServiceConstants.AUTOFIX_MAX_NAME_LENGTH,
                "max_namespace_name_length": ServiceConstants.AUTOFIX_MAX_NAME_LENGTH,
                "k8s_timeout": ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                "workflow_timeout": ServiceConstants.AUTOFIX_WORKFLOW_TIMEOUT,
            },
        }

        return config_info

    async def get_service_health_info(self) -> Dict[str, Any]:
        """
        获取自动修复服务详细健康信息

        Returns:
            健康信息字典
        """
        try:
            health_status = {
                "service": "autofix",
                "status": (
                    ServiceConstants.STATUS_HEALTHY
                    if await self.health_check()
                    else ServiceConstants.STATUS_UNHEALTHY
                ),
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "k8s_fixer": "unknown",
                    "supervisor": "unknown",
                    "k8s_service": "unknown",
                },
            }

            # 检查各组件状态
            if self._k8s_fixer:
                health_status["components"][
                    "k8s_fixer"
                ] = ServiceConstants.STATUS_HEALTHY

                # 检查K8s服务连接
                try:
                    k8s_health = await self.execute_with_timeout(
                        self._k8s_fixer.k8s_service.health_check,
                        timeout=ServiceConstants.AUTOFIX_K8S_TIMEOUT,
                        operation_name="k8s_service_health",
                    )
                    health_status["components"]["k8s_service"] = (
                        ServiceConstants.STATUS_HEALTHY
                        if k8s_health
                        else ServiceConstants.STATUS_UNHEALTHY
                    )
                except Exception:
                    health_status["components"][
                        "k8s_service"
                    ] = ServiceConstants.STATUS_UNHEALTHY
            else:
                health_status["components"][
                    "k8s_fixer"
                ] = ServiceConstants.STATUS_UNHEALTHY

            if self._supervisor:
                health_status["components"][
                    "supervisor"
                ] = ServiceConstants.STATUS_HEALTHY
            else:
                health_status["components"][
                    "supervisor"
                ] = ServiceConstants.STATUS_UNHEALTHY

            return health_status

        except Exception as e:
            self.logger.error(f"获取自动修复服务健康信息失败: {str(e)}")
            return {
                "service": "autofix",
                "status": ServiceConstants.STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _validate_k8s_params(self, deployment: str, namespace: str) -> None:
        """
        验证Kubernetes参数

        Args:
            deployment: 部署名称
            namespace: 命名空间

        Raises:
            ValidationError: 参数验证失败
        """
        if not deployment or not isinstance(deployment, str):
            raise ValidationError("deployment", "部署名称不能为空")

        if len(deployment) > ServiceConstants.AUTOFIX_MAX_NAME_LENGTH:
            raise ValidationError(
                "deployment",
                f"部署名称长度不能超过 {ServiceConstants.AUTOFIX_MAX_NAME_LENGTH} 字符",
            )

        if not namespace or not isinstance(namespace, str):
            raise ValidationError("namespace", "命名空间不能为空")

        if len(namespace) > ServiceConstants.AUTOFIX_MAX_NAME_LENGTH:
            raise ValidationError(
                "namespace",
                f"命名空间长度不能超过 {ServiceConstants.AUTOFIX_MAX_NAME_LENGTH} 字符",
            )

    def _wrap_fix_result(
        self,
        fix_result: Any,
        deployment: str,
        namespace: str,
        agent_type: str,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        包装修复结果

        Args:
            fix_result: 原始修复结果
            deployment: 部署名称
            namespace: 命名空间
            agent_type: 代理类型
            dry_run: 是否为演练模式

        Returns:
            标准化的修复结果
        """
        if isinstance(fix_result, dict):
            wrapped_result = fix_result.copy()
        else:
            wrapped_result = {"result": str(fix_result)}

        # 添加元数据
        wrapped_result.update(
            {
                "deployment": deployment,
                "namespace": namespace,
                "agent_type": agent_type,
                "dry_run": dry_run,
                "timestamp": datetime.now().isoformat(),
                "success": wrapped_result.get("success", True),
            }
        )

        # 确保包含必要字段
        if "actions_taken" not in wrapped_result:
            wrapped_result["actions_taken"] = []

        if "status" not in wrapped_result:
            wrapped_result["status"] = (
                "completed" if wrapped_result["success"] else "failed"
            )

        return wrapped_result
