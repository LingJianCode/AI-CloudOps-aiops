#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 自动修复FastAPI路由 - 提供Kubernetes问题诊断和自动修复功能
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from app.api.decorators import api_response, log_api_call
from app.common.constants import ServiceConstants, AppConstants, ApiEndpoints
from app.common.response import ResponseWrapper
from app.models.response_models import AutoFixResponse
from app.services.autofix_service import AutoFixService
from app.services.notification import NotificationService

logger = logging.getLogger("aiops.api.autofix")

router = APIRouter(tags=["autofix"])
autofix_service = AutoFixService()
notification_service = NotificationService()

class AutoFixRequestModel(BaseModel):
    deployment: str = Field(..., description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    event: str = Field(..., description="问题描述或事件信息")
    auto_apply: bool = Field(False, description="是否自动应用修复")
    dry_run: bool = Field(False, description="是否为演练模式")
    force_fix: bool = Field(False, description="是否强制修复")
    severity: str = Field("medium", description="问题严重级别", pattern="^(low|medium|high|critical)$")
    timeout: int = Field(ServiceConstants.AUTOFIX_DEFAULT_TIMEOUT, description="修复超时时间(秒)", ge=60, le=1800)

class DiagnoseRequestModel(BaseModel):
    deployment: Optional[str] = Field(None, description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(True, description="是否包含日志分析")
    include_pods: bool = Field(True, description="是否包含Pod信息")
    include_events: bool = Field(True, description="是否包含事件信息")

class AutoFixResponseModel(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


async def send_fix_notification(deployment: str, namespace: str, status: str, actions_taken: list):
    try:
        await notification_service.send_notification(
            title=f"自动修复完成: {deployment}",
            message=f"命名空间 {namespace} 中的部署 {deployment} 修复状态: {status}",
            severity="info" if status == "completed" else "warning",
            metadata={
                "deployment": deployment,
                "namespace": namespace,
                "status": status,
                "actions_taken": actions_taken
            }
        )
    except Exception as e:
        logger.error(f"发送修复通知失败: {str(e)}")


@router.post("/autofix", summary="Kubernetes自动修复")
@api_response("Kubernetes自动修复")
@log_api_call(log_request=True)
async def autofix_k8s(request: AutoFixRequestModel, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    await autofix_service.initialize()

    fix_result = await autofix_service.fix_kubernetes_deployment(
        deployment=request.deployment,
        namespace=request.namespace,
        force_fix=request.force_fix,
        dry_run=request.dry_run
    )

    if request.auto_apply and fix_result.get("success", False):
        try:
            background_tasks.add_task(
                send_fix_notification,
                request.deployment,
                request.namespace,
                fix_result.get("status", "unknown"),
                fix_result.get("actions_taken", [])
            )
        except Exception as e:
            logger.warning(f"添加通知任务失败: {str(e)}")

    response = AutoFixResponse(
        event=request.event,
        deployment=request.deployment,
        namespace=request.namespace,
        execution_time=fix_result.get("execution_time", 0.0),
        success=fix_result.get("success", False),
        status=fix_result.get("status", "unknown"),
        actions_taken=fix_result.get("actions_taken", [])
    )

    return ResponseWrapper.success(
        data=response.dict(),
        message="success"
    )


@router.post("/autofix/diagnose", summary="Kubernetes问题诊断")
@api_response("Kubernetes问题诊断")
@log_api_call(log_request=True)
async def diagnose_k8s(request: DiagnoseRequestModel) -> Dict[str, Any]:
    await autofix_service.initialize()

    diagnosis_result = await autofix_service.diagnose_kubernetes_issues(
        deployment=request.deployment,
        namespace=request.namespace,
        include_pods=request.include_pods,
        include_logs=request.include_logs,
        include_events=request.include_events
    )

    return ResponseWrapper.success(
        data=diagnosis_result,
        message="success"
    )


@router.get("/autofix/health", summary="自动修复服务健康检查")
@api_response("自动修复服务健康检查")
async def autofix_health() -> Dict[str, Any]:
    await autofix_service.initialize()
    
    health_status = await autofix_service.get_service_health_info()

    return ResponseWrapper.success(
        data=health_status,
        message="success"
    )


@router.get("/autofix/config", summary="获取自动修复配置")
@api_response("获取自动修复配置")
async def get_autofix_config() -> Dict[str, Any]:
    await autofix_service.initialize()
    
    config_info = await autofix_service.get_autofix_config()

    return ResponseWrapper.success(
        data=config_info,
        message="success"
    )


@router.get("/autofix/info", summary="自动修复服务信息")
@api_response("自动修复服务信息")
async def autofix_info() -> Dict[str, Any]:
    info = {
        "service": "自动修复",
        "version": AppConstants.APP_VERSION,
        "description": "基于智能代理的Kubernetes自动修复服务",
        "capabilities": [
            "部署问题诊断",
            "自动修复建议",
            "工作流执行",
            "故障自愈"
        ],
        "endpoints": {
            "autofix": ApiEndpoints.AUTOFIX,
            "diagnose": ApiEndpoints.AUTOFIX_DIAGNOSE,
            "health": ApiEndpoints.AUTOFIX_HEALTH,
            "config": ApiEndpoints.AUTOFIX_CONFIG,
            "info": ApiEndpoints.AUTOFIX_INFO
        },
        "supported_resources": [
            "deployments",
            "pods",
            "services",
            "configmaps",
            "secrets"
        ],
        "fix_strategies": [
            "restart_deployment",
            "scale_deployment", 
            "update_image",
            "fix_configuration",
            "resource_adjustment"
        ],
        "constraints": {
            "max_name_length": ServiceConstants.AUTOFIX_MAX_NAME_LENGTH,
            "k8s_timeout": ServiceConstants.AUTOFIX_K8S_TIMEOUT,
            "workflow_timeout": ServiceConstants.AUTOFIX_WORKFLOW_TIMEOUT,
            "analysis_timeout": ServiceConstants.AUTOFIX_ANALYSIS_TIMEOUT
        },
        "status": "available" if autofix_service else "unavailable"
    }

    return ResponseWrapper.success(
        data=info,
        message="success"
    )


__all__ = ["router"]
