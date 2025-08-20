#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 自动修复FastAPI路由 - 提供Kubernetes问题自动诊断、修复和工作流管理
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.config.settings import config
from app.core.agents.k8s_fixer import K8sFixerAgent
from app.core.agents.notifier import NotifierAgent
from app.core.agents.supervisor import SupervisorAgent
from app.models.request_models import AutoFixRequest
from app.models.response_models import APIResponse, AutoFixResponse
from app.services.notification import NotificationService
from app.utils.validators import (
    sanitize_input,
    validate_deployment_name,
    validate_namespace,
)

logger = logging.getLogger("aiops.autofix")

# 创建路由器
router = APIRouter(tags=["autofix"])

# 初始化Agent
supervisor_agent = SupervisorAgent()
k8s_fixer_agent = K8sFixerAgent()
notifier_agent = NotifierAgent()
notification_service = NotificationService()

# 请求模型
class AutoFixRequestModel(BaseModel):
    deployment: str = Field(..., description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    event: str = Field(..., description="问题描述或事件信息")
    auto_apply: bool = Field(False, description="是否自动应用修复")
    severity: str = Field("medium", description="问题严重级别", pattern="^(low|medium|high|critical)$")
    timeout: int = Field(300, description="修复超时时间(秒)", ge=60, le=1800)

class DiagnoseRequestModel(BaseModel):
    deployment: str = Field(..., description="Kubernetes Deployment名称")
    namespace: str = Field("default", description="Kubernetes命名空间")
    include_logs: bool = Field(True, description="是否包含日志分析")
    include_events: bool = Field(True, description="是否包含事件分析")

# 响应模型
class AutoFixResponseModel(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]


@router.post("/autofix", response_model=AutoFixResponseModel, summary="Kubernetes自动修复")
async def autofix_k8s(request: AutoFixRequestModel, background_tasks: BackgroundTasks) -> AutoFixResponseModel:
    """自动修复Kubernetes问题"""
    try:
        logger.info(f"收到自动修复请求: {request.dict()}")

        # 验证Kubernetes资源名称
        if not validate_deployment_name(request.deployment):
            raise HTTPException(status_code=400, detail="无效的Deployment名称")

        if not validate_namespace(request.namespace):
            raise HTTPException(status_code=400, detail="无效的命名空间名称")

        # 清理输入
        event_description = sanitize_input(request.event, 2000)

        logger.info(f"开始自动修复: deployment={request.deployment}, namespace={request.namespace}")

        # 创建AutoFixRequest对象
        autofix_request = AutoFixRequest(
            deployment=request.deployment,
            namespace=request.namespace,
            event=event_description,
            auto_apply=request.auto_apply,
            severity=request.severity,
            timeout=request.timeout
        )

        # 执行自动修复
        try:
            # 首先检查部署状态，判断是否有CrashLoopBackOff问题
            deployment = None
            pods = []
            
            try:
                deployment = await k8s_fixer_agent.k8s_service.get_deployment(
                    autofix_request.deployment, autofix_request.namespace
                )

                if deployment:
                    pods = await k8s_fixer_agent.k8s_service.get_pods(
                        namespace=autofix_request.namespace,
                        label_selector=f"app={autofix_request.deployment}",
                    )

                    # 检查是否有CrashLoopBackOff问题
                    for pod in pods:
                        if pod.get("status", {}).get("phase") == "Running":
                            continue
                        
                        container_statuses = pod.get("status", {}).get("containerStatuses", [])
                        for status in container_statuses:
                            waiting = status.get("state", {}).get("waiting", {})
                            if waiting.get("reason") == "CrashLoopBackOff":
                                logger.warning(f"检测到CrashLoopBackOff: Pod {pod.get('metadata', {}).get('name')}")
                                # 可以添加特殊处理逻辑

            except Exception as k8s_e:
                logger.warning(f"获取Kubernetes资源状态失败: {str(k8s_e)}")

            # 如果有工作流系统，使用Supervisor Agent
            start_time = time.time()
            
            if hasattr(supervisor_agent, 'execute_workflow'):
                logger.info("使用工作流执行自动修复")
                
                workflow_result = await supervisor_agent.execute_workflow(
                    problem_description=f"Kubernetes问题: {event_description}",
                    deployment_name=autofix_request.deployment,
                    namespace=autofix_request.namespace,
                    auto_apply=autofix_request.auto_apply
                )
                
                result = workflow_result
            else:
                # 直接使用K8s修复Agent
                logger.info("使用K8s修复Agent执行修复")
                
                result = await k8s_fixer_agent.analyze_and_fix_deployment(
                    autofix_request.deployment,
                    autofix_request.namespace,
                    event_description
                )

            execution_time = time.time() - start_time

            # 包装响应
            response_data = AutoFixResponse(
                deployment=autofix_request.deployment,
                namespace=autofix_request.namespace,
                event=event_description,
                **({} if isinstance(result, str) else result),
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )

            # 如果需要，异步发送通知
            if hasattr(notification_service, 'send_notification'):
                background_tasks.add_task(
                    notification_service.send_notification,
                    "autofix_completed",
                    response_data.dict()
                )

            return AutoFixResponseModel(
                code=0,
                message="自动修复完成",
                data=response_data.dict()
            )

        except Exception as e:
            logger.error(f"自动修复执行失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"自动修复失败: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自动修复请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"自动修复请求处理失败: {str(e)}")


@router.post("/autofix/diagnose", response_model=AutoFixResponseModel, summary="Kubernetes问题诊断")
async def diagnose_k8s(request: DiagnoseRequestModel) -> AutoFixResponseModel:
    """诊断Kubernetes问题（不进行修复）"""
    try:
        logger.info(f"收到诊断请求: {request.dict()}")

        # 验证参数
        if not validate_deployment_name(request.deployment):
            raise HTTPException(status_code=400, detail="无效的Deployment名称")

        if not validate_namespace(request.namespace):
            raise HTTPException(status_code=400, detail="无效的命名空间名称")

        # 执行诊断
        try:
            # 获取部署信息
            deployment = await k8s_fixer_agent.k8s_service.get_deployment(
                request.deployment, request.namespace
            )

            if not deployment:
                raise HTTPException(status_code=404, detail="找不到指定的Deployment")

            # 获取Pod信息
            pods = await k8s_fixer_agent.k8s_service.get_pods(
                namespace=request.namespace,
                label_selector=f"app={request.deployment}"
            )

            # 分析问题
            diagnosis = {
                "deployment_status": deployment.get("status", {}),
                "pods": [],
                "issues": [],
                "recommendations": []
            }

            # 分析每个Pod
            for pod in pods:
                pod_info = {
                    "name": pod.get("metadata", {}).get("name"),
                    "status": pod.get("status", {}),
                    "issues": []
                }

                # 检查Pod状态
                phase = pod.get("status", {}).get("phase")
                if phase != "Running":
                    pod_info["issues"].append(f"Pod状态异常: {phase}")

                # 检查容器状态
                container_statuses = pod.get("status", {}).get("containerStatuses", [])
                for status in container_statuses:
                    if not status.get("ready", False):
                        waiting = status.get("state", {}).get("waiting", {})
                        if waiting:
                            pod_info["issues"].append(f"容器等待: {waiting}")

                # 获取日志（如果需要）
                if request.include_logs and pod_info["issues"]:
                    try:
                        logs = await k8s_fixer_agent.k8s_service.get_pod_logs(
                            pod_info["name"], request.namespace
                        )
                        pod_info["recent_logs"] = logs[-10:] if logs else []
                    except Exception as log_e:
                        logger.warning(f"获取Pod日志失败: {str(log_e)}")

                diagnosis["pods"].append(pod_info)

            # 获取事件（如果需要）
            if request.include_events:
                try:
                    events = await k8s_fixer_agent.k8s_service.get_events(
                        namespace=request.namespace,
                        field_selector=f"involvedObject.name={request.deployment}"
                    )
                    diagnosis["events"] = events[-20:] if events else []
                except Exception as event_e:
                    logger.warning(f"获取事件失败: {str(event_e)}")

            # 生成总体分析
            total_pods = len(pods)
            healthy_pods = len([p for p in pods if p.get("status", {}).get("phase") == "Running"])
            
            if healthy_pods < total_pods:
                diagnosis["issues"].append(f"有 {total_pods - healthy_pods} 个Pod不健康")

            return AutoFixResponseModel(
                code=0,
                message="诊断完成",
                data={
                    "deployment": request.deployment,
                    "namespace": request.namespace,
                    "diagnosis": diagnosis,
                    "summary": {
                        "total_pods": total_pods,
                        "healthy_pods": healthy_pods,
                        "issues_found": len(diagnosis["issues"])
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"诊断失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"诊断失败: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"诊断请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"诊断请求处理失败: {str(e)}")


@router.get("/autofix/health", response_model=AutoFixResponseModel, summary="自动修复服务健康检查")
async def autofix_health() -> AutoFixResponseModel:
    """自动修复服务健康检查"""
    try:
        health_status = {
            "service": "autofix",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "supervisor_agent": "unknown",
                "k8s_fixer_agent": "unknown",
                "notifier_agent": "unknown",
                "kubernetes_api": "unknown"
            }
        }

        # 检查各组件状态
        try:
            if supervisor_agent:
                health_status["components"]["supervisor_agent"] = "healthy"
        except Exception:
            health_status["components"]["supervisor_agent"] = "unhealthy"

        try:
            if k8s_fixer_agent:
                health_status["components"]["k8s_fixer_agent"] = "healthy"
                
                # 检查Kubernetes API连接
                if hasattr(k8s_fixer_agent, 'k8s_service'):
                    k8s_health = await k8s_fixer_agent.k8s_service.health_check()
                    health_status["components"]["kubernetes_api"] = "healthy" if k8s_health else "unhealthy"
        except Exception:
            health_status["components"]["k8s_fixer_agent"] = "unhealthy"

        try:
            if notifier_agent:
                health_status["components"]["notifier_agent"] = "healthy"
        except Exception:
            health_status["components"]["notifier_agent"] = "unhealthy"

        # 检查整体健康状态
        unhealthy_components = [k for k, v in health_status["components"].items() if v == "unhealthy"]
        if unhealthy_components:
            health_status["status"] = "degraded"
            health_status["unhealthy_components"] = unhealthy_components

        return AutoFixResponseModel(
            code=0,
            message="健康检查完成",
            data=health_status
        )

    except Exception as e:
        logger.error(f"自动修复健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/autofix/ready", response_model=AutoFixResponseModel, summary="自动修复服务就绪检查")
async def autofix_ready() -> AutoFixResponseModel:
    """自动修复服务就绪检查"""
    try:
        is_ready = (
            supervisor_agent and
            k8s_fixer_agent and
            notifier_agent and
            hasattr(k8s_fixer_agent, 'k8s_service') and k8s_fixer_agent.k8s_service
        )

        if not is_ready:
            raise HTTPException(status_code=503, detail="自动修复服务未就绪")

        return AutoFixResponseModel(
            code=0,
            message="自动修复服务已就绪",
            data={
                "ready": True,
                "timestamp": datetime.now().isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自动修复就绪检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"就绪检查失败: {str(e)}")


@router.get("/autofix/info", response_model=AutoFixResponseModel, summary="自动修复服务信息")
async def autofix_info() -> AutoFixResponseModel:
    """获取自动修复服务信息"""
    try:
        info = {
            "service": "自动修复",
            "version": "1.0.0",
            "description": "基于AI的Kubernetes问题自动诊断和修复服务",
            "capabilities": [
                "Kubernetes问题诊断",
                "自动修复建议",
                "工作流管理",
                "智能通知"
            ],
            "endpoints": {
                "fix": "/api/v1/autofix",
                "diagnose": "/api/v1/autofix/diagnose",
                "health": "/api/v1/autofix/health",
                "ready": "/api/v1/autofix/ready",
                "info": "/api/v1/autofix/info"
            },
            "supported_resources": [
                "Deployment",
                "Pod", 
                "Service",
                "ConfigMap",
                "Secret"
            ],
            "fix_strategies": [
                "资源配置优化",
                "健康检查修复",
                "镜像问题解决",
                "网络连接修复"
            ],
            "agents": {
                "supervisor": "工作流协调器",
                "k8s_fixer": "Kubernetes修复器",
                "notifier": "通知服务"
            },
            "status": "available",
            "timestamp": datetime.now().isoformat()
        }

        return AutoFixResponseModel(
            code=0,
            message="获取信息成功", 
            data=info
        )

    except Exception as e:
        logger.error(f"获取自动修复服务信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取服务信息失败: {str(e)}")


# 导出
__all__ = ["router"]