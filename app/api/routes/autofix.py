#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI-CloudOps智能自动修复API接口
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.decorators import api_response, log_api_call
from app.common.constants import ApiEndpoints, AppConstants, ServiceConstants
from app.common.response import ResponseWrapper
from app.models import (
    AutoFixRequest,
    AutoFixResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    ServiceConfigResponse,
    ServiceHealthResponse,
    ServiceInfoResponse,
)
from app.services.autofix_service import AutoFixService
from app.services.notification import NotificationService

logger = logging.getLogger("aiops.api.autofix")

router = APIRouter(tags=["autofix"])
autofix_service = AutoFixService()
notification_service = NotificationService()

# 性能统计
_api_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0.0,
    "last_reset": datetime.now(),
}


async def send_fix_notification(
    deployment: str, namespace: str, status: str, actions_taken: list
):
    """发送修复通知"""
    max_retries = ServiceConstants.DEFAULT_REQUEST_TIMEOUT // 10
    retry_delay = ServiceConstants.DEFAULT_RETRY_DELAY

    for attempt in range(max_retries):
        try:
            await asyncio.wait_for(
                notification_service.send_notification(
                    title=f"AI-CloudOps自动修复完成: {deployment}",
                    message=f"AI-CloudOps在命名空间 {namespace} 中的部署 {deployment} 修复状态: {status}",
                    severity="info" if status == "completed" else "warning",
                    metadata={
                        "deployment": deployment,
                        "namespace": namespace,
                        "status": status,
                        "actions_taken": actions_taken,
                        "attempt": attempt + 1,
                        "timestamp": datetime.now().isoformat(),
                    },
                ),
                timeout=ServiceConstants.DEFAULT_REQUEST_TIMEOUT,
            )
            logger.info(f"通知发送成功 - 部署: {deployment}")
            return
        except asyncio.TimeoutError:
            logger.warning(f"通知发送超时 - 尝试 {attempt + 1}/{max_retries}")
        except Exception as e:
            logger.warning(
                f"发送修复通知失败 - 尝试 {attempt + 1}/{max_retries}: {str(e)}"
            )

        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))

    logger.error(f"通知发送最终失败 - 部署: {deployment}")


@router.post("/repair", summary="AI-CloudOps Kubernetes自动修复")
@api_response("AI-CloudOps Kubernetes自动修复")
@log_api_call(log_request=True)
async def autofix_k8s(
    request: AutoFixRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """执行Kubernetes自动修复"""
    start_time = time.time()
    global _api_stats
    _api_stats["total_requests"] += 1

    # 参数验证
    if not request.deployment or not request.deployment.strip():
        _api_stats["failed_requests"] += 1
        raise HTTPException(status_code=400, detail="部署名称不能为空")

    if not request.namespace or not request.namespace.strip():
        _api_stats["failed_requests"] += 1
        raise HTTPException(status_code=400, detail="命名空间不能为空")

    try:
        await asyncio.wait_for(
            autofix_service.initialize(),
            timeout=ServiceConstants.DEFAULT_WARMUP_TIMEOUT,
        )

        # 执行修复操作，带超时控制
        fix_result = await asyncio.wait_for(
            autofix_service.fix_kubernetes_deployment(
                deployment=request.deployment,
                namespace=request.namespace,
                force=request.force,
                auto_restart=request.auto_restart,
            ),
            timeout=ServiceConstants.AUTOFIX_WORKFLOW_TIMEOUT,
        )

        # 发送通知（如果需要）
        if request.auto_restart and fix_result.get("success", False):
            try:
                background_tasks.add_task(
                    send_fix_notification,
                    request.deployment,
                    request.namespace,
                    fix_result.get("status", "unknown"),
                    fix_result.get("actions_taken", []),
                )
            except Exception as e:
                logger.warning(f"添加通知任务失败: {str(e)}")

        # 构建响应
        response = AutoFixResponse(
            event=request.event,
            deployment=request.deployment,
            namespace=request.namespace,
            execution_time=fix_result.get("execution_time", 0.0),
            success=fix_result.get("success", False),
            status=fix_result.get("status", "unknown"),
            actions_taken=fix_result.get("actions_taken", []),
        )

        # 更新统计
        processing_time = time.time() - start_time
        _api_stats["successful_requests"] += 1
        _api_stats["average_response_time"] = (
            _api_stats["average_response_time"]
            * (_api_stats["successful_requests"] - 1)
            + processing_time
        ) / _api_stats["successful_requests"]

        logger.info(
            f"自动修复完成 - 部署: {request.deployment}, 耗时: {processing_time:.2f}s"
        )
        return ResponseWrapper.success(data=response.dict(), message="success")

    except asyncio.TimeoutError:
        _api_stats["failed_requests"] += 1
        logger.error(f"自动修复超时 - 部署: {request.deployment}")
        raise HTTPException(status_code=504, detail="修复操作超时")
    except Exception as e:
        _api_stats["failed_requests"] += 1
        logger.error(f"自动修复失败 - 部署: {request.deployment}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"修复操作失败: {str(e)}")


@router.post("/diagnose", summary="AI-CloudOps Kubernetes问题诊断")
@api_response("AI-CloudOps Kubernetes问题诊断")
@log_api_call(log_request=True)
async def diagnose_k8s(request: DiagnoseRequest) -> Dict[str, Any]:
    """执行Kubernetes问题诊断"""
    start_time = time.time()

    try:
        await asyncio.wait_for(
            autofix_service.initialize(),
            timeout=ServiceConstants.DEFAULT_WARMUP_TIMEOUT,
        )

        # 执行诊断，带超时控制
        diagnosis_result = await asyncio.wait_for(
            autofix_service.diagnose_kubernetes_issues(
                deployment=request.deployment,
                namespace=request.namespace,
                include_pods=request.include_pods,
                include_logs=request.include_logs,
                include_events=request.include_events,
            ),
            timeout=ServiceConstants.AUTOFIX_ANALYSIS_TIMEOUT,
        )

        # 构建响应
        response = DiagnoseResponse(
            deployment=request.deployment,
            namespace=request.namespace,
            status=diagnosis_result.get("status", "completed"),
            issues_found=diagnosis_result.get("issues_found", []),
            recommendations=diagnosis_result.get("recommendations", []),
            pods_status=diagnosis_result.get("pods_status"),
            logs_summary=diagnosis_result.get("logs_summary"),
            events_summary=diagnosis_result.get("events_summary"),
            timestamp=datetime.now().isoformat(),
        )

        processing_time = time.time() - start_time
        logger.info(
            f"诊断完成 - 部署: {request.deployment}, 耗时: {processing_time:.2f}s"
        )
        return ResponseWrapper.success(data=response.dict(), message="success")

    except asyncio.TimeoutError:
        logger.error(f"诊断超时 - 部署: {request.deployment}")
        raise HTTPException(status_code=504, detail="诊断操作超时")
    except Exception as e:
        logger.error(f"诊断失败 - 部署: {request.deployment}, 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"诊断操作失败: {str(e)}")


@router.get("/health", summary="AI-CloudOps自动修复服务健康检查")
@api_response("AI-CloudOps自动修复服务健康检查")
async def autofix_health() -> Dict[str, Any]:
    """健康检查"""
    try:
        await asyncio.wait_for(
            autofix_service.initialize(),
            timeout=ServiceConstants.DEFAULT_WARMUP_TIMEOUT,
        )

        health_status = await asyncio.wait_for(
            autofix_service.get_service_health_info(),
            timeout=ServiceConstants.DEFAULT_REQUEST_TIMEOUT,
        )

        # 添加API统计信息
        health_status["api_stats"] = _api_stats.copy()
        health_status["api_stats"]["success_rate"] = (
            _api_stats["successful_requests"]
            / max(_api_stats["total_requests"], 1)
            * 100
        )

        response = ServiceHealthResponse(
            status=health_status.get("status", "healthy"),
            service="autofix",
            version=health_status.get("version"),
            dependencies=health_status.get("dependencies"),
            last_check_time=datetime.now().isoformat(),
            uptime=health_status.get("uptime"),
        )

        return ResponseWrapper.success(data=response.dict(), message="success")

    except asyncio.TimeoutError:
        logger.error("健康检查超时")
        raise HTTPException(status_code=504, detail="健康检查超时")
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")


@router.get("/config", summary="AI-CloudOps获取自动修复配置")
@api_response("AI-CloudOps获取自动修复配置")
async def get_autofix_config() -> Dict[str, Any]:
    await autofix_service.initialize()

    config_info = await autofix_service.get_autofix_config()

    response = ServiceConfigResponse(
        service="autofix", config=config_info, timestamp=datetime.now().isoformat()
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/info", summary="AI-CloudOps自动修复服务信息")
@api_response("AI-CloudOps自动修复服务信息")
async def autofix_info() -> Dict[str, Any]:
    info = {
        "service": "自动修复",
        "version": AppConstants.APP_VERSION,
        "description": "Kubernetes自动修复服务",
        "capabilities": ["部署问题诊断", "自动修复建议", "工作流执行", "故障自愈"],
        "endpoints": {
            "autofix": ApiEndpoints.AUTOFIX,
            "diagnose": ApiEndpoints.AUTOFIX_DIAGNOSE,
            "health": ApiEndpoints.AUTOFIX_HEALTH,
            "config": ApiEndpoints.AUTOFIX_CONFIG,
            "info": ApiEndpoints.AUTOFIX_INFO,
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
            "max_name_length": ServiceConstants.AUTOFIX_MAX_NAME_LENGTH,
            "k8s_timeout": ServiceConstants.AUTOFIX_K8S_TIMEOUT,
            "workflow_timeout": ServiceConstants.AUTOFIX_WORKFLOW_TIMEOUT,
            "analysis_timeout": ServiceConstants.AUTOFIX_ANALYSIS_TIMEOUT,
        },
        "performance": {
            "total_requests": _api_stats["total_requests"],
            "success_rate": f"{_api_stats['successful_requests'] / max(_api_stats['total_requests'], 1) * 100:.2f}%",
            "average_response_time": f"{_api_stats['average_response_time']:.3f}s",
        },
        "status": "available" if autofix_service else "unavailable",
    }

    response = ServiceInfoResponse(
        service=info["service"],
        version=info["version"],
        description=info["description"],
        capabilities=info["capabilities"],
        endpoints=info["endpoints"],
        constraints=info["constraints"],
        status=info["status"],
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/stats/reset", summary="重置API统计信息")
@api_response("重置API统计信息")
async def reset_stats() -> Dict[str, Any]:
    """重置API统计信息"""
    global _api_stats
    old_stats = _api_stats.copy()
    _api_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time": 0.0,
        "last_reset": datetime.now(),
    }

    logger.info("API统计信息已重置")
    return ResponseWrapper.success(
        data={"old_stats": old_stats, "reset_time": datetime.now().isoformat()},
        message="统计信息已重置",
    )


__all__ = ["router"]
