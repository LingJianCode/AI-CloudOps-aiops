#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 系统健康检查API模块 - 提供系统级健康监控和状态检查功能
"""

import logging
from datetime import datetime

from flask import Blueprint, jsonify

from app.models.response_models import APIResponse
from .health_manager import health_manager

logger = logging.getLogger("aiops.health")

# 创建健康检查蓝图
health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    系统综合健康检查API
    
    Returns:
        JSON: 系统整体健康状态
    """
    try:
        health_data = health_manager.get_overall_health()
        return jsonify(APIResponse(code=0, message="健康检查完成", data=health_data).dict())

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return (
            jsonify(
                APIResponse(
                    code=500,
                    message=f"健康检查失败: {str(e)}",
                    data={"timestamp": datetime.utcnow().isoformat()},
                ).dict()
            ),
            500,
        )


@health_bp.route("/health/components", methods=["GET"])
def components_health():
    """
    组件详细健康检查API
    
    Returns:
        JSON: 所有组件的详细健康状态
    """
    try:
        components_detail = health_manager.check_all_components()
        
        return jsonify(
            APIResponse(
                code=0,
                message="组件健康检查完成",
                data={
                    "timestamp": datetime.utcnow().isoformat(),
                    "components": components_detail,
                },
            ).dict()
        )

    except Exception as e:
        logger.error(f"组件健康检查失败: {str(e)}")
        return (
            jsonify(
                APIResponse(
                    code=500,
                    message=f"组件健康检查失败: {str(e)}",
                    data={"timestamp": datetime.utcnow().isoformat()},
                ).dict()
            ),
            500,
        )


@health_bp.route("/health/metrics", methods=["GET"])
def health_metrics():
    """
    系统健康指标API
    
    Returns:
        JSON: 详细的系统资源监控指标
    """
    try:
        metrics = health_manager.get_system_metrics()
        metrics["uptime"] = health_manager.get_uptime()
        
        return jsonify(APIResponse(code=0, message="健康指标获取成功", data=metrics).dict())

    except Exception as e:
        logger.error(f"获取健康指标失败: {str(e)}")
        return (
            jsonify(
                APIResponse(
                    code=500,
                    message=f"获取健康指标失败: {str(e)}",
                    data={"timestamp": datetime.utcnow().isoformat()},
                ).dict()
            ),
            500,
        )


@health_bp.route("/health/ready", methods=["GET"])
def readiness_probe():
    """
    Kubernetes就绪性探针API
    
    Returns:
        JSON: 服务就绪状态检查结果
    """
    try:
        components = health_manager.check_all_components()
        
        # 定义核心组件
        required_components = ["prometheus", "prediction"]
        
        # 检查核心组件是否就绪
        ready = all(
            components.get(comp, {}).get("healthy", False) 
            for comp in required_components
        )

        if ready:
            return jsonify(
                APIResponse(
                    code=0,
                    message="服务就绪",
                    data={"status": "ready", "timestamp": datetime.utcnow().isoformat()},
                ).dict()
            )
        else:
            return (
                jsonify(
                    APIResponse(
                        code=503,
                        message="服务未就绪",
                        data={
                            "status": "not ready",
                            "timestamp": datetime.utcnow().isoformat(),
                            "components": {name: comp.get("healthy", False) for name, comp in components.items()},
                        },
                    ).dict()
                ),
                503,
            )

    except Exception as e:
        logger.error(f"就绪性检查失败: {str(e)}")
        return (
            jsonify(
                APIResponse(
                    code=500,
                    message=f"就绪性检查失败: {str(e)}",
                    data={"status": "error", "timestamp": datetime.utcnow().isoformat()},
                ).dict()
            ),
            500,
        )


@health_bp.route("/health/live", methods=["GET"])
def liveness_probe():
    """
    Kubernetes存活性探针API
    
    Returns:
        JSON: 服务存活状态
    """
    try:
        return jsonify(
            APIResponse(
                code=0,
                message="服务存活",
                data={
                    "status": "alive",
                    "timestamp": datetime.utcnow().isoformat(),
                    "uptime": health_manager.get_uptime(),
                },
            ).dict()
        )

    except Exception as e:
        logger.error(f"存活性检查失败: {str(e)}")
        return (
            jsonify(
                APIResponse(
                    code=500,
                    message=f"存活性检查失败: {str(e)}",
                    data={"status": "error", "timestamp": datetime.utcnow().isoformat()},
                ).dict()
            ),
            500,
        )


