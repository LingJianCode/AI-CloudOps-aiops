#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查API模块，提供AI-CloudOps系统的服务健康监控和状态检查功能。支持系统综合健康检查、组件详细健康检查和系统指标监控。

本模块提供了一套完整的健康检查API接口，用于监控和验证AI-CloudOps系统各个组件的运行状态。
通过这些接口，管理员和监控系统可以实时了解系统的健康状况，及时发现和解决问题。

主要功能：
1. 系统综合健康检查 - 整体状态评估和组件状态汇总
2. 组件详细健康检查 - 各个服务组件的深度状态检查
3. 系统指标监控 - CPU、内存、磁盘、网络等资源使用情况
4. 就绪性探针 - Kubernetes就绪性检查，确保服务可以接收流量
5. 存活性探针 - Kubernetes存活性检查，确保服务正常运行
6. 资源使用监控 - 实时系统资源和进程状态监控

支持的组件检查：
- Prometheus: 监控数据收集服务
- Kubernetes: 容器编排平台服务
- LLM: 大语言模型服务
- Notification: 通知和告警服务
- Prediction: 负载预测和分析服务

API设计特性：
- RESTful风格的HTTP接口
- 统一的JSON响应格式
- 详细的错误信息和日志记录
- 灵活的健康状态级别
- 符合Kubernetes探针规范

Author: AI-CloudOps Team
Date: 2024
"""

from flask import Blueprint, jsonify
from datetime import datetime
import time
import psutil
import logging
from app.services.prometheus import PrometheusService
from app.services.kubernetes import KubernetesService
from app.services.llm import LLMService
from app.services.notification import NotificationService
from app.core.prediction.predictor import PredictionService
from app.models.response_models import APIResponse

logger = logging.getLogger("aiops.health")

# 创建健康检查蓝图，用于组织相关的路由和视图函数
health_bp = Blueprint('health', __name__)

# 应用启动时间戳，用于计算系统运行时间（uptime）
start_time = time.time()

@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    系统综合健康检查API - 主要的健康状态检查接口
    
    这是系统的主要健康检查接口，提供整体系统健康状态的综合评估。
    它会检查所有核心组件的状态，收集系统资源信息，并返回整体健康评估结果。
    
    返回内容：
    - 整体健康状态（healthy/unhealthy）
    - 系统运行时间（uptime）
    - 各组件健康状态汇总
    - 系统资源使用情况
    - 版本信息和时间戳
    
    使用场景：
    - 系统状态监控面板
    - 自动化健康检查脚本
    - 负载均衡器健康检查
    - 运维监控和告警系统
    
    Returns:
        JSON: 包含详细健康状态信息的响应，格式如下：
        {
            "code": 0,
            "message": "健康检查完成",
            "data": {
                "status": "healthy|unhealthy",
                "timestamp": "ISO时间戳",
                "uptime": "运行时间(秒)",
                "version": "版本号",
                "components": {"组件名": true/false},
                "system": {"cpu_usage_percent": 10.5, ...}
            }
        }
    """
    try:
        # 获取当前UTC时间，确保时间戳的一致性
        current_time = datetime.utcnow()
        # 计算系统运行时间（从应用启动到现在的秒数）
        uptime = time.time() - start_time
        
        # 检查各组件健康状态，这是核心的健康评估步骤
        components_status = check_components_health()
        
        # 获取系统资源状态，包括CPU、内存、磁盘使用情况
        system_status = get_system_status()
        
        # 判断整体健康状态 - 只有当所有组件都健康时，系统才被认为是健康的
        is_healthy = all(components_status.values())
        
        # 构建健康检查响应数据
        health_data = {
            "status": "healthy" if is_healthy else "unhealthy",  # 整体状态
            "timestamp": current_time.isoformat(),  # ISO格式的时间戳
            "uptime": round(uptime, 2),  # 运行时间，保留2位小数
            "version": "1.0.0",  # 系统版本号
            "components": components_status,  # 各组件详细状态
            "system": system_status  # 系统资源状态
        }
        
        # 返回标准化的API响应格式
        return jsonify(APIResponse(
            code=0,
            message="健康检查完成",
            data=health_data
        ).dict())
        
    except Exception as e:
        # 异常处理：记录错误日志并返回500错误响应
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"健康检查失败: {str(e)}",
            data={"timestamp": datetime.utcnow().isoformat()}
        ).dict()), 500

@health_bp.route('/health/components', methods=['GET'])
def components_health():
    """
    组件详细健康检查API - 深度组件状态分析接口
    
    提供所有系统组件的详细健康状态信息，包括组件的配置信息、
    连接状态、错误信息等深度诊断数据。这个接口比基础健康检查
    提供更多的调试和故障排查信息。
    
    检查的组件包括：
    1. Prometheus服务 - 监控数据收集和存储
    2. Kubernetes服务 - 容器编排和集群管理
    3. LLM服务 - 大语言模型和AI推理
    4. 通知服务 - 告警和消息推送
    5. 预测服务 - 负载预测和容量规划
    
    返回内容：
    - 每个组件的详细健康状态
    - 组件配置信息（URL、模型等）
    - 错误信息和诊断数据
    - 组件特定的状态指标
    
    使用场景：
    - 系统故障诊断和排查
    - 组件配置验证
    - 运维监控面板详情
    - 自动化测试和验证
    
    Returns:
        JSON: 包含所有组件详细状态的响应，格式如下：
        {
            "code": 0,
            "message": "组件健康检查完成",
            "data": {
                "timestamp": "ISO时间戳",
                "components": {
                    "prometheus": {"healthy": true, "url": "...", ...},
                    "kubernetes": {"healthy": false, "error": "...", ...},
                    ...
                }
            }
        }
    """
    try:
        # 初始化组件详细信息字典
        components_detail = {}
        
        # Prometheus服务健康检查 - 监控数据收集服务
        prometheus_service = PrometheusService()
        prometheus_healthy = prometheus_service.is_healthy()
        components_detail["prometheus"] = {
            "healthy": prometheus_healthy,  # 健康状态
            "url": prometheus_service.base_url,  # 服务URL
            "timeout": prometheus_service.timeout  # 超时配置
        }
        
        # Kubernetes服务健康检查 - 容器编排平台服务
        try:
            k8s_service = KubernetesService()
            k8s_healthy = k8s_service.is_healthy()
            components_detail["kubernetes"] = {
                "healthy": k8s_healthy,  # 健康状态
                # 集群内部部署标识，判断是否在K8s集群内运行
                "in_cluster": k8s_service.k8s_config.in_cluster if hasattr(k8s_service, 'k8s_config') else False
            }
        except Exception as e:
            # Kubernetes服务异常时记录错误信息
            components_detail["kubernetes"] = {
                "healthy": False,
                "error": str(e)  # 详细错误信息
            }
        
        # LLM服务健康检查 - 大语言模型服务
        try:
            llm_service = LLMService()
            llm_healthy = llm_service.is_healthy()
            components_detail["llm"] = {
                "healthy": llm_healthy,  # 健康状态
                "model": llm_service.model,  # 当前使用的模型名称
                "base_url": llm_service.client.base_url  # 模型服务的基础URL
            }
        except Exception as e:
            # LLM服务异常时记录错误信息
            components_detail["llm"] = {
                "healthy": False,
                "error": str(e)  # 详细错误信息
            }
        
        # 通知服务健康检查 - 告警和消息推送服务
        try:
            notification_service = NotificationService()
            notification_healthy = notification_service.is_healthy()
            components_detail["notification"] = {
                "healthy": notification_healthy,  # 健康状态
                "enabled": notification_service.enabled  # 服务是否启用
            }
        except Exception as e:
            # 通知服务异常时记录错误信息
            components_detail["notification"] = {
                "healthy": False,
                "error": str(e)  # 详细错误信息
            }
        
        # 预测服务健康检查 - 负载预测和容量规划服务
        try:
            prediction_service = PredictionService()
            prediction_healthy = prediction_service.is_healthy()
            components_detail["prediction"] = {
                "healthy": prediction_healthy,  # 健康状态
                "model_loaded": prediction_service.model_loaded,  # ML模型加载状态
                "scaler_loaded": prediction_service.scaler_loaded  # 特征缩放器加载状态
            }
        except Exception as e:
            # 预测服务异常时记录错误信息
            components_detail["prediction"] = {
                "healthy": False,
                "error": str(e)  # 详细错误信息
            }
        
        # 返回组件详细健康检查结果
        return jsonify(APIResponse(
            code=0,
            message="组件健康检查完成",
            data={
                "timestamp": datetime.utcnow().isoformat(),  # 检查时间戳
                "components": components_detail  # 所有组件的详细状态
            }
        ).dict())
        
    except Exception as e:
        # 异常处理：记录错误日志并返回500错误响应
        logger.error(f"组件健康检查失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"组件健康检查失败: {str(e)}",
            data={"timestamp": datetime.utcnow().isoformat()}
        ).dict()), 500

@health_bp.route('/health/metrics', methods=['GET'])
def health_metrics():
    """
    系统健康指标API - 详细的系统资源监控接口
    
    提供详细的系统资源使用情况和性能指标，包括CPU、内存、磁盘、
    网络和进程级别的统计数据。这些指标用于系统性能监控、
    容量规划和性能优化。
    
    监控指标类别：
    1. 系统级指标：
       - CPU使用率（百分比）
       - 内存使用情况（总量、可用量、使用率）
       - 磁盘空间使用情况（总量、已用、可用）
    
    2. 网络指标：
       - 网络流量统计（发送/接收字节数）
       - 数据包统计（发送/接收包数）
    
    3. 进程指标：
       - 进程内存使用（RSS、VMS）
       - 进程CPU使用率
       - 线程数量和创建时间
    
    4. 运行时指标：
       - 系统运行时间（uptime）
    
    使用场景：
    - 性能监控仪表板
    - 资源使用趋势分析
    - 容量规划和预警
    - 系统优化决策支持
    
    Returns:
        JSON: 包含详细系统指标的响应，格式如下：
        {
            "code": 0,
            "message": "健康指标获取成功",
            "data": {
                "timestamp": "ISO时间戳",
                "system": {"cpu_percent": 15.2, "memory_percent": 68.5, ...},
                "network": {"bytes_sent": 1024000, ...},
                "process": {"memory_rss": 256000, ...},
                "uptime": 3600.0
            }
        }
    """
    try:
        # 系统级指标收集 - 获取CPU、内存、磁盘使用情况
        cpu_percent = psutil.cpu_percent(interval=1)  # CPU使用率，1秒采样间隔
        memory = psutil.virtual_memory()  # 内存使用情况
        disk = psutil.disk_usage('/')  # 根目录磁盘使用情况
        
        # 网络指标收集 - 获取网络I/O统计信息
        network = psutil.net_io_counters()
        
        # 当前进程指标收集 - 获取AI-CloudOps进程的资源使用情况
        process = psutil.Process()  # 当前进程
        process_memory = process.memory_info()  # 进程内存信息
        
        # 构建完整的指标数据结构
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),  # 指标收集时间戳
            # 系统级指标
            "system": {
                "cpu_percent": cpu_percent,  # CPU使用率百分比
                "memory_percent": memory.percent,  # 内存使用率百分比
                "memory_available": memory.available,  # 可用内存字节数
                "memory_total": memory.total,  # 总内存字节数
                "disk_percent": (disk.used / disk.total) * 100,  # 磁盘使用率百分比
                "disk_free": disk.free,  # 可用磁盘空间字节数
                "disk_total": disk.total  # 总磁盘空间字节数
            },
            # 网络I/O指标
            "network": {
                "bytes_sent": network.bytes_sent,  # 总发送字节数
                "bytes_recv": network.bytes_recv,  # 总接收字节数
                "packets_sent": network.packets_sent,  # 总发送包数
                "packets_recv": network.packets_recv  # 总接收包数
            },
            # 进程级指标
            "process": {
                "memory_rss": process_memory.rss,  # 常驻内存大小（字节）
                "memory_vms": process_memory.vms,  # 虚拟内存大小（字节）
                "cpu_percent": process.cpu_percent(),  # 进程CPU使用率
                "num_threads": process.num_threads(),  # 线程数量
                "create_time": process.create_time()  # 进程创建时间戳
            },
            # 运行时指标
            "uptime": time.time() - start_time  # 系统运行时间（秒）
        }
        
        # 返回成功的指标数据响应
        return jsonify(APIResponse(
            code=0,
            message="健康指标获取成功",
            data=metrics
        ).dict())
        
    except Exception as e:
        # 异常处理：记录错误日志并返回500错误响应
        logger.error(f"获取健康指标失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"获取健康指标失败: {str(e)}",
            data={"timestamp": datetime.utcnow().isoformat()}
        ).dict()), 500

@health_bp.route('/health/ready', methods=['GET'])
def readiness_probe():
    """
    Kubernetes就绪性探针API - 服务就绪状态检查接口
    
    这是符合Kubernetes规范的就绪性探针接口，用于判断服务是否
    已经准备好接收流量。只有当核心组件都就绪时，服务才被认为
    可以处理用户请求。
    
    就绪性检查标准：
    1. 核心组件必须正常运行
       - Prometheus服务：监控数据收集必须可用
       - 预测服务：机器学习模型必须加载完成
    
    2. 可选组件的故障不影响就绪状态
       - LLM服务、通知服务等可以降级运行
    
    就绪性与存活性的区别：
    - 就绪性：服务是否可以处理请求（流量路由决策）
    - 存活性：服务进程是否正在运行（重启决策）
    
    使用场景：
    - Kubernetes Service的流量路由
    - 负载均衡器的健康检查
    - 服务发现和注册
    - 滚动更新期间的流量控制
    
    Returns:
        JSON: 就绪状态响应，HTTP状态码表示就绪程度：
        - 200: 服务就绪，可以接收流量
        - 503: 服务未就绪，暂时不能接收流量
        - 500: 检查过程出错
        
        响应格式：
        {
            "code": 0|503|500,
            "message": "服务就绪|服务未就绪|就绪性检查失败",
            "data": {
                "status": "ready|not ready|error",
                "timestamp": "ISO时间戳",
                "components": {"组件状态"} (仅在未就绪时包含)
            }
        }
    """
    try:
        # 检查所有组件的健康状态
        components_status = check_components_health()
        
        # 定义核心组件列表 - 这些组件必须健康才能认为服务就绪
        # Prometheus: 监控数据收集，是所有分析功能的基础
        # Prediction: 负载预测服务，核心AI功能之一
        required_components = ["prometheus", "prediction"]
        
        # 检查所有必需组件是否都处于健康状态
        ready = all(components_status.get(comp, False) for comp in required_components)
        
        if ready:
            # 服务就绪：所有核心组件都健康
            return jsonify(APIResponse(
                code=0,
                message="服务就绪",
                data={
                    "status": "ready",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ).dict())
        else:
            # 服务未就绪：至少有一个核心组件不健康
            return jsonify(APIResponse(
                code=503,  # HTTP 503 Service Unavailable
                message="服务未就绪",
                data={
                    "status": "not ready",
                    "timestamp": datetime.utcnow().isoformat(),
                    "components": components_status  # 提供详细的组件状态信息
                }
            ).dict()), 503
            
    except Exception as e:
        # 异常处理：就绪性检查过程本身出现错误
        logger.error(f"就绪性检查失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"就绪性检查失败: {str(e)}",
            data={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
        ).dict()), 500

@health_bp.route('/health/live', methods=['GET'])
def liveness_probe():
    """存活性探针"""
    try:
        # 简单的存活性检查
        return jsonify(APIResponse(
            code=0,
            message="服务存活",
            data={
                "status": "alive",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime": time.time() - start_time
            }
        ).dict())
        
    except Exception as e:
        logger.error(f"存活性检查失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"存活性检查失败: {str(e)}",
            data={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
        ).dict()), 500

def check_components_health():
    """检查各组件健康状态"""
    components_status = {}
    
    # Prometheus
    try:
        prometheus_service = PrometheusService()
        components_status["prometheus"] = prometheus_service.is_healthy()
    except Exception:
        components_status["prometheus"] = False
    
    # Kubernetes
    try:
        k8s_service = KubernetesService()
        components_status["kubernetes"] = k8s_service.is_healthy()
    except Exception:
        components_status["kubernetes"] = False
    
    # LLM
    try:
        llm_service = LLMService()
        components_status["llm"] = llm_service.is_healthy()
    except Exception:
        components_status["llm"] = False
    
    # 通知服务
    try:
        notification_service = NotificationService()
        components_status["notification"] = notification_service.is_healthy()
    except Exception:
        components_status["notification"] = False
    
    # 预测服务
    try:
        prediction_service = PredictionService()
        components_status["prediction"] = prediction_service.is_healthy()
    except Exception:
        components_status["prediction"] = False
    
    return components_status

def get_system_status():
    """获取系统资源状态"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_status = {
        "cpu_usage_percent": cpu_percent,
        "memory_usage_percent": memory.percent,
        "disk_usage_percent": (disk.used / disk.total) * 100,
        "memory_available_mb": memory.available / (1024 * 1024),
        "disk_free_gb": disk.free / (1024 * 1024 * 1024)
    }
    
    return system_status