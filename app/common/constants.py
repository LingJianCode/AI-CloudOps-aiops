#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 常量定义模块
"""

from typing import Any, Dict


class AppConstants:
    """应用级常量"""

    # 应用信息
    APP_NAME = "AIOps Platform"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "智能云原生运维平台"

    # API版本
    API_VERSION_V1 = "/api/v1"

    # 默认超时设置（秒）
    DEFAULT_REQUEST_TIMEOUT = 30
    DEFAULT_WARMUP_TIMEOUT = 60
    DEFAULT_RETRY_DELAY = 2

    # 分页设置
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100


class ServiceConstants:
    """服务相关常量"""

    # 默认服务超时设置（秒）
    DEFAULT_SERVICE_TIMEOUT = 60
    DEFAULT_WARMUP_TIMEOUT = 60
    DEFAULT_RETRY_DELAY = 2

    # LLM服务常量
    LLM_MAX_RETRIES = 3
    LLM_TEMPERATURE_MIN = 0.0
    LLM_TEMPERATURE_MAX = 2.0
    LLM_HEALTH_CHECK_TOKENS = 5  # 健康检查使用的最小令牌数
    LLM_DEFAULT_TEMPERATURE = 0.7  # 默认温度值

    # 服务状态
    STATUS_HEALTHY = "healthy"
    STATUS_UNHEALTHY = "unhealthy"
    STATUS_DEGRADED = "degraded"
    STATUS_READY = "ready"
    STATUS_NOT_READY = "not_ready"
    STATUS_ALIVE = "alive"

    # 预测服务常量
    PREDICTION_MIN_QPS = 0.1
    PREDICTION_MAX_QPS = 10000.0
    PREDICTION_TIMEOUT = 120  # 2分钟

    # RCA服务
    RCA_MIN_METRICS = 1
    RCA_MAX_METRICS = 50
    RCA_DEFAULT_SEVERITY_THRESHOLD = 0.7
    RCA_TIMEOUT = 90

    # 自动修复服务
    AUTOFIX_K8S_TIMEOUT = 30
    AUTOFIX_LOGS_TIMEOUT = 15
    AUTOFIX_WORKFLOW_TIMEOUT = 300  # 5分钟
    AUTOFIX_ANALYSIS_TIMEOUT = 120  # 2分钟
    AUTOFIX_MAX_PODS_FOR_LOGS = 5
    AUTOFIX_MAX_NAME_LENGTH = 63  # K8s资源名称最大长度
    AUTOFIX_MIN_TIMEOUT = 60
    AUTOFIX_MAX_TIMEOUT = 1800  # 30分钟
    AUTOFIX_DEFAULT_TIMEOUT = 300  # 5分钟

    # 智能助手服务
    ASSISTANT_MAX_CONTEXT_DOCS = 10
    ASSISTANT_DEFAULT_CONTEXT_DOCS = 3
    ASSISTANT_TIMEOUT = 360  # 6分钟
    ASSISTANT_SESSION_TIMEOUT = 3600  # 1小时

    # 预测服务常量  
    LOW_QPS_THRESHOLD = 5.0
    PREDICTION_VARIATION_FACTOR = 0.1  # 10%波动

    # QPS置信度阈值
    QPS_CONFIDENCE_THRESHOLDS = {
        "low": 100,
        "medium": 500,
        "high": 1000,
        "very_high": 2000,
    }

    # 时间模式常量
    HOUR_FACTORS = {
        0: 0.3,
        1: 0.2,
        2: 0.15,
        3: 0.1,
        4: 0.1,
        5: 0.2,
        6: 0.4,
        7: 0.6,
        8: 0.8,
        9: 0.9,
        10: 1.0,
        11: 0.95,
        12: 0.9,
        13: 0.95,
        14: 1.0,
        15: 1.0,
        16: 0.95,
        17: 0.9,
        18: 0.8,
        19: 0.7,
        20: 0.6,
        21: 0.5,
        22: 0.4,
        23: 0.3,
    }

    DAY_FACTORS = {
        0: 0.95,  # 周一
        1: 1.0,  # 周二
        2: 1.05,  # 周三
        3: 1.05,  # 周四
        4: 0.95,  # 周五
        5: 0.7,  # 周六
        6: 0.6,  # 周日
    }


class ErrorMessages:
    """错误消息常量"""

    # 通用错误
    INTERNAL_SERVER_ERROR = "服务器内部错误"
    SERVICE_UNAVAILABLE = "服务暂时不可用"
    INVALID_PARAMETERS = "请求参数无效"
    TIMEOUT_ERROR = "请求超时"

    # 服务特定错误
    PREDICTION_SERVICE_ERROR = "预测服务错误"
    RCA_SERVICE_ERROR = "根因分析服务错误"
    AUTOFIX_SERVICE_ERROR = "自动修复服务错误"
    ASSISTANT_SERVICE_ERROR = "智能助手服务错误"

    # 验证错误
    INVALID_QPS = "QPS值必须大于0"
    INVALID_TIME_RANGE = "时间范围无效"
    INVALID_DEPLOYMENT_NAME = "部署名称格式无效"
    INVALID_NAMESPACE = "命名空间名称格式无效"


class HttpStatusCodes:
    """HTTP状态码常量"""

    # 成功响应
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204

    # 客户端错误
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422

    # 服务器错误
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class Messages:
    """系统消息常量"""

    # 错误消息
    ERROR_MESSAGES = {
        "invalid_input": "输入参数无效",
        "service_unavailable": "服务暂时不可用",
        "timeout": "请求超时",
        "not_found": "请求的资源未找到",
        "unauthorized": "未授权访问",
        "rate_limited": "请求频率超限",
        "internal_error": "内部服务错误",
    }

    # 成功消息
    SUCCESS_MESSAGES = {
        "operation_completed": "操作成功完成",
        "data_updated": "数据更新成功",
        "analysis_finished": "分析完成",
        "model_trained": "模型训练完成",
    }


class ApiEndpoints:
    """API端点常量"""

    # 根端点
    ROOT = "/"
    DOCS = "/docs"
    REDOC = "/redoc"
    OPENAPI = "/openapi.json"

    # 健康检查端点
    HEALTH = f"{AppConstants.API_VERSION_V1}/health"
    HEALTH_COMPONENTS = f"{HEALTH}/components"
    HEALTH_METRICS = f"{HEALTH}/metrics"
    HEALTH_READY = f"{HEALTH}/ready"
    HEALTH_LIVE = f"{HEALTH}/live"
    HEALTH_STARTUP = f"{HEALTH}/startup"
    HEALTH_DEPENDENCIES = f"{HEALTH}/dependencies"
    HEALTH_DETAIL = f"{HEALTH}/detail"

    # 预测端点
    PREDICT = f"{AppConstants.API_VERSION_V1}/predict"
    PREDICT_TREND = f"{PREDICT}/trend"
    PREDICT_HEALTH = f"{PREDICT}/health"
    PREDICT_READY = f"{PREDICT}/ready"
    PREDICT_INFO = f"{PREDICT}/info"
    PREDICT_MODELS = f"{PREDICT}/models"

    # RCA端点
    RCA = f"{AppConstants.API_VERSION_V1}/rca"
    RCA_METRICS = f"{RCA}/metrics"
    RCA_CONFIG = f"{RCA}/config"
    RCA_HEALTH = f"{RCA}/health"
    RCA_READY = f"{RCA}/ready"
    RCA_INFO = f"{RCA}/info"

    # 自动修复端点
    AUTOFIX = f"{AppConstants.API_VERSION_V1}/autofix"
    AUTOFIX_DIAGNOSE = f"{AUTOFIX}/diagnose"
    AUTOFIX_HEALTH = f"{AUTOFIX}/health"
    AUTOFIX_CONFIG = f"{AUTOFIX}/config"
    AUTOFIX_READY = f"{AUTOFIX}/ready"
    AUTOFIX_INFO = f"{AUTOFIX}/info"

    # 智能助手端点
    ASSISTANT = f"{AppConstants.API_VERSION_V1}/assistant"
    ASSISTANT_QUERY = f"{ASSISTANT}/query"
    ASSISTANT_SESSION = f"{ASSISTANT}/session"
    ASSISTANT_REFRESH = f"{ASSISTANT}/refresh"
    ASSISTANT_HEALTH = f"{ASSISTANT}/health"
    ASSISTANT_READY = f"{ASSISTANT}/ready"
    ASSISTANT_INFO = f"{ASSISTANT}/info"


def get_api_info() -> Dict[str, Any]:
    """
    获取API信息

    Returns:
        包含所有API端点信息的字典
    """
    return {
        "service": AppConstants.APP_NAME,
        "version": AppConstants.APP_VERSION,
        "status": "running",
        "description": AppConstants.APP_DESCRIPTION,
        "endpoints": {
            "health": {
                "main": ApiEndpoints.HEALTH,
                "components": ApiEndpoints.HEALTH_COMPONENTS,
                "metrics": ApiEndpoints.HEALTH_METRICS,
                "ready": ApiEndpoints.HEALTH_READY,
                "live": ApiEndpoints.HEALTH_LIVE,
                "startup": ApiEndpoints.HEALTH_STARTUP,
                "dependencies": ApiEndpoints.HEALTH_DEPENDENCIES,
                "detail": ApiEndpoints.HEALTH_DETAIL,
            },
            "prediction": {
                "predict": ApiEndpoints.PREDICT,
                "trend": ApiEndpoints.PREDICT_TREND,
                "health": ApiEndpoints.PREDICT_HEALTH,
                "ready": ApiEndpoints.PREDICT_READY,
                "info": ApiEndpoints.PREDICT_INFO,
                "models": ApiEndpoints.PREDICT_MODELS,
            },
            "rca": {
                "analyze": ApiEndpoints.RCA,
                "metrics": ApiEndpoints.RCA_METRICS,
                "config": ApiEndpoints.RCA_CONFIG,
                "health": ApiEndpoints.RCA_HEALTH,
                "ready": ApiEndpoints.RCA_READY,
                "info": ApiEndpoints.RCA_INFO,
            },
            "autofix": {
                "fix": ApiEndpoints.AUTOFIX,
                "diagnose": ApiEndpoints.AUTOFIX_DIAGNOSE,
                "health": ApiEndpoints.AUTOFIX_HEALTH,
                "ready": ApiEndpoints.AUTOFIX_READY,
                "info": ApiEndpoints.AUTOFIX_INFO,
            },
            "assistant": {
                "query": ApiEndpoints.ASSISTANT_QUERY,
                "session": ApiEndpoints.ASSISTANT_SESSION,
                "refresh": ApiEndpoints.ASSISTANT_REFRESH,
                "health": ApiEndpoints.ASSISTANT_HEALTH,
                "ready": ApiEndpoints.ASSISTANT_READY,
                "info": ApiEndpoints.ASSISTANT_INFO,
            },
        },
        "features": ["智能负载预测", "根因分析", "自动修复", "智能问答", "健康检查"],
    }
