#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 服务模块初始化文件，集成Prometheus、Kubernetes等外部服务
"""

from .prometheus import PrometheusService
from .llm import LLMService
from .notification import NotificationService

try:
    from .kubernetes import KubernetesService
    __all__ = ["PrometheusService", "KubernetesService", "LLMService", "NotificationService"]
except ImportError:
    KubernetesService = None
    __all__ = ["PrometheusService", "LLMService", "NotificationService"]