#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Core层预测模块初始化文件
"""

from .anomaly_detector import AnomalyDetector
from .cost_analyzer import CostAnalyzer
from .feature_extractor import FeatureExtractor

# AI增强预测组件
from .intelligent_predictor import IntelligentPredictor
from .intelligent_report_generator import IntelligentReportGenerator, ReportContext
from .model_manager import ModelManager
from .prediction_analyzer import PredictionAnalyzer
from .prompt_templates import (
    PredictionPromptBuilder,
    PromptTemplateManager,
    prompt_builder,
    template_manager,
)
from .scaling_advisor import ScalingAdvisor
from .unified_predictor import UnifiedPredictor

__all__ = [
    # 基础预测组件
    "UnifiedPredictor",
    "FeatureExtractor",
    "AnomalyDetector",
    "ScalingAdvisor",
    "CostAnalyzer",
    "ModelManager",
    # AI增强预测组件
    "IntelligentPredictor",
    "PredictionAnalyzer",
    "IntelligentReportGenerator",
    "ReportContext",
    # 提示词模板组件
    "PromptTemplateManager",
    "PredictionPromptBuilder",
    "template_manager",
    "prompt_builder",
]
