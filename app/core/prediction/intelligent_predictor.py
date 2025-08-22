#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI-CloudOps智能预测引擎 - 结合大模型的全流程预测分析系统
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from app.common.exceptions import PredictionError
from app.core.prediction.intelligent_report_generator import (
    IntelligentReportGenerator,
    ReportContext,
)
from app.core.prediction.prediction_analyzer import PredictionAnalyzer
from app.core.prediction.unified_predictor import UnifiedPredictor
from app.models import PredictionDataPoint, PredictionGranularity, PredictionType
from app.services.llm import LLMService

logger = logging.getLogger("aiops.core.prediction.intelligent")


class IntelligentPredictor:
    """智能预测引擎 - 在预测全流程中集成大模型分析"""

    def __init__(self, model_manager, feature_extractor):
        # 基础预测组件
        self.unified_predictor = UnifiedPredictor(model_manager, feature_extractor)

        # AI增强组件
        self.analyzer = PredictionAnalyzer()
        self.report_generator = IntelligentReportGenerator()
        self.llm_service = LLMService()

        # 状态管理
        self._initialized = False
        self._analysis_cache = {}  # 分析结果缓存

    async def initialize(self):
        """初始化智能预测引擎"""
        try:
            # 初始化基础预测器
            await self.unified_predictor.initialize()
            self._initialized = True
            logger.info("智能预测引擎初始化完成")

        except Exception as e:
            logger.error(f"智能预测引擎初始化失败: {str(e)}")
            self._initialized = False
            raise PredictionError(f"智能预测引擎初始化失败: {str(e)}")

    async def predict_with_ai_analysis(
        self,
        prediction_type: PredictionType,
        current_value: float,
        historical_data: List[Dict[str, Any]],
        prediction_hours: int,
        granularity: PredictionGranularity,
        consider_pattern: bool = True,
        enable_ai_insights: bool = True,
        report_style: str = "professional",
    ) -> Dict[str, Any]:
        """执行AI增强的完整预测分析"""

        if not self._initialized:
            raise PredictionError("智能预测引擎未初始化")

        start_time = datetime.now()
        analysis_id = f"{prediction_type.value}_{int(start_time.timestamp())}"

        try:
            logger.info(f"开始执行AI增强预测分析 - ID: {analysis_id}")

            # 阶段1: 预测前 - AI分析历史上下文
            context_analysis = None
            if enable_ai_insights:
                logger.info("阶段1: AI分析历史数据上下文")
                context_analysis = await self.analyzer.analyze_historical_context(
                    prediction_type=prediction_type,
                    current_value=current_value,
                    historical_data=historical_data,
                )
                logger.debug(
                    f"上下文分析完成: {context_analysis.get('status', 'unknown')}"
                )

            # 阶段2: 执行预测
            logger.info("阶段2: 执行预测计算")
            predictions = await self.unified_predictor.predict(
                prediction_type=prediction_type,
                current_value=current_value,
                historical_data=historical_data,
                prediction_hours=prediction_hours,
                granularity=granularity,
                consider_pattern=consider_pattern,
            )

            if not predictions:
                raise PredictionError("预测计算失败，未生成预测数据")

            logger.info(f"预测计算完成，生成{len(predictions)}个预测点")

            # 构建基础预测结果
            base_prediction_results = await self._build_base_prediction_results(
                prediction_type=prediction_type,
                current_value=current_value,
                predictions=predictions,
                prediction_hours=prediction_hours,
                granularity=granularity,
            )

            # 阶段3: 预测后 - AI解读和分析
            interpretation = None
            insights = []

            if enable_ai_insights:
                logger.info("阶段3: AI解读预测结果")

                # 执行预测解读
                interpretation = await self.analyzer.interpret_prediction_results(
                    prediction_type=prediction_type,
                    prediction_results=base_prediction_results,
                    analysis_context=context_analysis,
                )
                logger.debug(
                    f"预测解读完成: {interpretation.get('status', 'unknown') if isinstance(interpretation, dict) else 'string_response'}"
                )

                # 生成洞察（包含解读结果）
                insights = await self.analyzer.generate_insights(
                    prediction_type=prediction_type,
                    prediction_results=base_prediction_results,
                    context_analysis=context_analysis or {},
                    interpretation=interpretation or {},
                )
                logger.debug(f"生成洞察: {len(insights)}条")

            # 阶段4: 生成AI报告
            ai_report = None
            if enable_ai_insights and context_analysis and interpretation:
                logger.info("阶段4: 生成AI分析报告")

                # 安全获取字典数据，防止类型错误
                safe_base_results = (
                    base_prediction_results
                    if isinstance(base_prediction_results, dict)
                    else {}
                )
                safe_interpretation = (
                    interpretation if isinstance(interpretation, dict) else {}
                )

                report_context = ReportContext(
                    prediction_type=prediction_type,
                    analysis_context=context_analysis,
                    prediction_results=safe_base_results,
                    interpretation=safe_interpretation,
                    insights=insights,
                    scaling_recommendations=safe_base_results.get(
                        "scaling_recommendations", []
                    ),
                    cost_analysis=safe_base_results.get("cost_analysis"),
                    quantitative_metrics=safe_interpretation.get(
                        "quantitative_metrics", {}
                    ),
                )

                # 并行生成多种报告
                report_tasks = [
                    self.report_generator.generate_comprehensive_report(
                        report_context, report_style
                    ),
                    self.report_generator.generate_executive_summary(report_context),
                    self.report_generator.generate_action_plan(report_context),
                ]

                # 如果有成本分析，也生成成本优化报告
                if base_prediction_results.get("cost_analysis"):
                    report_tasks.append(
                        self.report_generator.generate_cost_optimization_report(
                            report_context
                        )
                    )

                reports = await asyncio.gather(*report_tasks, return_exceptions=True)

                ai_report = {
                    "comprehensive_report": (
                        reports[0] if not isinstance(reports[0], Exception) else None
                    ),
                    "executive_summary": (
                        reports[1] if not isinstance(reports[1], Exception) else None
                    ),
                    "action_plan": (
                        reports[2] if not isinstance(reports[2], Exception) else None
                    ),
                    "cost_optimization": (
                        reports[3]
                        if len(reports) > 3 and not isinstance(reports[3], Exception)
                        else None
                    ),
                }

                logger.info("AI分析报告生成完成")

            # 构建完整响应
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            response = {
                # 基础预测结果
                **base_prediction_results,
                # AI增强结果
                "ai_enhanced": enable_ai_insights,
                "analysis_context": context_analysis,
                "prediction_interpretation": interpretation,
                "ai_insights": insights,
                "ai_reports": ai_report,
                # 元数据
                "analysis_id": analysis_id,
                "processing_time_seconds": processing_time,
                "ai_processing_stages": self._get_processing_stages_summary(
                    enable_ai_insights, context_analysis, interpretation, ai_report
                ),
                "data_quality_assessment": self._assess_overall_data_quality(
                    historical_data, predictions, interpretation
                ),
                "timestamp": end_time,
            }

            # 缓存结果（可选）
            self._cache_analysis_result(analysis_id, response)

            logger.info(
                f"AI增强预测分析完成 - ID: {analysis_id}, 耗时: {processing_time:.2f}秒"
            )
            return response

        except Exception as e:
            logger.error(f"AI增强预测分析失败 - ID: {analysis_id}: {str(e)}")
            # 如果AI增强失败，返回基础预测结果
            return await self._fallback_to_basic_prediction(
                prediction_type,
                current_value,
                historical_data,
                prediction_hours,
                granularity,
                consider_pattern,
                str(e),
            )

    async def predict_multi_dimension_with_correlation(
        self,
        prediction_configs: List[Dict[str, Any]],
        enable_correlation_analysis: bool = True,
        report_style: str = "professional",
    ) -> Dict[str, Any]:
        """多维度预测与关联分析"""

        if not self._initialized:
            raise PredictionError("智能预测引擎未初始化")

        start_time = datetime.now()
        analysis_id = f"multi_dim_{int(start_time.timestamp())}"

        try:
            logger.info(f"开始多维度预测分析 - ID: {analysis_id}")

            # 并行执行所有维度的预测
            prediction_tasks = []
            for config in prediction_configs:
                task = self.predict_with_ai_analysis(
                    prediction_type=PredictionType(config["prediction_type"]),
                    current_value=config["current_value"],
                    historical_data=config.get("historical_data", []),
                    prediction_hours=config.get("prediction_hours", 24),
                    granularity=PredictionGranularity(
                        config.get("granularity", "hour")
                    ),
                    consider_pattern=config.get("consider_pattern", True),
                    enable_ai_insights=False,  # 单独处理AI洞察
                    report_style=report_style,
                )
                prediction_tasks.append(task)

            # 等待所有预测完成
            prediction_results = await asyncio.gather(*prediction_tasks)

            # 组织结果
            results_by_type = {}
            for i, result in enumerate(prediction_results):
                pred_type = PredictionType(prediction_configs[i]["prediction_type"])
                results_by_type[pred_type] = result

            logger.info(f"多维度预测完成，包含{len(results_by_type)}个维度")

            # 执行关联分析
            correlation_analysis = None
            if enable_correlation_analysis and len(results_by_type) >= 2:
                logger.info("执行多维度关联分析")
                correlation_analysis = (
                    await self.analyzer.analyze_multi_dimension_correlation(
                        results_by_type
                    )
                )

            # 生成综合洞察
            multi_dim_insights = await self._generate_multi_dimension_insights(
                results_by_type, correlation_analysis
            )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return {
                "analysis_id": analysis_id,
                "prediction_results": {k.value: v for k, v in results_by_type.items()},
                "correlation_analysis": correlation_analysis,
                "multi_dimension_insights": multi_dim_insights,
                "summary_statistics": self._calculate_multi_dimension_stats(
                    results_by_type
                ),
                "processing_time_seconds": processing_time,
                "analyzed_dimensions": [k.value for k in results_by_type.keys()],
                "timestamp": end_time,
            }

        except Exception as e:
            logger.error(f"多维度预测分析失败 - ID: {analysis_id}: {str(e)}")
            raise PredictionError(f"多维度预测分析失败: {str(e)}")

    async def _build_base_prediction_results(
        self,
        prediction_type: PredictionType,
        current_value: float,
        predictions: List[PredictionDataPoint],
        prediction_hours: int,
        granularity: PredictionGranularity,
    ) -> Dict[str, Any]:
        """构建基础预测结果"""

        # 计算预测摘要
        values = [p.predicted_value for p in predictions]
        prediction_summary = {
            "max_value": max(values) if values else 0,
            "min_value": min(values) if values else 0,
            "avg_value": sum(values) / len(values) if values else 0,
            "trend": self._detect_basic_trend(values),
            "peak_time": self._find_peak_time(predictions),
        }

        # 模拟其他组件的结果（这里应该调用实际的组件）
        # 这些组件在原始代码中已存在，这里简化处理

        return {
            "prediction_type": prediction_type,
            "prediction_hours": prediction_hours,
            "granularity": granularity,
            "current_value": current_value,
            "predicted_data": [p.model_dump() for p in predictions],
            "prediction_summary": prediction_summary,
            "scaling_recommendations": [],  # 应该调用ScalingAdvisor
            "anomaly_predictions": [],  # 应该调用AnomalyDetector
            "cost_analysis": None,  # 应该调用CostAnalyzer
            "resource_utilization": [],
            "pattern_analysis": {},
            "trend_insights": [],
            "model_accuracy": 0.85,
            "timestamp": datetime.now(),
        }

    async def _fallback_to_basic_prediction(
        self,
        prediction_type: PredictionType,
        current_value: float,
        historical_data: List[Dict[str, Any]],
        prediction_hours: int,
        granularity: PredictionGranularity,
        consider_pattern: bool,
        error_message: str,
    ) -> Dict[str, Any]:
        """降级到基础预测"""
        try:
            logger.warning("降级到基础预测模式")

            predictions = await self.unified_predictor.predict(
                prediction_type=prediction_type,
                current_value=current_value,
                historical_data=historical_data,
                prediction_hours=prediction_hours,
                granularity=granularity,
                consider_pattern=consider_pattern,
            )

            base_result = await self._build_base_prediction_results(
                prediction_type,
                current_value,
                predictions,
                prediction_hours,
                granularity,
            )

            # 添加降级标记
            base_result.update(
                {
                    "ai_enhanced": False,
                    "fallback_mode": True,
                    "fallback_reason": error_message,
                    "ai_insights": [
                        "AI增强功能暂时不可用",
                        "当前为基础预测模式",
                        "建议稍后重试获取完整分析",
                    ],
                    "timestamp": datetime.now(),
                }
            )

            return base_result

        except Exception as e:
            logger.error(f"基础预测也失败: {str(e)}")
            raise PredictionError(f"预测完全失败: {str(e)}")

    async def _generate_multi_dimension_insights(
        self,
        results_by_type: Dict[PredictionType, Dict[str, Any]],
        correlation_analysis: Optional[Dict[str, Any]],
    ) -> List[str]:
        """生成多维度综合洞察"""
        try:
            # 构建多维度洞察提示词
            dimensions_summary = {}
            for pred_type, result in results_by_type.items():
                dimensions_summary[pred_type.value] = {
                    "trend": result.get("prediction_summary", {}).get(
                        "trend", "unknown"
                    ),
                    "peak_value": result.get("prediction_summary", {}).get(
                        "max_value", 0
                    ),
                    "insights_count": len(result.get("ai_insights", [])),
                }

            insights_prompt = f"""基于多维度预测分析结果，生成综合洞察：

各维度预测概况：
{json.dumps(dimensions_summary, ensure_ascii=False, indent=2)}

关联分析结果：
{correlation_analysis.get("correlation_analysis", "未提供") if correlation_analysis else "未执行关联分析"}

请生成5-7条跨维度的综合洞察，关注：
1. 不同资源维度之间的相互影响
2. 整体系统的瓶颈和风险点
3. 协调优化的机会
4. 整体架构的建议

每条洞察应该体现多维度思考，避免单一维度的重复。"""

            insights_response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": insights_prompt}],
                temperature=0.4,
                max_tokens=600,
                use_task_model=False,  # 复杂操作：生成综合洞察，使用主模型
            )

            if insights_response:
                # 解析洞察
                insights = []
                for line in insights_response.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # 清理格式
                        line = line.lstrip("- *•").strip()
                        if len(line) > 20:
                            insights.append(line)

                return insights[:7]
            else:
                return self._get_fallback_multi_dimension_insights(results_by_type)

        except Exception as e:
            logger.error(f"生成多维度洞察失败: {str(e)}")
            return self._get_fallback_multi_dimension_insights(results_by_type)

    def _detect_basic_trend(self, values: List[float]) -> str:
        """检测基础趋势"""
        if len(values) < 2:
            return "stable"

        first_half = values[: len(values) // 2]
        second_half = values[len(values) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        change_percent = (
            (second_avg - first_avg) / first_avg * 100 if first_avg > 0 else 0
        )

        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"

    def _find_peak_time(self, predictions: List[PredictionDataPoint]) -> Optional[str]:
        """找出峰值时间"""
        if not predictions:
            return None

        max_pred = max(predictions, key=lambda p: p.predicted_value)
        return max_pred.timestamp.isoformat()

    def _get_processing_stages_summary(
        self,
        enable_ai: bool,
        context_analysis: Optional[Dict],
        interpretation: Optional[Dict],
        ai_report: Optional[Dict],
    ) -> Dict[str, str]:
        """获取处理阶段摘要"""
        stages = {
            "prediction": "completed",
            "context_analysis": (
                "skipped"
                if not enable_ai
                else ("completed" if context_analysis else "failed")
            ),
            "interpretation": (
                "skipped"
                if not enable_ai
                else ("completed" if interpretation else "failed")
            ),
            "report_generation": (
                "skipped" if not enable_ai else ("completed" if ai_report else "failed")
            ),
        }
        return stages

    def _assess_overall_data_quality(
        self,
        historical_data: List[Dict],
        predictions: List[PredictionDataPoint],
        interpretation: Optional[Dict],
    ) -> Dict[str, Union[float, str]]:
        """评估整体数据质量"""

        # 历史数据质量
        historical_quality = 0.8
        if historical_data:
            if len(historical_data) >= 48:  # 足够的历史数据
                historical_quality += 0.1
            elif len(historical_data) < 12:  # 数据不足
                historical_quality -= 0.2

        # 预测质量
        prediction_quality = 0.8
        if predictions:
            if len(predictions) >= 24:
                prediction_quality += 0.1
            # 检查是否有置信度信息
            confidences = [
                p.confidence_level
                for p in predictions
                if p.confidence_level is not None
            ]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence > 0.8:
                    prediction_quality += 0.1

        # AI分析质量
        ai_quality = 0.5  # 基础值
        if interpretation and interpretation.get("status") == "success":
            ai_quality = 0.8

        overall_score = (
            historical_quality * 0.3 + prediction_quality * 0.5 + ai_quality * 0.2
        )

        return {
            "overall_score": min(1.0, max(0.0, overall_score)),
            "historical_data_quality": historical_quality,
            "prediction_quality": prediction_quality,
            "ai_analysis_quality": ai_quality,
            "assessment": (
                "excellent"
                if overall_score > 0.9
                else (
                    "good"
                    if overall_score > 0.7
                    else "fair" if overall_score > 0.5 else "poor"
                )
            ),
        }

    def _cache_analysis_result(self, analysis_id: str, result: Dict[str, Any]) -> None:
        """缓存分析结果"""
        try:
            # 简单的内存缓存，实际应用可以使用Redis等
            cache_data = {
                "timestamp": datetime.now(),
                "result_summary": {
                    "prediction_type": result.get("prediction_type"),
                    "ai_enhanced": result.get("ai_enhanced", False),
                    "processing_time": result.get("processing_time_seconds", 0),
                },
            }
            self._analysis_cache[analysis_id] = cache_data

            # 限制缓存大小
            if len(self._analysis_cache) > 100:
                oldest_key = min(
                    self._analysis_cache.keys(),
                    key=lambda k: self._analysis_cache[k]["timestamp"],
                )
                del self._analysis_cache[oldest_key]

        except Exception as e:
            logger.warning(f"缓存分析结果失败: {str(e)}")

    def _calculate_multi_dimension_stats(
        self, results_by_type: Dict[PredictionType, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算多维度统计信息"""

        stats = {
            "total_dimensions": len(results_by_type),
            "successful_predictions": 0,
            "total_prediction_points": 0,
            "avg_confidence": 0,
            "trend_distribution": {"increasing": 0, "decreasing": 0, "stable": 0},
        }

        total_confidence = 0
        confidence_count = 0

        for pred_type, result in results_by_type.items():
            if result.get("predicted_data"):
                stats["successful_predictions"] += 1
                stats["total_prediction_points"] += len(result["predicted_data"])

                # 统计置信度
                for point in result["predicted_data"]:
                    if point.get("confidence_level"):
                        total_confidence += point["confidence_level"]
                        confidence_count += 1

                # 统计趋势
                trend = result.get("prediction_summary", {}).get("trend", "stable")
                if trend in stats["trend_distribution"]:
                    stats["trend_distribution"][trend] += 1

        if confidence_count > 0:
            stats["avg_confidence"] = total_confidence / confidence_count

        return stats

    def _get_fallback_multi_dimension_insights(
        self, results_by_type: Dict[PredictionType, Dict[str, Any]]
    ) -> List[str]:
        """获取多维度降级洞察"""
        fallback_insights = [
            f"完成了{len(results_by_type)}个维度的预测分析",
            "建议综合考虑各维度预测结果制定策略",
            "关注不同资源间的平衡和协调",
            "监控各维度实际值与预测偏差",
        ]

        # 基于趋势添加简单洞察
        trends = []
        for result in results_by_type.values():
            trend = result.get("prediction_summary", {}).get("trend", "stable")
            trends.append(trend)

        if trends.count("increasing") > len(trends) / 2:
            fallback_insights.append("多个维度呈上升趋势，建议提前准备扩容")
        elif trends.count("decreasing") > len(trends) / 2:
            fallback_insights.append("多个维度呈下降趋势，可考虑成本优化")

        return fallback_insights

    async def is_healthy(self) -> bool:
        """健康检查"""
        try:
            # 检查基础预测器
            if not await self.unified_predictor.is_healthy():
                return False

            # 检查LLM服务
            if not await self.llm_service.is_healthy():
                logger.warning("LLM服务不健康，但智能预测引擎可降级运行")

            return self._initialized

        except Exception as e:
            logger.error(f"智能预测引擎健康检查失败: {str(e)}")
            return False
