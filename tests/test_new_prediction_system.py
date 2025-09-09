#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 新预测系统验证脚本 - 完整测试所有预测功能
"""

import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config.settings import config  # noqa: E402
from app.models.predict_models import ResourceConstraints  # noqa: E402
from app.services.prediction_service import PredictionService  # noqa: E402

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_prediction_system")


class PredictionSystemValidator:
    """预测系统验证器"""

    def __init__(self):
        self.service = PredictionService()
        self.results = {}

    async def initialize_service(self):
        """初始化预测服务"""
        logger.info("🚀 初始化预测服务...")
        try:
            await self.service.initialize()
            logger.info("✅ 预测服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"❌ 预测服务初始化失败: {str(e)}")
            return False

    async def test_service_health(self):
        """测试服务健康状态"""
        logger.info("🏥 测试服务健康状态...")

        try:
            # 健康检查
            is_healthy = await self.service.health_check()
            logger.info(f"健康状态: {'✅ 健康' if is_healthy else '❌ 不健康'}")

            # 获取详细健康信息
            health_info = await self.service.get_service_health_info()
            logger.info(f"服务状态: {health_info.get('service_status')}")
            logger.info(f"模型状态: {health_info.get('model_status')}")
            logger.info(
                f"支持的预测类型: {health_info.get('supported_prediction_types', [])}"
            )

            self.results["health_check"] = {
                "healthy": is_healthy,
                "service_status": health_info.get("service_status"),
                "model_status": health_info.get("model_status"),
            }

            return is_healthy

        except Exception as e:
            logger.error(f"❌ 健康检查失败: {str(e)}")
            self.results["health_check"] = {"error": str(e)}
            return False

    async def test_model_info(self):
        """测试模型信息获取"""
        logger.info("📊 获取模型信息...")

        try:
            model_info = await self.service.get_model_info()
            logger.info(f"模型总数: {model_info.get('total_models', 0)}")
            logger.info(f"模型已加载: {model_info.get('models_loaded', False)}")

            if "models" in model_info:
                for model in model_info["models"]:
                    logger.info(
                        f"  - {model.get('type', 'unknown')}: {model.get('name', 'unnamed')}"
                    )

            self.results["model_info"] = model_info
            return True

        except Exception as e:
            logger.error(f"❌ 获取模型信息失败: {str(e)}")
            self.results["model_info"] = {"error": str(e)}
            return False

    async def test_qps_prediction(self):
        """测试QPS预测"""
        logger.info("📈 测试QPS预测...")

        try:
            result = await self.service.predict_qps(
                current_qps=150.0,
                prediction_hours=12,
                granularity="hour",
                include_confidence=True,
                include_anomaly_detection=True,
                consider_historical_pattern=True,
                target_utilization=0.7,
                sensitivity=0.8,
            )

            logger.info("✅ QPS预测成功")
            logger.info(f"  当前QPS: {result['current_value']}")
            logger.info(f"  预测点数: {len(result['predicted_data'])}")
            logger.info(
                f"  扩缩容建议数: {len(result.get('scaling_recommendations', []))}"
            )
            logger.info(f"  异常预测数: {len(result.get('anomaly_predictions', []))}")

            # 显示前3个预测点
            for i, pred in enumerate(result["predicted_data"][:3]):
                logger.info(
                    f"  预测点{i + 1}: {pred['predicted_value']:.2f} (置信度: {pred.get('confidence_level', 0):.2f})"
                )

            self.results["qps_prediction"] = {
                "success": True,
                "prediction_count": len(result["predicted_data"]),
                "recommendations_count": len(result.get("scaling_recommendations", [])),
                "anomalies_count": len(result.get("anomaly_predictions", [])),
            }

            return True

        except Exception as e:
            logger.error(f"❌ QPS预测失败: {str(e)}")
            self.results["qps_prediction"] = {"error": str(e)}
            return False

    async def test_cpu_prediction(self):
        """测试CPU预测"""
        logger.info("🖥️ 测试CPU预测...")

        try:
            constraints = ResourceConstraints(
                cpu_cores=8.0,
                memory_gb=32.0,
                max_instances=10,
                min_instances=2,
                cost_per_hour=1.5,
            )

            result = await self.service.predict_cpu_utilization(
                current_cpu_percent=75.5,
                prediction_hours=24,
                granularity="hour",
                resource_constraints=constraints.dict(),
                target_utilization=0.65,
                include_anomaly_detection=True,
            )

            logger.info("✅ CPU预测成功")
            logger.info(f"  当前CPU: {result['current_value']}%")
            logger.info(f"  预测点数: {len(result['predicted_data'])}")
            logger.info(
                f"  资源利用率预测数: {len(result.get('resource_utilization', []))}"
            )

            # 成本分析
            if result.get("cost_analysis"):
                cost = result["cost_analysis"]
                logger.info(
                    f"  当前成本: ${cost.get('current_hourly_cost', 0):.2f}/小时"
                )
                logger.info(
                    f"  预测成本: ${cost.get('predicted_hourly_cost', 0):.2f}/小时"
                )
                logger.info(f"  节省潜力: {cost.get('cost_savings_potential', 0):.1f}%")

            self.results["cpu_prediction"] = {
                "success": True,
                "prediction_count": len(result["predicted_data"]),
                "has_cost_analysis": result.get("cost_analysis") is not None,
            }

            return True

        except Exception as e:
            logger.error(f"❌ CPU预测失败: {str(e)}")
            self.results["cpu_prediction"] = {"error": str(e)}
            return False

    async def test_memory_prediction(self):
        """测试内存预测"""
        logger.info("🧠 测试内存预测...")

        try:
            result = await self.service.predict_memory_utilization(
                current_memory_percent=68.2,
                prediction_hours=48,
                granularity="hour",
                include_confidence=True,
                target_utilization=0.75,
            )

            logger.info("✅ 内存预测成功")
            logger.info(f"  当前内存: {result['current_value']}%")
            logger.info(f"  预测时间范围: {result['prediction_hours']}小时")
            logger.info(f"  模式分析: {result.get('pattern_analysis', {})}")

            # 趋势洞察
            insights = result.get("trend_insights", [])
            if insights:
                logger.info("  趋势洞察:")
                for insight in insights[:3]:  # 显示前3个洞察
                    logger.info(f"    - {insight}")

            self.results["memory_prediction"] = {
                "success": True,
                "prediction_count": len(result["predicted_data"]),
                "insights_count": len(insights),
            }

            return True

        except Exception as e:
            logger.error(f"❌ 内存预测失败: {str(e)}")
            self.results["memory_prediction"] = {"error": str(e)}
            return False

    async def test_disk_prediction(self):
        """测试磁盘预测"""
        logger.info("💽 测试磁盘预测...")

        try:
            result = await self.service.predict_disk_utilization(
                current_disk_percent=82.5,
                prediction_hours=72,
                granularity="day",
                sensitivity=0.9,
                include_anomaly_detection=True,
            )

            logger.info("✅ 磁盘预测成功")
            logger.info(f"  当前磁盘: {result['current_value']}%")
            logger.info(f"  预测粒度: {result['granularity']}")

            # 预测摘要
            summary = result.get("prediction_summary", {})
            if summary:
                logger.info(f"  最大预测值: {summary.get('max_value', 0):.1f}%")
                logger.info(f"  最小预测值: {summary.get('min_value', 0):.1f}%")
                logger.info(f"  平均预测值: {summary.get('avg_value', 0):.1f}%")
                logger.info(f"  趋势: {summary.get('trend', 'unknown')}")

            self.results["disk_prediction"] = {
                "success": True,
                "prediction_count": len(result["predicted_data"]),
                "granularity": result["granularity"],
            }

            return True

        except Exception as e:
            logger.error(f"❌ 磁盘预测失败: {str(e)}")
            self.results["disk_prediction"] = {"error": str(e)}
            return False

    async def test_validation_errors(self):
        """测试参数验证"""
        logger.info("🔍 测试参数验证...")

        validation_tests = []

        # 测试无效QPS值
        try:
            await self.service.predict_qps(current_qps=-10.0)
            validation_tests.append(("invalid_qps", False, "未捕获无效QPS值"))
        except Exception as e:
            validation_tests.append(
                ("invalid_qps", True, f"正确捕获: {str(e)[:50]}...")
            )

        # 测试无效利用率
        try:
            await self.service.predict_cpu_utilization(current_cpu_percent=150.0)
            validation_tests.append(("invalid_cpu", False, "未捕获无效CPU值"))
        except Exception as e:
            validation_tests.append(
                ("invalid_cpu", True, f"正确捕获: {str(e)[:50]}...")
            )

        # 测试无效预测时长
        try:
            await self.service.predict_memory_utilization(
                current_memory_percent=50.0, prediction_hours=0
            )
            validation_tests.append(("invalid_hours", False, "未捕获无效时长"))
        except Exception as e:
            validation_tests.append(
                ("invalid_hours", True, f"正确捕获: {str(e)[:50]}...")
            )

        # 输出验证结果
        passed_count = sum(1 for _, passed, _ in validation_tests if passed)
        logger.info(f"参数验证测试: {passed_count}/{len(validation_tests)} 通过")

        for test_name, passed, message in validation_tests:
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {test_name}: {message}")

        self.results["validation_tests"] = {
            "passed": passed_count,
            "total": len(validation_tests),
            "details": validation_tests,
        }

        return passed_count == len(validation_tests)

    async def test_configuration_integration(self):
        """测试配置集成"""
        logger.info("⚙️ 测试配置集成...")

        try:
            # 检查配置文件读取
            pred_config = config.prediction
            logger.info(f"模型基础路径: {pred_config.model_base_path}")
            logger.info(f"默认预测时长: {pred_config.default_prediction_hours}小时")
            logger.info(f"最大预测时长: {pred_config.max_prediction_hours}小时")
            logger.info(f"默认目标利用率: {pred_config.default_target_utilization}")

            # 检查扩缩容阈值配置
            thresholds = pred_config.scaling_thresholds
            logger.info(
                f"QPS扩容阈值: {thresholds.get('qps', {}).get('scale_up', 'N/A')}"
            )
            logger.info(
                f"CPU扩容阈值: {thresholds.get('cpu', {}).get('scale_up', 'N/A')}"
            )

            # 检查冷却时间配置
            cooldown = pred_config.cooldown_periods
            logger.info(f"扩容冷却时间: {cooldown.get('scale_up', 'N/A')}分钟")
            logger.info(f"缩容冷却时间: {cooldown.get('scale_down', 'N/A')}分钟")

            self.results["configuration"] = {
                "model_base_path": pred_config.model_base_path,
                "scaling_thresholds_configured": len(thresholds) > 0,
                "cooldown_configured": len(cooldown) > 0,
            }

            logger.info("✅ 配置集成测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 配置集成测试失败: {str(e)}")
            self.results["configuration"] = {"error": str(e)}
            return False

    async def generate_report(self):
        """生成测试报告"""
        logger.info("📋 生成测试报告...")

        report = {
            "test_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(
                    1
                    for r in self.results.values()
                    if isinstance(r, dict)
                    and r.get("success", False)
                    and "error" not in r
                ),
                "failed_tests": sum(
                    1
                    for r in self.results.values()
                    if isinstance(r, dict) and "error" in r
                ),
            },
            "detailed_results": self.results,
            "configuration": {
                "app_version": getattr(config, "app_version", "unknown"),
                "prediction_config": {
                    "model_base_path": config.prediction.model_base_path,
                    "default_hours": config.prediction.default_prediction_hours,
                    "max_hours": config.prediction.max_prediction_hours,
                },
            },
        }

        # 保存报告到文件
        report_file = (
            project_root
            / "logs"
            / f"prediction_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"📄 测试报告已保存到: {report_file}")

        # 输出摘要
        logger.info("=" * 60)
        logger.info("🎯 测试摘要")
        logger.info("=" * 60)
        logger.info(f"总测试数: {report['summary']['total_tests']}")
        logger.info(f"通过测试: {report['summary']['passed_tests']}")
        logger.info(f"失败测试: {report['summary']['failed_tests']}")

        success_rate = (
            report["summary"]["passed_tests"] / report["summary"]["total_tests"]
        ) * 100
        logger.info(f"成功率: {success_rate:.1f}%")

        if success_rate >= 80:
            logger.info("🎉 预测系统验证成功！")
        else:
            logger.warning("⚠️ 预测系统存在问题，请检查失败的测试")

        return report

    async def run_all_tests(self):
        """运行所有测试"""
        logger.info("🚀 开始预测系统完整验证...")

        # 测试列表
        tests = [
            ("服务初始化", self.initialize_service),
            ("健康检查", self.test_service_health),
            ("模型信息", self.test_model_info),
            ("QPS预测", self.test_qps_prediction),
            ("CPU预测", self.test_cpu_prediction),
            ("内存预测", self.test_memory_prediction),
            ("磁盘预测", self.test_disk_prediction),
            ("参数验证", self.test_validation_errors),
            ("配置集成", self.test_configuration_integration),
        ]

        start_time = time.time()

        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'=' * 50}")
                logger.info(f"🧪 执行测试: {test_name}")
                logger.info(f"{'=' * 50}")

                success = await test_func()

                if success:
                    logger.info(f"✅ {test_name} 测试通过")
                else:
                    logger.warning(f"⚠️ {test_name} 测试未完全通过")

            except Exception as e:
                logger.error(f"❌ {test_name} 测试异常: {str(e)}")
                self.results[test_name.lower().replace(" ", "_")] = {"error": str(e)}

        duration = time.time() - start_time
        logger.info(f"\n⏱️ 总测试时间: {duration:.2f}秒")

        # 生成报告
        report = await self.generate_report()

        return report


async def main():
    """主函数"""
    logger.info("🎯 AI-CloudOps 新预测系统验证")
    logger.info("=" * 60)

    validator = PredictionSystemValidator()

    try:
        report = await validator.run_all_tests()

        # 根据测试结果退出
        if report["summary"]["failed_tests"] == 0:
            logger.info("🎉 所有测试通过！预测系统运行正常。")
            sys.exit(0)
        else:
            logger.error(f"❌ 有 {report['summary']['failed_tests']} 个测试失败")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ 验证过程出现异常: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
