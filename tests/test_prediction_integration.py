#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 预测系统集成测试脚本
"""

import asyncio
from datetime import datetime
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.settings import config
from app.models import ResourceConstraints
from app.services.prediction_service import PredictionService


async def test_service_initialization():
    """测试服务初始化"""
    print("\n" + "=" * 60)
    print("测试1: 服务初始化")
    print("=" * 60)

    service = PredictionService()

    try:
        # 初始化服务
        await service.initialize()
        print("✅ 服务初始化成功")

        # 检查健康状态
        is_healthy = await service.health_check()
        print(f"✅ 健康检查: {'通过' if is_healthy else '失败'}")

        # 获取服务健康信息
        health_info = await service.get_service_health_info()
        print(f"✅ 服务状态: {health_info['service_status']}")
        print(f"✅ 模型状态: {health_info['model_status']}")

        return service

    except Exception as e:
        print(f"❌ 服务初始化失败: {str(e)}")
        return None


async def test_qps_prediction(service: PredictionService):
    """测试QPS预测"""
    print("\n" + "=" * 60)
    print("测试2: QPS负载预测")
    print("=" * 60)

    try:
        result = await service.predict_qps(
            current_qps=100.0,
            prediction_hours=24,
            granularity="hour",
            include_confidence=True,
            include_anomaly_detection=True,
            target_utilization=0.7,
        )

        print("✅ QPS预测成功")
        print(f"   - 预测数据点数: {len(result['predicted_data'])}")
        print(f"   - 扩缩容建议数: {len(result['scaling_recommendations'])}")
        print(f"   - 异常预测数: {len(result['anomaly_predictions'])}")

        # 显示预测摘要
        if "prediction_summary" in result:
            summary = result["prediction_summary"]
            print(f"   - 最大QPS: {summary['max_value']:.2f}")
            print(f"   - 最小QPS: {summary['min_value']:.2f}")
            print(f"   - 平均QPS: {summary['avg_value']:.2f}")
            print(f"   - 趋势: {summary['trend']}")

        return True

    except Exception as e:
        print(f"❌ QPS预测失败: {str(e)}")
        return False


async def test_cpu_prediction(service: PredictionService):
    """测试CPU预测"""
    print("\n" + "=" * 60)
    print("测试3: CPU使用率预测")
    print("=" * 60)

    try:
        result = await service.predict_cpu_utilization(
            current_cpu_percent=65.0,
            prediction_hours=12,
            granularity="hour",
            include_confidence=True,
            target_utilization=0.6,
        )

        print("✅ CPU预测成功")
        print(f"   - 预测数据点数: {len(result['predicted_data'])}")
        print(f"   - 扩缩容建议数: {len(result['scaling_recommendations'])}")

        # 显示趋势洞察
        if "trend_insights" in result:
            print("   - 趋势洞察:")
            for insight in result["trend_insights"][:3]:
                print(f"     • {insight}")

        return True

    except Exception as e:
        print(f"❌ CPU预测失败: {str(e)}")
        return False


async def test_memory_prediction(service: PredictionService):
    """测试内存预测"""
    print("\n" + "=" * 60)
    print("测试4: 内存使用率预测")
    print("=" * 60)

    try:
        result = await service.predict_memory_utilization(
            current_memory_percent=75.0,
            prediction_hours=6,
            granularity="hour",
            sensitivity=0.9,
        )

        print("✅ 内存预测成功")
        print(f"   - 预测数据点数: {len(result['predicted_data'])}")

        # 显示模式分析
        if "pattern_analysis" in result:
            pattern = result["pattern_analysis"]
            print("   - 模式分析:")
            print(f"     • 周期性: {pattern.get('has_periodicity', False)}")
            print(f"     • 波动性: {pattern.get('volatility', 0):.2f}")

        return True

    except Exception as e:
        print(f"❌ 内存预测失败: {str(e)}")
        return False


async def test_disk_prediction(service: PredictionService):
    """测试磁盘预测"""
    print("\n" + "=" * 60)
    print("测试5: 磁盘使用率预测")
    print("=" * 60)

    try:
        # 带资源约束的预测
        constraints = ResourceConstraints(disk_gb=100, cost_per_hour=0.1)

        result = await service.predict_disk_utilization(
            current_disk_percent=55.0,
            prediction_hours=48,
            granularity="hour",
            resource_constraints=constraints.dict(),
        )

        print("✅ 磁盘预测成功")
        print(f"   - 预测数据点数: {len(result['predicted_data'])}")

        # 显示成本分析
        if result.get("cost_analysis"):
            cost = result["cost_analysis"]
            print("   - 成本分析:")
            if cost.get("current_hourly_cost"):
                print(f"     • 当前成本: ${cost['current_hourly_cost']:.4f}/小时")
            if cost.get("predicted_hourly_cost"):
                print(f"     • 预测成本: ${cost['predicted_hourly_cost']:.4f}/小时")

        return True

    except Exception as e:
        print(f"❌ 磁盘预测失败: {str(e)}")
        return False


async def test_model_info(service: PredictionService):
    """测试模型信息获取"""
    print("\n" + "=" * 60)
    print("测试6: 模型信息获取")
    print("=" * 60)

    try:
        model_info = await service.get_model_info()

        if model_info.get("status") == "not_initialized":
            print("⚠️  模型未初始化")
        else:
            print("✅ 获取模型信息成功")
            print(f"   - 模型状态: {model_info.get('status', 'unknown')}")

            if "models" in model_info and model_info["models"]:
                print(f"   - 已加载模型数: {len(model_info['models'])}")
                for model in model_info["models"][:2]:
                    print(f"     • {model}")

        return True

    except Exception as e:
        print(f"❌ 获取模型信息失败: {str(e)}")
        return False


async def test_configuration():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试7: 配置验证")
    print("=" * 60)

    try:
        print("✅ 配置加载成功")
        print(f"   - 模型基础路径: {config.prediction.model_base_path}")
        print(f"   - 最大预测时长: {config.prediction.max_prediction_hours}小时")
        print(f"   - 最小预测时长: {config.prediction.min_prediction_hours}小时")
        print(f"   - 默认预测时长: {config.prediction.default_prediction_hours}小时")
        print(f"   - 默认粒度: {config.prediction.default_granularity}")
        print(f"   - 默认目标利用率: {config.prediction.default_target_utilization}")

        # 检查模型路径配置
        model_paths = config.prediction.model_paths
        if model_paths:
            print("   - 模型路径配置:")
            for model_type in ["qps", "cpu", "memory", "disk"]:
                if model_type in model_paths:
                    model_file = model_paths[model_type].get("model", "")
                    exists = os.path.exists(model_file) if model_file else False
                    status = "✅ 存在" if exists else "❌ 不存在"
                    print(f"     • {model_type.upper()}: {status}")

        return True

    except Exception as e:
        print(f"❌ 配置验证失败: {str(e)}")
        return False


async def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("AI-CloudOps 预测系统集成测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().isoformat()}")

    # 测试配置
    await test_configuration()

    # 初始化服务
    service = await test_service_initialization()
    if not service:
        print("\n❌ 服务初始化失败，无法继续测试")
        return

    # 运行各项测试
    test_results = []

    test_results.append(await test_qps_prediction(service))
    test_results.append(await test_cpu_prediction(service))
    test_results.append(await test_memory_prediction(service))
    test_results.append(await test_disk_prediction(service))
    test_results.append(await test_model_info(service))

    # 清理资源
    await service.cleanup()

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed = sum(1 for r in test_results if r)
    total = len(test_results)

    print(f"通过测试: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print(f"⚠️  有 {total - passed} 个测试失败")

    print(f"结束时间: {datetime.now().isoformat()}")


if __name__ == "__main__":
    # 设置事件循环策略（解决某些环境下的兼容性问题）
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行主测试
    asyncio.run(main())
