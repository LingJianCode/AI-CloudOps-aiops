#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Prometheus集成和特征名称修复测试脚本
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import warnings
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置日志级别，捕获警告
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('error')  # 将警告转换为错误以便捕获

from app.services.prediction_service import PredictionService
from app.services.prometheus import PrometheusService
from app.models import PredictionType


async def test_prometheus_connection():
    """测试Prometheus连接"""
    print("\n" + "="*60)
    print("测试1: Prometheus连接和健康检查")
    print("="*60)
    
    prom_service = PrometheusService()
    
    try:
        # 健康检查
        is_healthy = prom_service.is_healthy()
        print(f"Prometheus健康状态: {'✅ 正常' if is_healthy else '❌ 异常'}")
        
        if is_healthy:
            # 获取可用指标
            metrics = await prom_service.get_available_metrics()
            print(f"可用指标数量: {len(metrics)}")
            
            # 显示部分指标
            if metrics:
                print("前10个指标:")
                for metric in metrics[:10]:
                    print(f"  - {metric}")
            
            return True
        else:
            print("⚠️  Prometheus服务不可用，将使用模拟数据进行测试")
            return False
            
    except Exception as e:
        print(f"❌ Prometheus连接测试失败: {str(e)}")
        return False


async def test_prometheus_data_query():
    """测试Prometheus数据查询"""
    print("\n" + "="*60)
    print("测试2: Prometheus数据查询")
    print("="*60)
    
    prom_service = PrometheusService()
    
    if not prom_service.is_healthy():
        print("⚠️  Prometheus不可用，跳过数据查询测试")
        return False
    
    try:
        # 测试基本查询
        test_queries = [
            'up',
            'node_cpu_seconds_total',
            'container_cpu_usage_seconds_total'
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            try:
                result = await prom_service.query_instant(query)
                if result:
                    print(f"✅ 查询成功，返回 {len(result)} 个结果")
                else:
                    print("⚠️  查询无结果")
            except Exception as e:
                print(f"❌ 查询失败: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据查询测试失败: {str(e)}")
        return False


async def test_feature_name_consistency():
    """测试特征名称一致性（无警告）"""
    print("\n" + "="*60)
    print("测试3: 特征名称一致性（无sklearn警告）")
    print("="*60)
    
    service = PredictionService()
    
    try:
        await service.initialize()
        print("✅ 预测服务初始化成功")
        
        # 捕获警告的计数器
        warning_count = 0
        original_warn = warnings.warn
        
        def count_warnings(message, category=None, *args, **kwargs):
            nonlocal warning_count
            if "feature names" in str(message).lower():
                warning_count += 1
            return original_warn(message, category, *args, **kwargs)
        
        warnings.warn = count_warnings
        
        try:
            # 测试各种预测类型
            test_cases = [
                {"type": PredictionType.QPS, "value": 150.0, "name": "QPS预测"},
                {"type": PredictionType.CPU, "value": 65.0, "name": "CPU预测"},
                {"type": PredictionType.MEMORY, "value": 70.0, "name": "内存预测"},
                {"type": PredictionType.DISK, "value": 55.0, "name": "磁盘预测"}
            ]
            
            for case in test_cases:
                print(f"\n  测试 {case['name']}...")
                
                try:
                    if case["type"] == PredictionType.QPS:
                        result = await service.predict_qps(
                            current_qps=case["value"],
                            prediction_hours=6
                        )
                    elif case["type"] == PredictionType.CPU:
                        result = await service.predict_cpu_utilization(
                            current_cpu_percent=case["value"],
                            prediction_hours=6
                        )
                    elif case["type"] == PredictionType.MEMORY:
                        result = await service.predict_memory_utilization(
                            current_memory_percent=case["value"],
                            prediction_hours=6
                        )
                    else:  # DISK
                        result = await service.predict_disk_utilization(
                            current_disk_percent=case["value"],
                            prediction_hours=6
                        )
                    
                    if result and 'predicted_data' in result:
                        print(f"    ✅ {case['name']}成功 - {len(result['predicted_data'])} 个预测点")
                    else:
                        print(f"    ❌ {case['name']}失败 - 无预测数据")
                        
                except Exception as e:
                    print(f"    ❌ {case['name']}异常: {str(e)}")
            
        finally:
            warnings.warn = original_warn
        
        print(f"\n特征名称警告统计: {warning_count} 个")
        if warning_count == 0:
            print("🎉 没有特征名称不匹配的警告！")
            return True
        else:
            print(f"⚠️  仍有 {warning_count} 个特征名称警告")
            return False
            
    except Exception as e:
        print(f"❌ 特征名称测试失败: {str(e)}")
        return False
    
    finally:
        await service.cleanup()


async def test_prometheus_data_integration():
    """测试Prometheus数据集成"""
    print("\n" + "="*60)
    print("测试4: Prometheus数据集成到预测服务")
    print("="*60)
    
    service = PredictionService()
    
    try:
        await service.initialize()
        
        # 使用自定义Prometheus查询进行预测
        custom_query = "rate(nginx_ingress_controller_requests_total[5m])"
        
        print(f"使用自定义查询: {custom_query}")
        
        result = await service.predict_qps(
            current_qps=100.0,
            metric_query=custom_query,
            prediction_hours=12
        )
        
        if result:
            print("✅ Prometheus数据集成成功")
            print(f"   - 预测数据点: {len(result.get('predicted_data', []))}")
            print(f"   - 扩缩容建议: {len(result.get('scaling_recommendations', []))}")
            return True
        else:
            print("❌ Prometheus数据集成失败")
            return False
            
    except Exception as e:
        print(f"❌ Prometheus数据集成测试失败: {str(e)}")
        return False
    
    finally:
        await service.cleanup()


async def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("AI-CloudOps Prometheus集成和特征优化测试")
    print("="*80)
    print(f"开始时间: {datetime.now().isoformat()}")
    
    test_results = []
    
    # 1. Prometheus连接测试
    test_results.append(await test_prometheus_connection())
    
    # 2. Prometheus数据查询测试
    test_results.append(await test_prometheus_data_query())
    
    # 3. 特征名称一致性测试（最重要）
    test_results.append(await test_feature_name_consistency())
    
    # 4. Prometheus数据集成测试
    test_results.append(await test_prometheus_data_integration())
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    test_names = [
        "Prometheus连接",
        "Prometheus数据查询", 
        "特征名称一致性",
        "Prometheus数据集成"
    ]
    
    passed = sum(1 for r in test_results if r)
    total = len(test_results)
    
    print(f"通过测试: {passed}/{total}")
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    # 重点关注特征名称修复
    feature_test_passed = test_results[2] if len(test_results) > 2 else False
    
    print(f"\n结束时间: {datetime.now().isoformat()}")
    
    if feature_test_passed:
        print("\n🎉 特征名称警告已修复！")
    else:
        print("\n⚠️  特征名称警告仍需进一步修复")
    
    if passed >= 3:  # 至少3个测试通过
        print("🎊 核心功能正常工作")
    else:
        print("⚠️  部分功能需要检查")


if __name__ == "__main__":
    # 设置事件循环策略
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行测试
    asyncio.run(main())
