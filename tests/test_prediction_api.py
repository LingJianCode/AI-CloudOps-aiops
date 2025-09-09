#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 预测API端到端测试脚本
"""

from datetime import datetime
import json
from typing import Any, Dict, Optional

import requests

BASE_URL = "http://localhost:8080/api/v1"


def print_response(response_data: Dict[str, Any], indent: int = 2):
    """格式化打印响应数据"""
    print(json.dumps(response_data, indent=indent, ensure_ascii=False, default=str))


def test_api_endpoint(
    method: str, endpoint: str, data: Optional[Dict] = None, test_name: str = ""
) -> bool:
    """测试API端点"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'=' * 60}")
    print(f"测试: {test_name or endpoint}")
    print(f"{'=' * 60}")
    print(f"请求: {method} {url}")

    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            print("请求体:")
            print_response(data)
            response = requests.post(url, json=data, timeout=60)
        else:
            print(f"不支持的方法: {method}")
            return False

        print(f"\n响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ 测试通过")
                print(f"响应消息: {result.get('message', '')}")

                # 显示部分数据
                if result.get("data"):
                    data = result["data"]
                    if isinstance(data, dict):
                        # 显示关键信息
                        if "prediction_type" in data:
                            print(f"   - 预测类型: {data['prediction_type']}")
                        if "prediction_hours" in data:
                            print(f"   - 预测时长: {data['prediction_hours']}小时")
                        if "predicted_data" in data and isinstance(
                            data["predicted_data"], list
                        ):
                            print(f"   - 预测点数: {len(data['predicted_data'])}")
                        if "scaling_recommendations" in data:
                            print(
                                f"   - 扩缩容建议: {len(data['scaling_recommendations'])}条"
                            )
                        if "service_status" in data:
                            print(f"   - 服务状态: {data['service_status']}")
                        if "model_status" in data:
                            print(f"   - 模型状态: {data['model_status']}")
                        if "supported_prediction_types" in data:
                            print(
                                f"   - 支持类型: {data['supported_prediction_types']}"
                            )

                return True
            else:
                print(f"❌ 测试失败: {result.get('message', '未知错误')}")
                return False
        else:
            print(f"❌ HTTP错误: {response.status_code}")
            try:
                error_data = response.json()
                print(f"错误详情: {error_data}")
            except Exception:
                print(f"响应内容: {response.text[:500]}")
            return False

    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请确保服务已启动")
        return False
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("AI-CloudOps 预测API端到端测试")
    print("=" * 80)
    print(f"开始时间: {datetime.now().isoformat()}")
    print(f"目标服务: {BASE_URL}")

    # 检查服务是否运行
    print("\n检查服务状态...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务未运行，请先启动服务：python app/main.py")
            return
        print("✅ 服务正在运行")
    except Exception:
        print("❌ 无法连接到服务，请先启动服务：python app/main.py")
        return

    test_results = []

    # 1. 测试预测服务信息端点
    test_results.append(
        test_api_endpoint("GET", "/predict/info", test_name="预测服务信息")
    )

    # 2. 测试预测服务就绪检查
    test_results.append(
        test_api_endpoint("GET", "/predict/ready", test_name="预测服务就绪检查")
    )

    # 3. 测试预测服务就绪检查
    test_results.append(
        test_api_endpoint("GET", "/predict/ready", test_name="预测服务就绪检查")
    )

    # 4. 测试模型信息
    test_results.append(
        test_api_endpoint("GET", "/predict/models", test_name="模型信息")
    )

    # 5. 测试QPS预测
    qps_request = {
        "prediction_type": "qps",
        "current_value": 150.0,
        "prediction_hours": 12,
        "granularity": "hour",
        "include_confidence": True,
        "include_anomaly_detection": True,
        "target_utilization": 0.7,
        "sensitivity": 0.8,
    }
    test_results.append(
        test_api_endpoint(
            "POST", "/predict/qps", data=qps_request, test_name="QPS负载预测"
        )
    )

    # 6. 测试CPU预测
    cpu_request = {
        "prediction_type": "cpu",
        "current_value": 65.0,
        "prediction_hours": 6,
        "granularity": "hour",
        "include_confidence": True,
        "target_utilization": 0.6,
    }
    test_results.append(
        test_api_endpoint(
            "POST", "/predict/cpu", data=cpu_request, test_name="CPU使用率预测"
        )
    )

    # 7. 测试内存预测
    memory_request = {
        "prediction_type": "memory",
        "current_value": 70.0,
        "prediction_hours": 24,
        "granularity": "hour",
        "include_anomaly_detection": False,
        "target_utilization": 0.75,
    }
    test_results.append(
        test_api_endpoint(
            "POST", "/predict/memory", data=memory_request, test_name="内存使用率预测"
        )
    )

    # 8. 测试磁盘预测（带资源约束）
    disk_request = {
        "prediction_type": "disk",
        "current_value": 60.0,
        "prediction_hours": 48,
        "granularity": "hour",
        "resource_constraints": {"disk_gb": 200, "cost_per_hour": 0.05},
        "include_confidence": True,
        "sensitivity": 0.9,
    }
    test_results.append(
        test_api_endpoint(
            "POST", "/predict/disk", data=disk_request, test_name="磁盘使用率预测"
        )
    )

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed = sum(1 for r in test_results if r)
    total = len(test_results)

    print(f"通过测试: {passed}/{total}")

    # 详细结果
    test_names = [
        "预测服务信息",
        "预测服务健康检查",
        "预测服务就绪检查",
        "模型信息",
        "QPS负载预测",
        "CPU使用率预测",
        "内存使用率预测",
        "磁盘使用率预测",
    ]

    print("\n测试结果详情:")
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i + 1}. {name}: {status}")

    if passed == total:
        print("\n🎉 所有测试通过！预测API正常工作")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查服务和配置")

    print(f"\n结束时间: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
