#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能测试脚本 - 测试优化后的小助手性能
"""

import time
import asyncio
import requests
import json
from datetime import datetime
from typing import List, Dict, Any

# 测试配置
BASE_URL = "http://localhost:8000"
TEST_QUESTIONS = [
    "AI-CloudOps平台有哪些主要功能？",
    "如何部署Kubernetes集群？",
    "监控系统如何配置？",
    "故障排查有哪些步骤？",
    "自动化运维包括哪些内容？"
]

class PerformanceTestResult:
    def __init__(self):
        self.response_times = []
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_requests = 0
        self.answer_lengths = []
        self.start_time = None
        self.end_time = None
    
    def add_result(self, response_time: float, success: bool, answer_length: int = 0, cache_hit: bool = False):
        self.response_times.append(response_time)
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        if answer_length > 0:
            self.answer_lengths.append(answer_length)
        
        if cache_hit:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.response_times:
            return {"error": "No test results"}
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        max_response_time = max(self.response_times)
        min_response_time = min(self.response_times)
        
        avg_answer_length = sum(self.answer_lengths) / len(self.answer_lengths) if self.answer_lengths else 0
        
        cache_hit_rate = (self.cache_hit_count / self.total_requests * 100) if self.total_requests > 0 else 0
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        total_duration = (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0
        
        return {
            "测试总结": {
                "总请求数": self.total_requests,
                "成功请求数": self.successful_requests,
                "失败请求数": self.failed_requests,
                "成功率": f"{success_rate:.1f}%",
                "测试总时长": f"{total_duration:.2f}秒"
            },
            "响应时间": {
                "平均响应时间": f"{avg_response_time:.3f}秒",
                "最大响应时间": f"{max_response_time:.3f}秒",
                "最小响应时间": f"{min_response_time:.3f}秒",
                "响应时间分布": self._get_response_time_distribution()
            },
            "答案质量": {
                "平均答案长度": f"{avg_answer_length:.0f}字符",
                "答案长度分布": self._get_answer_length_distribution()
            },
            "缓存性能": {
                "缓存命中数": self.cache_hit_count,
                "缓存未命中数": self.cache_miss_count,
                "缓存命中率": f"{cache_hit_rate:.1f}%"
            }
        }
    
    def _get_response_time_distribution(self) -> Dict[str, int]:
        """获取响应时间分布"""
        distribution = {
            "0-2秒": 0,
            "2-5秒": 0,
            "5-10秒": 0,
            "10秒以上": 0
        }
        
        for time_val in self.response_times:
            if time_val <= 2:
                distribution["0-2秒"] += 1
            elif time_val <= 5:
                distribution["2-5秒"] += 1
            elif time_val <= 10:
                distribution["5-10秒"] += 1
            else:
                distribution["10秒以上"] += 1
        
        return distribution
    
    def _get_answer_length_distribution(self) -> Dict[str, int]:
        """获取答案长度分布"""
        distribution = {
            "0-100字符": 0,
            "100-200字符": 0,
            "200-300字符": 0,
            "300字符以上": 0
        }
        
        for length in self.answer_lengths:
            if length <= 100:
                distribution["0-100字符"] += 1
            elif length <= 200:
                distribution["100-200字符"] += 1
            elif length <= 300:
                distribution["200-300字符"] += 1
            else:
                distribution["300字符以上"] += 1
        
        return distribution


def test_assistant_performance():
    """测试小助手性能"""
    result = PerformanceTestResult()
    result.start_time = datetime.now()
    
    print("开始性能测试...")
    print(f"测试时间: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试问题数量: {len(TEST_QUESTIONS)}")
    print("-" * 50)
    
    # 第一轮测试 - 冷启动
    print("\n第一轮测试 (冷启动):")
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  {i}. {question}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response_time = time.time() - start_time
            success = response.status_code == 200
            
            if success:
                data = response.json()
                answer = data.get("data", {}).get("answer", "")
                answer_length = len(answer)
                print(f"     ✓ 响应时间: {response_time:.3f}秒, 答案长度: {answer_length}字符")
                result.add_result(response_time, success, answer_length, cache_hit=False)
            else:
                print(f"     ✗ 请求失败: {response.status_code}")
                result.add_result(response_time, success)
                
        except Exception as e:
            response_time = time.time() - start_time
            print(f"     ✗ 请求异常: {str(e)}")
            result.add_result(response_time, False)
    
    # 第二轮测试 - 缓存测试
    print("\n第二轮测试 (缓存测试):")
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  {i}. {question}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            response_time = time.time() - start_time
            success = response.status_code == 200
            
            if success:
                data = response.json()
                answer = data.get("data", {}).get("answer", "")
                answer_length = len(answer)
                # 假设2秒以下的响应很可能是缓存命中
                cache_hit = response_time < 2.0
                cache_indicator = "缓存命中" if cache_hit else "缓存未命中"
                print(f"     ✓ 响应时间: {response_time:.3f}秒, 答案长度: {answer_length}字符 ({cache_indicator})")
                result.add_result(response_time, success, answer_length, cache_hit=cache_hit)
            else:
                print(f"     ✗ 请求失败: {response.status_code}")
                result.add_result(response_time, success)
                
        except Exception as e:
            response_time = time.time() - start_time
            print(f"     ✗ 请求异常: {str(e)}")
            result.add_result(response_time, False)
    
    result.end_time = datetime.now()
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("性能测试结果:")
    print("=" * 50)
    
    summary = result.get_summary()
    
    # 美化输出
    for category, metrics in summary.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # 性能评估
    avg_response_time = sum(result.response_times) / len(result.response_times)
    success_rate = (result.successful_requests / result.total_requests * 100)
    
    print(f"\n性能评估:")
    if avg_response_time <= 3:
        print("  ✓ 响应速度: 优秀 (平均≤3秒)")
    elif avg_response_time <= 5:
        print("  ◐ 响应速度: 良好 (平均≤5秒)")
    elif avg_response_time <= 10:
        print("  ◑ 响应速度: 一般 (平均≤10秒)")
    else:
        print("  ✗ 响应速度: 需要优化 (平均>10秒)")
    
    if success_rate >= 95:
        print("  ✓ 成功率: 优秀 (≥95%)")
    elif success_rate >= 90:
        print("  ◐ 成功率: 良好 (≥90%)")
    elif success_rate >= 80:
        print("  ◑ 成功率: 一般 (≥80%)")
    else:
        print("  ✗ 成功率: 需要优化 (<80%)")
    
    if result.answer_lengths:
        avg_length = sum(result.answer_lengths) / len(result.answer_lengths)
        if 100 <= avg_length <= 250:
            print("  ✓ 答案长度: 优秀 (100-250字符)")
        elif 50 <= avg_length <= 350:
            print("  ◐ 答案长度: 良好 (50-350字符)")
        else:
            print("  ◑ 答案长度: 需要调整")
    
    print(f"\n测试完成时间: {result.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return summary


if __name__ == "__main__":
    # 检查服务是否可用
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print("❌ 服务不可用，请确保应用正在运行")
            exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到服务: {e}")
        print("请确保应用在 http://localhost:8000 上运行")
        exit(1)
    
    # 运行性能测试
    test_assistant_performance()