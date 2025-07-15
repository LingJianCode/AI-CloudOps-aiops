#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 健康检查接口完整测试套件
"""

import pytest
import requests
from datetime import datetime
from test_utils import (
    DEFAULT_API_BASE_URL,
    setup_test_logger,
    make_request,
    validate_response_structure,
    TestResult
)

logger = setup_test_logger("test_health")


class TestHealthAPI:
    """健康检查API测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.api_base_url = DEFAULT_API_BASE_URL
        cls.test_result = TestResult("健康检查API测试")
        
    def test_basic_health_check(self):
        """测试基础健康检查接口"""
        logger.info("测试基础健康检查接口 /api/v1/health")
        
        url = f"{self.api_base_url}/health"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "健康检查请求失败"
        assert response.status_code == 200, f"健康检查返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "status" in data, "响应数据缺少status字段"
        assert "uptime" in data, "响应数据缺少uptime字段"
        assert "timestamp" in data, "响应数据缺少timestamp字段"
        
        self.test_result.add_test_result(
            "basic_health_check", 
            True, 
            status_code=response.status_code,
            status=data.get("status")
        )
        
    def test_components_health_check(self):
        """测试组件健康检查接口"""
        logger.info("测试组件健康检查接口 /api/v1/health/components")
        
        url = f"{self.api_base_url}/health/components"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "组件健康检查请求失败"
        assert response.status_code == 200, f"组件健康检查返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "components" in data, "响应数据缺少components字段"
        assert "timestamp" in data, "响应数据缺少timestamp字段"
        
        # 验证组件列表
        components = data["components"]
        expected_components = ["prometheus", "kubernetes", "llm", "notification", "prediction"]
        for component in expected_components:
            assert component in components, f"缺少组件: {component}"
            assert "healthy" in components[component], f"组件 {component} 缺少健康状态"
        
        self.test_result.add_test_result(
            "components_health_check", 
            True, 
            status_code=response.status_code,
            components_count=len(components)
        )
        
    def test_health_metrics(self):
        """测试健康指标接口"""
        logger.info("测试健康指标接口 /api/v1/health/metrics")
        
        url = f"{self.api_base_url}/health/metrics"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "健康指标请求失败"
        assert response.status_code == 200, f"健康指标返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        
        # 验证系统指标
        expected_metrics = ["cpu", "memory", "disk", "network", "process"]
        for metric in expected_metrics:
            assert metric in data, f"缺少系统指标: {metric}"
            
        # 验证CPU指标
        cpu_data = data["cpu"]
        assert "usage_percent" in cpu_data, "CPU数据缺少使用率"
        assert "count" in cpu_data, "CPU数据缺少核心数"
        assert isinstance(cpu_data["usage_percent"], (int, float)), "CPU使用率类型错误"
        
        # 验证内存指标
        memory_data = data["memory"]
        assert "usage_percent" in memory_data, "内存数据缺少使用率"
        assert "available_bytes" in memory_data, "内存数据缺少可用字节数"
        assert "total_bytes" in memory_data, "内存数据缺少总字节数"
        
        self.test_result.add_test_result(
            "health_metrics", 
            True, 
            status_code=response.status_code,
            cpu_usage=cpu_data.get("usage_percent"),
            memory_usage=memory_data.get("usage_percent")
        )
        
    def test_readiness_probe(self):
        """测试就绪性探针接口"""
        logger.info("测试就绪性探针接口 /api/v1/health/ready")
        
        url = f"{self.api_base_url}/health/ready"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "就绪性探针请求失败"
        # 就绪性探针可能返回503(未就绪)或200(就绪)
        assert response.status_code in [200, 503], f"就绪性探针返回不预期状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "status" in data, "响应数据缺少status字段"
        assert "timestamp" in data, "响应数据缺少timestamp字段"
        
        status = data["status"]
        assert status in ["ready", "not ready"], f"未知的就绪状态: {status}"
        
        self.test_result.add_test_result(
            "readiness_probe", 
            True, 
            status_code=response.status_code,
            ready_status=status
        )
        
    def test_liveness_probe(self):
        """测试存活性探针接口"""
        logger.info("测试存活性探针接口 /api/v1/health/live")
        
        url = f"{self.api_base_url}/health/live"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "存活性探针请求失败"
        assert response.status_code == 200, f"存活性探针返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "status" in data, "响应数据缺少status字段"
        assert "timestamp" in data, "响应数据缺少timestamp字段"
        assert "uptime" in data, "响应数据缺少uptime字段"
        
        assert data["status"] == "alive", f"存活状态错误: {data['status']}"
        assert isinstance(data["uptime"], (int, float)), "运行时间类型错误"
        
        self.test_result.add_test_result(
            "liveness_probe", 
            True, 
            status_code=response.status_code,
            uptime=data.get("uptime")
        )
        
    def test_health_response_timing(self):
        """测试健康检查响应时间"""
        logger.info("测试健康检查响应时间")
        
        import time
        start_time = time.time()
        
        url = f"{self.api_base_url}/health"
        response = make_request("get", url, timeout=5, logger=logger)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response is not None, "健康检查请求失败"
        assert response.status_code == 200, f"健康检查返回状态码 {response.status_code}"
        assert response_time < 5.0, f"健康检查响应时间过长: {response_time:.2f}秒"
        
        self.test_result.add_test_result(
            "health_response_timing", 
            True, 
            status_code=response.status_code,
            response_time=response_time
        )
        
        logger.info(f"健康检查响应时间: {response_time:.2f}秒")
        
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        results = cls.test_result.finalize()
        
        # 打印测试摘要
        from test_utils import print_test_summary
        print_test_summary(results, results.get("summary", {}).get("duration_seconds", 0))
        
        # 保存测试结果
        from test_utils import save_test_results
        save_test_results(results, "logs/health_test_results.json", logger)


def test_root_endpoint():
    """测试根路径端点"""
    logger.info("测试根路径端点 /")
    
    api_base_url = DEFAULT_API_BASE_URL.replace("/api/v1", "")
    url = f"{api_base_url}/"
    
    response = make_request("get", url, logger=logger)
    
    assert response is not None, "根路径请求失败"
    assert response.status_code == 200, f"根路径返回状态码 {response.status_code}"
    
    # 验证响应结构
    try:
        data = response.json()
        assert "service" in data, "根路径响应缺少service字段"
        assert "version" in data, "根路径响应缺少version字段"
        assert "endpoints" in data, "根路径响应缺少endpoints字段"
        
        logger.info(f"服务名称: {data.get('service')}")
        logger.info(f"服务版本: {data.get('version')}")
        
    except Exception as e:
        pytest.fail(f"根路径响应解析失败: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])