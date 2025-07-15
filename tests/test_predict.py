#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 负载预测服务完整测试套件
"""

import pytest
import requests
from datetime import datetime, timedelta
from test_utils import (
    DEFAULT_API_BASE_URL,
    setup_test_logger,
    make_request,
    validate_response_structure,
    TestResult
)

logger = setup_test_logger("test_predict")


class TestPredictAPI:
    """负载预测API测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.api_base_url = DEFAULT_API_BASE_URL
        cls.test_result = TestResult("负载预测API测试")
        
    def test_predict_health_check(self):
        """测试预测服务健康检查"""
        logger.info("测试预测服务健康检查 /api/v1/predict/health")
        
        url = f"{self.api_base_url}/predict/health"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "预测健康检查请求失败"
        assert response.status_code == 200, f"预测健康检查返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "status" in data, "响应数据缺少status字段"
        assert "healthy" in data, "响应数据缺少healthy字段"
        assert "model_loaded" in data, "响应数据缺少model_loaded字段"
        assert "scaler_loaded" in data, "响应数据缺少scaler_loaded字段"
        
        self.test_result.add_test_result(
            "predict_health_check", 
            True, 
            status_code=response.status_code,
            healthy=data.get("healthy"),
            model_loaded=data.get("model_loaded")
        )
        
    def test_predict_ready_check(self):
        """测试预测服务就绪性检查"""
        logger.info("测试预测服务就绪性检查 /api/v1/predict/ready")
        
        url = f"{self.api_base_url}/predict/ready"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "预测就绪性检查请求失败"
        # 就绪性检查可能返回503(未就绪)或200(就绪)
        assert response.status_code in [200, 503], f"预测就绪性检查返回不预期状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "status" in data, "响应数据缺少status字段"
        
        self.test_result.add_test_result(
            "predict_ready_check", 
            True, 
            status_code=response.status_code,
            ready_status=data.get("status")
        )
        
    def test_predict_info(self):
        """测试预测服务信息接口"""
        logger.info("测试预测服务信息接口 /api/v1/predict/info")
        
        url = f"{self.api_base_url}/predict/info"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "预测服务信息请求失败"
        assert response.status_code == 200, f"预测服务信息返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "service" in data, "响应数据缺少service字段"
        assert "version" in data, "响应数据缺少version字段"
        
        self.test_result.add_test_result(
            "predict_info", 
            True, 
            status_code=response.status_code,
            service=data.get("service")
        )
        
    def test_predict_get_request(self):
        """测试GET请求预测接口"""
        logger.info("测试GET请求预测接口 /api/v1/predict")
        
        url = f"{self.api_base_url}/predict"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "GET预测请求失败"
        assert response.status_code == 200, f"GET预测请求返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "instances" in data, "响应数据缺少instances字段"
        assert "current_qps" in data, "响应数据缺少current_qps字段"
        assert "timestamp" in data, "响应数据缺少timestamp字段"
        
        # 验证数据类型
        assert isinstance(data["instances"], int), "实例数应为整数"
        assert isinstance(data["current_qps"], (int, float)), "QPS应为数字"
        
        self.test_result.add_test_result(
            "predict_get_request", 
            True, 
            status_code=response.status_code,
            instances=data.get("instances"),
            current_qps=data.get("current_qps")
        )
        
    def test_predict_post_request_with_qps(self):
        """测试带QPS参数的POST请求预测接口"""
        logger.info("测试带QPS参数的POST请求预测接口")
        
        url = f"{self.api_base_url}/predict"
        payload = {
            "current_qps": 100.5,
            "include_confidence": True
        }
        
        response = make_request("post", url, json_data=payload, logger=logger)
        
        assert response is not None, "POST预测请求失败"
        assert response.status_code == 200, f"POST预测请求返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        assert "instances" in data, "响应数据缺少instances字段"
        assert "current_qps" in data, "响应数据缺少current_qps字段"
        assert data["current_qps"] == 100.5, "返回的QPS与请求不符"
        
        # 由于设置了include_confidence=True，检查置信度
        if "confidence" in data:
            assert isinstance(data["confidence"], (int, float)), "置信度应为数字"
        
        self.test_result.add_test_result(
            "predict_post_request_with_qps", 
            True, 
            status_code=response.status_code,
            instances=data.get("instances"),
            confidence=data.get("confidence")
        )
        
    def test_predict_post_request_invalid_qps(self):
        """测试无效QPS参数的POST请求"""
        logger.info("测试无效QPS参数的POST请求")
        
        url = f"{self.api_base_url}/predict"
        payload = {
            "current_qps": -10  # 负数QPS应该被拒绝
        }
        
        response = make_request("post", url, json_data=payload, logger=logger)
        
        assert response is not None, "POST预测请求失败"
        assert response.status_code == 400, f"无效QPS应返回400状态码，实际: {response.status_code}"
        
        self.test_result.add_test_result(
            "predict_post_request_invalid_qps", 
            True, 
            status_code=response.status_code
        )
        
    def test_predict_trend_get_request(self):
        """测试GET趋势预测接口"""
        logger.info("测试GET趋势预测接口 /api/v1/predict/trend")
        
        url = f"{self.api_base_url}/predict/trend?hours=24&qps=50"
        response = make_request("get", url, logger=logger)
        
        assert response is not None, "GET趋势预测请求失败"
        assert response.status_code == 200, f"GET趋势预测返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        data = validation["data"]["data"]
        # 趋势预测可能包含forecast等字段
        
        self.test_result.add_test_result(
            "predict_trend_get_request", 
            True, 
            status_code=response.status_code
        )
        
    def test_predict_trend_post_request(self):
        """测试POST趋势预测接口"""
        logger.info("测试POST趋势预测接口")
        
        url = f"{self.api_base_url}/predict/trend"
        payload = {
            "hours_ahead": 12,
            "current_qps": 75.0
        }
        
        response = make_request("post", url, json_data=payload, logger=logger)
        
        assert response is not None, "POST趋势预测请求失败"
        assert response.status_code == 200, f"POST趋势预测返回状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        self.test_result.add_test_result(
            "predict_trend_post_request", 
            True, 
            status_code=response.status_code
        )
        
    def test_predict_trend_invalid_hours(self):
        """测试无效小时数的趋势预测请求"""
        logger.info("测试无效小时数的趋势预测请求")
        
        url = f"{self.api_base_url}/predict/trend"
        payload = {
            "hours_ahead": 200  # 超过最大限制168小时
        }
        
        response = make_request("post", url, json_data=payload, logger=logger)
        
        assert response is not None, "POST趋势预测请求失败"
        assert response.status_code == 400, f"无效小时数应返回400状态码，实际: {response.status_code}"
        
        self.test_result.add_test_result(
            "predict_trend_invalid_hours", 
            True, 
            status_code=response.status_code
        )
        
    def test_predict_models_reload(self):
        """测试模型重新加载接口"""
        logger.info("测试模型重新加载接口 /api/v1/predict/models/reload")
        
        url = f"{self.api_base_url}/predict/models/reload"
        response = make_request("post", url, logger=logger)
        
        assert response is not None, "模型重新加载请求失败"
        # 模型加载可能成功(200)或失败(500)，取决于模型文件是否存在
        assert response.status_code in [200, 500], f"模型重新加载返回不预期状态码 {response.status_code}"
        
        # 验证响应结构
        required_fields = ["code", "message", "data"]
        validation = validate_response_structure(response, required_fields, logger)
        assert validation["valid"], f"响应结构验证失败: {validation['missing_fields']}"
        
        self.test_result.add_test_result(
            "predict_models_reload", 
            True, 
            status_code=response.status_code
        )
        
    def test_predict_response_timing(self):
        """测试预测接口响应时间"""
        logger.info("测试预测接口响应时间")
        
        import time
        start_time = time.time()
        
        url = f"{self.api_base_url}/predict"
        response = make_request("get", url, timeout=10, logger=logger)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response is not None, "预测请求失败"
        assert response.status_code == 200, f"预测请求返回状态码 {response.status_code}"
        assert response_time < 10.0, f"预测响应时间过长: {response_time:.2f}秒"
        
        self.test_result.add_test_result(
            "predict_response_timing", 
            True, 
            status_code=response.status_code,
            response_time=response_time
        )
        
        logger.info(f"预测接口响应时间: {response_time:.2f}秒")
        
    @classmethod
    def teardown_class(cls):
        """测试类清理"""
        results = cls.test_result.finalize()
        
        # 打印测试摘要
        from test_utils import print_test_summary
        print_test_summary(results, results.get("summary", {}).get("duration_seconds", 0))
        
        # 保存测试结果
        from test_utils import save_test_results
        save_test_results(results, "logs/predict_test_results.json", logger)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])