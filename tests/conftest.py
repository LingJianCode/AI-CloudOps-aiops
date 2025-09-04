#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import inspect
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import requests
import yaml


def pytest_pycollect_makeitem(collector, name, obj):
    # 跳过收集 tests/test_prediction_api.py 中的工具函数 test_api_endpoint
    try:
        module = getattr(collector, "module", None)
        if (
            module
            and getattr(module, "__file__", "").endswith("tests/test_prediction_api.py")
            and name == "test_api_endpoint"
            and inspect.isfunction(obj)
        ):
            return []
    except Exception:
        pass
    # 其他项按默认流程
    return None


def _remove_prediction_api_helper(items):
    # 防止将工具函数 test_api_endpoint 视为测试用例（其参数非fixture）
    for item in list(items):
        if item.nodeid.endswith("tests/test_prediction_api.py::test_api_endpoint"):
            items.remove(item)


@pytest.fixture
def service():
    # 为异步集成测试提供一个占位fixture，真正的实例在测试内部创建
    # 这些测试中的函数签名包含 service，但并未使用pytest注入
    # 提供该fixture以防止“fixture 'service' not found”错误
    return None


# 添加项目路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session", autouse=True)
def setup_test_config():
    """设置测试配置文件"""
    # 设置环境为测试环境
    os.environ["ENV"] = "test"

    # 创建测试配置文件
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)

    test_config_path = config_dir / "config.test.yaml"
    if not test_config_path.exists():
        # 基于默认配置创建测试配置
        default_config_path = config_dir / "config.yaml"
        if default_config_path.exists():
            with open(default_config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # 修改配置适应测试环境
            config_data["app"]["debug"] = True
            config_data["app"]["log_level"] = "WARNING"
            config_data["testing"] = {"skip_llm_tests": True}  # 默认跳过LLM测试

            with open(test_config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, allow_unicode=True)

    yield

    # 清理环境变量
    os.environ.pop("ENV", None)


@pytest.fixture
def app():
    """创建测试用的Flask应用"""
    from app.main import create_app

    app = create_app()
    app.config["TESTING"] = True
    app.config["DEBUG"] = True

    return app


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return app.test_client()


@pytest.fixture
def prometheus_service():
    """获取Prometheus服务实例"""
    from app.services.prometheus import PrometheusService

    return PrometheusService()


@pytest.fixture
def k8s_service():
    """获取Kubernetes服务实例"""
    from app.services.kubernetes import KubernetesService

    return KubernetesService()


@pytest.fixture
def llm_service():
    """获取LLM服务实例"""
    from app.services.llm import LLMService

    return LLMService()


@pytest.fixture
def prediction_service():
    """获取预测服务实例"""
    from app.services.prediction_service import PredictionService

    return PredictionService()


@pytest.fixture
def sample_rca_request():
    """示例RCA请求数据"""
    return {
        "start_time": "2024-01-01T10:00:00Z",
        "end_time": "2024-01-01T11:00:00Z",
        "metrics": ["container_cpu_usage_seconds_total"],
    }


@pytest.fixture
def sample_autofix_request():
    """示例自动修复请求数据"""
    return {"deployment": "test-app", "namespace": "default", "event": "Pod启动失败"}


@pytest.fixture
def real_knowledge_base():
    """使用真实知识库目录"""
    from app.config.settings import config

    return config.rag.knowledge_base_path


@pytest.fixture
def sample_document():
    """示例知识库文档"""
    return """
# AIOps平台说明文档

## 简介

AIOps平台是一个智能运维系统，提供根因分析、自动修复和负载预测功能。

## 核心功能

1. 智能根因分析
2. Kubernetes自动修复
3. 基于机器学习的负载预测

## 系统架构

AIOps平台采用微服务架构，包括API网关、核心业务逻辑和服务层。

## 联系方式

如有问题请联系开发团队：support@example.com
"""


@pytest.fixture
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """模拟环境变量"""
    old_vars = {}

    def _set_vars(**kwargs):
        for key, value in kwargs.items():
            old_vars[key] = os.environ.get(key)
            os.environ[key] = value

        return old_vars

    yield _set_vars

    # 恢复原始环境变量
    for key, value in old_vars.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# 新增的API测试夹具


@pytest.fixture(scope="session")
def api_base_url():
    """API基础URL"""
    host = os.getenv("API_HOST", "localhost")
    port = os.getenv("API_PORT", "8080")
    return f"http://{host}:{port}/api/v1"


@pytest.fixture(scope="session")
def api_timeout():
    """API请求超时时间"""
    return int(os.getenv("API_TIMEOUT", "30"))


@pytest.fixture
def test_logger():
    """测试专用日志器"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@pytest.fixture
def api_client(api_base_url, api_timeout):
    """API测试客户端"""

    class APIClient:
        def __init__(self, base_url, timeout):
            self.base_url = base_url
            self.timeout = timeout
            self.session = requests.Session()

        def get(self, endpoint, **kwargs):
            return self.session.get(
                f"{self.base_url}{endpoint}", timeout=self.timeout, **kwargs
            )

        def post(self, endpoint, **kwargs):
            return self.session.post(
                f"{self.base_url}{endpoint}", timeout=self.timeout, **kwargs
            )

        def put(self, endpoint, **kwargs):
            return self.session.put(
                f"{self.base_url}{endpoint}", timeout=self.timeout, **kwargs
            )

        def delete(self, endpoint, **kwargs):
            return self.session.delete(
                f"{self.base_url}{endpoint}", timeout=self.timeout, **kwargs
            )

        def close(self):
            self.session.close()

    client = APIClient(api_base_url, api_timeout)
    yield client
    client.close()


@pytest.fixture
def sample_predict_request():
    """示例预测请求数据"""
    return {"current_qps": 100.5, "include_confidence": True}


@pytest.fixture
def sample_trend_request():
    """示例趋势预测请求数据"""
    return {"hours_ahead": 12, "current_qps": 75.0}


@pytest.fixture
def sample_rca_incident():
    """示例RCA事件数据"""
    return {
        "affected_services": ["nginx", "mysql"],
        "symptoms": ["高CPU使用率", "内存泄漏", "响应超时"],
        "start_time": (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z",
        "end_time": datetime.utcnow().isoformat() + "Z",
    }


@pytest.fixture
def sample_assistant_query():
    """示例智能助手查询数据"""
    return {"question": "AI-CloudOps平台是什么？", "max_context_docs": 4}


@pytest.fixture
def sample_notification():
    """示例通知数据"""
    return {"title": "测试通知", "message": "这是一条测试通知消息", "type": "info"}


@pytest.fixture
def temp_log_dir():
    """临时日志目录"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    yield log_dir


@pytest.fixture
def test_result_collector():
    """测试结果收集器"""
    results = {"start_time": datetime.utcnow(), "tests": [], "summary": {}}

    def add_result(test_name, success, **kwargs):
        results["tests"].append(
            {
                "name": test_name,
                "success": success,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs,
            }
        )

    results["add"] = add_result
    yield results

    # 计算统计信息
    total_tests = len(results["tests"])
    passed_tests = sum(1 for test in results["tests"] if test["success"])
    failed_tests = total_tests - passed_tests

    results["summary"] = {
        "total": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
        "duration": (datetime.utcnow() - results["start_time"]).total_seconds(),
    }


@pytest.fixture(scope="session")
def service_health_checker(api_base_url):
    """服务健康检查器"""

    def check_service_health(service_name=""):
        try:
            # 统一检查对应服务的就绪接口
            if not service_name:
                endpoint = "/predict/ready"
            else:
                endpoint = f"/{service_name}/ready"
            response = requests.get(f"{api_base_url}{endpoint}", timeout=10)
            return response.status_code in [200, 503]  # 503也表示服务在运行但未就绪
        except Exception:
            return False

    return check_service_health


@pytest.fixture
def skip_if_service_down(service_health_checker):
    """如果服务未运行则跳过测试"""

    def _skip_if_down(service_name=""):
        if not service_health_checker(service_name):
            pytest.skip(f"服务 {service_name or '主服务'} 未运行")

    return _skip_if_down


@pytest.fixture
def performance_monitor():
    """性能监控器"""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = []

        def start(self):
            self.start_time = time.time()

        def stop(self, operation_name=""):
            if self.start_time:
                duration = time.time() - self.start_time
                self.measurements.append(
                    {
                        "operation": operation_name,
                        "duration": duration,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                self.start_time = None
                return duration
            return 0

        def get_stats(self):
            if not self.measurements:
                return {}

            durations = [m["duration"] for m in self.measurements]
            return {
                "count": len(durations),
                "total": sum(durations),
                "average": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "measurements": self.measurements,
            }

    return PerformanceMonitor()


def pytest_addoption(parser):
    """添加pytest命令行选项"""
    parser.addoption(
        "--api-host", action="store", default="localhost", help="API服务器主机地址"
    )
    parser.addoption("--api-port", action="store", default="8080", help="API服务器端口")
    parser.addoption(
        "--skip-integration", action="store_true", default=False, help="跳过集成测试"
    )
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="跳过耗时较长的测试"
    )


def pytest_configure(config):
    """pytest配置"""
    # 注册自定义标记
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "slow: 耗时较长的测试")
    config.addinivalue_line("markers", "api: API接口测试")
    config.addinivalue_line("markers", "unit: 单元测试")

    # 设置环境变量
    os.environ["API_HOST"] = config.getoption("--api-host")
    os.environ["API_PORT"] = config.getoption("--api-port")


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    skip_integration = pytest.mark.skip(reason="使用 --skip-integration 跳过")
    skip_slow = pytest.mark.skip(reason="使用 --skip-slow 跳过")

    # 移除不应被收集的辅助测试函数
    _remove_prediction_api_helper(items)

    for item in items:
        if "integration" in item.keywords and config.getoption("--skip-integration"):
            item.add_marker(skip_integration)
        if "slow" in item.keywords and config.getoption("--skip-slow"):
            item.add_marker(skip_slow)
