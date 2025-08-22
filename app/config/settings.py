#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 配置管理模块
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
load_dotenv()
ENV = os.getenv("ENV", "development")


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_file = (
        ROOT_DIR / "config" / f"config{'.' + ENV if ENV != 'development' else ''}.yaml"
    )
    default_config_file = ROOT_DIR / "config" / "config.yaml"

    try:
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        elif default_config_file.exists():
            with open(default_config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print(
                f"警告: 未找到配置文件 {config_file} 或 {default_config_file}，将使用环境变量默认值"
            )
            return {}
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return {}


CONFIG = load_config()


def get_env_or_config(
    env_key: str, config_path: str, default: Any = None, transform: Any = None
) -> Any:
    """
    从环境变量或配置文件获取值

    Args:
        env_key: 环境变量键名
        config_path: 配置文件中的路径(使用点表示法)
        default: 默认值
        transform: 类型转换函数

    Returns:
        Any: 获取的配置值
    """
    parts = config_path.split(".")
    config_value = CONFIG
    for part in parts:
        config_value = (
            config_value.get(part, {}) if isinstance(config_value, dict) else {}
        )

    value = os.getenv(env_key) or config_value or default

    if transform and value is not None:
        if transform == bool and isinstance(value, str):
            return value.lower() == "true"
        return transform(value)
    return value


@dataclass
class PrometheusConfig:
    """Prometheus 监控系统配置"""

    host: str = field(
        default_factory=lambda: get_env_or_config(
            "PROMETHEUS_HOST", "prometheus.host", "127.0.0.1:9090"
        )
    )
    timeout: int = field(
        default_factory=lambda: get_env_or_config(
            "PROMETHEUS_TIMEOUT", "prometheus.timeout", 30, int
        )
    )

    @property
    def url(self) -> str:
        return f"http://{self.host}"


@dataclass
class LLMConfig:
    """大语言模型配置"""

    provider: str = field(
        default_factory=lambda: get_env_or_config(
            "LLM_PROVIDER", "llm.provider", "openai"
        )
        .split("#")[0]
        .strip()
    )
    model: str = field(
        default_factory=lambda: get_env_or_config(
            "LLM_MODEL", "llm.model", "Qwen/Qwen3-14B"
        )
    )
    task_model: str = field(
        default_factory=lambda: get_env_or_config(
            "LLM_TASK_MODEL", "llm.task_model", "Qwen/Qwen2.5-14B-Instruct"
        )
    )
    api_key: str = field(
        default_factory=lambda: get_env_or_config(
            "LLM_API_KEY", "llm.api_key", "sk-xxx"
        )
    )
    base_url: str = field(
        default_factory=lambda: get_env_or_config(
            "LLM_BASE_URL", "llm.base_url", "https://api.siliconflow.cn/v1"
        )
    )
    temperature: float = field(
        default_factory=lambda: get_env_or_config(
            "LLM_TEMPERATURE", "llm.temperature", 0.7, float
        )
    )
    max_tokens: int = field(
        default_factory=lambda: get_env_or_config(
            "LLM_MAX_TOKENS", "llm.max_tokens", 2048, int
        )
    )
    request_timeout: int = field(
        default_factory=lambda: get_env_or_config(
            "LLM_REQUEST_TIMEOUT", "llm.request_timeout", 360, int
        )
    )

    ollama_model: str = field(
        default_factory=lambda: get_env_or_config(
            "OLLAMA_MODEL", "llm.ollama_model", "qwen2.5:3b"
        )
    )
    ollama_base_url: str = field(
        default_factory=lambda: get_env_or_config(
            "OLLAMA_BASE_URL", "llm.ollama_base_url", "http://127.0.0.1:11434/v1"
        )
    )

    embedding_model: str = field(
        default_factory=lambda: get_env_or_config(
            "LLM_EMBEDDING_MODEL", "llm.embedding_model", "Pro/BAAI/bge-m3"
        )
    )

    @property
    def effective_model(self) -> str:
        if self.provider.lower() == "ollama":
            return self.ollama_model.split("#")[0].strip() if self.ollama_model else ""
        return self.model.split("#")[0].strip() if self.model else ""

    @property
    def effective_base_url(self) -> str:
        if self.provider.lower() == "ollama":
            return (
                self.ollama_base_url.split("#")[0].strip()
                if self.ollama_base_url
                else ""
            )
        return self.base_url.split("#")[0].strip() if self.base_url else ""

    @property
    def effective_api_key(self) -> str:
        return "ollama" if self.provider.lower() == "ollama" else self.api_key


@dataclass
class K8sConfig:
    """Kubernetes 集群配置"""

    in_cluster: bool = field(
        default_factory=lambda: get_env_or_config(
            "K8S_IN_CLUSTER", "kubernetes.in_cluster", False, bool
        )
    )
    config_path: Optional[str] = field(
        default_factory=lambda: get_env_or_config(
            "K8S_CONFIG_PATH", "kubernetes.config_path"
        )
        or str(ROOT_DIR / "deploy/kubernetes/config")
    )
    namespace: str = field(
        default_factory=lambda: get_env_or_config(
            "K8S_NAMESPACE", "kubernetes.namespace", "default"
        )
    )


@dataclass
class RCAConfig:
    """根因分析配置"""

    default_time_range: int = field(
        default_factory=lambda: get_env_or_config(
            "RCA_DEFAULT_TIME_RANGE", "rca.default_time_range", 30, int
        )
    )
    max_time_range: int = field(
        default_factory=lambda: get_env_or_config(
            "RCA_MAX_TIME_RANGE", "rca.max_time_range", 1440, int
        )
    )
    anomaly_threshold: float = field(
        default_factory=lambda: get_env_or_config(
            "RCA_ANOMALY_THRESHOLD", "rca.anomaly_threshold", 0.65, float
        )
    )
    correlation_threshold: float = field(
        default_factory=lambda: get_env_or_config(
            "RCA_CORRELATION_THRESHOLD", "rca.correlation_threshold", 0.7, float
        )
    )

    default_metrics: List[str] = field(
        default_factory=lambda: CONFIG.get("rca", {}).get(
            "default_metrics",
            [
                "container_cpu_usage_seconds_total",
                "container_memory_working_set_bytes",
                "kube_pod_container_status_restarts_total",
                "kube_pod_status_phase",
                "node_cpu_seconds_total",
                "node_memory_MemFree_bytes",
                "kubelet_http_requests_duration_seconds_count",
                "kubelet_http_requests_duration_seconds_sum",
            ],
        )
    )


@dataclass
class PredictionConfig:
    """预测模型配置"""

    # 模型路径配置
    model_base_path: str = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_MODEL_BASE_PATH", "prediction.model_base_path", "data/models"
        )
    )

    # 实例配置
    max_instances: int = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_MAX_INSTANCES", "prediction.max_instances", 20, int
        )
    )
    min_instances: int = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_MIN_INSTANCES", "prediction.min_instances", 1, int
        )
    )

    # Prometheus查询配置 - 使用node_exporter标准指标
    prometheus_query: str = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_PROMETHEUS_QUERY",
            "prediction.prometheus_query",
            'rate(node_network_receive_bytes_total{device!="lo"}[10m])',
        )
    )

    # 预测参数
    default_prediction_hours: int = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_DEFAULT_HOURS", "prediction.default_prediction_hours", 24, int
        )
    )
    max_prediction_hours: int = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_MAX_HOURS", "prediction.max_prediction_hours", 168, int
        )
    )
    min_prediction_hours: int = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_MIN_HOURS", "prediction.min_prediction_hours", 1, int
        )
    )
    default_granularity: str = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_DEFAULT_GRANULARITY", "prediction.default_granularity", "hour"
        )
    )
    default_target_utilization: float = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_DEFAULT_TARGET_UTILIZATION",
            "prediction.default_target_utilization",
            0.7,
            float,
        )
    )
    default_sensitivity: float = field(
        default_factory=lambda: get_env_or_config(
            "PREDICTION_DEFAULT_SENSITIVITY",
            "prediction.default_sensitivity",
            0.8,
            float,
        )
    )

    @property
    def model_paths(self) -> Dict[str, Dict[str, str]]:
        """获取模型路径配置"""
        return CONFIG.get("prediction", {}).get("model_paths", {})

    @property
    def scaling_thresholds(self) -> Dict[str, Dict[str, float]]:
        """获取扩缩容阈值配置"""
        return CONFIG.get("prediction", {}).get("scaling_thresholds", {})

    @property
    def cooldown_periods(self) -> Dict[str, int]:
        """获取冷却时间配置"""
        return CONFIG.get("prediction", {}).get("cooldown_periods", {})

    @property
    def cost_analysis_config(self) -> Dict[str, Any]:
        """获取成本分析配置"""
        return CONFIG.get("prediction", {}).get("cost_analysis", {})

    @property
    def anomaly_detection_config(self) -> Dict[str, Any]:
        """获取异常检测配置"""
        return CONFIG.get("prediction", {}).get("anomaly_detection", {})


@dataclass
class NotificationConfig:
    """通知配置"""

    feishu_webhook: str = field(
        default_factory=lambda: get_env_or_config(
            "FEISHU_WEBHOOK", "notification.feishu_webhook", ""
        )
    )
    enabled: bool = field(
        default_factory=lambda: get_env_or_config(
            "NOTIFICATION_ENABLED", "notification.enabled", True, bool
        )
    )


@dataclass
class TavilyConfig:
    """Tavily 搜索引擎配置"""

    api_key: str = field(
        default_factory=lambda: get_env_or_config(
            "TAVILY_API_KEY", "tavily.api_key", ""
        )
    )
    max_results: int = field(
        default_factory=lambda: get_env_or_config(
            "TAVILY_MAX_RESULTS", "tavily.max_results", 5, int
        )
    )


@dataclass
class RedisConfig:
    """Redis 向量存储配置"""

    host: str = field(
        default_factory=lambda: get_env_or_config(
            "REDIS_HOST", "redis.host", "localhost"
        )
    )
    port: int = field(
        default_factory=lambda: get_env_or_config("REDIS_PORT", "redis.port", 6379, int)
    )
    db: int = field(
        default_factory=lambda: get_env_or_config("REDIS_DB", "redis.db", 0, int)
    )
    password: str = field(
        default_factory=lambda: get_env_or_config(
            "REDIS_PASSWORD", "redis.password", ""
        )
    )
    connection_timeout: int = field(
        default_factory=lambda: get_env_or_config(
            "REDIS_CONNECTION_TIMEOUT", "redis.connection_timeout", 5, int
        )
    )
    socket_timeout: int = field(
        default_factory=lambda: get_env_or_config(
            "REDIS_SOCKET_TIMEOUT", "redis.socket_timeout", 5, int
        )
    )
    max_connections: int = field(
        default_factory=lambda: get_env_or_config(
            "REDIS_MAX_CONNECTIONS", "redis.max_connections", 10, int
        )
    )
    decode_responses: bool = field(
        default_factory=lambda: get_env_or_config(
            "REDIS_DECODE_RESPONSES", "redis.decode_responses", True, bool
        )
    )


@dataclass
class RAGConfig:
    """RAG 智能助手配置"""

    vector_db_path: str = field(
        default_factory=lambda: get_env_or_config(
            "RAG_VECTOR_DB_PATH", "rag.vector_db_path", "data/vector_db"
        )
    )
    collection_name: str = field(
        default_factory=lambda: get_env_or_config(
            "RAG_COLLECTION_NAME", "rag.collection_name", "aiops-assistant"
        )
    )
    knowledge_base_path: str = field(
        default_factory=lambda: get_env_or_config(
            "RAG_KNOWLEDGE_BASE_PATH", "rag.knowledge_base_path", "data/knowledge_base"
        )
    )
    chunk_size: int = field(
        default_factory=lambda: get_env_or_config(
            "RAG_CHUNK_SIZE", "rag.chunk_size", 1000, int
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: get_env_or_config(
            "RAG_CHUNK_OVERLAP", "rag.chunk_overlap", 200, int
        )
    )
    top_k: int = field(
        default_factory=lambda: get_env_or_config("RAG_TOP_K", "rag.top_k", 4, int)
    )
    similarity_threshold: float = field(
        default_factory=lambda: get_env_or_config(
            "RAG_SIMILARITY_THRESHOLD", "rag.similarity_threshold", 0.7, float
        )
    )
    openai_embedding_model: str = field(
        default_factory=lambda: get_env_or_config(
            "RAG_OPENAI_EMBEDDING_MODEL",
            "rag.openai_embedding_model",
            "Pro/BAAI/bge-m3",
        )
    )
    ollama_embedding_model: str = field(
        default_factory=lambda: get_env_or_config(
            "RAG_OLLAMA_EMBEDDING_MODEL",
            "rag.ollama_embedding_model",
            "nomic-embed-text",
        )
    )
    max_context_length: int = field(
        default_factory=lambda: get_env_or_config(
            "RAG_MAX_CONTEXT_LENGTH", "rag.max_context_length", 4000, int
        )
    )
    temperature: float = field(
        default_factory=lambda: get_env_or_config(
            "RAG_TEMPERATURE", "rag.temperature", 0.1, float
        )
    )
    timeout: int = field(
        default_factory=lambda: get_env_or_config(
            "RAG_TIMEOUT", "rag.timeout", 360, int
        )
    )
    max_docs_per_query: int = field(
        default_factory=lambda: get_env_or_config(
            "RAG_MAX_DOCS_PER_QUERY", "rag.max_docs_per_query", 10, int
        )
    )
    cache_expiry: int = field(
        default_factory=lambda: get_env_or_config(
            "RAG_CACHE_EXPIRY", "rag.cache_expiry", 3600, int
        )
    )

    @property
    def effective_embedding_model(self) -> str:
        llm_provider = get_env_or_config(
            "LLM_PROVIDER", "llm.provider", "openai"
        ).lower()
        return (
            self.ollama_embedding_model
            if llm_provider == "ollama"
            else self.openai_embedding_model
        )


@dataclass
class MCPConfig:
    """MCP配置"""

    server_url: str = field(
        default_factory=lambda: get_env_or_config(
            "MCP_SERVER_URL", "mcp.server_url", "http://localhost:9000"
        )
    )
    timeout: int = field(
        default_factory=lambda: get_env_or_config("MCP_TIMEOUT", "mcp.timeout", 30, int)
    )
    max_retries: int = field(
        default_factory=lambda: get_env_or_config(
            "MCP_MAX_RETRIES", "mcp.max_retries", 3, int
        )
    )
    health_check_interval: int = field(
        default_factory=lambda: get_env_or_config(
            "MCP_HEALTH_CHECK_INTERVAL", "mcp.health_check_interval", 30, int
        )
    )


@dataclass
class TimeConfig:
    """时间配置"""

    holidays: List[str] = field(
        default_factory=lambda: CONFIG.get("time", {}).get(
            "holidays",
            [
                "0101",
                "0102",
                "0103",  # 元旦
                "0210",
                "0211",
                "0212",
                "0213",
                "0214",
                "0215",
                "0216",
                "0217",  # 春节
                "0404",
                "0405",
                "0406",  # 清明节
                "0501",
                "0502",
                "0503",
                "0504",
                "0505",  # 劳动节
                "0610",
                "0611",
                "0612",  # 端午节
                "0917",
                "0918",
                "0919",  # 中秋节
                "1001",
                "1002",
                "1003",
                "1004",
                "1005",
                "1006",
                "1007",  # 国庆节
            ],
        )
    )


@dataclass
class AppConfig:
    """应用程序主配置类"""

    debug: bool = field(
        default_factory=lambda: get_env_or_config("DEBUG", "app.debug", False, bool)
    )
    host: str = field(
        default_factory=lambda: get_env_or_config("HOST", "app.host", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: get_env_or_config("PORT", "app.port", 8080, int)
    )
    log_level: str = field(
        default_factory=lambda: get_env_or_config("LOG_LEVEL", "app.log_level", "INFO")
    )

    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    k8s: K8sConfig = field(default_factory=K8sConfig)
    rca: RCAConfig = field(default_factory=RCAConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    time: TimeConfig = field(default_factory=TimeConfig)


config = AppConfig()
