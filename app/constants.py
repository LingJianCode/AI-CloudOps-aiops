#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 常量定义模块，包含系统中使用的各种常量和配置
"""

# LLM 服务常量
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 3
OPENAI_TEST_MAX_TOKENS = 5
LLM_CONFIDENCE_THRESHOLD = 0.1
LLM_TEMPERATURE_MIN = 0.0
LLM_TEMPERATURE_MAX = 2.0

# 负载预测常量
LOW_QPS_THRESHOLD = 5.0
MAX_PREDICTION_HOURS = 168  # 7 天
DEFAULT_PREDICTION_HOURS = 24
PREDICTION_VARIATION_FACTOR = 0.1  # 10% 波动

# QPS置信度阈值
QPS_CONFIDENCE_THRESHOLDS = {"low": 100, "medium": 500, "high": 1000, "very_high": 2000}

# 时间模式常量
HOUR_FACTORS = {
    0: 0.3,
    1: 0.2,
    2: 0.15,
    3: 0.1,
    4: 0.1,
    5: 0.2,
    6: 0.4,
    7: 0.6,
    8: 0.8,
    9: 0.9,
    10: 1.0,
    11: 0.95,
    12: 0.9,
    13: 0.95,
    14: 1.0,
    15: 1.0,
    16: 0.95,
    17: 0.9,
    18: 0.8,
    19: 0.7,
    20: 0.6,
    21: 0.5,
    22: 0.4,
    23: 0.3,
}

DAY_FACTORS = {
    0: 0.95,  # 周一
    1: 1.0,  # 周二
    2: 1.05,  # 周三
    3: 1.05,  # 周四
    4: 0.95,  # 周五
    5: 0.7,  # 周六
    6: 0.6,  # 周日
}

# RAG 助手常量
DEFAULT_TOP_K = 4
MAX_CONTEXT_LENGTH = 4000
SIMILARITY_THRESHOLD = 0.1
MAX_HISTORY_LENGTH = 20
HALLUCINATION_COVERAGE_THRESHOLD = 0.3
DEFAULT_MAX_CONTEXT_DOCS = 6
MIN_RELEVANCE_SCORE = 0.6

# 助手性能常量
ASSISTANT_PERFORMANCE_METRICS_MAX_SAMPLES = 100
ASSISTANT_DEFAULT_VECTOR_DIM = 1536
ASSISTANT_INIT_SLEEP_INTERVAL = 0.1
ASSISTANT_MANAGER_WAIT_CYCLES = 20
ASSISTANT_MANAGER_SLEEP_INTERVAL = 0.5
ASSISTANT_SESSION_HISTORY_MAX_LENGTH = 4
ASSISTANT_ANSWER_TRUNCATE_LENGTH = 50
ASSISTANT_RELEVANCE_CACHE_THRESHOLD = 0.7
ASSISTANT_CACHE_TTL_SECONDS = 1800
ASSISTANT_DEFAULT_MAX_CONTEXT_DOCS = 1

# 缓存配置常量
ASSISTANT_CACHE_DB_OFFSET = 1
ASSISTANT_CACHE_DEFAULT_TTL = 3600
ASSISTANT_CACHE_MAX_SIZE = 1000

# 向量数据库常量
VECTOR_DB_COLLECTION_NAME = "aiops_knowledge"
EMBEDDING_BATCH_SIZE = 50
VECTOR_SEARCH_TIMEOUT = 30

# ==================== RCA 分析常量 ====================
# 异常检测阈值
RCA_ANOMALY_THRESHOLD = 0.65
# 相关性分析阈值
RCA_CORRELATION_THRESHOLD = 0.6
# 最大候选根因数量
RCA_MAX_CANDIDATES = 10
# 最小置信度
RCA_MIN_CONFIDENCE = 0.5
# 历史数据回溯天数
RCA_HISTORICAL_LOOKBACK_DAYS = 30

# 异常检测算法常量
# Z-Score异常检测阈值
Z_SCORE_THRESHOLD = 2.5
# 孤立森林算法的污染率
ISOLATION_FOREST_CONTAMINATION = 0.1
# DBSCAN聚类算法的邻域半径
DBSCAN_EPS = 0.5
# DBSCAN聚类算法的最小样本数
DBSCAN_MIN_SAMPLES = 5

# ==================== API 响应常量 ====================
# API默认分页大小
API_DEFAULT_PAGE_SIZE = 20
# API最大分页大小
API_MAX_PAGE_SIZE = 100
# API请求超时时间（秒）
API_REQUEST_TIMEOUT = 30
# 限流：每分钟最大请求数
API_RATE_LIMIT_REQUESTS = 100
# 限流：时间窗口（秒）
API_RATE_LIMIT_WINDOW = 60  # 秒

# HTTP 状态码
HTTP_STATUS_OK = 200
HTTP_STATUS_CREATED = 201
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_UNAUTHORIZED = 401
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_INTERNAL_ERROR = 500

# ==================== Kubernetes 自动修复常量 ====================
# 最大副本数
K8S_MAX_REPLICAS = 50
# 最小副本数
K8S_MIN_REPLICAS = 1
# 默认副本数
K8S_DEFAULT_REPLICAS = 3
# 扩容阈值
K8S_SCALE_UP_THRESHOLD = 0.8
# 缩容阈值
K8S_SCALE_DOWN_THRESHOLD = 0.3
# 冷却期（秒）
K8S_COOLDOWN_PERIOD = 300  # 5 分钟

# Pod 健康检查常量
# 检查超时时间（秒） - 仅保留使用的常量
DEFAULT_TIMEOUT_SECONDS = 5

# ==================== 监控和告警常量 ====================
# Prometheus查询超时时间（秒）
PROMETHEUS_QUERY_TIMEOUT = 30
# Prometheus最大数据点数
PROMETHEUS_MAX_POINTS = 11000
# Prometheus默认步长
PROMETHEUS_DEFAULT_STEP = "1m"

# 健康检查所需组件列表
REQUIRED_HEALTH_COMPONENTS = [
    "prometheus",  # Prometheus监控
    "llm",  # 大语言模型
    "vector_store",  # 向量存储
    "prediction",  # 预测模型
]

# ==================== 日志常量 ====================
# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# 日志文件最大字节数
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
# 日志文件备份数量
LOG_BACKUP_COUNT = 5

# 日志级别映射
LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

# ==================== 通知系统常量 ====================
# 通知重试次数
NOTIFICATION_RETRY_ATTEMPTS = 3
# 通知重试延迟（秒）
NOTIFICATION_RETRY_DELAY = 5  # 秒
# 通知超时时间（秒）
NOTIFICATION_TIMEOUT = 10  # 秒

# 通知严重程度级别
NOTIFICATION_SEVERITY = {"low": "低", "medium": "中", "high": "高", "critical": "紧急"}

# ==================== 文件和路径常量 ====================
# 默认知识库路径
DEFAULT_KNOWLEDGE_BASE_PATH = "data/knowledge_base"
# 默认向量数据库路径
DEFAULT_VECTOR_DB_PATH = "data/vector_db"
# 默认模型文件路径
DEFAULT_MODELS_PATH = "data/models"
# 默认日志文件路径
DEFAULT_LOGS_PATH = "logs"
# 默认配置文件路径
DEFAULT_CONFIG_PATH = "config"

# 支持的文档格式列表
SUPPORTED_DOC_FORMATS = [".md", ".txt", ".pdf", ".csv", ".json", ".html", ".xml"]

# ==================== 性能和限制常量 ====================
# 最大并发请求数
MAX_CONCURRENT_REQUESTS = 100
# 最大内存使用量（MB）
MAX_MEMORY_USAGE_MB = 1024
# 最大文件大小（MB）
MAX_FILE_SIZE_MB = 100
# 最大批处理大小
MAX_BATCH_SIZE = 1000

# 缓存配置
# 缓存默认生存时间（秒）
CACHE_DEFAULT_TTL = 3600  # 1 小时
# 缓存最大条目数
CACHE_MAX_SIZE = 1000
# 缓存淘汰策略
CACHE_EVICTION_POLICY = "LRU"

# ==================== 安全常量 ====================
# 最大登录尝试次数
MAX_LOGIN_ATTEMPTS = 5
# 会话超时时间（秒）
SESSION_TIMEOUT = 3600  # 1 小时
# 密码最小长度
PASSWORD_MIN_LENGTH = 8
# 令牌过期时间（小时）
TOKEN_EXPIRY_HOURS = 24

# API 密钥长度限制
# 最小API密钥长度
MIN_API_KEY_LENGTH = 32
# 最大API密钥长度
MAX_API_KEY_LENGTH = 256

# ==================== 模型和算法常量 ====================
# 模型版本
MODEL_VERSION = "1.0"
# 模型重新训练间隔（天）
MODEL_RETRAIN_INTERVAL_DAYS = 7
# 模型准确率阈值
MODEL_ACCURACY_THRESHOLD = 0.8
# 模型置信度阈值
MODEL_CONFIDENCE_THRESHOLD = 0.7

# 特征工程常量
# 时间窗口大小（分钟）
TIME_WINDOW_MINUTES = 60
# 特征窗口大小（小时）
FEATURE_WINDOW_HOURS = 24
# 最大特征数量
MAX_FEATURE_COUNT = 50

# ==================== 环境和部署常量 ====================
# 支持的环境类型
ENVIRONMENTS = ["development", "staging", "production"]
# 默认环境
DEFAULT_ENVIRONMENT = "development"

# 资源配置建议
# 根据不同规模提供资源配置建议
RESOURCE_REQUIREMENTS = {
    "small": {"cpu": "2", "memory": "4Gi"},
    "medium": {"cpu": "4", "memory": "8Gi"},
    "large": {"cpu": "8", "memory": "16Gi"},
}

# ==================== 错误消息常量 ====================
# 错误消息字典
ERROR_MESSAGES = {
    "invalid_input": "输入参数无效",
    "service_unavailable": "服务暂时不可用",
    "timeout": "请求超时",
    "not_found": "请求的资源未找到",
    "unauthorized": "未授权访问",
    "rate_limited": "请求频率超限",
    "internal_error": "内部服务错误",
}

# 成功消息字典
SUCCESS_MESSAGES = {
    "operation_completed": "操作成功完成",
    "data_updated": "数据更新成功",
    "analysis_finished": "分析完成",
    "model_trained": "模型训练完成",
}
