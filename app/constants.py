"""
AI-CloudOps 应用常量定义

这个模块定义了整个应用中使用的常量，包括：
- 系统配置常量（超时时间、重试次数等）
- 各功能模块的阈值和参数
- API相关常量
- 错误和成功消息
- 资源限制和性能参数

常量的分类和命名遵循以下原则：
1. 使用大写字母和下划线命名
2. 按功能模块分组
3. 包含详细的注释说明
4. 提供合理的默认值

使用方式：
from app.constants import LLM_TIMEOUT_SECONDS, QPS_CONFIDENCE_THRESHOLDS
"""

# ==================== 时间相关常量 ====================
# 系统默认超时时间（秒）
DEFAULT_TIMEOUT_SECONDS = 30
# 最大重试次数
MAX_RETRIES = 3
# 缓存过期时间（秒）
CACHE_EXPIRY_SECONDS = 3600
# 健康检查间隔（秒）
HEALTH_CHECK_INTERVAL = 60

# ==================== LLM 服务常量 ====================
# LLM服务请求超时时间（秒）
LLM_TIMEOUT_SECONDS = 30
# LLM服务最大重试次数
LLM_MAX_RETRIES = 3
# OpenAI测试时的最大token数量
OPENAI_TEST_MAX_TOKENS = 5
# LLM响应置信度阈值
LLM_CONFIDENCE_THRESHOLD = 0.1
# 温度参数的最小值
LLM_TEMPERATURE_MIN = 0.0
# 温度参数的最大值
LLM_TEMPERATURE_MAX = 2.0

# ==================== 负载预测常量 ====================
# 低QPS阈值，用于判断系统负载水平
LOW_QPS_THRESHOLD = 5.0
# QPS变化的除数，用于计算负载变化率
QPS_CHANGE_DIVISOR = 1.0
# 最大预测时间范围（小时）
MAX_PREDICTION_HOURS = 168  # 7 天
# 默认预测时间范围（小时）
DEFAULT_PREDICTION_HOURS = 24
# 预测结果的波动系数
PREDICTION_VARIATION_FACTOR = 0.1  # 10% 波动

# QPS置信度阈值字典
# 根据不同的QPS水平定义置信度分类
QPS_CONFIDENCE_THRESHOLDS = {
    'low': 100,        # 低负载
    'medium': 500,     # 中等负载
    'high': 1000,      # 高负载
    'very_high': 2000  # 极高负载
}

# 时间模式常量
# 一天中不同小时的负载系数（0-23小时）
# 数值越大表示该时间段负载越高
HOUR_FACTORS = {
    0: 0.3, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.1, 5: 0.2,    # 凌晨时段，负载较低
    6: 0.4, 7: 0.6, 8: 0.8, 9: 0.9, 10: 1.0, 11: 0.95,  # 上午时段，负载逐渐上升
    12: 0.9, 13: 0.95, 14: 1.0, 15: 1.0, 16: 0.95, 17: 0.9,  # 下午时段，负载保持高位
    18: 0.8, 19: 0.7, 20: 0.6, 21: 0.5, 22: 0.4, 23: 0.3   # 晚上时段，负载逐渐下降
}

# 一周中不同日期的负载系数（0-6，周一到周日）
# 数值越大表示该日期负载越高
DAY_FACTORS = {
    0: 0.95,  # 周一
    1: 1.0,   # 周二
    2: 1.05,  # 周三
    3: 1.05,  # 周四
    4: 0.95,  # 周五
    5: 0.7,   # 周六
    6: 0.6    # 周日
}

# ==================== RAG 助手常量 ====================
# 默认的Top-K检索数量
DEFAULT_TOP_K = 4
# 最大上下文长度
MAX_CONTEXT_LENGTH = 4000
# 相似度阈值
SIMILARITY_THRESHOLD = 0.1
# 最大对话历史长度
MAX_HISTORY_LENGTH = 20
# 幻觉检测覆盖率阈值
HALLUCINATION_COVERAGE_THRESHOLD = 0.3
# 默认最大上下文文档数量
DEFAULT_MAX_CONTEXT_DOCS = 6
# 最小相关性得分
MIN_RELEVANCE_SCORE = 0.6

# 向量数据库常量
# 向量数据库集合名称
VECTOR_DB_COLLECTION_NAME = "aiops_knowledge"
# 嵌入处理的批量大小
EMBEDDING_BATCH_SIZE = 50
# 向量搜索超时时间（秒）
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
# 初始延迟时间（秒）
DEFAULT_INITIAL_DELAY_SECONDS = 30
# 检查周期（秒）
DEFAULT_PERIOD_SECONDS = 10
# 检查超时时间（秒）
DEFAULT_TIMEOUT_SECONDS = 5
# 失败阈值
DEFAULT_FAILURE_THRESHOLD = 3
# 成功阈值
DEFAULT_SUCCESS_THRESHOLD = 1

# 资源配置常量
# 默认CPU请求量
DEFAULT_CPU_REQUEST = "100m"
# 默认内存请求量
DEFAULT_MEMORY_REQUEST = "128Mi"
# 默认CPU限制
DEFAULT_CPU_LIMIT = "500m"
# 默认内存限制
DEFAULT_MEMORY_LIMIT = "512Mi"

# ==================== 监控和告警常量 ====================
# Prometheus查询超时时间（秒）
PROMETHEUS_QUERY_TIMEOUT = 30
# Prometheus最大数据点数
PROMETHEUS_MAX_POINTS = 11000
# Prometheus默认步长
PROMETHEUS_DEFAULT_STEP = "1m"

# 健康检查所需组件列表
REQUIRED_HEALTH_COMPONENTS = [
    "prometheus",    # Prometheus监控
    "llm",          # 大语言模型
    "vector_store", # 向量存储
    "prediction"    # 预测模型
]

# ==================== 日志常量 ====================
# 日志格式
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# 日志文件最大字节数
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
# 日志文件备份数量
LOG_BACKUP_COUNT = 5

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# ==================== 通知系统常量 ====================
# 通知重试次数
NOTIFICATION_RETRY_ATTEMPTS = 3
# 通知重试延迟（秒）
NOTIFICATION_RETRY_DELAY = 5  # 秒
# 通知超时时间（秒）
NOTIFICATION_TIMEOUT = 10  # 秒

# 通知严重程度级别
NOTIFICATION_SEVERITY = {
    'low': '低',
    'medium': '中',
    'high': '高',
    'critical': '紧急'
}

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
SUPPORTED_DOC_FORMATS = [
    '.md', '.txt', '.pdf', '.csv', '.json', '.html', '.xml'
]

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
ENVIRONMENTS = ['development', 'staging', 'production']
# 默认环境
DEFAULT_ENVIRONMENT = 'development'

# 资源配置建议
# 根据不同规模提供资源配置建议
RESOURCE_REQUIREMENTS = {
    'small': {'cpu': '2', 'memory': '4Gi'},
    'medium': {'cpu': '4', 'memory': '8Gi'},
    'large': {'cpu': '8', 'memory': '16Gi'}
}

# ==================== 错误消息常量 ====================
# 错误消息字典
ERROR_MESSAGES = {
    'invalid_input': '输入参数无效',
    'service_unavailable': '服务暂时不可用',
    'timeout': '请求超时',
    'not_found': '请求的资源未找到',
    'unauthorized': '未授权访问',
    'rate_limited': '请求频率超限',
    'internal_error': '内部服务错误'
}

# 成功消息字典
SUCCESS_MESSAGES = {
    'operation_completed': '操作成功完成',
    'data_updated': '数据更新成功',
    'analysis_finished': '分析完成',
    'model_trained': '模型训练完成'
}
