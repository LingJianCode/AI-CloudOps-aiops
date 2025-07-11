"""
AI-CloudOps 应用配置管理模块

该模块负责管理整个AI-CloudOps应用的配置，包括：
- 环境变量加载和处理
- 配置文件加载（支持多环境）
- 各组件的配置类定义
- 统一的配置获取接口

主要功能：
1. 自动检测并加载环境对应的配置文件
2. 支持环境变量覆盖配置文件的值
3. 提供类型安全的配置类
4. 统一的配置访问接口

使用方式：
from app.config.settings import config
prometheus_url = config.prometheus.url
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# 定义项目根目录
# 通过当前文件路径向上三级目录获取项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 加载环境变量文件(.env)
# 这将读取项目根目录下的.env文件中的环境变量
load_dotenv()

# 确定运行环境
# 优先从环境变量ENV获取，如果没有则默认为development
ENV = os.getenv("ENV", "development")

# 加载配置文件
def load_config() -> Dict[str, Any]:
    """
    加载配置文件，优先使用环境对应的配置，如果不存在则使用默认配置
    
    配置文件加载优先级：
    1. config.{ENV}.yaml（如config.production.yaml）
    2. config.yaml（默认配置文件）
    3. 空字典（如果没有找到配置文件）
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    # 构建环境特定的配置文件路径
    config_file = ROOT_DIR / "config" / f"config{'.' + ENV if ENV != 'development' else ''}.yaml"
    # 默认配置文件路径
    default_config_file = ROOT_DIR / "config" / "config.yaml"
    
    try:
        # 优先使用环境特定的配置文件
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        # 如果环境特定配置不存在，使用默认配置
        elif default_config_file.exists():
            with open(default_config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print(f"警告: 未找到配置文件 {config_file} 或 {default_config_file}，将使用环境变量默认值")
            return {}
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return {}

# 加载全局配置
# 这个全局变量在模块导入时就会被初始化
CONFIG = load_config()

def get_env_or_config(env_key, config_path, default=None, transform=None):
    """
    从环境变量或配置文件获取值，支持类型转换
    
    优先级：环境变量 > 配置文件 > 默认值
    
    Args:
        env_key (str): 环境变量名称
        config_path (str): 配置文件中的路径，使用点分隔符（如：'llm.model'）
        default: 默认值
        transform: 类型转换函数（如：int, float, bool）
    
    Returns:
        转换后的配置值
    """
    # 从配置文件中解析路径获取值
    parts = config_path.split('.')
    config_value = CONFIG
    for part in parts:
        config_value = config_value.get(part, {}) if isinstance(config_value, dict) else {}
    
    # 按优先级获取值：环境变量 > 配置文件 > 默认值
    value = os.getenv(env_key) or config_value or default
    
    # 如果指定了类型转换函数，进行类型转换
    if transform and value is not None:
        if transform == bool and isinstance(value, str):
            # 布尔类型的特殊处理，将字符串转换为布尔值
            return value.lower() == "true"
        return transform(value)
    return value

@dataclass
class PrometheusConfig:
    """
    Prometheus 监控系统配置类
    
    Prometheus 是一个开源的监控和报警工具，用于收集和存储时间序列数据
    该配置类定义了连接 Prometheus 服务器所需的参数
    """
    # Prometheus 服务器地址和端口
    host: str = get_env_or_config("PROMETHEUS_HOST", "prometheus.host", "127.0.0.1:9090")
    # 查询超时时间（秒）
    timeout: int = get_env_or_config("PROMETHEUS_TIMEOUT", "prometheus.timeout", 30, int)
    
    @property
    def url(self) -> str:
        """
        返回完整的 Prometheus URL
        
        Returns:
            str: 完整的 HTTP URL，格式为 http://host:port
        """
        return f"http://{self.host}"

@dataclass
class LLMConfig:
    """
    大语言模型（LLM）配置类
    
    该配置类管理与各种大语言模型提供商的连接参数，包括：
    - OpenAI 兼容的 API 服务
    - Ollama 本地模型服务
    - 模型参数配置
    """
    # LLM 提供商类型：openai、ollama 等
    provider: str = (get_env_or_config("LLM_PROVIDER", "llm.provider", "openai")).split('#')[0].strip()
    # 默认使用的模型名称
    model: str = get_env_or_config("LLM_MODEL", "llm.model", "Qwen/Qwen3-14B")
    # 用于特定任务的模型（如代码生成、推理等）
    task_model: str = get_env_or_config("LLM_TASK_MODEL", "llm.task_model", "Qwen/Qwen2.5-14B-Instruct")
    # API 密钥，用于身份验证
    api_key: str = get_env_or_config("LLM_API_KEY", "llm.api_key", "sk-xxx")
    # API 基础 URL
    base_url: str = get_env_or_config("LLM_BASE_URL", "llm.base_url", "https://api.siliconflow.cn/v1")
    # 模型温度参数，控制输出的随机性（0.0-2.0）
    temperature: float = get_env_or_config("LLM_TEMPERATURE", "llm.temperature", 0.7, float)
    # 最大输出 token 数量
    max_tokens: int = get_env_or_config("LLM_MAX_TOKENS", "llm.max_tokens", 2048, int)
    
    # Ollama 本地模型服务配置
    ollama_model: str = get_env_or_config("OLLAMA_MODEL", "llm.ollama_model", "qwen2.5:3b")
    ollama_base_url: str = get_env_or_config("OLLAMA_BASE_URL", "llm.ollama_base_url", "http://127.0.0.1:11434/v1")
    
    @property
    def effective_model(self) -> str:
        """
        根据提供商返回有效的模型名称
        
        Returns:
            str: 当前提供商对应的模型名称
        """
        if self.provider.lower() == "ollama":
            return self.ollama_model.split('#')[0].strip() if self.ollama_model else ""
        return self.model.split('#')[0].strip() if self.model else ""
    
    @property
    def effective_base_url(self) -> str:
        """
        根据提供商返回有效的基础URL
        
        Returns:
            str: 当前提供商对应的 API 基础 URL
        """
        if self.provider.lower() == "ollama":
            return self.ollama_base_url.split('#')[0].strip() if self.ollama_base_url else ""
        return self.base_url.split('#')[0].strip() if self.base_url else ""
    
    @property
    def effective_api_key(self) -> str:
        """
        根据提供商返回有效的API密钥
        
        Returns:
            str: 当前提供商对应的 API 密钥
        """
        return "ollama" if self.provider.lower() == "ollama" else self.api_key

@dataclass
class K8sConfig:
    """
    Kubernetes 集群配置类
    
    该配置类管理与 Kubernetes 集群的连接参数，包括：
    - 集群内部访问模式配置
    - kubeconfig 文件路径
    - 默认命名空间设置
    
    支持两种连接模式：
    1. 集群内模式：应用部署在 K8s 集群内，自动使用 ServiceAccount 认证
    2. 集群外模式：应用在集群外运行，需要指定 kubeconfig 文件路径
    """
    # 是否在集群内部运行（True：集群内模式，False：集群外模式）
    in_cluster: bool = get_env_or_config("K8S_IN_CLUSTER", "kubernetes.in_cluster", False, bool)
    # kubeconfig 配置文件路径，仅在集群外模式时使用
    config_path: Optional[str] = get_env_or_config("K8S_CONFIG_PATH", "kubernetes.config_path") or str(ROOT_DIR / "deploy/kubernetes/config")
    # 默认的 Kubernetes 命名空间
    namespace: str = get_env_or_config("K8S_NAMESPACE", "kubernetes.namespace", "default")

@dataclass
class RCAConfig:
    """
    根因分析（RCA - Root Cause Analysis）配置类
    
    该配置类管理根因分析模块的各种参数，包括：
    - 时间范围配置
    - 异常检测阈值
    - 相关性分析阈值
    - 默认监控指标列表
    
    RCA 模块用于自动分析系统异常的根本原因，通过监控指标的相关性分析
    和异常检测算法，帮助运维人员快速定位问题源头。
    """
    # 默认时间范围（分钟），用于分析历史数据
    default_time_range: int = get_env_or_config("RCA_DEFAULT_TIME_RANGE", "rca.default_time_range", 30, int)
    # 最大时间范围（分钟），限制分析的时间窗口上限
    max_time_range: int = get_env_or_config("RCA_MAX_TIME_RANGE", "rca.max_time_range", 1440, int)
    # 异常检测阈值（0.0-1.0），超过此值认为是异常
    anomaly_threshold: float = get_env_or_config("RCA_ANOMALY_THRESHOLD", "rca.anomaly_threshold", 0.65, float)
    # 相关性分析阈值（0.0-1.0），超过此值认为指标间存在强相关性
    correlation_threshold: float = get_env_or_config("RCA_CORRELATION_THRESHOLD", "rca.correlation_threshold", 0.7, float)
    
    # 默认监控指标列表，用于根因分析的核心指标
    default_metrics: List[str] = field(default_factory=lambda: CONFIG.get("rca", {}).get("default_metrics", [
        'container_cpu_usage_seconds_total',          # 容器 CPU 使用率
        'container_memory_working_set_bytes',         # 容器内存工作集
        'kube_pod_container_status_restarts_total',   # Pod 容器重启次数
        'kube_pod_status_phase',                      # Pod 状态阶段
        'node_cpu_seconds_total',                     # 节点 CPU 总使用时间
        'node_memory_MemFree_bytes',                  # 节点可用内存
        'kubelet_http_requests_duration_seconds_count', # kubelet HTTP 请求计数
        'kubelet_http_requests_duration_seconds_sum'    # kubelet HTTP 请求耗时总和
    ]))

@dataclass
class PredictionConfig:
    """
    预测模型配置类
    
    该配置类管理 AI 预测模型的相关参数，包括：
    - 机器学习模型文件路径
    - 数据预处理器路径
    - 自动扩缩容实例数量限制
    - Prometheus 查询语句
    
    预测模块使用机器学习算法分析历史数据，预测未来的资源需求，
    为自动扩缩容提供智能决策支持。
    """
    # 预测模型文件路径（pkl 格式）
    model_path: str = get_env_or_config("PREDICTION_MODEL_PATH", "prediction.model_path", "data/models/time_qps_auto_scaling_model.pkl")
    # 数据标准化器文件路径（pkl 格式），用于数据预处理
    scaler_path: str = get_env_or_config("PREDICTION_SCALER_PATH", "prediction.scaler_path", "data/models/time_qps_auto_scaling_scaler.pkl")
    # 自动扩缩容的最大实例数量
    max_instances: int = get_env_or_config("PREDICTION_MAX_INSTANCES", "prediction.max_instances", 20, int)
    # 自动扩缩容的最小实例数量
    min_instances: int = get_env_or_config("PREDICTION_MIN_INSTANCES", "prediction.min_instances", 1, int)
    # Prometheus 查询语句，用于获取预测所需的指标数据
    prometheus_query: str = get_env_or_config("PREDICTION_PROMETHEUS_QUERY", "prediction.prometheus_query", 
        'rate(nginx_ingress_controller_nginx_process_requests_total{service="ingress-nginx-controller-metrics"}[10m])')

@dataclass
class NotificationConfig:
    """
    通知配置类
    
    该配置类管理系统通知功能的参数，包括：
    - 飞书 Webhook 配置
    - 通知功能开关
    
    通知模块负责在系统发生异常、预警或重要事件时，
    通过各种渠道（如飞书、钉钉、邮件等）通知相关人员。
    """
    # 飞书群组 Webhook URL，用于发送通知到飞书群
    feishu_webhook: str = get_env_or_config("FEISHU_WEBHOOK", "notification.feishu_webhook", "")
    # 是否启用通知功能
    enabled: bool = get_env_or_config("NOTIFICATION_ENABLED", "notification.enabled", True, bool)

@dataclass
class TavilyConfig:
    """
    Tavily 搜索引擎配置类
    
    该配置类管理 Tavily AI 搜索引擎的连接参数，包括：
    - API 密钥配置
    - 搜索结果数量限制
    
    Tavily 是一个专为 AI 代理和应用程序设计的搜索引擎，
    提供实时、准确的信息搜索服务，用于增强 AI 助手的知识获取能力。
    """
    # Tavily API 密钥，用于身份验证
    api_key: str = get_env_or_config("TAVILY_API_KEY", "tavily.api_key", "")
    # 搜索结果的最大数量
    max_results: int = get_env_or_config("TAVILY_MAX_RESULTS", "tavily.max_results", 5, int)

@dataclass
class RAGConfig:
    """
    RAG（检索增强生成）智能助手配置类
    
    该配置类管理 RAG 系统的各种参数，包括：
    - 向量数据库配置
    - 知识库路径和集合名称
    - 文档分块和检索参数
    - 嵌入模型配置
    - 生成模型参数
    
    RAG 系统结合了检索和生成技术，通过检索相关知识库内容
    来增强大语言模型的回答质量，提供更准确、更具体的答案。
    """
    # 向量数据库存储路径
    vector_db_path: str = get_env_or_config("RAG_VECTOR_DB_PATH", "rag.vector_db_path", "data/vector_db")
    # 向量数据库中的集合名称
    collection_name: str = get_env_or_config("RAG_COLLECTION_NAME", "rag.collection_name", "aiops-assistant")
    # 知识库文档存储路径
    knowledge_base_path: str = get_env_or_config("RAG_KNOWLEDGE_BASE_PATH", "rag.knowledge_base_path", "data/knowledge_base")
    # 文档分块大小（字符数）
    chunk_size: int = get_env_or_config("RAG_CHUNK_SIZE", "rag.chunk_size", 1000, int)
    # 文档分块重叠大小（字符数）
    chunk_overlap: int = get_env_or_config("RAG_CHUNK_OVERLAP", "rag.chunk_overlap", 200, int)
    # 检索时返回的最相似文档数量
    top_k: int = get_env_or_config("RAG_TOP_K", "rag.top_k", 4, int)
    # 相似度阈值（0.0-1.0），低于此值的文档将被过滤
    similarity_threshold: float = get_env_or_config("RAG_SIMILARITY_THRESHOLD", "rag.similarity_threshold", 0.7, float)
    # OpenAI 兼容的嵌入模型名称
    openai_embedding_model: str = get_env_or_config("RAG_OPENAI_EMBEDDING_MODEL", "rag.openai_embedding_model", "Pro/BAAI/bge-m3")
    # Ollama 本地嵌入模型名称
    ollama_embedding_model: str = get_env_or_config("RAG_OLLAMA_EMBEDDING_MODEL", "rag.ollama_embedding_model", "nomic-embed-text")
    # 上下文最大长度（token 数）
    max_context_length: int = get_env_or_config("RAG_MAX_CONTEXT_LENGTH", "rag.max_context_length", 4000, int)
    # 生成温度参数，控制输出的随机性（0.0-2.0）
    temperature: float = get_env_or_config("RAG_TEMPERATURE", "rag.temperature", 0.1, float)

    @property
    def effective_embedding_model(self) -> str:
        """
        根据 LLM 提供商返回有效的嵌入模型
        
        该方法根据当前配置的 LLM 提供商自动选择对应的嵌入模型：
        - 如果使用 Ollama，返回 ollama_embedding_model
        - 如果使用 OpenAI 兼容服务，返回 openai_embedding_model
        
        Returns:
            str: 当前有效的嵌入模型名称
        """
        llm_provider = get_env_or_config("LLM_PROVIDER", "llm.provider", "openai").lower()
        return self.ollama_embedding_model if llm_provider == "ollama" else self.openai_embedding_model

@dataclass
class AppConfig:
    """
    应用程序主配置类
    
    该配置类是整个应用程序的配置根节点，包含：
    - 应用基础配置（调试模式、服务地址、日志级别等）
    - 各子模块的配置实例
    
    通过统一的配置类管理所有模块的配置，提供：
    1. 类型安全的配置访问
    2. 环境变量和配置文件的统一管理
    3. 配置的集中式管理和验证
    
    使用方式：
    from app.config.settings import config
    debug_mode = config.debug
    prometheus_url = config.prometheus.url
    """
    # 是否启用调试模式
    debug: bool = get_env_or_config("DEBUG", "app.debug", False, bool)
    # 服务绑定的主机地址
    host: str = get_env_or_config("HOST", "app.host", "0.0.0.0")
    # 服务监听端口
    port: int = get_env_or_config("PORT", "app.port", 8080, int)
    # 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
    log_level: str = get_env_or_config("LOG_LEVEL", "app.log_level", "INFO")

    # Prometheus 监控配置
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    # 大语言模型配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Kubernetes 集群配置
    k8s: K8sConfig = field(default_factory=K8sConfig)
    # 根因分析配置
    rca: RCAConfig = field(default_factory=RCAConfig)
    # 预测模型配置
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    # 通知服务配置
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    # Tavily 搜索引擎配置
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    # RAG 智能助手配置
    rag: RAGConfig = field(default_factory=RAGConfig)

# 全局配置实例
# 这是整个应用程序的统一配置入口点，在应用启动时自动初始化
# 使用方式：from app.config.settings import config
config = AppConfig()