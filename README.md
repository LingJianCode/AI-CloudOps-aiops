# AIOps Platform - 智能云原生运维平台

## 📋 项目概述

AIOps Platform 是一个企业级智能云原生运维平台，基于人工智能技术提供全方位的运维自动化解决方案。平台整合了机器学习、大语言模型、向量检索和工具调用等先进技术，为现代化运维团队提供智能化、自动化的运维管理能力。

### 🎯 核心功能

1. **智能负载预测** - 基于时间序列分析和机器学习的QPS预测与实例数建议
2. **根因分析（RCA）** - 多数据源智能根因分析，整合指标、事件和日志进行深度分析
3. **自动化修复** - Kubernetes资源的智能诊断、修复和优化建议
4. **企业级智能助手** - 支持RAG+MCP双模式架构的智能运维问答系统
5. **健康检查** - 多组件系统健康监控和依赖关系检测
6. **实时监控** - 与Prometheus深度集成的实时指标监控和告警

### 🏗️ 技术架构

```
AIOps Platform 企业级架构
┌─────────────────────────────────────────────────────────────┐
│                     接口与协议层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  REST API   │  │     MCP     │  │   WebSocket │          │
│  │   (FastAPI) │  │  Tool Calls │  │     SSE     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                     智能代理层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │企业级助手    │  │  自动修复    │  │  根因分析    │          │
│  │(RAG+MCP)   │  │ (K8s Fixer) │  │ (RCA Engine)│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  负载预测    │  │  健康监控    │  │  通知管理    │          │
│  │(ML Predictor)│  │(Health Mgr) │  │ (Notifier)  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                     数据与存储层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │Redis向量库   │  │时序数据库    │  │  缓存管理    │          │
│  │(Vector+KV)  │  │(Prometheus) │  │(Redis Cache)│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                     基础设施层                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Kubernetes  │  │    Docker   │  │   云原生     │          │
│  │   集群管理   │  │   容器化     │  │   基础设施   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
Ai-CloudOps-aiops/
├── app/                          # 应用主目录
│   ├── __init__.py              # 应用初始化
│   ├── main.py                  # FastAPI应用入口
│   ├── api/                     # API接口层
│   │   ├── decorators.py        # API装饰器
│   │   ├── middleware/          # 中间件
│   │   │   ├── cors.py          # CORS处理
│   │   │   └── error_handler.py # 全局错误处理
│   │   └── routes/              # API路由模块
│   │       ├── assistant.py     # 企业级智能助手API
│   │       ├── autofix.py       # K8s自动修复API
│   │       ├── health.py        # 系统健康检查API
│   │       ├── predict.py       # 负载预测API
│   │       └── rca.py           # 根因分析API
│   ├── common/                  # 通用模块
│   │   ├── constants.py         # 系统常量定义
│   │   ├── exceptions.py        # 自定义异常
│   │   └── response.py          # 响应包装器
│   ├── config/                  # 配置管理
│   │   ├── logging.py           # 日志配置
│   │   └── settings.py          # 应用配置
│   ├── core/                    # 核心业务逻辑
│   │   ├── agents/              # 智能代理系统
│   │   │   ├── enterprise_assistant.py  # 企业级RAG助手
│   │   │   ├── fallback_models.py       # 备用模型管理
│   │   │   ├── k8s_fixer.py            # K8s自动修复代理
│   │   │   ├── notifier.py             # 通知代理
│   │   │   └── supervisor.py           # 主管代理
│   │   ├── cache/               # 缓存管理
│   │   │   └── redis_cache_manager.py  # Redis缓存管理
│   │   ├── prediction/          # 智能预测模块
│   │   │   ├── model_loader.py  # ML模型加载器
│   │   │   └── predictor.py     # 负载预测引擎
│   │   ├── rca/                 # 根因分析引擎
│   │   │   ├── base_collector.py    # 基础数据收集器
│   │   │   ├── events_collector.py  # K8s事件收集器
│   │   │   ├── logs_collector.py    # 日志收集器
│   │   │   ├── metrics_collector.py # 指标收集器
│   │   │   └── rca_engine.py       # RCA分析引擎
│   │   └── vector/              # 向量数据库
│   │       └── redis_vector_store.py   # Redis向量存储
│   ├── mcp/                     # MCP工具调用系统
│   │   ├── main.py              # MCP服务器入口
│   │   ├── mcp_client.py        # MCP客户端
│   │   ├── server/              # MCP服务器实现
│   │   │   ├── main.py          # 服务器主程序
│   │   │   ├── mcp_server.py    # MCP协议实现
│   │   │   └── tools/           # 工具集合
│   │   │       ├── k8s_*.py     # Kubernetes工具集
│   │   │       ├── system_info_tool.py  # 系统信息工具
│   │   │       └── time_tool.py         # 时间工具
│   │   └── tests/               # MCP测试
│   ├── models/                  # 数据模型
│   │   ├── data_models.py       # 核心数据模型
│   │   ├── rca_models.py        # RCA专用模型
│   │   ├── request_models.py    # API请求模型
│   │   └── response_models.py   # API响应模型
│   ├── services/                # 业务服务层
│   │   ├── assistant_service.py # 智能助手服务
│   │   ├── autofix_service.py   # 自动修复服务
│   │   ├── health_service.py    # 健康检查服务
│   │   ├── kubernetes.py        # K8s集成服务
│   │   ├── llm.py               # LLM模型服务
│   │   ├── mcp_service.py       # MCP集成服务
│   │   ├── prediction_service.py # 预测服务
│   │   ├── prometheus.py        # Prometheus集成
│   │   └── startup.py           # 启动服务管理
│   └── utils/                   # 工具类
│       ├── error_handlers.py    # 错误处理工具
│       ├── time_utils.py        # 时间处理工具
│       └── validators.py        # 数据验证工具
├── config/                      # 配置文件目录
│   ├── config.yaml              # 开发环境配置
│   ├── config.production.yaml   # 生产环境配置
│   └── config.test.yaml         # 测试环境配置
├── data/                        # 数据目录
│   ├── knowledge_base/          # RAG知识库
│   ├── models/                  # ML模型文件
│   └── sample/                  # 示例配置文件
├── deploy/                      # 部署配置
│   ├── kubernetes/              # K8s部署文件
│   ├── predict_operator/        # 预测Operator
│   ├── grafana/                 # Grafana配置
│   └── prometheus/              # Prometheus配置
├── docs/                        # 项目文档
├── scripts/                     # 运维脚本
├── tests/                       # 测试文件
├── requirements.txt             # Python依赖
├── docker-compose.yml           # Docker编排
└── Dockerfile                   # Docker镜像构建
```

## 🚀 快速开始

### 环境要求

- **Python 3.8+** - 核心运行环境
- **Docker & Docker Compose** - 容器化部署
- **Redis 6.0+** - 向量存储和缓存
- **Kubernetes (可选)** - 集群管理和自动修复功能
- **Prometheus (推荐)** - 监控指标收集
- **Node.js 16+ (MCP功能需要)** - MCP工具调用支持

### 安装步骤

1. **克隆项目**

```bash
git clone 'https://github.com/GoSimplicity/AI-CloudOps.git'
cd Ai-CloudOps-aiops
```

2. **安装Python依赖**

```bash
pip install -r requirements.txt
```

3. **配置环境变量**

```bash
cp env.example .env
# 编辑 .env 文件，配置以下关键变量：
# - OPENAI_API_KEY 或 OLLAMA_BASE_URL (LLM服务)
# - REDIS_HOST, REDIS_PORT, REDIS_PASSWORD (Redis配置)
# - PROMETHEUS_URL (监控集成)
```

4. **启动Redis服务**

```bash
# 使用Docker启动Redis
docker run -d --name redis-aiops -p 6379:6379 redis:latest

# 或使用docker-compose
docker-compose up redis -d
```

5. **启动主服务**

```bash
# 开发环境
python app/main.py

# 生产环境
python app/main.py --env production

# 或使用启动脚本
bash scripts/start.sh
```

6. **启动MCP服务（可选，用于工具调用功能）**

```bash
# 在新终端窗口启动MCP服务器
python -m app.mcp.main

# 后台启动
bash scripts/start_mcp.sh
```

### 配置说明

主要配置文件：`config/config.yaml`

核心配置项：
- **llm**: LLM模型配置 (OpenAI/Ollama)
- **redis**: Redis连接配置
- **rag**: RAG知识库配置
- **mcp**: MCP工具调用配置
- **prometheus**: 监控集成配置
- **kubernetes**: K8s集群配置

## 📊 核心模块详解

### 1. 智能负载预测 (Prediction)

**位置**: `app/core/prediction/`

**功能特性**:

- 基于时间序列的QPS预测分析
- 支持1-168小时（7天）的预测窗口
- 智能实例数量和资源配置建议
- 多维度置信度评估和趋势分析
- 考虑时间模式和周期性因素

**核心算法**:

- 时间序列分析和周期性模式识别
- 机器学习预测模型（线性回归）
- 负载峰值检测和异常识别
- 资源使用率优化计算

**API端点**:
- `POST /api/v1/predict/predict` - QPS预测
- `GET /api/v1/predict/trend` - 负载趋势分析
- `GET /api/v1/predict/models` - 模型信息

**使用示例**:

```python
from app.services.prediction_service import PredictionService

prediction_service = PredictionService()
result = await prediction_service.predict_instances(
    service_name="my-service",
    current_qps=100,
    hours=24,
    instance_cpu=1.0,
    instance_memory=2.0
)
```

### 2. 根因分析 (RCA)

**位置**: `app/core/rca/`

**功能特性**:

- 多数据源智能根因分析引擎
- 整合Prometheus指标、K8s事件、Pod日志
- 异常检测和相关性分析
- 快速问题诊断和智能建议
- 支持自定义分析规则和阈值

**核心组件**:

- **MetricsCollector**: Prometheus指标收集和异常检测
- **EventsCollector**: Kubernetes事件收集和模式分析
- **LogsCollector**: Pod日志收集和错误模式识别
- **RCAEngine**: 综合分析引擎和根因推理

**分析方法**:

- 统计异常检测和时间序列分析
- 事件关联和因果推理
- 日志模式匹配和错误分类
- 多维度相关性分析

**API端点**:
- `POST /api/v1/rca/analyze` - 综合根因分析
- `GET /api/v1/rca/metrics` - 获取所有可用的Prometheus指标
- `GET /api/v1/rca/health` - RCA服务健康检查
- `GET /api/v1/rca/quick-diagnosis` - 快速问题诊断
- `GET /api/v1/rca/event-patterns` - 事件模式分析
- `GET /api/v1/rca/error-summary` - 错误摘要

**使用示例**:

```python
from app.services.rca_service import RCAService

rca_service = RCAService()
result = await rca_service.analyze_root_cause(
    namespace="default",
    time_window_hours=1.0,
    metrics=["cpu_usage", "memory_usage"],
    severity_threshold=0.7
)
```

### 3. 企业级智能助手 (Assistant)

**位置**: `app/core/agents/enterprise_assistant.py`

**核心特性**:

- **双模式架构**: 支持RAG和MCP两种工作模式，互不干扰
- **企业级RAG**: 基于Redis向量存储的知识检索增强
- **MCP工具调用**: 支持Kubernetes操作、系统信息查询等工具
- **智能路由**: 自动识别用户意图，选择最适合的处理模式
- **多轮对话**: 支持会话状态管理和上下文理解

**技术架构**:

- **RAG模式**: 向量检索 + LLM生成，适用于知识问答
- **MCP模式**: 工具调用协议，适用于操作执行
- **LangGraph工作流**: 企业级工作流引擎
- **Redis向量存储**: 高性能向量检索和缓存
- **多级质量评估**: 响应质量监控和自动优化

**支持的工具类型**:

- Kubernetes集群操作（Pod、Service、Deployment等）
- 系统信息查询和监控
- 时间和计算相关工具
- 可扩展的自定义工具接口

**API端点**:
- `POST /api/v1/assistant/query` - 智能问答（支持mode参数）
- `GET /api/v1/assistant/session` - 会话管理
- `POST /api/v1/assistant/refresh` - 刷新知识库

**使用示例**:

```python
# RAG模式 - 知识问答
response = await post("/api/v1/assistant/query", {
    "question": "如何优化Kubernetes集群性能？",
    "mode": "rag",
    "session_id": "user123"
})

# MCP模式 - 工具调用
response = await post("/api/v1/assistant/query", {
    "question": "获取default命名空间下的Pod列表",
    "mode": "mcp", 
    "session_id": "user123"
})
```

### 4. 自动修复 (AutoFix)

**位置**: `app/core/agents/k8s_fixer.py`

**功能特性**:

- Kubernetes资源智能诊断和自动修复
- 多维度问题检测和分析
- 智能修复建议和风险评估
- 支持批量资源处理和安全修复
- 集成监控和日志分析能力

**核心能力**:

- **资源诊断**: Pod、Deployment、Service等资源状态分析
- **问题检测**: CPU/内存异常、镜像问题、配置错误等
- **智能修复**: 资源重启、配置调整、扩缩容建议
- **风险控制**: 修复前预检、回滚机制、安全限制

**修复类型**:

- Pod异常重启和资源调整
- Deployment副本数优化
- Service连通性修复
- 资源配额和限制调整
- 配置错误自动纠正

**API端点**:
- `POST /api/v1/autofix/fix` - 执行自动修复
- `POST /api/v1/autofix/diagnose` - 资源诊断
- `GET /api/v1/autofix/config` - 获取修复配置

**使用示例**:

```python
from app.services.autofix_service import AutoFixService

autofix_service = AutoFixService()
result = await autofix_service.fix_resources(
    namespace="default",
    resource_type="deployment",
    resource_name="my-app",
    timeout=300
)
```

### 5. 健康检查 (Health Management)

**位置**: `app/services/health_service.py`

**功能特性**:

- 多组件系统健康状态监控
- 依赖关系检测和状态聚合
- 实时健康指标收集和分析
- 启动就绪和存活性检查
- 详细的组件状态报告

**监控组件**:

- **LLM服务**: 模型响应时间和可用性
- **向量存储**: Redis连接和查询性能
- **Prometheus**: 监控系统连通性
- **Kubernetes**: 集群连接状态
- **缓存系统**: Redis缓存性能

**健康检查级别**:

- **Basic**: 基础组件可用性检查
- **Detail**: 详细性能指标和响应时间
- **Deep**: 深度依赖关系和功能测试

### 6. 监控集成 (Prometheus Integration)

**位置**: `app/services/prometheus.py`

**集成功能**:

- Prometheus指标查询和聚合
- 实时监控数据获取和处理
- 多维度指标分析和计算
- 时间序列数据处理
- 自定义查询语言支持

**支持的指标类型**:

- **系统指标**: CPU、内存、磁盘、网络使用率
- **应用指标**: QPS、响应时间、错误率
- **业务指标**: 用户活跃度、交易量等
- **Kubernetes指标**: Pod、节点、集群状态
- **自定义指标**: 业务特定的监控指标

## 🔧 API 接口文档

### 健康检查 API

#### 基础健康检查
```
GET /api/v1/health
```

#### 详细组件状态
```
GET /api/v1/health/detail
```

#### 依赖关系检查
```
GET /api/v1/health/dependencies
```

**响应示例**:

```json
{
  "code": 0,
  "message": "系统运行正常",
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z",
    "version": "1.0.0",
    "uptime": 3600.5,
    "components": {
      "llm_service": {
        "status": "healthy",
        "response_time": 0.12,
        "details": "OpenAI GPT-4 连接正常"
      },
      "vector_store": {
        "status": "healthy", 
        "response_time": 0.03,
        "details": "Redis向量存储运行正常"
      },
      "prometheus": {
        "status": "healthy",
        "response_time": 0.05,
        "details": "监控系统连接正常"
      },
      "kubernetes": {
        "status": "healthy",
        "response_time": 0.08,
        "details": "K8s集群连接正常"
      }
    }
  }
}
```

### 负载预测 API

#### QPS预测
```
POST /api/v1/predict/predict
Content-Type: application/json

{
  "service_name": "my-service",
  "current_qps": 100.0,
  "hours": 24,
  "instance_cpu": 1.0,
  "instance_memory": 2.0
}
```

#### 负载趋势分析
```
GET /api/v1/predict/trend?service_name=my-service&hours=48
```

#### 模型信息
```
GET /api/v1/predict/models
```

**响应示例**:

```json
{
  "code": 0,
  "message": "预测完成",
  "data": {
    "service_name": "my-service",
    "prediction_hours": 24,
    "current_qps": 100.0,
    "predictions": [
      {
        "hour": 1,
        "predicted_qps": 105.2,
        "confidence_score": 0.92,
        "peak_probability": 0.15
      },
      {
        "hour": 24, 
        "predicted_qps": 180.5,
        "confidence_score": 0.85,
        "peak_probability": 0.78
      }
    ],
    "recommendations": {
      "suggested_instances": 4,
      "cpu_recommendation": 1.2,
      "memory_recommendation": 2.5,
      "scale_up_time": "2024-01-01T14:00:00Z"
    },
    "analysis": {
      "max_predicted_qps": 180.5,
      "avg_predicted_qps": 142.8,
      "growth_rate": 0.805,
      "volatility": 0.23
    }
  }
}
```

### 根因分析 API

#### 综合根因分析
```
POST /api/v1/rca/analyze
Content-Type: application/json

{
  "namespace": "default",
  "time_window_hours": 1.0,
  "metrics": ["cpu_usage", "memory_usage", "disk_io"],
  "severity_threshold": 0.7,
  "include_logs": true,
  "include_events": true
}
```

#### 获取所有可用指标
```
GET /api/v1/rca/metrics
```

**响应示例**:

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "metrics": [
      "up",
      "node_cpu_seconds_total",
      "node_memory_MemTotal_bytes",
      "node_memory_MemAvailable_bytes",
      "node_load1",
      "node_load5",
      "node_load15",
      "kubernetes_pod_cpu_usage_seconds_total",
      "kubernetes_pod_memory_usage_bytes",
      "container_cpu_usage_seconds_total",
      "container_memory_usage_bytes",
      "container_memory_working_set_bytes",
      "kube_pod_status_phase",
      "kube_deployment_status_replicas",
      "prometheus_http_requests_total",
      "process_cpu_seconds_total",
      "process_resident_memory_bytes"
    ],
    "count": 17,
    "timestamp": "2024-01-01T10:00:00.123456"
  }
}
```

**根因分析响应示例**:

```json
{
  "code": 0,
  "message": "根因分析完成",
  "data": {
    "analysis_id": "rca-20240101-001",
    "namespace": "default",
    "time_window": "1.0小时",
    "analysis_timestamp": "2024-01-01T10:00:00Z",
    "root_causes": [
      {
        "cause_id": "cpu-spike-001",
        "cause_type": "resource_exhaustion",
        "title": "CPU使用率异常飙升",
        "description": "检测到多个Pod的CPU使用率在过去30分钟内持续超过80%",
        "confidence_score": 0.92,
        "severity": "high",
        "affected_resources": [
          {"type": "pod", "name": "my-app-5f7b8", "namespace": "default"},
          {"type": "pod", "name": "my-app-6c8d9", "namespace": "default"}
        ],
        "recommendations": [
          {
            "action": "scale_up",
            "description": "建议将Deployment副本数从2增加到4",
            "priority": "high",
            "estimated_impact": "解决当前CPU瓶颈问题"
          },
          {
            "action": "optimize_config", 
            "description": "建议调整CPU资源限制从1核增加到2核",
            "priority": "medium",
            "estimated_impact": "防止未来CPU限流"
          }
        ],
        "supporting_evidence": [
          {
            "type": "metric",
            "source": "prometheus",
            "description": "CPU使用率峰值达到95%",
            "timestamp": "2024-01-01T09:45:00Z"
          },
          {
            "type": "event",
            "source": "kubernetes", 
            "description": "Pod重启事件：OOMKilled",
            "timestamp": "2024-01-01T09:50:00Z"
          }
        ]
      }
    ],
    "correlations": [
      {
        "metric_pair": ["cpu_usage", "response_time"],
        "correlation_coefficient": 0.89,
        "strength": "强正相关",
        "description": "CPU使用率与响应时间呈强正相关关系"
      }
    ],
    "summary": {
      "total_issues_detected": 3,
      "high_priority_issues": 1,
      "medium_priority_issues": 2,
      "affected_resources_count": 5,
      "analysis_duration": 2.3
    }
  }
}
```

### 自动修复 API

#### 执行自动修复
```
POST /api/v1/autofix/fix
Content-Type: application/json

{
  "namespace": "default",
  "resource_type": "deployment", 
  "resource_name": "my-app",
  "timeout": 300
}
```

#### 资源诊断
```
POST /api/v1/autofix/diagnose
Content-Type: application/json

{
  "namespace": "default",
  "deployment": "my-app",
  "include_events": true,
  "include_logs": true,
  "log_lines": 50
}
```

#### 获取修复配置
```
GET /api/v1/autofix/config
```

**响应示例**:

```json
{
  "code": 0,
  "message": "修复完成",
  "data": {
    "fix_id": "autofix-20240101-001",
    "namespace": "default",
    "resource_type": "deployment",
    "resource_name": "my-app",
    "fix_timestamp": "2024-01-01T10:00:00Z",
    "issues_detected": [
      {
        "issue_type": "resource_limit",
        "severity": "medium",
        "description": "CPU资源限制过低，可能导致性能问题",
        "affected_pods": ["my-app-7d8f9", "my-app-8e9a0"]
      },
      {
        "issue_type": "replica_count",
        "severity": "high", 
        "description": "副本数量不足，存在单点故障风险",
        "current_replicas": 1,
        "recommended_replicas": 3
      }
    ],
    "actions_taken": [
      {
        "action_type": "scale_up",
        "description": "将副本数从1增加到3",
        "status": "completed",
        "execution_time": 1.2
      },
      {
        "action_type": "update_resources",
        "description": "调整CPU限制从0.5核增加到1核",
        "status": "completed", 
        "execution_time": 0.8
      }
    ],
    "verification": {
      "all_pods_running": true,
      "health_check_passed": true,
      "performance_improved": true
    },
    "summary": {
      "total_issues": 2,
      "issues_fixed": 2,
      "execution_duration": 2.5,
      "risk_level_before": "high",
      "risk_level_after": "low"
    }
  }
}
```

### 企业级智能助手 API

#### 智能问答（支持双模式）
```
POST /api/v1/assistant/query
Content-Type: application/json

{
  "question": "如何优化Kubernetes集群性能？",
  "mode": "rag",  // 或 "mcp" 
  "session_id": "user123",
  "max_context_docs": 5,
  "stream": false
}
```

#### 会话管理
```
GET /api/v1/assistant/session?session_id=user123
```

#### 刷新知识库
```
POST /api/v1/assistant/refresh
```

#### 服务配置
```
GET /api/v1/assistant/config
```

**RAG模式响应示例**:

```json
{
  "code": 0,
  "message": "查询成功",
  "data": {
    "answer": "基于您的生产环境高负载情况，我建议采取以下优化措施：\n\n1. **资源配置优化**\n   - 合理设置Pod的CPU和内存请求/限制\n   - 使用HPA（水平Pod自动扩缩容）根据负载自动调整副本数\n   - 配置VPA（垂直Pod自动扩缩容）优化资源分配\n\n2. **调度策略优化**\n   - 使用Pod反亲和性规则避免单点故障\n   - 配置节点亲和性实现合理的工作负载分布\n   - 设置优先级类确保关键应用优先调度...",
    "mode": "rag",
    "session_id": "user123",
    "sources": [
      {
        "title": "Kubernetes性能优化指南",
        "content": "性能优化是Kubernetes运维的关键环节...",
        "relevance_score": 0.94,
        "source_type": "knowledge_base",
        "file_path": "kubernetes_ops_guide.md"
      },
      {
        "title": "监控故障排查手册",
        "content": "在高负载场景下，需要重点关注以下指标...",
        "relevance_score": 0.89,
        "source_type": "knowledge_base", 
        "file_path": "monitoring_troubleshooting_manual.md"
      }
    ],
    "suggestions": [
      "检查资源配额设置",
      "优化Pod调度策略",
      "配置HPA自动扩缩容",
      "设置监控告警规则",
      "定期进行性能测试"
    ],
    "metadata": {
      "processing_time": 1.24,
      "vector_search_time": 0.15,
      "llm_generation_time": 1.09,
      "context_docs_used": 3,
      "quality_score": 0.91
    }
  }
}
```

**MCP模式响应示例**:

```json
{
  "code": 0,
  "message": "查询成功",
  "data": {
    "answer": "已成功获取default命名空间下的Pod列表：\n\n**运行中的Pod (3个)**:\n1. **my-app-5f7b8** (Running) - CPU: 0.2/1.0, Memory: 512Mi/1Gi\n2. **nginx-deployment-6c8d9** (Running) - CPU: 0.1/0.5, Memory: 128Mi/512Mi\n3. **redis-master-abc123** (Running) - CPU: 0.3/1.0, Memory: 256Mi/512Mi\n\n**待调度的Pod (1个)**:\n4. **worker-job-xyz789** (Pending) - 等待调度到合适的节点\n\n**总结**: 集群整体运行稳定，资源利用率合理。建议关注待调度的worker-job Pod，可能需要检查节点资源或调度策略。",
    "mode": "mcp",
    "session_id": "user123",
    "tool_calls": [
      {
        "tool_name": "k8s_pod_tool",
        "function": "list_pods",
        "parameters": {"namespace": "default"},
        "execution_time": 0.45,
        "status": "success",
        "result": {
          "pods": [
            {
              "name": "my-app-5f7b8",
              "status": "Running",
              "cpu_usage": "0.2",
              "memory_usage": "512Mi"
            }
          ]
        }
      }
    ],
    "metadata": {
      "processing_time": 0.68,
      "tool_execution_time": 0.45,
      "response_generation_time": 0.23,
      "tools_used": 1
    }
  }
}
```

### WebSocket 流式 API

```
WS /api/v1/assistant/stream
```

**消息格式**:

```json
{
  "type": "query",
  "data": {
    "query": "用户问题",
    "session_id": "unique-session-id"
  }
}
```

## 🛠️ 开发指南

### 代码规范

1. **命名规范**

   - 使用 Python PEP 8 标准
   - 类名使用驼峰命名法
   - 函数和变量使用下划线命名法
   - 常量使用大写字母和下划线

2. **文档规范**

   - 所有模块、类、函数都需要 docstring
   - 使用中文注释说明复杂逻辑
   - 参数和返回值需要类型注解

3. **错误处理**
   - 使用自定义异常类
   - 记录详细的错误日志
   - 提供有意义的错误消息

### 测试规范

1. **单元测试**

   - 测试文件放在 `tests/` 目录
   - 使用 pytest 作为测试框架
   - 测试覆盖率要求 > 80%

2. **集成测试**

   - API 接口测试
   - 数据库连接测试
   - 外部服务集成测试

3. **性能测试**
   - 负载测试
   - 压力测试
   - 内存使用测试

### 部署指南

1. **本地部署**

```bash
# 启动开发环境
python app/main.py
```

2. **Kubernetes 部署**

```bash
# TODO: 待实现
```

3. **生产部署**

```bash
# 设置环境变量
export ENV=production

# 启动生产服务
python app/main.py
```

## 📈 性能优化

### 1. 系统性能

- **异步处理**: 使用 asyncio 处理 I/O 密集操作
- **连接池**: 数据库和 HTTP 连接池管理
- **缓存策略**: 多级缓存提升响应速度
- **负载均衡**: 支持水平扩展

### 2. 内存优化

- **对象池**: 复用大对象减少 GC 压力
- **流式处理**: 大数据集分批处理
- **内存监控**: 实时监控内存使用情况

### 3. 网络优化

- **压缩传输**: 启用 gzip 压缩
- **长连接**: 复用 HTTP 连接
- **CDN 加速**: 静态资源 CDN 分发

## 🔒 安全说明

### 1. 数据安全

- **加密存储**: 敏感数据加密存储
- **传输加密**: HTTPS/TLS 加密传输
- **访问控制**: 基于角色的访问控制

### 2. API 安全

- **身份验证**: JWT 令牌认证
- **授权控制**: 细粒度权限控制
- **限流保护**: API 请求限流

### 3. 系统安全

- **输入验证**: 严格的输入参数验证
- **SQL 注入防护**: 使用参数化查询
- **XSS 防护**: 输出数据转义

## 📝 更新日志

### v1.0.0 (2025-07-11)

- 初始版本发布
- 完整的 AI-CloudOps 功能实现
- 支持多种部署方式
- 完善的 API 文档和使用指南

### 规划中的功能

- [ ] 图形化 Dashboard
- [ ] 更多云平台支持
- [ ] 增强的 AI 分析能力
- [ ] 移动端应用支持

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📞 联系我们

- 项目主页: [https://github.com/GoSimplicity/AI-CloudOps]
- 问题报告: [https://github.com/GoSimplicity/AI-CloudOps/issues]
- 邮件联系: [13664854532@163.com]

## 📄 许可证

本项目采用 MIT 许可证，详情请参阅 [LICENSE](LICENSE) 文件。

---

_本文档最后更新: 2025-07-11_
_版本: 1.0.0_
