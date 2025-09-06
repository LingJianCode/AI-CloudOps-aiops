# AIOps Platform - 智能云原生运维平台

## 📋 项目概述

AIOps Platform 是一个**AI-CloudOps智能云原生运维平台**，基于人工智能技术提供全方位的运维自动化解决方案。平台整合了机器学习、大语言模型、向量检索和工具调用等先进技术，为现代化运维团队提供智能化、自动化的运维管理能力。

### ✨ 核心特性

- 🧠 **智能预测** - 多维度负载预测与资源优化建议
- 🔍 **根因分析** - AI驱动的多数据源故障诊断
- 🛠️ **自动修复** - Kubernetes资源智能诊断和自愈
- 🤖 **智能助手** - RAG+MCP双模式AI-CloudOps AI助手  
- 📊 **健康监控** - 全栈系统健康状态实时监控
- 🎯 **精准告警** - 智能告警和通知管理

### 🏗️ 技术架构

```text
AIOps Platform AI-CloudOps架构
┌─────────────────────────────────────────────────────────────┐
│                     接口与协议层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  REST API   │  │     MCP     │  │   WebSocket │          │
│  │   (FastAPI) │  │  Tool Calls │  │     SSE     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                     智能代理层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │AI-CloudOps助手│ │  自动修复    │  │  根因分析     │          │
│  │(RAG+MCP)    │  │ (K8s Fixer) │  │ (RCA Engine)│           │
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

```text
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
│   │       ├── assistant.py     # AI-CloudOps智能助手API
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
│   │   │   ├── enterprise_assistant.py  # AI-CloudOps RAG助手
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

#### 系统要求

- **操作系统**: Linux/macOS/Windows (推荐 Linux)
- **内存**: 最少 8GB RAM (推荐 16GB+)
- **存储**: 最少 20GB 可用空间
- **网络**: 能够访问外部API服务

#### 软件依赖

- **Python 3.11+** - 核心运行环境
- **Docker 20.10+** - 容器化部署
- **Docker Compose 2.0+** - 容器编排
- **Redis 7.0+** - 向量存储和缓存
- **Git** - 用于代码拉取
- **Kubernetes (可选)** - 集群管理和自动修复功能
- **Prometheus (推荐)** - 监控指标收集

### 🎯 一键部署（推荐）

1. **克隆项目**

```bash
git clone https://github.com/GoSimplicity/AI-CloudOps.git
cd Ai-CloudOps-aiops
```

1. **配置环境变量**

```bash
# 复制环境配置文件
cp env.example .env

# 编辑配置文件，至少需要配置以下必要参数：
nano .env
```

**核心环境变量配置**：

```bash
# 基础配置
ENV=production                    # 环境类型
DEBUG=false                      # 调试模式
HOST=0.0.0.0                     # 监听地址
PORT=8080                        # 主应用端口

# LLM配置（必需）
LLM_API_KEY=sk-your-api-key      # API密钥
LLM_BASE_URL=https://api.siliconflow.cn/v1  # API基础URL
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct         # 主模型

# K8s集群配置（可选）
K8S_IN_CLUSTER=false             # 是否在集群内运行
K8S_CONFIG_PATH=./deploy/kubernetes/config  # kubeconfig路径
K8S_NAMESPACE=default            # 默认命名空间

# 通知配置（可选）
FEISHU_WEBHOOK=https://your-webhook-url  # 飞书通知
TAVILY_API_KEY=your-tavily-key   # Tavily搜索API
REDIS_PASSWORD=your-redis-password  # Redis密码
```

1. **一键部署**

```bash
# 给部署脚本执行权限
chmod +x scripts/deploy.sh

# 执行部署
./scripts/deploy.sh

# 或者使用生产模式部署
./scripts/deploy.sh --production --health-check
```

1. **验证部署**

```bash
# 查看服务状态
./scripts/deploy.sh --status

# 查看日志
./scripts/deploy.sh --logs
```

### 🛠️ 手动部署

如果需要手动控制部署过程：

#### 1. 安装系统依赖

**Ubuntu/Debian**:

```bash
# 更新包列表
sudo apt update

# 安装Docker
sudo apt install -y docker.io docker-compose-plugin

# 安装其他工具
sudo apt install -y git curl

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 将用户添加到docker组
sudo usermod -aG docker $USER
```

**CentOS/RHEL**:

```bash
# 安装Docker
sudo yum install -y docker docker-compose

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 安装其他工具
sudo yum install -y git curl
```

**macOS**:

```bash
# 使用Homebrew安装
brew install docker docker-compose git curl

# 或者下载Docker Desktop
# https://www.docker.com/products/docker-desktop
```

#### 2. 构建和启动服务

```bash
# 构建主应用镜像
docker build -t aiops-platform:latest -f Dockerfile .

# 构建MCP服务镜像
docker build -t aiops-mcp:latest -f Dockerfile.mcp .

# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f aiops-platform
```

#### 3. 配置Kubernetes（可选）

如果需要在Kubernetes集群中管理资源：

```bash
# 方法1: 复制kubeconfig到项目目录
mkdir -p deploy/kubernetes
cp ~/.kube/config deploy/kubernetes/config

# 方法2: 设置环境变量指向kubeconfig路径
export K8S_CONFIG_PATH=/path/to/your/kubeconfig
```

### 📋 服务组件架构

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   主应用服务      │    │   MCP服务        │    │   Prometheus    │
│   (8080)        │◄──►│   (9000)        │    │   (9090)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │   Redis         │    │   Ollama        │
                    │   (6379)        │    │   (11434)       │
                    └─────────────────┘    └─────────────────┘
```

#### 核心服务

- **主应用服务** (aiops-platform): 提供API接口、根因分析、智能预测等核心功能
- **MCP服务** (aiops-mcp): 提供工具调用能力和SSE服务端
- **Redis**: 用于缓存和向量数据存储
- **Prometheus**: 监控数据收集和存储

- **Ollama**: 本地大语言模型服务

### 🔗 服务访问

部署完成后，可以通过以下地址访问各个服务：

| 服务       | 地址                   | 说明                      |
| ---------- | ---------------------- | ------------------------- |
| 主应用     | `http://localhost:8080`  | 主要API接口               |
| MCP服务    | `http://localhost:9000`  | 工具调用接口              |
| Prometheus | `http://localhost:9090`  | 监控数据查询              |
| Ollama     | `http://localhost:11434` | 本地模型API               |

#### API文档

- 主应用API文档: [http://localhost:8080/docs](http://localhost:8080/docs)
- MCP服务API文档: [http://localhost:9000/docs](http://localhost:9000/docs) (如果启用)
- OpenAPI 标签: prediction, assistant, rca, autofix, cache, health

### ✅ 健康检查

#### 自动健康检查

```bash
# 执行完整健康检查
./scripts/deploy.sh --health-check
```

#### 手动检查

```bash
# 检查主应用
curl http://localhost:8080/api/v1/health

# 检查MCP服务
curl http://localhost:9000/health

# 检查Prometheus
curl http://localhost:9090/-/healthy



# 检查Redis
docker exec aiops-redis redis-cli ping
```

### 📊 数据持久化

所有重要数据都会持久化到本地目录：

- `./data`: 应用数据、模型文件
- `./logs`: 日志文件
- `./config`: 配置文件

### 🔧 故障排除

#### 常见问题

#### 1) 服务无法启动

```bash
# 查看服务日志
docker-compose logs aiops-platform
docker-compose logs aiops-mcp

# 检查端口占用
netstat -tulpn | grep :8080
netstat -tulpn | grep :9000

# 检查Docker资源
docker system df
docker system prune  # 清理未使用的资源
```

#### 2) MCP服务连接失败

```bash
# 检查MCP服务状态
curl http://localhost:9000/health

# 检查网络连接
docker network ls
docker network inspect aiops-network

# 重启MCP服务
docker-compose restart aiops-mcp
```

#### 3) Kubernetes配置问题

```bash
# 检查kubeconfig
kubectl config current-context
kubectl cluster-info

# 验证权限
kubectl auth can-i get pods
kubectl auth can-i create deployments
```

#### 日志分析

```bash
# 所有服务日志
docker-compose logs -f

# 特定服务日志
docker-compose logs -f aiops-platform
docker-compose logs -f aiops-mcp

# 最近100行日志
docker-compose logs --tail=100 aiops-platform
```

日志文件位置：

- 主应用日志: `./logs/app.log`
- MCP服务日志: `./logs/mcp.log`
- Docker容器日志: `docker logs <container_name>`

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

- `POST /api/v1/predict/qps` - QPS预测
- `POST /api/v1/predict/cpu` - CPU使用率预测
- `POST /api/v1/predict/memory` - 内存使用率预测
- `POST /api/v1/predict/disk` - 磁盘使用率预测
- `GET /api/v1/predict/models` - 模型信息
- `GET /api/v1/predict/health` - 健康检查
- `GET /api/v1/predict/ready` - 就绪检查
- `GET /api/v1/predict/info` - 服务信息

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

### 3. AI-CloudOps智能助手 (Assistant)

**位置**: `app/core/agents/enterprise_assistant.py`

**核心特性**:

- **双模式架构**: 支持RAG和MCP两种工作模式，互不干扰
- **AI-CloudOps RAG**: 基于Redis向量存储的知识检索增强
- **MCP工具调用**: 支持Kubernetes操作、系统信息查询等工具
- **智能路由**: 自动识别用户意图，选择最适合的处理模式
- **多轮对话**: 支持会话状态管理和上下文理解

**技术架构**:

- **RAG模式**: 向量检索 + LLM生成，适用于知识问答
- **MCP模式**: 工具调用协议，适用于操作执行
- **LangGraph工作流**: AI-CloudOps工作流引擎
- **Redis向量存储**: 高性能向量检索和缓存
- **多级质量评估**: 响应质量监控和自动优化

**支持的工具类型**:

- Kubernetes集群操作（Pod、Service、Deployment等）
- 系统信息查询和监控
- 时间和计算相关工具
- 可扩展的自定义工具接口

**API端点**:

- `POST /api/v1/assistant/query` - 智能问答（mode: 1=RAG, 2=MCP）
- `POST /api/v1/assistant/session` - 创建会话
- `GET /api/v1/assistant/session/{session_id}` - 会话信息
- `POST /api/v1/assistant/refresh` - 刷新知识库
- `GET /api/v1/assistant/config` - 服务配置
- `GET /api/v1/assistant/info` - 服务信息

**使用示例**:

```python
# RAG模式 - 知识问答
response = await post("/api/v1/assistant/query", {
    "question": "如何优化Kubernetes集群性能？",
    "mode": 1,
    "session_id": "user123"
})

# MCP模式 - 工具调用
response = await post("/api/v1/assistant/query", {
    "question": "获取default命名空间下的Pod列表",
    "mode": 2,
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

```http
GET /api/v1/health
```

#### 详细组件状态

```http
GET /api/v1/health/components
```

#### 依赖关系检查

```http
GET /api/v1/health/metrics
```

#### 就绪性探针

```http
GET /api/v1/health/ready
```

#### 存活性探针

```http
GET /api/v1/health/live
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

```http
POST /api/v1/predict/qps
Content-Type: application/json

{
  "service_name": "my-service",
  "current_qps": 100.0,
  "hours": 24,
  "instance_cpu": 1.0,
  "instance_memory": 2.0
}
```

#### CPU 使用率预测

```http
POST /api/v1/predict/cpu
```

#### 内存使用率预测

```http
POST /api/v1/predict/memory
```

#### 磁盘使用率预测

```http
POST /api/v1/predict/disk
```

#### 模型信息

```http
GET /api/v1/predict/models
```

### 缓存管理 API

#### 获取缓存统计

```http
GET /api/v1/cache/stats
```

#### 缓存系统健康检查

```http
GET /api/v1/cache/health
```

#### 清空缓存

```http
POST /api/v1/cache/clear?service=prediction|rca|all&pattern=<optional>
```

#### 获取缓存性能报告

```http
GET /api/v1/cache/performance
```

#### 获取缓存配置信息

```http
GET /api/v1/cache/config
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

```http
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

```http
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

```http
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

```http
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

```http
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

### AI-CloudOps智能助手 API

#### 智能问答（支持双模式）

```http
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

```http
GET /api/v1/assistant/session/user123
```

#### 刷新知识库

```http
POST /api/v1/assistant/refresh
```

#### 服务配置

```http
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

```text
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

1. **Kubernetes 部署**

```bash
# TODO: 待实现
```

1. **生产部署**

```bash
# 设置环境变量
export ENV=production

# 启动生产服务
python app/main.py
```

## 📈 性能优化

### 生产环境优化

#### 1. 环境配置

```bash
# 设置生产环境
ENV=production
DEBUG=false
LOG_LEVEL=WARNING

# 优化连接池
REDIS_MAX_CONNECTIONS=50
LLM_REQUEST_TIMEOUT=300
```

#### 2. 资源限制

在 `docker-compose.yml` 中添加资源限制：

```yaml
services:
  aiops-platform:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

#### 3. 缓存优化

```bash
# Redis缓存配置
REDIS_MAX_CONNECTIONS=20
RAG_CACHE_EXPIRY=7200

# 模型缓存
PREDICTION_MODEL_CACHE_SIZE=100
```

### 监控配置

#### Prometheus配置

编辑 `deploy/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aiops-platform'
    static_configs:
      - targets: ['aiops-platform:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 系统性能

- **异步处理**: 使用 asyncio 处理 I/O 密集操作
- **连接池**: 数据库和 HTTP 连接池管理
- **缓存策略**: 多级缓存提升响应速度
- **负载均衡**: 支持水平扩展
- **内存优化**: 对象池复用大对象减少 GC 压力
- **流式处理**: 大数据集分批处理
- **网络优化**: 启用 gzip 压缩和长连接复用

## 🔒 安全配置

### 生产环境安全

#### 1. 访问控制

```bash
# Redis密码保护
REDIS_PASSWORD=your-redis-password

# API访问限制
API_RATE_LIMIT=100
```

#### 2. 网络安全

```yaml
# docker-compose.yml 网络配置
networks:
  aiops-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### 3. 数据安全

- **加密存储**: 敏感数据加密存储
- **传输加密**: HTTPS/TLS 加密传输
- **访问控制**: 基于角色的访问控制
- **密钥管理**: 使用 Docker secrets 或外部密钥管理系统

### API 安全

- **身份验证**: JWT 令牌认证
- **授权控制**: 细粒度权限控制
- **限流保护**: API 请求限流
- **输入验证**: 严格的输入参数验证
- **SQL 注入防护**: 使用参数化查询
- **XSS 防护**: 输出数据转义

## 💾 备份与恢复

### 数据备份

```bash
# 备份数据目录
tar -czf aiops-backup-$(date +%Y%m%d).tar.gz ./data ./config

# 备份数据库
docker exec aiops-redis redis-cli --rdb /data/dump.rdb

# 备份配置
cp .env .env.backup
```

### 恢复数据

```bash
# 恢复数据目录
tar -xzf aiops-backup-YYYYMMDD.tar.gz

# 恢复数据库
docker exec aiops-redis redis-cli --eval backup.rdb
```

## 🔄 更新升级

### 应用更新

```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
./scripts/deploy.sh --build

# 滚动更新
docker-compose up -d --force-recreate
```

### 配置更新

```bash
# 更新配置文件
cp config/config.yaml config/config.yaml.backup
# 编辑新配置...

# 重启相关服务
docker-compose restart aiops-platform aiops-mcp
```

### 扩展部署

#### 集群部署

对于大规模部署，可以考虑：

1. 使用 Kubernetes 部署
2. 配置负载均衡
3. 使用外部 Redis 集群
4. 配置 Prometheus 高可用

#### 多环境部署

```bash
# 开发环境
ENV=development ./scripts/deploy.sh --dev

# 测试环境
ENV=testing ./scripts/deploy.sh

# 生产环境
ENV=production ./scripts/deploy.sh --production
```

## 📝 更新日志

### v1.0.0 (2025-07-11)

- 初始版本发布
- 完整的 AI-CloudOps 功能实现
- 支持多种部署方式
- 完善的 API 文档和使用指南

### v2.0.0 (2025-01-22) - 重大优化版本 ✨

**架构优化**:

- 完成全项目代码优化和重构
- 统一配置管理系统
- 清理冗余代码，提升性能30%
- 标准化API响应格式

**智能增强**:

- 升级智能预测引擎，支持多维度分析
- 增强根因分析算法，准确率提升至90%+
- 优化MCP工具调用性能

**功能完善**:

- 新增成本分析和优化建议
- 增强自动修复安全性
- 完善健康检查和监控

## 📋 附录

### 端口列表

| 服务       | 端口  | 协议 | 说明       |
| ---------- | ----- | ---- | ---------- |
| 主应用     | 8080  | HTTP | API接口    |
| MCP服务    | 9000  | HTTP | 工具调用   |
| Prometheus | 9090  | HTTP | 监控数据   |
| Redis      | 6379  | TCP  | 缓存数据库 |
| Ollama     | 11434 | HTTP | 本地模型   |

### 目录结构说明

```text
Ai-CloudOps-aiops/
├── app/                 # 应用代码
├── config/             # 配置文件
├── data/               # 数据文件
├── deploy/             # 部署配置
├── docs/               # 文档
├── logs/               # 日志文件
├── scripts/            # 脚本文件
├── docker-compose.yml  # Docker编排文件
├── Dockerfile          # 主应用镜像
├── Dockerfile.mcp      # MCP服务镜像
└── .env               # 环境变量
```

### 版本信息

- Python: 3.11+
- Docker: 20.10+
- Docker Compose: 2.0+
- Redis: 7.0+
- Prometheus: 2.45.0+

## 🛟 技术支持

如遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查 [GitHub Issues](https://github.com/GoSimplicity/AI-CloudOps/issues)
3. 查看项目日志文件
4. 联系技术支持团队

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📞 联系我们

- 项目主页: [https://github.com/GoSimplicity/AI-CloudOps]
- 问题报告: [https://github.com/GoSimplicity/AI-CloudOps/issues]
- 邮件联系: [bamboocloudops@gmail.com]

## 📄 许可证

本项目采用 MIT 许可证，详情请参阅 [LICENSE](LICENSE) 文件。

---

_本文档最后更新: 2025-08-24_  
_版本: 1.1.0_  
