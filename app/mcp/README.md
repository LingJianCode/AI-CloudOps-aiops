# MCP (Model-Context-Protocol) 功能文档

## 简介

本项目为AI-CloudOps-aiops增加了MCP（Model-Context-Protocol）工具调用能力，与现有RAG功能完全隔离，支持RAG ⇄ MCP模式切换。

## 功能特性

- ✅ **独立运行**：MCP服务端独立进程运行，与现有系统解耦
- ✅ **SSE传输**：支持Server-Sent Events实时数据流
- ✅ **模式切换**：支持RAG/MCP模式互斥切换，不共享上下文
- ✅ **工具扩展**：提供示例工具`get_current_time`，支持自定义扩展
- ✅ **完整兼容**：保持现有RAG功能100%兼容

## 目录结构

```
mcp/
├── client/               # 客户端实现
│   └── mcp_client.py    # 命令行客户端
├── server/               # 服务端实现
│   ├── main.py          # FastAPI服务端入口
│   ├── mcp_server.py    # MCP核心实现
│   └── tools/
│       ├── __init__.py
│       └── time_tool.py # 示例工具
├── tests/
│   └── test_mcp.sh      # 验收测试脚本
├── mcp_client.py        # 集成客户端
└── README.md            # 本文档
```

## 快速开始

### 1. 安装依赖

确保已安装所需的Python依赖：

```bash
pip install aiohttp fastapi uvicorn
```

### 2. 启动MCP服务端

```bash
# 启动服务端（监听9000端口）
python app/mcp/server/main.py

# 后台启动
python app/mcp/server/main.py &
```

### 3. 验证服务状态

```bash
# 健康检查
curl http://localhost:9000/health

# 获取工具列表
curl http://localhost:9000/tools

# 调用示例工具
curl -X POST http://localhost:9000/tools/execute \
  -H "Content-Type: application/json" \
  -d '{"tool":"get_current_time","parameters":{"format":"iso"}}'
```

### 4. 连接SSE端点

```bash
# 测试SSE连接
curl -N -H "Accept:text/event-stream" http://localhost:9000/sse
```

### 5. 使用MCP客户端

```bash
# 命令行客户端
python app/mcp/client/mcp_client.py --mode interactive

# 单次工具调用
python app/mcp/client/mcp_client.py --mode mcp --tool get_current_time

# 连接SSE
python app/mcp/client/mcp_client.py --sse
```

### 6. 通过API使用

```bash
# MCP模式调用
curl -X POST http://localhost:8080/api/v1/assistant/query \
  -H "Content-Type: application/json" \
  -d '{"question":"获取当前时间", "mode":"mcp", "session_id":"1234567890"}'

# RAG模式调用（保持兼容）
curl -X POST http://localhost:8080/api/v1/assistant/query \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是Kubernetes", "mode":"rag", "session_id":"1234567890"}'
```

## API接口

### 服务端接口 (端口: 9000)

#### 健康检查
```
GET /health
```

#### 获取工具列表
```
GET /tools
```

#### 执行工具调用
```
POST /tools/execute
Content-Type: application/json

{
  "tool": "工具名称",
  "parameters": {
    "参数名": "参数值"
  }
}
```

#### SSE实时流
```
GET /sse
Accept: text/event-stream
```

### 主API接口 (端口: 8080)

#### 查询接口
```
POST /api/v1/assistant/query
Content-Type: application/json

{
  "question": "用户问题",
  "mode": "mcp|rag",  // 模式选择
  "session_id": "会话ID",
  "max_context_docs": 4
}
```

## 使用示例

### 示例1：获取当前时间

```bash
# 通过API
response=$(curl -s -X POST http://localhost:8080/api/v1/assistant/query \
  -H "Content-Type: application/json" \
  -d '{"question":"获取当前时间", "mode":"mcp", "session_id":"1234567890"}')

echo $response
# 输出: {"code":0,"message":"查询成功","data":{"answer":"当前时间是: 2025-07-16T12:34:56Z"...}}
```

### 示例2：命令行交互

```bash
python app/mcp/client/mcp_client.py --mode interactive

# 交互示例:
# 🔧 MCP客户端交互模式
# mcp> tools
# 📋 可用工具:
#   - get_current_time: 获取当前时间，返回ISO-8601格式的时间字符串
# 
# mcp> execute get_current_time {"format":"iso"}
# ✅ 执行结果: {"time": "2025-07-16T12:34:56Z", "format": "ISO-8601", "timezone": "UTC"}
```

## 工具扩展

### 添加新工具

1. 创建工具类，继承BaseTool：

```python
from app.mcp.server.mcp_server import BaseTool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="工具描述"
        )
    
    def get_parameters(self):
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "参数1"}
            },
            "required": ["param1"]
        }
    
    async def execute(self, parameters):
        # 实现工具逻辑
        return {"result": "success"}
```

2. 在服务端注册工具：

```python
# 在 app/mcp/server/main.py 中
from .tools.my_tool import MyTool

# 在初始化时注册
my_tool = MyTool()
await mcp_server.register_tool(my_tool)
```

## 配置说明

### 配置文件

在 `config/config.yaml` 中添加MCP配置：

```yaml
# MCP配置
mcp:
  server_url: "http://localhost:9000"  # MCP服务端地址
  timeout: 30                           # 请求超时时间(秒)
  max_retries: 3                        # 最大重试次数
```

### 环境变量

也可以通过环境变量配置：

```bash
export MCP_SERVER_URL="http://localhost:9000"
export MCP_TIMEOUT="30"
```

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   lsof -i :9000
   
   # 修改端口
   python app/mcp/server/main.py --port 9001
   ```

2. **依赖问题**
   ```bash
   # 安装缺失依赖
   pip install aiohttp fastapi uvicorn requests
   ```

3. **服务连接失败**
   ```bash
   # 检查服务状态
   curl http://localhost:9000/health
   
   # 查看日志
   tail -f logs/mcp_server.log
   ```

### 调试模式

```bash
# 服务端调试模式
python app/mcp/server/main.py --log-level debug

# 客户端调试模式
python app/mcp/client/mcp_client.py --mode mcp --tool get_current_time --log-level debug
```

## 一键启动

使用提供的测试脚本一键启动和验证：

```bash
# 一键测试所有功能
./app/mcp/tests/test_mcp.sh

# 手动启动流程
python app/mcp/server/main.py &
python app/mcp/client/mcp_client.py --mode interactive
```

## 性能优化

- **连接池**：客户端使用连接池复用连接
- **缓存**：支持响应缓存（可配置）
- **超时控制**：完善的超时和重试机制
- **错误处理**：详细的错误日志和恢复机制

## 安全考虑

- **输入验证**：所有参数都经过验证和清洗
- **错误处理**：不会暴露敏感信息
- **超时控制**：防止长时间阻塞
- **日志记录**：完整的操作日志记录

## 开发指南

### 本地开发

```bash
# 1. 启动服务端（开发模式）
uvicorn app.mcp.server.main:app --reload --port 9000

# 2. 启动客户端测试
python app/mcp/client/mcp_client.py --mode interactive
```

### 测试

```bash
# 运行所有测试
./app/mcp/tests/test_mcp.sh

# 单独测试
python -m pytest tests/test_mcp.py
```

## 许可证

Apache 2.0 License - 详见项目根目录LICENSE文件