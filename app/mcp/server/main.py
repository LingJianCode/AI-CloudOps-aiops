#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops MCP服务端
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: MCP(Model-Context-Protocol)服务端，提供SSE传输接口
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.mcp.server.mcp_server import MCPServer
from app.mcp.server.tools import tools as mcp_tools
from app.config.settings import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/mcp_server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("aiops.mcp.server")

# 全局变量
mcp_server: Optional[MCPServer] = None


class ToolRequest(BaseModel):
    """工具调用请求模型"""
    tool: str
    parameters: Dict[str, Any] = {}
    request_id: Optional[str] = None


class ToolResponse(BaseModel):
    """工具调用响应模型"""
    request_id: Optional[str] = None
    tool: str
    result: Any
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global mcp_server
    
    # 启动时初始化
    logger.info("正在启动MCP服务端...")
    mcp_server = MCPServer()
    
    # 注册所有工具
    registered_count = 0
    for tool in mcp_tools:
        await mcp_server.register_tool(tool)
        registered_count += 1
        logger.info(f"已注册工具: {tool.name}")
    
    logger.info(f"MCP服务端启动完成，共注册 {registered_count} 个工具")
    yield
    
    # 关闭时清理
    if mcp_server:
        await mcp_server.shutdown()
        logger.info("MCP服务端已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="AI-CloudOps MCP服务端",
    description="提供MCP工具调用能力的SSE服务端",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "tools": list(mcp_server.tools.keys()) if mcp_server else []
    }


@app.get("/sse")
async def sse_endpoint() -> StreamingResponse:
    """SSE端点，提供实时数据流"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器未初始化")
    
    async def event_generator():
        """事件生成器"""
        try:
            # 发送连接事件
            yield f"data: {json.dumps({'type': 'connected', 'message': 'MCP连接已建立'})}\n\n"
            
            # 发送可用工具列表
            tools_info = {
                'type': 'tools_list',
                'tools': [
                    {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': tool.parameters
                    }
                    for tool in mcp_server.tools.values()
                ]
            }
            yield f"data: {json.dumps(tools_info)}\n\n"
            
            # 保持连接活跃
            while True:
                # 每30秒发送心跳
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': asyncio.get_event_loop().time()})}\n\n"
                
        except asyncio.CancelledError:
            logger.info("SSE连接已关闭")
            yield f"data: {json.dumps({'type': 'disconnected', 'message': 'MCP连接已断开'})}\n\n"
        except Exception as e:
            logger.error(f"SSE连接错误: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/tools/execute")
async def execute_tool(request: ToolRequest) -> ToolResponse:
    """执行工具调用"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器未初始化")
    
    try:
        logger.info(f"执行工具调用: {request.tool}")
        
        # 执行工具
        result = await mcp_server.execute_tool(
            tool_name=request.tool,
            parameters=request.parameters
        )
        
        return ToolResponse(
            request_id=request.request_id,
            tool=request.tool,
            result=result
        )
        
    except Exception as e:
        logger.error(f"工具调用失败: {str(e)}")
        return ToolResponse(
            request_id=request.request_id,
            tool=request.tool,
            result=None,
            error=str(e)
        )


@app.get("/tools")
async def list_tools() -> Dict[str, Any]:
    """获取可用工具列表"""
    if not mcp_server:
        raise HTTPException(status_code=503, detail="MCP服务器未初始化")
    
    tools = [
        {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters
        }
        for tool in mcp_server.tools.values()
    ]
    
    return {"tools": tools}


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"接收到信号 {signum}，正在优雅关闭...")
    sys.exit(0)


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务
    logger.info("正在启动MCP服务端...")
    
    # 从MCP服务器URL中提取端口
    server_url = config.mcp.server_url
    if "://" in server_url:
        server_url = server_url.split("://")[1]
    server_port = 9000
    if ":" in server_url:
        try:
            server_port = int(server_url.split(":")[-1])
        except ValueError:
            server_port = 9000
    
    uvicorn.run(
        "app.mcp.server.main:app",
        host=config.host,
        port=server_port,
        log_level=config.log_level.lower(),
        access_log=True,
        reload=config.debug
    )