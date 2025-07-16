#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops MCP客户端
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: MCP客户端实现，支持SSE连接和工具调用
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import Any, Dict, Optional

import aiohttp
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aiops.mcp.client")


class MCPClient:
    """MCP客户端类"""
    
    def __init__(self, server_url: str = None):
        from app.config.settings import config
        mcp_config = config.mcp
        self.server_url = (server_url or mcp_config.server_url).rstrip('/')
        self.timeout = mcp_config.timeout
        self.max_retries = mcp_config.max_retries
        self.health_check_interval = mcp_config.health_check_interval
        self.session = None
        logger.info(f"MCP客户端已初始化，服务端地址: {self.server_url}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"健康检查成功: {data}")
                        return True
                    else:
                        logger.error(f"健康检查失败，状态码: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"健康检查异常: {str(e)}")
            return False
    
    async def list_tools(self) -> Dict[str, Any]:
        """获取可用工具列表"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/tools") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        raise RuntimeError(f"获取工具列表失败，状态码: {response.status}")
        except Exception as e:
            logger.error(f"获取工具列表异常: {str(e)}")
            raise
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any] = None) -> Any:
        """执行工具调用"""
        if parameters is None:
            parameters = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "tool": tool_name,
                    "parameters": parameters
                }
                
                async with session.post(
                    f"{self.server_url}/tools/execute",
                    json=request_data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("error"):
                            raise RuntimeError(f"工具执行失败: {data['error']}")
                        return data.get("result")
                    else:
                        raise RuntimeError(f"工具调用失败，状态码: {response.status}")
        except Exception as e:
            logger.error(f"工具调用异常: {str(e)}")
            raise
    
    async def connect_sse(self) -> None:
        """连接SSE端点"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/sse",
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    if response.status == 200:
                        logger.info("SSE连接已建立")
                        
                        async for line in response.content:
                            if line:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        logger.info(f"收到SSE消息: {data}")
                                        
                                        # 根据消息类型处理
                                        if data.get('type') == 'connected':
                                            print(f"✅ {data.get('message')}")
                                        elif data.get('type') == 'tools_list':
                                            print("📋 可用工具:")
                                            for tool in data.get('tools', []):
                                                print(f"  - {tool['name']}: {tool['description']}")
                                        elif data.get('type') == 'heartbeat':
                                            print("💓 心跳")
                                        elif data.get('type') == 'disconnected':
                                            print("❌ 连接已断开")
                                            break
                                            
                                    except json.JSONDecodeError as e:
                                        logger.error(f"解析SSE消息失败: {e}")
                    else:
                        logger.error(f"SSE连接失败，状态码: {response.status}")
        except Exception as e:
            logger.error(f"SSE连接异常: {str(e)}")
    
    def execute_tool_sync(self, tool_name: str, parameters: Dict[str, Any] = None) -> Any:
        """同步执行工具调用"""
        if parameters is None:
            parameters = {}
        
        try:
            request_data = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            response = requests.post(
                f"{self.server_url}/tools/execute",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("error"):
                    raise RuntimeError(f"工具执行失败: {data['error']}")
                return data.get("result")
            else:
                raise RuntimeError(f"工具调用失败，状态码: {response.status_code}")
                
        except Exception as e:
            logger.error(f"工具调用异常: {str(e)}")
            raise


async def interactive_mode(client: MCPClient) -> None:
    """交互模式"""
    print("🔧 MCP客户端交互模式")
    print("可用的命令:")
    print("  tools - 显示可用工具列表")
    print("  execute <tool_name> [parameters] - 执行工具调用")
    print("  sse - 连接SSE端点")
    print("  quit - 退出")
    print()
    
    while True:
        try:
            command = input("mcp> ").strip()
            
            if command == "quit" or command == "exit":
                break
            elif command == "tools":
                tools = await client.list_tools()
                print("📋 可用工具:")
                for tool in tools.get('tools', []):
                    print(f"  - {tool['name']}: {tool['description']}")
                    print(f"    参数: {json.dumps(tool['parameters'], indent=4, ensure_ascii=False)}")
                    print()
            elif command.startswith("execute"):
                parts = command.split(" ", 2)
                if len(parts) < 2:
                    print("❌ 用法: execute <tool_name> [parameters]")
                    continue
                
                tool_name = parts[1]
                parameters = {}
                if len(parts) > 2:
                    try:
                        parameters = json.loads(parts[2])
                    except json.JSONDecodeError:
                        print("❌ 参数必须是有效的JSON格式")
                        continue
                
                try:
                    result = await client.execute_tool(tool_name, parameters)
                    print(f"✅ 执行结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
                except Exception as e:
                    print(f"❌ 执行失败: {str(e)}")
            elif command == "sse":
                print("🔗 正在连接SSE端点...")
                await client.connect_sse()
            else:
                print("❓ 未知命令，输入 'quit' 退出")
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {str(e)}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MCP客户端")
    parser.add_argument("--server", default="http://localhost:9000", 
                       help="MCP服务端地址 (默认: http://localhost:9000)")
    parser.add_argument("--mode", choices=["mcp", "interactive"], default="interactive",
                       help="运行模式: mcp(单次调用) 或 interactive(交互模式)")
    parser.add_argument("--tool", help="要执行的工具名称")
    parser.add_argument("--params", default="{}", help="工具参数(JSON格式)")
    parser.add_argument("--sse", action="store_true", help="连接SSE端点")
    
    args = parser.parse_args()
    
    client = MCPClient(args.server)
    
    # 健康检查
    if not await client.health_check():
        print("❌ MCP服务端连接失败")
        return
    
    print("✅ MCP服务端连接成功")
    
    if args.sse:
        await client.connect_sse()
    elif args.mode == "mcp":
        if not args.tool:
            print("❌ 在mcp模式下必须指定 --tool 参数")
            return
        
        try:
            parameters = json.loads(args.params)
            result = await client.execute_tool(args.tool, parameters)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("❌ 参数必须是有效的JSON格式")
        except Exception as e:
            print(f"❌ 执行失败: {str(e)}")
    else:
        await interactive_mode(client)


def handle_signal(signum, frame):
    """信号处理器"""
    print(f"\n接收到信号 {signum}，正在退出...")
    sys.exit(0)


if __name__ == "__main__":
    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        print(f"❌ 运行错误: {str(e)}")
        sys.exit(1)