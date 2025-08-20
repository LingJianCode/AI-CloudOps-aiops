#!/bin/bash
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops MCP服务快速启动脚本
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
"""

set -e

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "AI-CloudOps MCP服务启动脚本"
echo "项目目录: $PROJECT_DIR"
echo "当前时间: $(date)"
echo "========================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python"
    exit 1
fi

# 进入项目目录
cd "$PROJECT_DIR"

# 检查依赖
echo "检查依赖包..."
python -c "import fastapi, uvicorn, aiohttp" 2>/dev/null || {
    echo "错误: 缺少必要依赖，请运行: pip install fastapi uvicorn aiohttp"
    exit 1
}

# 检查配置文件
if [[ ! -f "config/config.yaml" ]]; then
    echo "错误: 未找到配置文件 config/config.yaml"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 启动MCP服务
echo "正在启动MCP服务..."
echo "启动命令: python -m app.mcp.main"
echo "========================================"

# 根据参数决定启动方式
if [[ "$1" == "--background" || "$1" == "-b" ]]; then
    echo "后台模式启动..."
    nohup python -m app.mcp.main > logs/mcp_startup.log 2>&1 &
    MCP_PID=$!
    echo "MCP服务已在后台启动，PID: $MCP_PID"
    echo "日志文件: logs/mcp_startup.log"
    echo "健康检查: curl http://localhost:9000/health"
elif [[ "$1" == "--dev" || "$1" == "-d" ]]; then
    echo "开发模式启动（自动重载）..."
    python -m app.mcp.main --reload
else
    echo "前台模式启动..."
    python -m app.mcp.main
fi

echo "MCP服务启动完成！"
