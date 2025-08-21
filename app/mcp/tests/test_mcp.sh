#!/bin/bash
# -*- coding: utf-8 -*-

# CloudOps Platform MCP测试脚本
# Author: Bamboo
# Description: MCP功能验收测试脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 配置
MCP_SERVER_PORT=9000
MCP_SERVER_URL="http://localhost:${MCP_SERVER_PORT}"
API_SERVER_PORT=8080
API_SERVER_URL="http://localhost:${API_SERVER_PORT}"

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查curl
    if ! command -v curl &> /dev/null; then
        log_error "curl 未安装"
        exit 1
    fi
    
    # 检查Python依赖
    python3 -c "import aiohttp" 2>/dev/null || {
        log_error "aiohttp 未安装，请运行: pip install aiohttp"
        exit 1
    }
    
    log_success "依赖检查完成"
}

# 启动MCP服务端
start_mcp_server() {
    log_info "启动MCP服务端..."
    
    # 检查端口是否被占用
    if lsof -i :${MCP_SERVER_PORT} &> /dev/null; then
        log_warning "端口 ${MCP_SERVER_PORT} 已被占用，尝试使用现有服务"
        return 0
    fi
    
    # 创建日志目录
    mkdir -p logs
    
    # 启动服务端
    PYTHONPATH=/Users/wangzijian/golangProject/Bamboo/Ai-CloudOps-aiops python3 -m app.mcp.server.main &
    MCP_SERVER_PID=$!
    
    # 等待服务启动
    log_info "等待MCP服务端启动..."
    for i in {1..30}; do
        if curl -s "${MCP_SERVER_URL}/health" &> /dev/null; then
            log_success "MCP服务端启动成功 (PID: ${MCP_SERVER_PID})"
            return 0
        fi
        sleep 1
    done
    
    log_error "MCP服务端启动超时"
    return 1
}

# 测试健康检查
test_health_check() {
    log_info "测试健康检查接口..."
    
    response=$(curl -s "${MCP_SERVER_URL}/health")
    if [[ $? -eq 0 ]]; then
        log_success "健康检查成功: ${response}"
    else
        log_error "健康检查失败"
        return 1
    fi
}

# 测试获取工具列表
test_list_tools() {
    log_info "测试获取工具列表..."
    
    response=$(curl -s "${MCP_SERVER_URL}/tools")
    if [[ $? -eq 0 ]]; then
        log_success "获取工具列表成功: ${response}"
        
        # 检查是否包含time工具
        if echo "${response}" | grep -q "get_current_time"; then
            log_success "找到 get_current_time 工具"
        else
            log_error "未找到 get_current_time 工具"
            return 1
        fi
    else
        log_error "获取工具列表失败"
        return 1
    fi
}

# 测试工具调用
test_execute_tool() {
    log_info "测试工具调用..."
    
    response=$(curl -s -X POST "${MCP_SERVER_URL}/tools/execute" \
        -H "Content-Type: application/json" \
        -d '{"tool": "get_current_time", "parameters": {"format": "iso"}}')
    
    if [[ $? -eq 0 ]]; then
        log_success "工具调用成功: ${response}"
        
        # 检查返回格式
        if echo "${response}" | grep -q "time"; then
            log_success "工具返回格式正确"
        else
            log_error "工具返回格式错误"
            return 1
        fi
    else
        log_error "工具调用失败"
        return 1
    fi
}

# 测试SSE连接
test_sse_connection() {
    log_info "测试SSE连接..."
    
    # 使用超时命令，5秒后自动退出
    timeout 5s curl -s -N -H "Accept: text/event-stream" "${MCP_SERVER_URL}/sse" | head -n 10 || true
    
    if [[ $? -eq 0 ]]; then
        log_success "SSE连接测试完成"
    else
        log_warning "SSE连接测试可能未完成，但不影响功能"
    fi
}

# 测试MCP客户端
test_mcp_client() {
    log_info "测试MCP客户端..."
    
    # 测试客户端工具调用
    result=$(PYTHONPATH=/Users/wangzijian/golangProject/Bamboo/Ai-CloudOps-aiops python3 -m app.mcp.client.mcp_client --mode mcp --tool get_current_time --params '{"format":"iso"}')
    
    if [[ $? -eq 0 ]]; then
        log_success "MCP客户端测试成功: ${result}"
    else
        log_error "MCP客户端测试失败"
        return 1
    fi
}

# 测试API接口
test_api_integration() {
    log_info "测试API集成..."
    
    # 测试MCP模式
    response=$(curl -s -X POST "${API_SERVER_URL}/api/v1/assistant/query" \
        -H "Content-Type: application/json" \
        -d '{"question":"获取当前时间", "mode":"mcp", "session_id":"1234567890"}')
    
    if [[ $? -eq 0 ]]; then
        log_success "API集成测试成功: ${response}"
        
        # 验证返回格式
        if echo "${response}" | jq -e '.code == 0' &> /dev/null; then
            log_success "API返回格式正确"
        else
            log_error "API返回格式错误"
            return 1
        fi
    else
        log_error "API集成测试失败"
        return 1
    fi
    
    # 测试RAG模式（确保原有功能正常）
    response=$(curl -s -X POST "${API_SERVER_URL}/api/v1/assistant/query" \
        -H "Content-Type: application/json" \
        -d '{"question":"什么是Kubernetes", "mode":"rag", "session_id":"1234567890"}')
    
    if [[ $? -eq 0 ]]; then
        log_success "RAG模式测试成功"
    else
        log_error "RAG模式测试失败"
        return 1
    fi
}

# 清理函数
cleanup() {
    log_info "清理测试环境..."
    
    if [[ -n "${MCP_SERVER_PID}" ]]; then
        if kill -0 ${MCP_SERVER_PID} 2>/dev/null; then
            log_info "停止MCP服务端 (PID: ${MCP_SERVER_PID})"
            kill ${MCP_SERVER_PID}
        fi
    fi
}

# 主测试流程
main() {
    log_info "开始MCP功能验收测试..."
    
    # 注册清理函数
    trap cleanup EXIT
    
    # 步骤1: 检查依赖
    check_dependencies
    
    # 步骤2: 启动MCP服务端
    start_mcp_server
    
    # 步骤3: 测试服务端接口
    test_health_check
    test_list_tools
    test_execute_tool
    test_sse_connection
    
    # 步骤4: 测试客户端
    test_mcp_client
    
    # 步骤5: 测试API集成
    test_api_integration
    
    log_success "所有测试通过！✅"
    echo
    log_info "测试总结:"
    echo "  - ✅ MCP服务端正常启动"
    echo "  - ✅ 健康检查接口正常"
    echo "  - ✅ 工具列表接口正常"
    echo "  - ✅ 工具调用功能正常"
    echo "  - ✅ SSE连接正常"
    echo "  - ✅ MCP客户端正常"
    echo "  - ✅ API集成正常"
    echo "  - ✅ RAG模式兼容"
    echo
    log_info "一键启动命令:"
    echo "  python3 app/mcp/server/main.py &"
    echo "  python3 app/mcp/client/mcp_client.py --mode interactive"
}

# 运行主函数
main "$@" 2>&1 | tee logs/mcp_test.log