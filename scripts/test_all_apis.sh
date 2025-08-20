#!/bin/bash

# AI-CloudOps-aiops 接口测试脚本
# 测试所有接口确保无死锁问题

BASE_URL="http://localhost:8080"
SUCCESS_COUNT=0
FAIL_COUNT=0

# 颜色定义
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
    ((SUCCESS_COUNT++))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((FAIL_COUNT++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 测试函数
test_api() {
    local method="$1"
    local endpoint="$2"
    local description="$3"
    local data="$4"
    local timeout="$5"
    
    if [ -z "$timeout" ]; then
        timeout=30
    fi
    
    log_info "测试: $description"
    log_info "请求: $method $endpoint"
    if [ -n "$data" ]; then
        log_info "请求体: ${data:0:100}..."
    fi
    
    # 构建curl命令
    local curl_cmd="curl -s -w \"%{http_code}\" --max-time $timeout"
    local curl_result
    
    if [ "$method" = "GET" ]; then
        curl_result=$(eval "$curl_cmd \"$BASE_URL$endpoint\"" 2>/dev/null)
    elif [ "$method" = "POST" ]; then
        if [ -n "$data" ]; then
            curl_result=$(eval "$curl_cmd -X POST -H \"Content-Type: application/json\" -d '$data' \"$BASE_URL$endpoint\"" 2>/dev/null)
        else
            curl_result=$(eval "$curl_cmd -X POST \"$BASE_URL$endpoint\"" 2>/dev/null)
        fi
    elif [ "$method" = "DELETE" ]; then
        curl_result=$(eval "$curl_cmd -X DELETE \"$BASE_URL$endpoint\"" 2>/dev/null)
    fi
    
    local curl_exit_code=$?
    
    if [ $curl_exit_code -eq 0 ] && [ -n "$curl_result" ]; then
        # 提取HTTP状态码（最后3位）
        local http_code="${curl_result: -3}"
        # 提取响应体（除了最后3位）
        local response_body="${curl_result%???}"
        
        if [[ "$http_code" =~ ^[2-3][0-9][0-9]$ ]]; then
            log_success "$description - HTTP $http_code"
            if [ ${#response_body} -gt 0 ]; then
                echo "响应: ${response_body:0:150}..."
            fi
        else
            log_error "$description - HTTP $http_code"
            if [ ${#response_body} -gt 0 ]; then
                echo "错误响应: ${response_body:0:300}"
            fi
        fi
    else
        log_error "$description - 请求失败 (退出码: $curl_exit_code)"
        if [ $curl_exit_code -eq 28 ]; then
            echo "错误: 请求超时 (${timeout}秒)"
        elif [ $curl_exit_code -eq 7 ]; then
            echo "错误: 无法连接到服务器"
        else
            echo "错误: curl命令执行失败"
        fi
    fi
    
    echo "----------------------------------------"
    sleep 1
}

# 检查服务可用性
check_service_availability() {
    log_info "检查服务可用性..."
    local health_response=$(curl -s --max-time 10 "$BASE_URL/" 2>/dev/null)
    local curl_exit_code=$?
    
    if [ $curl_exit_code -eq 0 ] && [ -n "$health_response" ]; then
        log_success "服务已启动并可访问"
        return 0
    else
        log_error "服务不可访问，请检查服务是否正常启动"
        log_error "确保服务运行在 $BASE_URL"
        return 1
    fi
}

# 开始测试
echo "=========================================="
echo "AI-CloudOps-aiops API 测试开始"
echo "基础URL: $BASE_URL"
echo "时间: $(date)"
echo "=========================================="

# 检查服务可用性
if ! check_service_availability; then
    echo "=========================================="
    echo "服务不可用，测试终止"
    echo "=========================================="
    exit 1
fi

# 1. 健康检查接口测试
log_info "开始测试健康检查接口..."
test_api "GET" "/api/v1/health" "系统综合健康检查" "" 10
test_api "GET" "/api/v1/health/components" "组件健康检查" "" 15
test_api "GET" "/api/v1/health/metrics" "系统指标检查" "" 10
test_api "GET" "/api/v1/health/ready" "就绪状态检查" "" 10
test_api "GET" "/api/v1/health/live" "存活状态检查" "" 10
test_api "GET" "/api/v1/health/startup" "启动状态检查" "" 10
test_api "GET" "/api/v1/health/dependencies" "依赖服务检查" "" 15
test_api "GET" "/api/v1/health/detail" "详细健康检查" "" 20

# 2. 智能助手接口测试
log_info "开始测试智能助手接口..."
test_api "POST" "/api/v1/assistant/session" "创建会话" "" 10
test_api "GET" "/api/v1/assistant/health" "助手健康检查" "" 10
test_api "GET" "/api/v1/assistant/ready" "助手就绪检查" "" 10
test_api "GET" "/api/v1/assistant/info" "助手信息" "" 10
test_api "POST" "/api/v1/assistant/refresh" "刷新助手" "" 30

# 智能助手查询测试（使用简单问题避免长时间等待）
query_data='{"question":"你好","session_id":"test-session-123","max_context_docs":1}'
test_api "POST" "/api/v1/assistant/query" "智能助手查询" "$query_data" 60

# 3. 预测服务接口测试
log_info "开始测试预测服务接口..."
test_api "GET" "/api/v1/predict/health" "预测服务健康检查" "" 15
test_api "GET" "/api/v1/predict/ready" "预测服务就绪检查" "" 10
test_api "GET" "/api/v1/predict/info" "预测服务信息" "" 10
test_api "GET" "/api/v1/predict/models" "模型信息" "" 15

# 预测接口测试
predict_data='{"service_name":"test-service","current_qps":100,"hours":1,"instance_cpu":2,"instance_memory":4}'
test_api "POST" "/api/v1/predict" "QPS预测" "$predict_data" 60
test_api "GET" "/api/v1/predict/trend?service_name=test-service&hours=1" "负载趋势分析" "" 30

# 4. RCA接口测试
log_info "开始测试RCA服务接口..."
test_api "GET" "/api/v1/rca/health" "RCA服务健康检查" "" 15
test_api "GET" "/api/v1/rca/ready" "RCA服务就绪检查" "" 10
test_api "GET" "/api/v1/rca/info" "RCA服务信息" "" 10
test_api "GET" "/api/v1/rca/config" "获取RCA配置" "" 10
test_api "GET" "/api/v1/rca/metrics" "获取可用指标" "" 15

# RCA分析测试（使用简单的测试数据）
rca_data='{"metrics":["cpu_usage"],"start_time":"2024-01-01T00:00:00","end_time":"2024-01-01T01:00:00","service_name":"test-service","namespace":"default","include_logs":false,"severity_threshold":0.7}'
test_api "POST" "/api/v1/rca" "根因分析" "$rca_data" 60

# 5. 自动修复接口测试
log_info "开始测试自动修复服务接口..."
test_api "GET" "/api/v1/autofix/health" "自动修复服务健康检查" "" 15
test_api "GET" "/api/v1/autofix/ready" "自动修复服务就绪检查" "" 10
test_api "GET" "/api/v1/autofix/info" "自动修复服务信息" "" 10

# 自动修复诊断测试
diagnose_data='{"deployment":"test-deployment","namespace":"default","include_logs":false,"include_events":false}'
test_api "POST" "/api/v1/autofix/diagnose" "Kubernetes问题诊断" "$diagnose_data" 30

# 自动修复测试（非应用模式，仅分析）
autofix_data='{"deployment":"test-deployment","namespace":"default","event":"测试问题描述","auto_apply":false,"severity":"low","timeout":60}'
test_api "POST" "/api/v1/autofix" "Kubernetes自动修复" "$autofix_data" 90

# 6. 基础接口测试
log_info "开始测试基础接口..."
test_api "GET" "/" "根路径" "" 5
test_api "GET" "/docs" "API文档" "" 10

# 测试总结
echo "=========================================="
echo "API 测试完成"
echo "=========================================="
echo "测试结果摘要:"
echo -e "  ${GREEN}成功: $SUCCESS_COUNT${NC}"
echo -e "  ${RED}失败: $FAIL_COUNT${NC}"
echo "  总计: $((SUCCESS_COUNT + FAIL_COUNT))"
echo ""
echo "成功率: $(( SUCCESS_COUNT * 100 / (SUCCESS_COUNT + FAIL_COUNT) ))%"
echo "测试完成时间: $(date)"

if [ $FAIL_COUNT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 所有接口测试通过！${NC}"
    echo -e "${GREEN}✓ 系统运行正常，无死锁或阻塞问题检测到${NC}"
    echo -e "${GREEN}✓ 所有核心功能可用${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}⚠️  有 $FAIL_COUNT 个接口测试失败${NC}"
    echo -e "${YELLOW}建议检查：${NC}"
    echo "  1. 服务是否完全启动"
    echo "  2. 依赖服务(Redis, Prometheus等)是否可用"
    echo "  3. 配置文件是否正确"
    echo "  4. 日志文件中的错误信息"
    exit 1
fi
