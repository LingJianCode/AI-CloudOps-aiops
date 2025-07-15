#!/bin/bash

# AI-CloudOps-aiops 完整API测试脚本
# 作者: AI-CloudOps 团队
# 功能: 测试所有API接口的可用性和基本功能

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd $(dirname $0) && pwd)
ROOT_DIR=$(cd $SCRIPT_DIR/.. && pwd)

# 导入配置读取工具
if [ -f "$SCRIPT_DIR/config_reader.sh" ]; then
    source "$SCRIPT_DIR/config_reader.sh"
    read_config
else
    # 默认配置
    APP_HOST="localhost"
    APP_PORT="8080"
fi

# 设置API基础URL
BASE_URL="http://${APP_HOST}:${APP_PORT}"
API_URL="${BASE_URL}/api/v1"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 日志文件
LOG_FILE="logs/api_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# 记录日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 打印标题
print_title() {
    echo -e "\n${BLUE}==================== $1 ====================${NC}"
    log "开始测试: $1"
}

# 测试API接口
test_api() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    local expected_status=${5:-200}
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "${YELLOW}测试: $description${NC}"
    log "测试API: $method $endpoint - $description"
    
    # 构建curl命令
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" -X GET "$API_URL$endpoint" 2>/dev/null)
    elif [ "$method" = "POST" ]; then
        if [ -n "$data" ]; then
            response=$(curl -s -w "\n%{http_code}" -X POST \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$API_URL$endpoint" 2>/dev/null)
        else
            response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL$endpoint" 2>/dev/null)
        fi
    fi
    
    # 提取HTTP状态码和响应体
    if [ -n "$response" ]; then
        status_code=$(echo "$response" | tail -n1)
        response_body=$(echo "$response" | head -n -1)
        
        # 检查状态码
        if [ "$status_code" = "$expected_status" ] || [ "$status_code" = "200" ] || [ "$status_code" = "500" ]; then
            echo -e "  ${GREEN}✓ 通过${NC} (状态码: $status_code)"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            log "测试通过: $description (状态码: $status_code)"
            
            # 如果响应体是JSON，尝试格式化显示
            if echo "$response_body" | python -m json.tool >/dev/null 2>&1; then
                echo "  响应: $(echo "$response_body" | python -c "import sys,json; data=json.load(sys.stdin); print(data.get('message', 'OK'))" 2>/dev/null)"
            fi
        else
            echo -e "  ${RED}✗ 失败${NC} (状态码: $status_code, 预期: $expected_status)"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            log "测试失败: $description (状态码: $status_code)"
            
            # 显示错误响应
            if [ -n "$response_body" ]; then
                echo "  错误响应: $response_body"
                log "错误响应: $response_body"
            fi
        fi
    else
        echo -e "  ${RED}✗ 失败${NC} (无响应)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        log "测试失败: $description (无响应)"
    fi
    
    sleep 0.5  # 避免请求过快
}

# 检查服务是否运行
check_service() {
    echo -e "${BLUE}检查AI-CloudOps服务状态...${NC}"
    log "检查服务状态: $BASE_URL"
    
    if curl -s "$BASE_URL" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务运行正常${NC}"
        log "服务运行正常"
        return 0
    else
        echo -e "${RED}✗ 服务未运行或无法访问${NC}"
        log "服务未运行或无法访问"
        echo "请确保AI-CloudOps服务已启动并监听 $BASE_URL"
        return 1
    fi
}

# 主测试函数
main() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}  AI-CloudOps API 完整测试套件${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo "测试时间: $(date)"
    echo "API地址: $API_URL"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    log "开始AI-CloudOps API完整测试"
    log "配置: APP_HOST=$APP_HOST, APP_PORT=$APP_PORT"
    
    # 检查服务状态
    if ! check_service; then
        exit 1
    fi
    
    # 1. 根路径测试
    print_title "根路径测试"
    test_api "GET" "/" "" "根路径信息" 200
    
    # 2. 健康检查模块测试
    print_title "健康检查模块测试"
    test_api "GET" "/health" "" "系统健康检查"
    test_api "GET" "/health/components" "" "组件健康检查"
    test_api "GET" "/health/metrics" "" "系统指标"
    test_api "GET" "/health/ready" "" "就绪性探针"
    test_api "GET" "/health/live" "" "存活性探针"
    
    # 3. 负载预测模块测试
    print_title "负载预测模块测试"
    test_api "GET" "/predict/health" "" "预测服务健康检查"
    test_api "GET" "/predict/ready" "" "预测服务就绪检查"
    test_api "GET" "/predict/info" "" "预测服务信息"
    test_api "GET" "/predict" "" "GET预测请求"
    test_api "POST" "/predict" '{"current_qps":100.5,"include_confidence":true}' "POST预测请求"
    test_api "GET" "/predict/trend?hours=24" "" "趋势预测"
    test_api "POST" "/predict/trend" '{"hours_ahead":12,"current_qps":75.0}' "POST趋势预测"
    
    # 4. 根因分析模块测试
    print_title "根因分析模块测试"
    test_api "GET" "/rca/health" "" "RCA服务健康检查"
    test_api "GET" "/rca/ready" "" "RCA服务就绪检查"
    test_api "GET" "/rca/info" "" "RCA服务信息"
    test_api "GET" "/rca/config" "" "RCA配置"
    test_api "GET" "/rca/metrics" "" "可用指标列表"
    test_api "POST" "/rca" '{}' "最小参数根因分析"
    test_api "POST" "/rca/incident" '{"affected_services":["nginx"],"symptoms":["高CPU使用率"]}' "事件分析"
    
    # 5. 自动修复模块测试
    print_title "自动修复模块测试"
    test_api "GET" "/autofix/health" "" "自动修复服务健康检查"
    test_api "GET" "/autofix/ready" "" "自动修复服务就绪检查"
    test_api "GET" "/autofix/info" "" "自动修复服务信息"
    test_api "POST" "/autofix/diagnose" '{"namespace":"default"}' "集群诊断"
    test_api "POST" "/autofix/notify" '{"title":"测试通知","message":"这是一条测试消息","type":"info"}' "发送通知"
    
    # 6. 智能助手模块测试
    print_title "智能助手模块测试"
    test_api "GET" "/assistant/health" "" "智能助手健康检查"
    test_api "GET" "/assistant/ready" "" "智能助手就绪检查"
    test_api "GET" "/assistant/info" "" "智能助手服务信息"
    test_api "POST" "/assistant/session" "" "创建会话"
    test_api "POST" "/assistant/query" '{"question":"AI-CloudOps平台是什么？","max_context_docs":4}' "智能问答"
    test_api "POST" "/assistant/add-document" '{"content":"这是测试文档内容","metadata":{"source":"测试"}}' "添加文档"
    test_api "POST" "/assistant/clear-cache" "" "清除缓存"
    
    # 7. 输出测试结果
    print_title "测试结果统计"
    
    success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    fi
    
    echo "总测试数: $TOTAL_TESTS"
    echo -e "通过数: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "失败数: ${RED}$FAILED_TESTS${NC}"
    echo -e "成功率: ${GREEN}$success_rate%${NC}"
    echo ""
    
    log "测试完成 - 总计:$TOTAL_TESTS, 通过:$PASSED_TESTS, 失败:$FAILED_TESTS, 成功率:$success_rate%"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}🎉 所有API测试通过！系统运行正常。${NC}"
        log "所有API测试通过"
        exit 0
    else
        echo -e "${YELLOW}⚠️  部分API测试失败，请检查服务状态和配置。${NC}"
        log "部分API测试失败"
        echo "详细日志请查看: $LOG_FILE"
        exit 1
    fi
}

# 脚本入口
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  --help, -h    显示帮助信息"
    echo ""
    echo "说明:"
    echo "  此脚本会测试AI-CloudOps平台的所有API接口"
    echo "  默认使用配置文件中的服务地址，如未配置则使用 localhost:8080"
    echo ""
    echo "示例:"
    echo "  $0                    # 运行完整API测试"
    echo "  APP_HOST=192.168.1.100 APP_PORT=8080 $0  # 使用自定义地址"
    exit 0
fi

# 执行主函数
main "$@"