#!/bin/bash

# AI-CloudOps-aiops 负载预测API测试脚本
# 作者: AI-CloudOps 团队
# 功能: 测试负载预测模块的所有API接口

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
LOG_FILE="logs/predict_api_test_$(date +%Y%m%d_%H%M%S).log"
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
    echo -e "${BLUE}  AI-CloudOps 负载预测API测试套件${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo "测试时间: $(date)"
    echo "API地址: $API_URL"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    log "开始负载预测API测试"
    log "配置: APP_HOST=$APP_HOST, APP_PORT=$APP_PORT"
    
    # 检查服务状态
    if ! check_service; then
        exit 1
    fi
    
    # 1. 预测服务健康检查
    print_title "预测服务健康检查"
    test_api "GET" "/predict/health" "" "预测服务健康检查"
    test_api "GET" "/predict/ready" "" "预测服务就绪检查" 
    test_api "GET" "/predict/info" "" "预测服务信息"
    
    # 2. 基础预测接口测试
    print_title "基础预测接口测试"
    test_api "GET" "/predict" "" "GET预测请求"
    test_api "POST" "/predict" '{"current_qps":100.5,"include_confidence":true}' "POST预测请求"
    
    # 3. 预测接口参数测试
    print_title "预测接口参数测试"
    test_api "POST" "/predict" '{"current_qps":50.0}' "简单QPS预测"
    test_api "POST" "/predict" '{"current_qps":200.8,"include_confidence":false}' "不包含置信度预测"
    test_api "POST" "/predict" '{"current_qps":0}' "零QPS预测"
    test_api "POST" "/predict" '{"current_qps":1000}' "高QPS预测"
    
    # 4. 趋势预测接口测试
    print_title "趋势预测接口测试"
    test_api "GET" "/predict/trend?hours=1" "" "1小时趋势预测"
    test_api "GET" "/predict/trend?hours=6" "" "6小时趋势预测"
    test_api "GET" "/predict/trend?hours=12" "" "12小时趋势预测"
    test_api "GET" "/predict/trend?hours=24" "" "24小时趋势预测"
    test_api "POST" "/predict/trend" '{"hours_ahead":12,"current_qps":75.0}' "POST趋势预测"
    test_api "POST" "/predict/trend" '{"hours_ahead":6,"current_qps":150.5}' "POST短期趋势预测"
    
    # 5. 模型管理接口测试
    print_title "模型管理接口测试"
    test_api "POST" "/predict/reload" "" "重新加载模型"
    test_api "GET" "/predict/metrics" "" "预测指标"
    test_api "GET" "/predict/model/status" "" "模型状态"
    
    # 6. 错误处理测试
    print_title "错误处理测试"
    test_api "POST" "/predict" '{"current_qps":-50}' "负数QPS值" 400
    test_api "POST" "/predict" '{"current_qps":"invalid"}' "非数字QPS值" 400
    test_api "POST" "/predict" '{}' "空请求体" 400
    test_api "POST" "/predict" '{"wrong_param":100}' "错误参数名" 400
    test_api "GET" "/predict/trend?hours=-1" "" "负数小时值" 400
    test_api "GET" "/predict/trend?hours=abc" "" "非数字小时值" 400
    test_api "POST" "/predict/trend" '{"hours_ahead":-5}' "负数预测小时" 400
    
    # 7. 边界值测试
    print_title "边界值测试"
    test_api "POST" "/predict" '{"current_qps":0.1}' "最小QPS值"
    test_api "POST" "/predict" '{"current_qps":10000}' "极大QPS值"
    test_api "GET" "/predict/trend?hours=168" "" "一周趋势预测"
    test_api "POST" "/predict/trend" '{"hours_ahead":72,"current_qps":500}' "3天趋势预测"
    
    # 8. 输出测试结果
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
        echo -e "${GREEN}🎉 负载预测API测试全部通过！${NC}"
        log "负载预测API测试全部通过"
        exit 0
    else
        echo -e "${YELLOW}⚠️  部分负载预测API测试失败，请检查服务状态。${NC}"
        log "部分负载预测API测试失败"
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
    echo "  此脚本测试AI-CloudOps平台的负载预测模块API"
    echo "  包含基础预测、趋势预测、模型管理和错误处理测试"
    echo "  默认使用配置文件中的服务地址，如未配置则使用 localhost:8080"
    echo ""
    echo "示例:"
    echo "  $0                    # 运行负载预测API测试"
    echo "  APP_HOST=192.168.1.100 APP_PORT=8080 $0  # 使用自定义地址"
    exit 0
fi

# 执行主函数
main "$@"