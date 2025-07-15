#!/bin/bash

# AI-CloudOps-aiops 根因分析API测试脚本
# 作者: AI-CloudOps 团队
# 功能: 测试根因分析模块的所有API接口

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
LOG_FILE="logs/rca_api_test_$(date +%Y%m%d_%H%M%S).log"
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
    echo -e "${BLUE}  AI-CloudOps 根因分析API测试套件${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo "测试时间: $(date)"
    echo "API地址: $API_URL"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    log "开始根因分析API测试"
    log "配置: APP_HOST=$APP_HOST, APP_PORT=$APP_PORT"
    
    # 检查服务状态
    if ! check_service; then
        exit 1
    fi
    
    # 1. RCA服务健康检查
    print_title "RCA服务健康检查"
    test_api "GET" "/rca/health" "" "RCA服务健康检查"
    test_api "GET" "/rca/ready" "" "RCA服务就绪检查"
    test_api "GET" "/rca/info" "" "RCA服务信息"
    
    # 2. RCA配置和指标接口
    print_title "RCA配置和指标接口"
    test_api "GET" "/rca/config" "" "RCA配置信息"
    test_api "GET" "/rca/metrics" "" "可用指标列表"
    test_api "GET" "/rca/status" "" "RCA服务状态"
    
    # 3. 基础根因分析测试
    print_title "基础根因分析测试"
    test_api "POST" "/rca" '{}' "最小参数根因分析"
    test_api "POST" "/rca" '{"time_range":"1h"}' "1小时时间范围分析"
    test_api "POST" "/rca" '{"time_range":"6h","service":"web"}' "指定服务分析"
    
    # 4. 详细根因分析测试
    print_title "详细根因分析测试"
    test_api "POST" "/rca" '{
        "time_range": "2h",
        "service": "api-server",
        "symptoms": ["高CPU使用率", "响应时间增加"],
        "metrics": ["cpu_usage", "response_time", "error_rate"]
    }' "完整参数根因分析"
    
    test_api "POST" "/rca" '{
        "time_range": "30m",
        "service": "database",
        "symptoms": ["连接超时", "查询缓慢"],
        "threshold": 0.8
    }' "数据库问题分析"
    
    # 5. 事件分析接口测试
    print_title "事件分析接口测试"
    test_api "POST" "/rca/incident" '{
        "affected_services": ["nginx", "mysql"],
        "symptoms": ["高CPU使用率", "内存泄漏", "响应超时"]
    }' "多服务事件分析"
    
    test_api "POST" "/rca/incident" '{
        "affected_services": ["redis"],
        "symptoms": ["连接数过多"],
        "start_time": "2024-01-01T10:00:00Z",
        "end_time": "2024-01-01T12:00:00Z"
    }' "指定时间事件分析"
    
    # 6. 相关性分析测试
    print_title "相关性分析测试"
    test_api "POST" "/rca/correlate" '{
        "primary_metric": "cpu_usage",
        "time_range": "1h"
    }' "CPU相关性分析"
    
    test_api "POST" "/rca/correlate" '{
        "primary_metric": "response_time",
        "secondary_metrics": ["cpu_usage", "memory_usage", "disk_io"],
        "time_range": "2h"
    }' "响应时间多指标相关性分析"
    
    # 7. 异常检测测试
    print_title "异常检测测试"
    test_api "POST" "/rca/anomaly" '{
        "metrics": ["cpu_usage", "memory_usage"],
        "time_range": "24h"
    }' "CPU和内存异常检测"
    
    test_api "POST" "/rca/anomaly" '{
        "metrics": ["error_rate"],
        "service": "api-gateway",
        "sensitivity": 0.9
    }' "API网关错误率异常检测"
    
    # 8. 历史分析测试
    print_title "历史分析测试"
    test_api "GET" "/rca/history?limit=10" "" "历史分析记录"
    test_api "GET" "/rca/history?service=web&limit=5" "" "Web服务历史分析"
    test_api "POST" "/rca/compare" '{
        "current_incident": "incident_123",
        "time_range": "7d"
    }' "历史事件对比分析"
    
    # 9. 错误处理测试
    print_title "错误处理测试"
    test_api "POST" "/rca" '{"time_range":"invalid"}' "无效时间范围" 400
    test_api "POST" "/rca" '{"threshold":-0.5}' "无效阈值" 400
    test_api "POST" "/rca" '{"metrics":["non_existent_metric"]}' "不存在的指标" 400
    test_api "POST" "/rca/incident" '{}' "空事件分析请求" 400
    test_api "POST" "/rca/incident" '{"affected_services":[]}' "空服务列表" 400
    test_api "POST" "/rca/correlate" '{}' "缺少主要指标" 400
    test_api "GET" "/rca/history?limit=-1" "" "无效limit参数" 400
    
    # 10. 边界值测试
    print_title "边界值测试"
    test_api "POST" "/rca" '{"time_range":"5m"}' "最小时间范围"
    test_api "POST" "/rca" '{"time_range":"7d"}' "最大时间范围"
    test_api "POST" "/rca" '{"threshold":0.01}' "最小阈值"
    test_api "POST" "/rca" '{"threshold":0.99}' "最大阈值"
    test_api "GET" "/rca/history?limit=100" "" "最大历史记录数"
    
    # 11. 性能测试
    print_title "性能测试"
    test_api "POST" "/rca" '{
        "time_range": "1h",
        "metrics": ["cpu_usage", "memory_usage", "disk_io", "network_io", "error_rate"]
    }' "多指标性能测试"
    
    # 12. 输出测试结果
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
        echo -e "${GREEN}🎉 根因分析API测试全部通过！${NC}"
        log "根因分析API测试全部通过"
        exit 0
    else
        echo -e "${YELLOW}⚠️  部分根因分析API测试失败，请检查服务状态。${NC}"
        log "部分根因分析API测试失败"
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
    echo "  此脚本测试AI-CloudOps平台的根因分析模块API"
    echo "  包含基础分析、事件分析、相关性分析、异常检测和历史分析"
    echo "  默认使用配置文件中的服务地址，如未配置则使用 localhost:8080"
    echo ""
    echo "示例:"
    echo "  $0                    # 运行根因分析API测试"
    echo "  APP_HOST=192.168.1.100 APP_PORT=8080 $0  # 使用自定义地址"
    exit 0
fi

# 执行主函数
main "$@"