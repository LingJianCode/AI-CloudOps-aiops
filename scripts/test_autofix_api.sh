#!/bin/bash

# AI-CloudOps-aiops 自动修复API测试脚本
# 作者: AI-CloudOps 团队
# 功能: 测试自动修复模块的所有API接口

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
LOG_FILE="logs/autofix_api_test_$(date +%Y%m%d_%H%M%S).log"
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
    echo -e "${BLUE}  AI-CloudOps 自动修复API测试套件${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo "测试时间: $(date)"
    echo "API地址: $API_URL"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    log "开始自动修复API测试"
    log "配置: APP_HOST=$APP_HOST, APP_PORT=$APP_PORT"
    
    # 检查服务状态
    if ! check_service; then
        exit 1
    fi
    
    # 1. 自动修复服务健康检查
    print_title "自动修复服务健康检查"
    test_api "GET" "/autofix/health" "" "自动修复服务健康检查"
    test_api "GET" "/autofix/ready" "" "自动修复服务就绪检查"
    test_api "GET" "/autofix/info" "" "自动修复服务信息"
    
    # 2. 集群诊断接口测试
    print_title "集群诊断接口测试"
    test_api "POST" "/autofix/diagnose" '{"namespace":"default"}' "默认命名空间诊断"
    test_api "POST" "/autofix/diagnose" '{"namespace":"kube-system"}' "系统命名空间诊断"
    test_api "POST" "/autofix/diagnose" '{"namespace":"monitoring"}' "监控命名空间诊断"
    
    # 3. K8s修复接口测试
    print_title "K8s修复接口测试"
    test_api "POST" "/autofix" '{
        "deployment": "test-app",
        "namespace": "default",
        "event": "Pod处于CrashLoopBackOff状态"
    }' "基本部署修复"
    
    test_api "POST" "/autofix" '{
        "deployment": "nginx-deployment",
        "namespace": "default",
        "event": "Pod启动失败，镜像拉取错误",
        "auto_restart": true
    }' "镜像拉取错误修复"
    
    test_api "POST" "/autofix" '{
        "deployment": "api-server",
        "namespace": "production",
        "event": "服务响应超时，需要重启",
        "force": false
    }' "生产环境修复"
    
    # 4. 工作流执行测试
    print_title "工作流执行测试"
    test_api "POST" "/autofix/workflow" '{
        "problem_description": "系统CPU使用率过高，内存不足"
    }' "CPU和内存问题工作流"
    
    test_api "POST" "/autofix/workflow" '{
        "problem_description": "数据库连接池耗尽，应用无法连接数据库",
        "priority": "high"
    }' "数据库连接问题工作流"
    
    test_api "POST" "/autofix/workflow" '{
        "problem_description": "网络延迟增加，服务间通信超时",
        "auto_execute": false
    }' "网络问题工作流"
    
    # 5. 通知发送测试
    print_title "通知发送测试"
    test_api "POST" "/autofix/notify" '{
        "title": "自动修复测试通知",
        "message": "这是一条测试通知消息",
        "type": "info"
    }' "信息通知"
    
    test_api "POST" "/autofix/notify" '{
        "title": "警告通知",
        "message": "检测到系统异常，需要关注",
        "type": "warning"
    }' "警告通知"
    
    test_api "POST" "/autofix/notify" '{
        "title": "错误通知",
        "message": "自动修复执行失败",
        "type": "error"
    }' "错误通知"
    
    # 6. 修复历史和状态
    print_title "修复历史和状态"
    test_api "GET" "/autofix/history?limit=10" "" "修复历史记录"
    test_api "GET" "/autofix/status" "" "修复服务状态"
    test_api "GET" "/autofix/metrics" "" "修复服务指标"
    
    # 7. 配置管理测试
    print_title "配置管理测试"
    test_api "GET" "/autofix/config" "" "获取修复配置"
    test_api "POST" "/autofix/config/reload" "" "重新加载配置"
    
    # 8. 错误处理测试
    print_title "错误处理测试"
    test_api "POST" "/autofix/diagnose" '{}' "缺少命名空间参数" 400
    test_api "POST" "/autofix/diagnose" '{"namespace":""}' "空命名空间" 400
    test_api "POST" "/autofix/diagnose" '{"namespace":"invalid-ns!"}' "无效命名空间名称" 400
    
    test_api "POST" "/autofix" '{}' "缺少必需参数" 400
    test_api "POST" "/autofix" '{"deployment":""}' "空部署名称" 400
    test_api "POST" "/autofix" '{"deployment":"test","namespace":"","event":"test"}' "空命名空间" 400
    test_api "POST" "/autofix" '{"deployment":"invalid-name!","namespace":"default","event":"test"}' "无效部署名称" 400
    
    test_api "POST" "/autofix/workflow" '{}' "缺少问题描述" 400
    test_api "POST" "/autofix/workflow" '{"problem_description":""}' "空问题描述" 400
    
    test_api "POST" "/autofix/notify" '{}' "缺少通知参数" 400
    test_api "POST" "/autofix/notify" '{"title":""}' "空标题" 400
    test_api "POST" "/autofix/notify" '{"title":"test"}' "缺少消息" 400
    test_api "POST" "/autofix/notify" '{"title":"test","message":"test","type":"invalid"}' "无效通知类型" 400
    
    # 9. 边界值测试
    print_title "边界值测试"
    test_api "POST" "/autofix" '{
        "deployment": "very-long-deployment-name-that-might-exceed-limits",
        "namespace": "default",
        "event": "测试长部署名称"
    }' "长部署名称测试"
    
    test_api "POST" "/autofix/workflow" '{
        "problem_description": "'$(printf 'A%.0s' {1..1000})'"
    }' "长问题描述测试"
    
    test_api "POST" "/autofix/notify" '{
        "title": "极长的通知标题'$(printf 'T%.0s' {1..100})'",
        "message": "测试长标题通知",
        "type": "info"
    }' "长标题通知测试"
    
    # 10. 并发和性能测试
    print_title "性能测试"
    test_api "GET" "/autofix/info" "" "信息接口性能测试"
    test_api "GET" "/autofix/health" "" "健康检查性能测试"
    test_api "GET" "/autofix/status" "" "状态接口性能测试"
    
    # 11. 高级功能测试
    print_title "高级功能测试"
    test_api "POST" "/autofix" '{
        "deployment": "complex-app",
        "namespace": "staging",
        "event": "多容器Pod中某个容器重启循环",
        "force": true,
        "auto_restart": true,
        "timeout": 300
    }' "复杂修复场景"
    
    test_api "POST" "/autofix/workflow" '{
        "problem_description": "集群整体性能下降，多个服务响应缓慢",
        "priority": "critical",
        "auto_execute": true,
        "affected_services": ["api-gateway", "user-service", "order-service"]
    }' "集群级别修复工作流"
    
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
        echo -e "${GREEN}🎉 自动修复API测试全部通过！${NC}"
        log "自动修复API测试全部通过"
        exit 0
    else
        echo -e "${YELLOW}⚠️  部分自动修复API测试失败，请检查服务状态。${NC}"
        log "部分自动修复API测试失败"
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
    echo "  此脚本测试AI-CloudOps平台的自动修复模块API"
    echo "  包含集群诊断、K8s修复、工作流执行和通知发送测试"
    echo "  默认使用配置文件中的服务地址，如未配置则使用 localhost:8080"
    echo ""
    echo "示例:"
    echo "  $0                    # 运行自动修复API测试"
    echo "  APP_HOST=192.168.1.100 APP_PORT=8080 $0  # 使用自定义地址"
    exit 0
fi

# 执行主函数
main "$@"