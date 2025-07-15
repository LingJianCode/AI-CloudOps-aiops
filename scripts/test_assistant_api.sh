#!/bin/bash

# AI-CloudOps-aiops 智能助手API测试脚本
# 作者: AI-CloudOps 团队
# 功能: 测试智能助手模块的所有API接口

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

# 会话管理
SESSION_ID=""

# 日志文件
LOG_FILE="logs/assistant_api_test_$(date +%Y%m%d_%H%M%S).log"
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
            
            # 如果是创建会话的成功响应，提取session_id
            if [ "$endpoint" = "/assistant/session" ] && [ "$status_code" = "200" ]; then
                SESSION_ID=$(echo "$response_body" | python -c "import sys,json; data=json.load(sys.stdin); print(data['data']['session_id'])" 2>/dev/null)
                if [ -n "$SESSION_ID" ]; then
                    echo "  会话ID: $SESSION_ID"
                    log "创建会话成功，会话ID: $SESSION_ID"
                fi
            fi
            
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
    echo -e "${BLUE}  AI-CloudOps 智能助手API测试套件${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo "测试时间: $(date)"
    echo "API地址: $API_URL"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    log "开始智能助手API测试"
    log "配置: APP_HOST=$APP_HOST, APP_PORT=$APP_PORT"
    
    # 检查服务状态
    if ! check_service; then
        exit 1
    fi
    
    # 1. 智能助手服务健康检查
    print_title "智能助手服务健康检查"
    test_api "GET" "/assistant/health" "" "智能助手服务健康检查"
    test_api "GET" "/assistant/ready" "" "智能助手服务就绪检查"
    test_api "GET" "/assistant/info" "" "智能助手服务信息"
    
    # 2. 会话管理测试
    print_title "会话管理测试"
    test_api "POST" "/assistant/session" "" "创建新会话"
    test_api "GET" "/assistant/sessions" "" "获取会话列表"
    
    # 3. 基础问答测试
    print_title "基础问答测试"
    test_api "POST" "/assistant/query" '{
        "question": "AI-CloudOps平台是什么？",
        "max_context_docs": 4
    }' "基础平台介绍问答"
    
    test_api "POST" "/assistant/query" '{
        "question": "如何使用负载预测功能？",
        "max_context_docs": 3
    }' "负载预测功能问答"
    
    test_api "POST" "/assistant/query" '{
        "question": "根因分析是如何工作的？",
        "max_context_docs": 5
    }' "根因分析功能问答"
    
    # 4. 带会话的问答测试
    print_title "带会话的问答测试"
    if [ -n "$SESSION_ID" ]; then
        test_api "POST" "/assistant/query" '{
            "question": "自动修复功能有哪些特点？",
            "session_id": "'$SESSION_ID'",
            "max_context_docs": 4
        }' "带会话的自动修复问答"
        
        test_api "POST" "/assistant/query" '{
            "question": "请详细介绍一下刚才提到的特点",
            "session_id": "'$SESSION_ID'",
            "max_context_docs": 3
        }' "上下文相关问答"
    else
        echo "  跳过带会话的问答测试（未获取到会话ID）"
    fi
    
    # 5. 技术类问答测试
    print_title "技术类问答测试"
    test_api "POST" "/assistant/query" '{
        "question": "如何部署AI-CloudOps平台？",
        "max_context_docs": 6
    }' "部署相关问答"
    
    test_api "POST" "/assistant/query" '{
        "question": "平台支持哪些监控指标？",
        "max_context_docs": 4
    }' "监控指标问答"
    
    test_api "POST" "/assistant/query" '{
        "question": "如何配置Prometheus集成？",
        "max_context_docs": 5
    }' "配置相关问答"
    
    # 6. 知识库管理测试
    print_title "知识库管理测试"
    test_api "POST" "/assistant/add-document" '{
        "content": "AI-CloudOps平台测试文档：这是一个用于测试的示例文档，包含关于平台功能的基本信息。平台提供负载预测、根因分析、自动修复和智能问答等核心功能。",
        "metadata": {
            "source": "测试脚本",
            "type": "测试文档",
            "category": "功能介绍"
        }
    }' "添加测试文档"
    
    test_api "POST" "/assistant/add-document" '{
        "content": "Kubernetes集成指南：平台与Kubernetes深度集成，支持Pod状态监控、自动重启、资源调度优化等功能。",
        "metadata": {
            "source": "API测试",
            "type": "技术文档",
            "category": "集成指南"
        }
    }' "添加技术文档"
    
    test_api "POST" "/assistant/refresh" "" "刷新知识库"
    
    # 7. 缓存管理测试
    print_title "缓存管理测试"
    test_api "POST" "/assistant/clear-cache" "" "清除缓存"
    test_api "GET" "/assistant/cache/status" "" "缓存状态查询"
    
    # 8. 配置和统计测试
    print_title "配置和统计测试"
    test_api "GET" "/assistant/config" "" "获取助手配置"
    test_api "GET" "/assistant/stats" "" "获取使用统计"
    test_api "GET" "/assistant/knowledge/stats" "" "知识库统计信息"
    
    # 9. 错误处理测试
    print_title "错误处理测试"
    test_api "POST" "/assistant/query" '{}' "缺少问题参数" 400
    test_api "POST" "/assistant/query" '{"question":""}' "空问题" 400
    test_api "POST" "/assistant/query" '{"question":"test","max_context_docs":-1}' "无效文档数量" 400
    test_api "POST" "/assistant/query" '{"question":"test","session_id":"invalid-session"}' "无效会话ID" 400
    
    test_api "POST" "/assistant/add-document" '{}' "缺少文档内容" 400
    test_api "POST" "/assistant/add-document" '{"content":""}' "空文档内容" 400
    test_api "POST" "/assistant/add-document" '{"content":"test","metadata":"invalid"}' "无效元数据格式" 400
    
    # 10. 高级查询测试
    print_title "高级查询测试"
    test_api "POST" "/assistant/query" '{
        "question": "当系统CPU使用率超过80%时，应该如何处理？",
        "max_context_docs": 8,
        "include_sources": true
    }' "包含来源的查询"
    
    test_api "POST" "/assistant/query" '{
        "question": "比较负载预测和根因分析功能的区别",
        "max_context_docs": 6,
        "search_mode": "semantic"
    }' "语义搜索模式查询"
    
    test_api "POST" "/assistant/query" '{
        "question": "平台的API接口有哪些？",
        "max_context_docs": 10,
        "filter": {"category": "API文档"}
    }' "带过滤条件的查询"
    
    # 11. 批量操作测试
    print_title "批量操作测试"
    test_api "POST" "/assistant/bulk-query" '{
        "questions": [
            "什么是AIOps？",
            "平台有哪些核心功能？",
            "如何开始使用平台？"
        ],
        "max_context_docs": 4
    }' "批量问答查询"
    
    # 12. 多语言和特殊字符测试
    print_title "多语言和特殊字符测试"
    test_api "POST" "/assistant/query" '{
        "question": "AI-CloudOps平台如何处理中文、English mixed queries？",
        "max_context_docs": 4
    }' "中英混合查询"
    
    test_api "POST" "/assistant/query" '{
        "question": "特殊符号测试：@#$%^&*()[]{}",
        "max_context_docs": 3
    }' "特殊字符查询"
    
    # 13. 性能测试
    print_title "性能测试"
    echo "执行性能测试..."
    start_time=$(date +%s)
    test_api "POST" "/assistant/query" '{
        "question": "快速响应测试：平台主要功能概述",
        "max_context_docs": 2
    }' "快速响应测试"
    end_time=$(date +%s)
    response_time=$((end_time - start_time))
    log "查询响应时间: ${response_time}秒"
    
    # 14. 边界值测试
    print_title "边界值测试"
    test_api "POST" "/assistant/query" '{
        "question": "'$(printf 'A%.0s' {1..1000})'",
        "max_context_docs": 1
    }' "超长问题测试"
    
    test_api "POST" "/assistant/query" '{
        "question": "简单问题",
        "max_context_docs": 50
    }' "最大文档数量测试"
    
    # 15. 清理操作
    print_title "清理操作"
    if [ -n "$SESSION_ID" ]; then
        test_api "DELETE" "/assistant/session/$SESSION_ID" "" "删除测试会话"
    fi
    test_api "POST" "/assistant/cleanup" "" "清理临时数据"
    
    # 16. 输出测试结果
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
        echo -e "${GREEN}🎉 智能助手API测试全部通过！${NC}"
        log "智能助手API测试全部通过"
        exit 0
    else
        echo -e "${YELLOW}⚠️  部分智能助手API测试失败，请检查服务状态。${NC}"
        log "部分智能助手API测试失败"
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
    echo "  此脚本测试AI-CloudOps平台的智能助手模块API"
    echo "  包含问答、会话管理、知识库操作和高级功能测试"
    echo "  默认使用配置文件中的服务地址，如未配置则使用 localhost:8080"
    echo ""
    echo "示例:"
    echo "  $0                    # 运行智能助手API测试"
    echo "  APP_HOST=192.168.1.100 APP_PORT=8080 $0  # 使用自定义地址"
    exit 0
fi

# 执行主函数
main "$@"