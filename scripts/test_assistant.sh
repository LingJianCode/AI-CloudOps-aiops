#!/bin/bash
# 测试小助手API的脚本 - 简单版本
# 如需完整测试，请使用 test_assistant_enhanced.sh

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd $(dirname $0) && pwd)
ROOT_DIR=$(cd $SCRIPT_DIR/.. && pwd)

# 导入配置读取工具
source "$SCRIPT_DIR/config_reader.sh"

# 读取配置
read_config

# 设置API基础URL，默认从配置文件读取
DEFAULT_URL="http://${APP_HOST}:${APP_PORT}/api/v1/assistant"
BASE_URL="${1:-$DEFAULT_URL}"
HEADER="Content-Type: application/json"

# 测试轮数（可通过参数调整）
ROUNDS=${2:-1}

# JSON解析函数 - 处理Unicode和控制字符
parse_json_safe() {
    local json="$1"
    local key="$2"
    echo "$json" | python3 -c "
import sys, json, re
try:
    content = sys.stdin.read()
    # 清理控制字符
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    content = content.strip()
    
    data = json.loads(content)
    keys = '$key'.split('.')
    result = data
    for k in keys:
        if isinstance(result, dict) and k in result:
            result = result[k]
        else:
            print('')
            exit()
    
    if isinstance(result, (int, float)):
        print(result)
    elif isinstance(result, str):
        print(result)
    else:
        print(str(result))
except Exception as e:
    print('')
" 2>/dev/null
}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}          智能小助手API测试脚本         ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "${YELLOW}API地址: ${BASE_URL}${NC}"
echo -e "${YELLOW}测试轮数: ${ROUNDS}${NC}"
echo -e "${CYAN}提示: 如需完整测试请使用 test_assistant_enhanced.sh${NC}"
echo ""

# 开始多轮测试
for ((round=1; round<=ROUNDS; round++)); do
    if [ $ROUNDS -gt 1 ]; then
        echo -e "${PURPLE}========== 第 $round 轮测试开始 ==========${NC}"
    fi

# 1. 测试创建会话
echo -e "${YELLOW}1. 测试创建会话...${NC}"
SESSION_RESPONSE=$(curl -s -X POST "${BASE_URL}/session" -H "${HEADER}")
# 使用安全的JSON解析函数
SESSION_ID=$(parse_json_safe "$SESSION_RESPONSE" "data.session_id")

if [ -z "$SESSION_ID" ]; then
  echo -e "${RED}创建会话失败!${NC}"
  echo $SESSION_RESPONSE
  exit 1
else
  echo -e "${GREEN}创建会话成功! 会话ID: $SESSION_ID${NC}"
fi
echo ""

# 2. 测试知识库刷新
echo -e "${YELLOW}2. 测试知识库刷新...${NC}"
REFRESH_RESPONSE=$(curl -s -X POST "${BASE_URL}/refresh" -H "${HEADER}")
REFRESH_CODE=$(parse_json_safe "$REFRESH_RESPONSE" "code")

if [ "$REFRESH_CODE" = "0" ]; then
  echo -e "${GREEN}知识库刷新成功!${NC}"
else
  echo -e "${YELLOW}知识库刷新失败或部分成功，继续测试...${NC}"
  echo $REFRESH_RESPONSE
fi
echo ""

# 3. 测试基本查询
echo -e "${YELLOW}3. 测试基本查询...${NC}"
QUERY_DATA='{"question":"AIOps平台是什么?","session_id":"'$SESSION_ID'"}'
echo -e "${BLUE}发送查询: $QUERY_DATA${NC}"

QUERY_RESPONSE=$(curl -s -X POST "${BASE_URL}/query" -H "${HEADER}" -d "$QUERY_DATA")
QUERY_CODE=$(parse_json_safe "$QUERY_RESPONSE" "code")

if [ "$QUERY_CODE" = "0" ]; then
  echo -e "${GREEN}查询成功!${NC}"
  # 提取并显示回答
  ANSWER=$(parse_json_safe "$QUERY_RESPONSE" "data.answer")
  echo -e "${BLUE}回答:${NC} $ANSWER"
else
  echo -e "${RED}查询失败!${NC}"
  echo $QUERY_RESPONSE
fi
echo ""

# 4. 测试上下文查询
echo -e "${YELLOW}4. 测试上下文查询...${NC}"
CONTEXT_DATA='{"question":"核心功能模块有哪些?","session_id":"'$SESSION_ID'"}'
echo -e "${BLUE}发送上下文查询: $CONTEXT_DATA${NC}"

CONTEXT_RESPONSE=$(curl -s -X POST "${BASE_URL}/query" -H "${HEADER}" -d "$CONTEXT_DATA")
CONTEXT_CODE=$(parse_json_safe "$CONTEXT_RESPONSE" "code")

if [ "$CONTEXT_CODE" = "0" ]; then
  echo -e "${GREEN}上下文查询成功!${NC}"
  # 提取并显示回答
  ANSWER=$(parse_json_safe "$CONTEXT_RESPONSE" "data.answer")
  echo -e "${BLUE}回答:${NC} $ANSWER"
else
  echo -e "${RED}上下文查询失败!${NC}"
  echo $CONTEXT_RESPONSE
fi
echo ""

# 5. 测试清除缓存
echo -e "${YELLOW}5. 测试清除缓存...${NC}"
CACHE_RESPONSE=$(curl -s -X POST "${BASE_URL}/clear-cache" -H "${HEADER}")
CACHE_CODE=$(parse_json_safe "$CACHE_RESPONSE" "code")

if [ "$CACHE_CODE" = "0" ]; then
  echo -e "${GREEN}缓存清除成功!${NC}"
else
  echo -e "${RED}缓存清除失败!${NC}"
  echo $CACHE_RESPONSE
fi
echo ""

echo -e "${GREEN}测试完成!${NC}"

    # 轮次间隔
    if [ $round -lt $ROUNDS ]; then
        echo -e "${YELLOW}等待 3 秒后开始下一轮测试...${NC}"
        sleep 3
        echo ""
    fi
done

# 测试总结
if [ $ROUNDS -gt 1 ]; then
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${GREEN}所有 $ROUNDS 轮测试已完成!${NC}"
    echo -e "${CYAN}详细测试请使用: ./test_assistant_enhanced.sh${NC}"
    echo -e "${BLUE}=========================================${NC}"
fi
