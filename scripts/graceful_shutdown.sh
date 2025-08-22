#!/bin/bash

# AI-CloudOps 优雅关闭脚本
# Author: Bamboo
# Description: 用于优雅地关闭AI-CloudOps应用程序

set -e

# 配置
APP_NAME="AI-CloudOps"
PID_FILE="/tmp/aiops.pid"
MAX_WAIT_TIME=60  # 最大等待时间（秒）
CHECK_INTERVAL=2  # 检查间隔（秒）

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 获取应用进程ID
get_app_pid() {
    # 方法1: 从PID文件读取
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    
    # 方法2: 通过进程名查找
    local pid=$(pgrep -f "app.main:app" 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
        echo "$pid"
        return 0
    fi
    
    # 方法3: 通过端口查找
    local pid=$(lsof -ti:8000 2>/dev/null | head -1)
    if [[ -n "$pid" ]]; then
        echo "$pid"
        return 0
    fi
    
    return 1
}

# 检查应用状态
check_app_status() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0  # 运行中
    else
        return 1  # 已停止
    fi
}

# 发送信号给应用
send_signal() {
    local pid="$1"
    local signal="$2"
    local signal_name="$3"
    
    log_info "发送 $signal_name 信号给进程 $pid..."
    if kill -"$signal" "$pid" 2>/dev/null; then
        return 0
    else
        log_error "发送信号失败"
        return 1
    fi
}

# 等待进程结束
wait_for_shutdown() {
    local pid="$1"
    local wait_time=0
    
    log_info "等待应用优雅关闭..."
    
    while [[ $wait_time -lt $MAX_WAIT_TIME ]]; do
        if ! check_app_status "$pid"; then
            log_success "应用已成功关闭"
            return 0
        fi
        
        sleep $CHECK_INTERVAL
        wait_time=$((wait_time + CHECK_INTERVAL))
        
        # 显示进度
        echo -n "."
    done
    
    echo
    log_warn "等待超时"
    return 1
}

# 强制关闭
force_shutdown() {
    local pid="$1"
    
    log_warn "强制关闭应用..."
    if send_signal "$pid" "KILL" "SIGKILL"; then
        sleep 2
        if ! check_app_status "$pid"; then
            log_success "应用已强制关闭"
            return 0
        fi
    fi
    
    log_error "强制关闭失败"
    return 1
}

# 清理资源
cleanup_resources() {
    log_info "清理资源..."
    
    # 清理PID文件
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
        log_info "已清理PID文件"
    fi
    
    # 清理临时文件
    if [[ -d "/tmp/aiops_temp" ]]; then
        rm -rf "/tmp/aiops_temp"
        log_info "已清理临时文件"
    fi
    
    log_success "资源清理完成"
}

# 获取应用状态报告
get_status_report() {
    local pid="$1"
    
    if check_app_status "$pid"; then
        log_info "发送状态查询信号..."
        send_signal "$pid" "USR1" "SIGUSR1"
        sleep 1
        echo "请查看应用日志获取详细状态报告"
    else
        log_warn "应用未运行，无法获取状态报告"
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "  $APP_NAME 优雅关闭脚本"
    echo "========================================"
    
    # 获取应用进程ID
    log_info "查找应用进程..."
    local app_pid
    if app_pid=$(get_app_pid); then
        log_info "找到应用进程: $app_pid"
    else
        log_warn "未找到运行中的应用进程"
        exit 0
    fi
    
    # 处理命令行参数
    case "${1:-}" in
        "status")
            get_status_report "$app_pid"
            exit 0
            ;;
        "force")
            log_warn "执行强制关闭..."
            force_shutdown "$app_pid"
            cleanup_resources
            exit $?
            ;;
        "help"|"-h"|"--help")
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  (无参数)    优雅关闭应用"
            echo "  status      获取应用状态报告"
            echo "  force       强制关闭应用"
            echo "  help        显示此帮助信息"
            exit 0
            ;;
    esac
    
    # 执行优雅关闭
    log_info "开始优雅关闭流程..."
    
    # 1. 发送SIGTERM信号
    if send_signal "$app_pid" "TERM" "SIGTERM"; then
        # 2. 等待优雅关闭
        if wait_for_shutdown "$app_pid"; then
            cleanup_resources
            log_success "$APP_NAME 已优雅关闭"
            exit 0
        fi
    fi
    
    # 3. 如果优雅关闭失败，尝试SIGINT
    log_warn "SIGTERM关闭失败，尝试SIGINT..."
    if send_signal "$app_pid" "INT" "SIGINT"; then
        if wait_for_shutdown "$app_pid"; then
            cleanup_resources
            log_success "$APP_NAME 已关闭"
            exit 0
        fi
    fi
    
    # 4. 询问是否强制关闭
    echo
    log_warn "优雅关闭失败"
    read -p "是否强制关闭应用? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        force_shutdown "$app_pid"
        cleanup_resources
    else
        log_info "取消关闭操作"
        exit 1
    fi
}

# 执行主函数
main "$@"
