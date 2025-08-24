#!/bin/bash
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops 完整部署脚本
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 一键部署所有服务，包括主应用、MCP服务、Prometheus、Redis等
"""

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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

# 显示帮助信息
show_help() {
    cat << EOF
AI-CloudOps AIOps 平台部署脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -d, --dev               开发模式部署（启用调试）
    -p, --production        生产模式部署（默认）
    --build                 强制重新构建镜像
    --no-cache              不使用缓存构建镜像
    --pull                  先拉取最新的基础镜像
    --logs                  部署完成后显示日志
    --stop                  停止所有服务
    --down                  停止并移除所有容器和网络
    --restart               重启所有服务
    --status                显示服务状态

环境变量:
    ENV                     环境类型（development/production）
    LLM_API_KEY            LLM API密钥
    LLM_BASE_URL           LLM API基础URL
    FEISHU_WEBHOOK         飞书通知Webhook
    TAVILY_API_KEY         Tavily搜索API密钥
    K8S_IN_CLUSTER         是否在K8s集群内运行
    K8S_NAMESPACE          K8s命名空间

示例:
    $0                      # 生产模式部署
    $0 --dev               # 开发模式部署
    $0 --build             # 强制重新构建
    $0 --status            # 查看状态
    $0 --down              # 完全清理

EOF
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查curl
    if ! command -v curl &> /dev/null; then
        log_warning "curl 未安装，部分功能可能受影响"
    fi
    
    log_success "依赖检查完成"
}

# 检查配置文件
check_config() {
    log_info "检查配置文件..."
    
    cd "$PROJECT_DIR"
    
    # 检查主配置文件
    if [[ ! -f "config/config.yaml" ]]; then
        log_error "未找到配置文件 config/config.yaml"
        exit 1
    fi
    
    # 检查环境配置文件
    if [[ "$ENV" == "production" && ! -f "config/config.production.yaml" ]]; then
        log_warning "未找到生产环境配置文件 config/config.production.yaml，将使用默认配置"
    fi
    
    # 检查.env文件
    if [[ ! -f ".env" ]]; then
        if [[ -f "env.example" ]]; then
            log_warning "未找到.env文件，正在从env.example创建..."
            cp env.example .env
        else
            log_warning "未找到.env文件和env.example，请手动创建"
        fi
    fi
    
    log_success "配置文件检查完成"
}

# 准备部署环境
prepare_deploy() {
    log_info "准备部署环境..."
    
    cd "$PROJECT_DIR"
    
    # 创建必要的目录
    mkdir -p logs data/models data/sample config
    
    # 设置权限
    chmod 755 logs data config
    
    # 检查Kubernetes配置
    if [[ "${K8S_IN_CLUSTER:-false}" == "false" ]]; then
        if [[ ! -f "deploy/kubernetes/config" && -f "$HOME/.kube/config" ]]; then
            log_info "复制Kubernetes配置文件..."
            mkdir -p deploy/kubernetes
            cp "$HOME/.kube/config" deploy/kubernetes/config
        fi
    fi
    
    log_success "部署环境准备完成"
}

# 构建镜像
build_images() {
    log_info "构建Docker镜像..."
    
    cd "$PROJECT_DIR"
    
    local build_args=""
    if [[ "$NO_CACHE" == "true" ]]; then
        build_args="$build_args --no-cache"
    fi
    
    if [[ "$PULL" == "true" ]]; then
        build_args="$build_args --pull"
    fi
    
    # 构建主应用镜像
    log_info "构建主应用镜像..."
    docker build $build_args -t aiops-platform:latest -f Dockerfile .
    
    # 构建MCP服务镜像
    log_info "构建MCP服务镜像..."
    docker build $build_args -t aiops-mcp:latest -f Dockerfile.mcp .
    
    log_success "镜像构建完成"
}

# 部署服务
deploy_services() {
    log_info "部署服务..."
    
    cd "$PROJECT_DIR"
    
    # 使用适当的docker-compose命令
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    local compose_args=""
    if [[ "$BUILD" == "true" ]]; then
        compose_args="$compose_args --build"
    fi
    
    # 启动服务
    $compose_cmd up -d $compose_args
    
    log_success "服务部署完成"
}

# 显示服务状态
show_status() {
    log_info "显示服务状态..."
    
    cd "$PROJECT_DIR"
    
    # 使用适当的docker-compose命令
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    $compose_cmd ps
    
    echo ""
    log_info "服务访问地址:"
    echo "  - 主应用:        http://localhost:8080"
    echo "  - MCP服务:       http://localhost:9000"
    echo "  - Prometheus:    http://localhost:9090"
    echo "  - Redis:         localhost:6379"
    echo "  - Ollama:        http://localhost:11434"
}

# 显示日志
show_logs() {
    log_info "显示服务日志..."
    
    cd "$PROJECT_DIR"
    
    # 使用适当的docker-compose命令
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    $compose_cmd logs -f --tail=100
}

# 停止服务
stop_services() {
    log_info "停止服务..."
    
    cd "$PROJECT_DIR"
    
    # 使用适当的docker-compose命令
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    $compose_cmd stop
    
    log_success "服务已停止"
}

# 完全清理
cleanup_services() {
    log_info "清理所有容器和网络..."
    
    cd "$PROJECT_DIR"
    
    # 使用适当的docker-compose命令
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    $compose_cmd down -v --remove-orphans
    
    log_success "清理完成"
}

# 重启服务
restart_services() {
    log_info "重启服务..."
    
    cd "$PROJECT_DIR"
    
    # 使用适当的docker-compose命令
    local compose_cmd="docker-compose"
    if docker compose version &> /dev/null; then
        compose_cmd="docker compose"
    fi
    
    $compose_cmd restart
    
    log_success "服务已重启"
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dev)
                export ENV="development"
                shift
                ;;
            -p|--production)
                export ENV="production"
                shift
                ;;
            --build)
                BUILD="true"
                shift
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --pull)
                PULL="true"
                shift
                ;;
            --logs)
                SHOW_LOGS="true"
                shift
                ;;
            --stop)
                stop_services
                exit 0
                ;;
            --down)
                cleanup_services
                exit 0
                ;;
            --restart)
                restart_services
                exit 0
                ;;
            --status)
                show_status
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认环境
    export ENV="${ENV:-production}"
    
    echo "=================================="
    echo "AI-CloudOps AIOps 平台部署脚本"
    echo "环境: $ENV"
    echo "时间: $(date)"
    echo "项目目录: $PROJECT_DIR"
    echo "=================================="
    
    # 执行部署流程
    check_dependencies
    check_config
    prepare_deploy
    
    if [[ "$BUILD" == "true" || "$NO_CACHE" == "true" || "$PULL" == "true" ]]; then
        build_images
    fi
    
    deploy_services
    
    # 等待服务启动
    log_info "等待服务启动..."
    sleep 10
    
    show_status
    
    if [[ "$SHOW_LOGS" == "true" ]]; then
        show_logs
    else
        echo ""
        log_success "部署完成！"
        log_info "使用 '$0 --logs' 查看日志"
        log_info "使用 '$0 --status' 查看状态"
    fi
}

# 捕获Ctrl+C信号
trap 'log_warning "部署被中断"; exit 1' INT

# 执行主函数
main "$@"
