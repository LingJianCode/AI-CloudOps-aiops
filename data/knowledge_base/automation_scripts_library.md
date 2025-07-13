# 智能运维系统自动化脚本库

## 概述
本脚本库为智能运维系统提供了全面的自动化运维脚本，涵盖系统监控、自动部署、故障恢复、性能优化等各个方面。

## 系统监控脚本

### 1. 系统健康检查脚本
```bash
#!/bin/bash
# system_health_check.sh - 系统健康状态检查

# 配置参数
CPU_THRESHOLD=80
MEMORY_THRESHOLD=85
DISK_THRESHOLD=90
LOAD_THRESHOLD=5

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================= 系统健康检查报告 ================="
echo "检查时间: $(date)"
echo "主机名: $(hostname)"
echo "=================================================="

# CPU使用率检查
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )); then
    echo -e "${RED}⚠ CPU使用率异常: ${cpu_usage}%${NC}"
else
    echo -e "${GREEN}✓ CPU使用率正常: ${cpu_usage}%${NC}"
fi

# 内存使用率检查
memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
if (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
    echo -e "${RED}⚠ 内存使用率异常: ${memory_usage}%${NC}"
else
    echo -e "${GREEN}✓ 内存使用率正常: ${memory_usage}%${NC}"
fi

# 磁盘使用率检查
df -h | awk 'NR>1 {print $5 " " $6}' | while read output; do
    usage=$(echo $output | awk '{print $1}' | cut -d'%' -f1)
    partition=$(echo $output | awk '{print $2}')
    if [ $usage -ge $DISK_THRESHOLD ]; then
        echo -e "${RED}⚠ 磁盘使用率异常: $partition $usage%${NC}"
    else
        echo -e "${GREEN}✓ 磁盘使用率正常: $partition $usage%${NC}"
    fi
done

# 系统负载检查
load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | cut -d',' -f1)
if (( $(echo "$load_avg > $LOAD_THRESHOLD" | bc -l) )); then
    echo -e "${RED}⚠ 系统负载异常: ${load_avg}${NC}"
else
    echo -e "${GREEN}✓ 系统负载正常: ${load_avg}${NC}"
fi

# 网络连接检查
network_connections=$(netstat -an | grep ESTABLISHED | wc -l)
echo -e "${GREEN}✓ 网络连接数: ${network_connections}${NC}"

# 服务状态检查
services=("nginx" "mysql" "redis" "docker")
for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        echo -e "${GREEN}✓ $service 服务运行正常${NC}"
    else
        echo -e "${RED}⚠ $service 服务未运行${NC}"
    fi
done

echo "=================================================="
echo "健康检查完成: $(date)"
```

### 2. 性能监控脚本
```python
#!/usr/bin/env python3
# performance_monitor.py - 性能实时监控

import psutil
import time
import json
import requests
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.alert_webhook = "http://localhost:8080/api/alerts"
        self.thresholds = {
            'cpu': 80.0,
            'memory': 85.0,
            'disk': 90.0,
            'network_io': 1000000  # 1MB/s
        }
    
    def collect_metrics(self):
        """收集系统性能指标"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'load_avg': psutil.getloadavg(),
                'core_count': psutil.cpu_count()
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {},
            'network': psutil.net_io_counters()._asdict(),
            'processes': len(psutil.pids())
        }
        
        # 磁盘使用情况
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                metrics['disk'][partition.mountpoint] = {
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': (usage.used / usage.total) * 100
                }
            except PermissionError:
                continue
        
        return metrics
    
    def check_alerts(self, metrics):
        """检查告警条件"""
        alerts = []
        
        # CPU告警
        if metrics['cpu']['usage_percent'] > self.thresholds['cpu']:
            alerts.append({
                'type': 'cpu_high',
                'message': f"CPU使用率过高: {metrics['cpu']['usage_percent']:.1f}%",
                'severity': 'warning'
            })
        
        # 内存告警
        if metrics['memory']['percent'] > self.thresholds['memory']:
            alerts.append({
                'type': 'memory_high',
                'message': f"内存使用率过高: {metrics['memory']['percent']:.1f}%",
                'severity': 'warning'
            })
        
        # 磁盘告警
        for mount, disk_info in metrics['disk'].items():
            if disk_info['percent'] > self.thresholds['disk']:
                alerts.append({
                    'type': 'disk_high',
                    'message': f"磁盘使用率过高: {mount} {disk_info['percent']:.1f}%",
                    'severity': 'critical'
                })
        
        return alerts
    
    def send_alert(self, alert):
        """发送告警通知"""
        try:
            response = requests.post(self.alert_webhook, json=alert, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            print(f"告警发送失败: {alert['message']}")
            return False
    
    def run_monitoring(self, interval=60):
        """运行性能监控"""
        print("性能监控已启动...")
        
        while True:
            try:
                metrics = self.collect_metrics()
                alerts = self.check_alerts(metrics)
                
                # 输出当前状态
                print(f"\n[{metrics['timestamp']}] 系统状态:")
                print(f"  CPU: {metrics['cpu']['usage_percent']:.1f}%")
                print(f"  内存: {metrics['memory']['percent']:.1f}%")
                print(f"  进程数: {metrics['processes']}")
                
                # 处理告警
                for alert in alerts:
                    print(f"⚠ 告警: {alert['message']}")
                    self.send_alert(alert)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n监控已停止")
                break
            except Exception as e:
                print(f"监控异常: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitoring()
```

## 自动部署脚本

### 3. 应用自动部署脚本
```bash
#!/bin/bash
# auto_deploy.sh - 应用自动部署脚本

# 配置参数
APP_NAME="aiops-platform"
GIT_REPO="https://github.com/company/aiops-platform.git"
DEPLOY_DIR="/opt/aiops"
BACKUP_DIR="/opt/backups"
LOG_FILE="/var/log/deploy.log"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 错误处理
handle_error() {
    log "ERROR: $1"
    exit 1
}

# 前置检查
pre_check() {
    log "开始部署前检查..."
    
    # 检查磁盘空间
    available_space=$(df $DEPLOY_DIR | awk 'NR==2 {print $4}')
    if [ $available_space -lt 1048576 ]; then # 1GB
        handle_error "磁盘空间不足，需要至少1GB可用空间"
    fi
    
    # 检查必要工具
    command -v git >/dev/null 2>&1 || handle_error "Git未安装"
    command -v docker >/dev/null 2>&1 || handle_error "Docker未安装"
    command -v docker-compose >/dev/null 2>&1 || handle_error "Docker Compose未安装"
    
    log "部署前检查完成"
}

# 创建备份
create_backup() {
    log "创建当前版本备份..."
    
    if [ -d "$DEPLOY_DIR" ]; then
        backup_name="$APP_NAME-$(date +%Y%m%d-%H%M%S)"
        tar -czf "$BACKUP_DIR/$backup_name.tar.gz" -C "$DEPLOY_DIR" .
        log "备份已创建: $BACKUP_DIR/$backup_name.tar.gz"
    fi
}

# 下载代码
download_code() {
    log "下载最新代码..."
    
    if [ -d "$DEPLOY_DIR/.git" ]; then
        cd $DEPLOY_DIR
        git fetch origin
        git reset --hard origin/main
    else
        rm -rf $DEPLOY_DIR
        git clone $GIT_REPO $DEPLOY_DIR
        cd $DEPLOY_DIR
    fi
    
    log "代码下载完成"
}

# 构建应用
build_application() {
    log "开始构建应用..."
    
    cd $DEPLOY_DIR
    
    # 构建Docker镜像
    docker-compose build --no-cache || handle_error "应用构建失败"
    
    log "应用构建完成"
}

# 执行数据库迁移
run_migrations() {
    log "执行数据库迁移..."
    
    cd $DEPLOY_DIR
    docker-compose run --rm app python manage.py migrate || handle_error "数据库迁移失败"
    
    log "数据库迁移完成"
}

# 部署应用
deploy_application() {
    log "部署应用..."
    
    cd $DEPLOY_DIR
    
    # 停止旧版本
    docker-compose down
    
    # 启动新版本
    docker-compose up -d || handle_error "应用启动失败"
    
    # 等待服务启动
    sleep 30
    
    log "应用部署完成"
}

# 健康检查
health_check() {
    log "执行健康检查..."
    
    max_attempts=10
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            log "健康检查通过"
            return 0
        fi
        
        log "健康检查失败，重试 $attempt/$max_attempts"
        sleep 10
        ((attempt++))
    done
    
    handle_error "健康检查失败，部署回滚"
}

# 部署后清理
post_deploy_cleanup() {
    log "执行部署后清理..."
    
    # 清理旧的Docker镜像
    docker image prune -f
    
    # 清理旧备份（保留最近10个）
    ls -t $BACKUP_DIR/$APP_NAME-*.tar.gz | tail -n +11 | xargs rm -f
    
    log "清理完成"
}

# 主函数
main() {
    log "========== 开始自动部署 =========="
    
    pre_check
    create_backup
    download_code
    build_application
    run_migrations
    deploy_application
    health_check
    post_deploy_cleanup
    
    log "========== 部署成功完成 =========="
}

# 执行主函数
main "$@"
```

## 数据库管理脚本

### 4. 数据库备份脚本
```bash
#!/bin/bash
# db_backup.sh - 数据库自动备份脚本

# 配置参数
DB_HOST="localhost"
DB_USER="backup_user"
DB_PASSWORD="backup_password"
BACKUP_DIR="/opt/backups/database"
RETENTION_DAYS=30
LOG_FILE="/var/log/db_backup.log"

# 数据库列表
DATABASES=("aiops_main" "aiops_metrics" "aiops_logs")

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 创建备份目录
create_backup_dir() {
    local backup_date=$(date +%Y%m%d)
    local backup_path="$BACKUP_DIR/$backup_date"
    
    mkdir -p $backup_path
    echo $backup_path
}

# 备份MySQL数据库
backup_mysql() {
    local db_name=$1
    local backup_path=$2
    local backup_file="$backup_path/${db_name}_$(date +%H%M%S).sql.gz"
    
    log "开始备份数据库: $db_name"
    
    mysqldump --host=$DB_HOST \
              --user=$DB_USER \
              --password=$DB_PASSWORD \
              --single-transaction \
              --routines \
              --triggers \
              $db_name | gzip > $backup_file
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log "数据库备份成功: $backup_file"
        
        # 验证备份文件
        if [ -s $backup_file ]; then
            log "备份文件验证通过"
        else
            log "ERROR: 备份文件为空"
            return 1
        fi
    else
        log "ERROR: 数据库备份失败: $db_name"
        return 1
    fi
}

# 清理旧备份
cleanup_old_backups() {
    log "清理 $RETENTION_DAYS 天前的备份文件"
    
    find $BACKUP_DIR -type f -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -type d -empty -delete
    
    log "旧备份清理完成"
}

# 发送备份报告
send_backup_report() {
    local status=$1
    local backup_path=$2
    
    local backup_size=$(du -sh $backup_path 2>/dev/null | cut -f1)
    local report="数据库备份报告\n"
    report+="时间: $(date)\n"
    report+="状态: $status\n"
    report+="备份路径: $backup_path\n"
    report+="备份大小: $backup_size\n"
    
    # 这里可以集成邮件或企业微信通知
    echo -e "$report" | tee -a $LOG_FILE
}

# 主函数
main() {
    log "========== 开始数据库备份 =========="
    
    # 创建备份目录
    backup_path=$(create_backup_dir)
    
    # 备份每个数据库
    backup_success=true
    for db in "${DATABASES[@]}"; do
        if ! backup_mysql $db $backup_path; then
            backup_success=false
        fi
    done
    
    # 清理旧备份
    cleanup_old_backups
    
    # 发送报告
    if $backup_success; then
        send_backup_report "成功" $backup_path
        log "========== 数据库备份完成 =========="
    else
        send_backup_report "失败" $backup_path
        log "========== 数据库备份有错误 =========="
        exit 1
    fi
}

# 执行主函数
main "$@"
```

## 容器管理脚本

### 5. Docker容器管理脚本
```python
#!/usr/bin/env python3
# docker_manager.py - Docker容器管理脚本

import docker
import json
import time
import logging
from datetime import datetime, timedelta

class DockerManager:
    def __init__(self):
        self.client = docker.from_env()
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/docker_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_container_stats(self):
        """获取容器统计信息"""
        containers = self.client.containers.list()
        stats = []
        
        for container in containers:
            try:
                stats_stream = container.stats(stream=False)
                
                # 计算CPU使用率
                cpu_delta = stats_stream['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats_stream['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats_stream['cpu_stats']['system_cpu_usage'] - \
                              stats_stream['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
                
                # 计算内存使用率
                memory_usage = stats_stream['memory_stats']['usage']
                memory_limit = stats_stream['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100.0
                
                container_info = {
                    'name': container.name,
                    'id': container.short_id,
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else 'unknown',
                    'cpu_percent': round(cpu_percent, 2),
                    'memory_usage_mb': round(memory_usage / 1024 / 1024, 2),
                    'memory_percent': round(memory_percent, 2),
                    'created': container.attrs['Created'],
                    'ports': container.ports
                }
                
                stats.append(container_info)
                
            except Exception as e:
                self.logger.error(f"获取容器 {container.name} 统计信息失败: {e}")
        
        return stats
    
    def monitor_containers(self):
        """监控容器状态"""
        self.logger.info("开始监控容器状态...")
        
        stats = self.get_container_stats()
        
        # 检查异常容器
        for container_stat in stats:
            # CPU使用率异常
            if container_stat['cpu_percent'] > 80:
                self.logger.warning(f"容器 {container_stat['name']} CPU使用率过高: {container_stat['cpu_percent']}%")
            
            # 内存使用率异常
            if container_stat['memory_percent'] > 85:
                self.logger.warning(f"容器 {container_stat['name']} 内存使用率过高: {container_stat['memory_percent']}%")
            
            # 容器状态异常
            if container_stat['status'] != 'running':
                self.logger.error(f"容器 {container_stat['name']} 状态异常: {container_stat['status']}")
        
        return stats
    
    def restart_unhealthy_containers(self):
        """重启不健康的容器"""
        containers = self.client.containers.list(all=True)
        
        for container in containers:
            if container.status == 'exited':
                self.logger.info(f"重启退出的容器: {container.name}")
                try:
                    container.restart()
                    self.logger.info(f"容器 {container.name} 重启成功")
                except Exception as e:
                    self.logger.error(f"容器 {container.name} 重启失败: {e}")
    
    def cleanup_resources(self):
        """清理Docker资源"""
        self.logger.info("开始清理Docker资源...")
        
        # 清理停止的容器
        try:
            self.client.containers.prune()
            self.logger.info("已清理停止的容器")
        except Exception as e:
            self.logger.error(f"清理容器失败: {e}")
        
        # 清理未使用的镜像
        try:
            self.client.images.prune(filters={'dangling': False})
            self.logger.info("已清理未使用的镜像")
        except Exception as e:
            self.logger.error(f"清理镜像失败: {e}")
        
        # 清理未使用的卷
        try:
            self.client.volumes.prune()
            self.logger.info("已清理未使用的卷")
        except Exception as e:
            self.logger.error(f"清理卷失败: {e}")
        
        # 清理未使用的网络
        try:
            self.client.networks.prune()
            self.logger.info("已清理未使用的网络")
        except Exception as e:
            self.logger.error(f"清理网络失败: {e}")
    
    def generate_report(self):
        """生成容器状态报告"""
        stats = self.get_container_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_containers': len(stats),
            'running_containers': len([s for s in stats if s['status'] == 'running']),
            'containers': stats,
            'system_info': {
                'docker_version': self.client.version()['Version'],
                'api_version': self.client.version()['ApiVersion']
            }
        }
        
        # 保存报告
        report_file = f"/tmp/docker_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"容器状态报告已生成: {report_file}")
        return report

def main():
    manager = DockerManager()
    
    # 监控容器
    manager.monitor_containers()
    
    # 重启不健康容器
    manager.restart_unhealthy_containers()
    
    # 清理资源
    manager.cleanup_resources()
    
    # 生成报告
    manager.generate_report()

if __name__ == "__main__":
    main()
```

## 网络管理脚本

### 6. 网络连通性检测脚本
```bash
#!/bin/bash
# network_check.sh - 网络连通性检测脚本

# 配置参数
HOSTS_FILE="/etc/network_monitor/hosts.conf"
LOG_FILE="/var/log/network_check.log"
ALERT_WEBHOOK="http://localhost:8080/api/alerts"

# 默认检测目标
DEFAULT_HOSTS=(
    "8.8.8.8:53"
    "114.114.114.114:53"
    "www.baidu.com:80"
    "www.google.com:443"
)

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# 发送告警
send_alert() {
    local message="$1"
    local severity="$2"
    
    curl -s -X POST "$ALERT_WEBHOOK" \
         -H "Content-Type: application/json" \
         -d "{\"message\":\"$message\",\"severity\":\"$severity\",\"type\":\"network\"}"
}

# 检测TCP连接
check_tcp_connection() {
    local host="$1"
    local port="$2"
    local timeout=5
    
    if timeout $timeout bash -c "</dev/tcp/$host/$port"; then
        return 0
    else
        return 1
    fi
}

# 检测HTTP服务
check_http_service() {
    local url="$1"
    local timeout=10
    
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $timeout "$url")
    
    if [[ "$response_code" -ge 200 && "$response_code" -lt 400 ]]; then
        return 0
    else
        return 1
    fi
}

# 检测DNS解析
check_dns_resolution() {
    local domain="$1"
    local dns_server="${2:-8.8.8.8}"
    
    if nslookup "$domain" "$dns_server" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 检测网络延迟
check_network_latency() {
    local host="$1"
    local count="${2:-4}"
    
    local result=$(ping -c $count "$host" 2>/dev/null | tail -1)
    
    if [[ $? -eq 0 ]]; then
        local avg_time=$(echo "$result" | awk -F'/' '{print $5}')
        echo "$avg_time"
        return 0
    else
        return 1
    fi
}

# 检测带宽
check_bandwidth() {
    local interface="${1:-eth0}"
    
    # 获取网络接口统计信息
    local rx_bytes1=$(cat /sys/class/net/$interface/statistics/rx_bytes)
    local tx_bytes1=$(cat /sys/class/net/$interface/statistics/tx_bytes)
    
    sleep 1
    
    local rx_bytes2=$(cat /sys/class/net/$interface/statistics/rx_bytes)
    local tx_bytes2=$(cat /sys/class/net/$interface/statistics/tx_bytes)
    
    local rx_rate=$(( (rx_bytes2 - rx_bytes1) * 8 / 1024 / 1024 ))  # Mbps
    local tx_rate=$(( (tx_bytes2 - tx_bytes1) * 8 / 1024 / 1024 ))  # Mbps
    
    echo "RX: ${rx_rate} Mbps, TX: ${tx_rate} Mbps"
}

# 主检测函数
run_network_checks() {
    log "========== 开始网络连通性检测 =========="
    
    local failed_checks=0
    local total_checks=0
    
    # 读取检测目标
    local hosts_to_check=()
    if [[ -f "$HOSTS_FILE" ]]; then
        while IFS= read -r line; do
            [[ ! "$line" =~ ^[[:space:]]*# ]] && [[ -n "$line" ]] && hosts_to_check+=("$line")
        done < "$HOSTS_FILE"
    else
        hosts_to_check=("${DEFAULT_HOSTS[@]}")
    fi
    
    # 检测每个目标
    for target in "${hosts_to_check[@]}"; do
        local host=$(echo "$target" | cut -d':' -f1)
        local port=$(echo "$target" | cut -d':' -f2)
        
        ((total_checks++))
        
        log "检测目标: $host:$port"
        
        # TCP连接检测
        if check_tcp_connection "$host" "$port"; then
            log "✓ TCP连接正常: $host:$port"
            
            # 延迟检测
            if latency=$(check_network_latency "$host"); then
                log "✓ 网络延迟: $host - $latency ms"
            fi
            
        else
            log "✗ TCP连接失败: $host:$port"
            send_alert "TCP连接失败: $host:$port" "warning"
            ((failed_checks++))
        fi
        
        # HTTP服务检测（针对Web端口）
        if [[ "$port" == "80" || "$port" == "443" ]]; then
            local protocol="http"
            [[ "$port" == "443" ]] && protocol="https"
            
            if check_http_service "${protocol}://${host}"; then
                log "✓ HTTP服务正常: $protocol://$host"
            else
                log "✗ HTTP服务异常: $protocol://$host"
                send_alert "HTTP服务异常: $protocol://$host" "warning"
            fi
        fi
    done
    
    # DNS解析检测
    local dns_domains=("www.baidu.com" "www.google.com" "github.com")
    for domain in "${dns_domains[@]}"; do
        ((total_checks++))
        
        if check_dns_resolution "$domain"; then
            log "✓ DNS解析正常: $domain"
        else
            log "✗ DNS解析失败: $domain"
            send_alert "DNS解析失败: $domain" "error"
            ((failed_checks++))
        fi
    done
    
    # 网络接口检测
    for interface in $(ls /sys/class/net/ | grep -v lo); do
        if [[ -d "/sys/class/net/$interface" ]]; then
            local status=$(cat /sys/class/net/$interface/operstate)
            local bandwidth=$(check_bandwidth "$interface")
            
            log "网络接口 $interface: 状态=$status, 带宽=$bandwidth"
            
            if [[ "$status" != "up" ]]; then
                send_alert "网络接口异常: $interface ($status)" "warning"
            fi
        fi
    done
    
    # 汇总报告
    local success_rate=$(( (total_checks - failed_checks) * 100 / total_checks ))
    log "检测完成: 总计 $total_checks 项，失败 $failed_checks 项，成功率 $success_rate%"
    
    if [[ $failed_checks -gt 0 ]]; then
        send_alert "网络连通性检测发现 $failed_checks 个问题，成功率 $success_rate%" "warning"
    fi
    
    log "========== 网络连通性检测完成 =========="
}

# 主函数
main() {
    # 创建日志目录
    mkdir -p "$(dirname $LOG_FILE)"
    
    # 运行网络检测
    run_network_checks
}

# 执行主函数
main "$@"
```

这个自动化脚本库涵盖了智能运维系统的核心自动化需求，包括系统监控、应用部署、数据库管理、容器运维和网络检测等方面，为运维团队提供了完整的自动化工具集。