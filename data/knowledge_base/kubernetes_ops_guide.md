# Kubernetes运维指南

## 概述
Kubernetes是一个用于自动部署、扩缩和管理容器化应用程序的开源系统。本指南提供了Kubernetes运维的最佳实践和故障排除方法。

## 集群架构
### 控制平面组件
- **kube-apiserver**: API服务器，是Kubernetes控制平面的前端
- **etcd**: 分布式键值存储，用于保存集群的所有配置数据
- **kube-scheduler**: 调度器，负责将Pod分配到节点
- **kube-controller-manager**: 控制器管理器，运行控制器进程

### 节点组件
- **kubelet**: 在每个节点上运行的代理，确保容器在Pod中运行
- **kube-proxy**: 网络代理，维护节点上的网络规则
- **容器运行时**: 负责运行容器的软件（如Docker、containerd）

## 常见运维任务

### 1. 集群监控
```bash
# 查看集群状态
kubectl cluster-info

# 查看节点状态
kubectl get nodes -o wide

# 查看系统Pod状态
kubectl get pods -n kube-system

# 检查资源使用情况
kubectl top nodes
kubectl top pods --all-namespaces
```

### 2. Pod管理
```bash
# 创建Pod
kubectl create -f pod.yaml

# 查看Pod详情
kubectl describe pod <pod-name> -n <namespace>

# 查看Pod日志
kubectl logs <pod-name> -n <namespace> -f

# 进入Pod执行命令
kubectl exec -it <pod-name> -n <namespace> -- /bin/bash
```

### 3. 服务管理
```bash
# 创建Service
kubectl expose deployment <deployment-name> --port=80 --target-port=8080

# 查看服务
kubectl get services -o wide

# 测试服务连通性
kubectl port-forward service/<service-name> 8080:80
```

## 故障排除

### Pod故障
1. **Pod处于Pending状态**
   - 检查节点资源是否充足
   - 验证Pod的资源请求和限制
   - 检查节点污点和Pod容忍度

2. **Pod启动失败**
   - 查看Pod事件：`kubectl describe pod <pod-name>`
   - 检查镜像是否存在和可访问
   - 验证配置文件和环境变量

3. **Pod运行异常**
   - 查看应用日志：`kubectl logs <pod-name>`
   - 检查健康检查配置
   - 验证网络连接和存储挂载

### 网络故障
1. **Service无法访问**
   - 检查Service和Endpoint：`kubectl get endpoints`
   - 验证标签选择器匹配
   - 测试Pod间网络连通性

2. **Ingress故障**
   - 检查Ingress控制器状态
   - 验证Ingress规则配置
   - 检查TLS证书和域名解析

### 存储故障
1. **PV/PVC问题**
   - 检查存储类配置
   - 验证PV和PVC的绑定状态
   - 检查存储后端的可用性

## 性能优化

### 资源管理
- 为所有容器设置资源请求和限制
- 使用HPA（水平Pod自动扩缩）
- 合理配置节点亲和性和反亲和性

### 网络优化
- 选择合适的CNI插件
- 优化Service网格配置
- 使用NodePort或LoadBalancer合理暴露服务

### 存储优化
- 选择合适的存储类型
- 配置存储的IOPS和带宽
- 实现数据备份和恢复策略

## 安全最佳实践

### RBAC配置
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
```

### 网络策略
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### Pod安全策略
- 使用非root用户运行容器
- 设置只读根文件系统
- 禁用特权模式和主机网络

## 备份和恢复
1. **etcd备份**
   ```bash
   ETCDCTL_API=3 etcdctl snapshot save backup.db
   ```

2. **应用数据备份**
   - 使用Velero进行集群备份
   - 配置定期备份策略
   - 测试恢复流程

## 监控和告警
- 部署Prometheus
- 配置关键指标监控
- 设置告警规则和通知

这份指南涵盖了Kubernetes运维的核心要点，帮助运维人员高效管理和维护Kubernetes集群。