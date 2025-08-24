# Kubernetes问题资源示例

本目录包含了各种有问题的Kubernetes资源配置文件，用于测试根因分析模块的效果。

## 文件说明

### 1. resource-limit-problem-pod.yaml
**问题类型**: 资源限制问题
- 请求过多CPU和内存资源（16 CPU, 64GB内存）
- 资源limit设置不合理
- 健康检查端口配置错误
- 节点选择器要求不存在的标签
- 亲和性规则过于严格

### 2. image-pull-failed-deployment.yaml  
**问题类型**: 镜像拉取失败
- 使用不存在的镜像
- 引用私有镜像但缺少认证
- 使用错误的镜像标签
- 镜像拉取密钥不存在
- 部署策略配置问题
- 资源配置不当（limit < request）

### 3. health-check-failed-service.yaml
**问题类型**: 健康检查失败  
- Service端口与Pod端口不匹配
- 健康检查路径不存在
- 就绪检查配置过于严格
- ConfigMap依赖不存在
- DNS策略配置错误
- 安全上下文配置有问题

### 4. configmap-dependency-problem-pod.yaml
**问题类型**: ConfigMap依赖问题
- 引用不存在的ConfigMap和Secret
- ConfigMap键不存在
- 卷挂载路径冲突
- 初始化容器ConfigMap依赖失败
- ConfigMap数据格式错误（JSON/YAML语法错误）
- 敏感信息存储在ConfigMap中（应该用Secret）

### 5. storage-mount-problem-statefulset.yaml
**问题类型**: 存储卷挂载问题
- 引用不存在的PVC
- 卷挂载路径冲突
- 存储类不存在
- 访问模式冲突
- 存储请求过大
- StatefulSet配置不当
- Headless Service配置错误

### 6. network-policy-conflicts.yaml
**问题类型**: 网络策略冲突
- 过度限制的网络策略
- 多个策略选择相同Pod但规则冲突
- CIDR格式错误
- 端口配置错误
- 协议名称错误
- 安全漏洞（允许所有流量）
- 性能影响（过于复杂的规则）

## 测试用途

这些文件设计用于：

1. **测试根因分析引擎**：验证系统能否正确识别和分析各种Kubernetes资源问题
2. **验证告警机制**：测试监控系统是否能及时发现这些配置问题
3. **评估修复建议**：检查系统是否能提供有效的问题修复建议
4. **压力测试**：使用多个有问题的资源进行系统稳定性测试

## 使用方法

```bash
# 部署测试资源（注意：这些资源故意包含错误）
kubectl apply -f data/sample/

# 查看资源状态
kubectl get pods,deployments,services,statefulsets,networkpolicies -l problem

# 查看事件（用于根因分析）
kubectl get events --sort-by='.lastTimestamp'

# 清理测试资源
kubectl delete -f data/sample/
```

## 注意事项

⚠️ **警告**：这些文件包含故意设计的错误配置，仅用于测试目的。不要在生产环境中使用。

- 部分资源可能会一直处于Pending或Failed状态
- 某些配置可能会占用大量资源
- 网络策略可能会阻断正常的集群通信
- 建议在独立的测试命名空间中使用

## 问题统计

总计包含以下类型的问题：
- 资源限制问题：6个
- 镜像拉取问题：13个  
- 健康检查问题：11个
- ConfigMap/Secret依赖问题：15个
- 存储挂载问题：18个
- 网络策略问题：20+个

这些问题覆盖了Kubernetes集群中最常见的故障场景，为根因分析系统提供了全面的测试用例。