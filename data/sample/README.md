# RCA测试用问题资源

本目录包含用于测试根因分析(RCA)模块的Kubernetes资源文件。这些资源被特意设计为包含各种常见的Kubernetes问题，以展示RCA引擎的检测和分析能力。

## 📋 问题类型覆盖

根据RCA引擎的检测能力，创建了以下7类问题资源：

### 1. OOM (内存不足) - `01-oom-problem.yaml`
- **问题**: 容器内存限制设置过小(50Mi)，而应用尝试消耗200MB内存
- **触发条件**: 应用启动后约30-60秒会触发OOMKilled
- **RCA检测点**:
  - 指标: `container_memory_usage_bytes`, `container_memory_working_set_bytes`
  - 事件: `OOMKilled`, `Killing`
  - 日志: `out of memory`, `oom`, `memory exhausted`

### 2. CPU限制 - `02-cpu-throttling-problem.yaml`
- **问题**: CPU限制设置过小(100m)，而应用启动4个CPU密集型并发任务
- **触发条件**: 立即开始CPU throttling，HPA会尝试扩容
- **RCA检测点**:
  - 指标: `container_cpu_cfs_throttled_periods_total`, `container_cpu_usage_seconds_total`
  - 事件: `CPUThrottling`, `HighCPU`
  - 日志: `cpu throttled`, `high cpu usage`

### 3. 崩溃循环 - `03-crash-loop-problem.yaml`
- **问题**: 应用随机崩溃(4种不同的崩溃模式)，健康检查失败
- **触发条件**: 启动后10-30秒随机崩溃，进入CrashLoopBackOff
- **RCA检测点**:
  - 指标: `kube_pod_container_status_restarts_total`
  - 事件: `CrashLoopBackOff`, `BackOff`, `Failed`
  - 日志: `panic`, `fatal error`, `segmentation fault`

### 4. 网络问题 - `04-network-problem.yaml`
- **问题**: DNS配置错误，网络策略限制，连接不存在的服务
- **触发条件**: Pod启动后立即出现网络连接失败
- **RCA检测点**:
  - 指标: `container_network_receive_errors_total`, `container_network_transmit_errors_total`
  - 事件: `NetworkNotReady`, `NetworkPluginNotReady`
  - 日志: `connection refused`, `timeout`, `network unreachable`

### 5. 镜像拉取失败 - `05-image-pull-problem.yaml`
- **问题**: 使用不存在的镜像、错误的私有仓库、无效的镜像标签
- **触发条件**: Pod创建后立即进入ImagePullBackOff状态
- **RCA检测点**:
  - 事件: `ImagePullBackOff`, `ErrImagePull`
  - 日志: `pull access denied`, `image not found`

### 6. 资源配额不足 - `06-resource-quota-problem.yaml`
- **问题**: 严格的ResourceQuota限制，多个Deployment竞争有限资源
- **触发条件**: 第二个和第三个Deployment无法创建Pod
- **RCA检测点**:
  - 指标: `kube_resourcequota`
  - 事件: `FailedScheduling`, `InsufficientCPU`, `InsufficientMemory`, `FailedCreate`
  - 日志: `exceeded quota`, `insufficient resources`, `forbidden`

### 7. 磁盘压力 - `07-disk-pressure-problem.yaml`
- **问题**: 大量磁盘I/O操作，持续写入大文件和日志
- **触发条件**: 启动后持续消耗磁盘空间，可能触发磁盘空间不足
- **RCA检测点**:
  - 指标: `node_filesystem_avail_bytes`, `node_filesystem_size_bytes`
  - 事件: `DiskPressure`, `EvictedByNodeCondition`
  - 日志: `no space left`, `disk full`

### 8. 综合问题 - `08-complex-problems.yaml`
- **问题**: 包含多种问题的复杂场景
- **组合**: 内存泄漏+网络问题, CPU密集+镜像拉取失败, 崩溃循环+磁盘压力
- **目的**: 测试RCA引擎的综合分析和关联分析能力

## 🚀 使用方法

### 1. 逐个测试单一问题类型
```bash
# 测试OOM问题
kubectl apply -f 01-oom-problem.yaml

# 等待问题出现后运行RCA分析
curl -X POST "http://localhost:8000/api/v1/rca/analyze" \
  -H "Content-Type: application/json" \
  -d '{"namespace": "rca-test-oom", "time_window_hours": 0.5}'

# 清理资源
kubectl delete -f 01-oom-problem.yaml
```

### 2. 批量测试多种问题
```bash
# 应用所有问题资源
kubectl apply -f .

# 等待5-10分钟让问题充分暴露

# 分析各个namespace
for ns in rca-test-oom rca-test-cpu rca-test-crashloop rca-test-network rca-test-imagepull rca-test-quota rca-test-disk rca-test-complex; do
  echo "分析namespace: $ns"
  curl -X POST "http://localhost:8000/api/v1/rca/analyze" \
    -H "Content-Type: application/json" \
    -d "{\"namespace\": \"$ns\", \"time_window_hours\": 1.0}"
  echo ""
done
```

### 3. 快速诊断测试
```bash
# 快速诊断所有问题namespace
for ns in rca-test-oom rca-test-cpu rca-test-crashloop rca-test-network rca-test-imagepull rca-test-quota rca-test-disk rca-test-complex; do
  curl -X POST "http://localhost:8000/api/v1/rca/quick-diagnosis" \
    -H "Content-Type: application/json" \
    -d "{\"namespace\": \"$ns\"}"
done
```

## 📊 预期的RCA分析结果

### OOM问题分析
- **根因类型**: OOM
- **置信度**: 0.9+
- **关键证据**: 
  - 内存使用指标异常
  - OOMKilled事件
  - 容器重启次数增加
- **建议**: 增加内存限制，优化内存使用

### CPU限制问题分析  
- **根因类型**: CPU_THROTTLING
- **置信度**: 0.85+
- **关键证据**:
  - CPU throttling指标异常
  - CPU使用率指标异常
  - HPA扩容事件
- **建议**: 增加CPU限制，优化CPU使用

### 崩溃循环问题分析
- **根因类型**: CRASH_LOOP
- **置信度**: 0.95+
- **关键证据**:
  - 容器重启次数指标异常
  - CrashLoopBackOff事件
  - panic/fatal error日志
- **建议**: 检查应用代码，修复崩溃原因

### 其他问题类型...
每种问题类型都应该能被RCA引擎准确识别，并提供相应的根因分析和解决建议。

## 🧹 清理资源

测试完成后，清理所有测试资源：

```bash
# 删除所有测试namespace及其资源
kubectl delete namespace rca-test-oom rca-test-cpu rca-test-crashloop rca-test-network rca-test-imagepull rca-test-quota rca-test-disk rca-test-complex

# 或者逐个删除资源文件
kubectl delete -f .
```

## ⚠️ 注意事项

1. **资源消耗**: 这些测试资源会消耗集群资源，建议在测试环境中使用
2. **时间窗口**: 某些问题需要时间才能充分暴露，建议等待5-10分钟后进行分析
3. **集群影响**: 磁盘压力和CPU密集型任务可能影响集群性能
4. **存储需求**: 某些测试需要PV支持，确保集群有足够的存储资源
5. **网络策略**: 如果集群启用了网络策略，某些网络问题测试可能需要调整

## 🔍 调试提示

如果RCA分析结果不如预期：

1. **检查数据收集**: 确保Prometheus、Kubernetes API、日志系统正常工作
2. **调整时间窗口**: 某些问题可能需要更长的时间窗口才能检测到
3. **查看详细日志**: 检查RCA引擎的日志输出，了解分析过程
4. **验证指标**: 使用Prometheus UI验证相关指标是否正常收集
5. **手动验证**: 使用kubectl命令手动检查Pod状态、事件和日志

这些测试资源覆盖了RCA引擎的所有主要检测模式，可以全面验证根因分析功能的准确性和完整性。