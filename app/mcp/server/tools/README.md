# MCP工具使用示例

## 已创建的工具

### 1. 时间工具 (time_tool.py)
- **工具名称**: `get_current_time`
- **描述**: 获取当前时间，支持多种格式和时区
- **使用示例**:
```json
{
  "format": "iso",
  "timezone": "UTC"
}
```

### 2. 系统信息工具 (system_info_tool.py)
- **工具名称**: `get_system_info`
- **描述**: 获取系统基本信息，包括CPU、内存、磁盘使用情况
- **使用示例**:
```json
{
  "info_type": "all"
}
```

### 3. 文件操作工具 (file_tool.py)

#### 3.1 读取文件
- **工具名称**: `read_file`
- **使用示例**:
```json
{
  "file_path": "/path/to/file.txt",
  "encoding": "utf-8",
  "max_lines": 100
}
```

#### 3.2 列出目录
- **工具名称**: `list_directory`
- **使用示例**:
```json
{
  "directory_path": "/path/to/directory",
  "show_hidden": false
}
```

#### 3.3 获取文件统计信息
- **工具名称**: `get_file_stats`
- **使用示例**:
```json
{
  "path": "/path/to/file_or_directory"
}
```

### 4. 计算器工具 (calculator_tool.py)

#### 4.1 基础计算器
- **工具名称**: `calculate`
- **使用示例**:
```json
{
  "expression": "2 + 3 * 4",
  "precision": 2
}
```

#### 4.2 统计计算
- **工具名称**: `calculate_statistics`
- **使用示例**:
```json
{
  "numbers": [1, 2, 3, 4, 5],
  "operations": ["mean", "std", "min", "max"]
}
```

#### 4.3 单位转换
- **工具名称**: `convert_units`
- **使用示例**:
```json
{
  "value": 100,
  "from_unit": "m",
  "to_unit": "ft",
  "category": "length"
}
```
### 5. k8s集群检查工具 (k8s_cluster_check_tool.py)
- **工具名称**: `k8s_cluster_check`
- **使用示例**:
```json
{
  "config_path": "/path/to/kubeconfig",
  "namespace": "default",
  "time_window_hours": 1
}
```

### 6. k8s Pod管理工具 (k8s_pod_tool.py)
- **工具名称**: `k8s_pod_management`
- **支持操作**: list_pods, get_pod_details, delete_pod, restart_pod, get_pod_events
- **使用示例**:
```json
{
  "operation": "list_pods",
  "namespace": "default",
  "label_selector": "app=nginx",
  "max_results": 50
}
```

### 7. k8s Service管理工具 (k8s_service_tool.py)
- **工具名称**: `k8s_service_management`
- **支持操作**: list_services, get_service_details, create_service, delete_service, get_endpoints
- **使用示例**:
```json
{
  "operation": "create_service",
  "namespace": "default",
  "service_config": {
    "name": "my-service",
    "type": "ClusterIP",
    "selector": {"app": "nginx"},
    "ports": [{"port": 80, "target_port": 80}]
  }
}
```

### 8. k8s Deployment管理工具 (k8s_deployment_tool.py)
- **工具名称**: `k8s_deployment_management`
- **支持操作**: list_deployments, get_deployment_status, update_image, rollback, scale, get_rollout_history, restart_deployment
- **使用示例**:
```json
{
  "operation": "update_image",
  "deployment_name": "nginx-deployment",
  "namespace": "default",
  "container_name": "nginx",
  "new_image": "nginx:1.20"
}
```

### 9. k8s配置管理工具 (k8s_config_tool.py)
- **工具名称**: `k8s_config_management`
- **支持操作**: list_configmaps, list_secrets, get_configmap, get_secret, create_configmap, create_secret, update_configmap, update_secret, delete_configmap, delete_secret
- **使用示例**:
```json
{
  "operation": "create_configmap",
  "namespace": "default",
  "resource_name": "my-config",
  "data": {
    "config.yaml": "key: value"
  }
}
```

### 10. k8s日志查看工具 (k8s_logs_tool.py)
- **工具名称**: `k8s_logs_viewer`
- **支持操作**: get_pod_logs, get_container_logs, get_previous_logs, tail_logs, search_logs
- **使用示例**:
```json
{
  "operation": "get_container_logs",
  "pod_name": "nginx-pod",
  "container_name": "nginx",
  "namespace": "default",
  "tail_lines": 100
}
```

### 11. k8s资源监控工具 (k8s_monitor_tool.py)
- **工具名称**: `k8s_resource_monitor`
- **支持操作**: get_node_metrics, get_pod_metrics, get_resource_quotas, get_limit_ranges, get_top_pods, get_top_nodes
- **使用示例**:
```json
{
  "operation": "get_top_pods",
  "namespace": "default",
  "max_results": 10
}
```

### 12. k8s命名空间管理工具 (k8s_namespace_tool.py)
- **工具名称**: `k8s_namespace_management`
- **支持操作**: list_namespaces, get_namespace, create_namespace, delete_namespace, get_namespace_resources
- **使用示例**:
```json
{
  "operation": "create_namespace",
  "namespace_name": "my-namespace",
  "labels": {"environment": "dev"}
}
```

### 13. k8s应用伸缩工具 (k8s_scaling_tool.py)
- **工具名称**: `k8s_application_scaling`
- **支持操作**: scale_deployment, scale_replicaset, get_hpa_status, list_hpa, create_hpa, delete_hpa
- **使用示例**:
```json
{
  "operation": "scale_deployment",
  "resource_name": "nginx-deployment",
  "namespace": "default",
  "replicas": 5
}
```

### 14. k8s Ingress管理工具 (k8s_ingress_tool.py)
- **工具名称**: `k8s_ingress_management`
- **支持操作**: list_ingresses, get_ingress, create_ingress, delete_ingress, get_ingress_classes
- **使用示例**:
```json
{
  "operation": "create_ingress",
  "namespace": "default",
  "ingress_config": {
    "name": "my-ingress",
    "ingress_class": "nginx",
    "rules": [
      {
        "host": "example.com",
        "paths": [
          {
            "path": "/",
            "service_name": "my-service",
            "service_port": 80
          }
        ]
      }
    ]
  }
}
```

### 15. k8s资源操作工具 (k8s_resource_tool.py)
- **工具名称**: `k8s_resource_operations`
- **支持操作**: describe_resource, add_labels, remove_labels, add_annotations, remove_annotations, apply_yaml, get_resource_yaml
- **使用示例**:
```json
{
  "operation": "add_labels",
  "resource_type": "pod",
  "resource_name": "nginx-pod",
  "namespace": "default",
  "labels": {
    "version": "v1.0"
  }
}
```

## 支持的单位类别

- **length**: 长度单位 (m, cm, mm, km, ft, in, yd)
- **weight**: 重量单位 (kg, g, mg, lb, oz)
- **temperature**: 温度单位 (c, f, k)
- **storage**: 存储单位 (b, kb, mb, gb, tb)
- **time**: 时间单位 (s, min, h, day, ms, us)

## 使用方法

所有工具都继承自 `BaseTool` 类，可以通过以下方式使用：

```python
from app.mcp.server.tools import tools

# 获取所有工具
for tool in tools:
    print(f"工具名称: {tool.name}")
    print(f"工具描述: {tool.description}")
    
    # 异步执行工具
    result = await tool.execute(parameters)
```

## 测试

运行测试脚本验证所有工具：
```bash
python test_mcp_tools.py
```