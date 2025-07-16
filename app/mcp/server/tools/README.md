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