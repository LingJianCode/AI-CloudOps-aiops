# AI-CloudOps 模型训练说明

## 概述

本训练系统用于训练AI-CloudOps平台的四个核心预测模型：
- **QPS预测模型**：预测流量并推荐实例数
- **CPU预测模型**：预测CPU使用率趋势
- **Memory预测模型**：预测内存使用率趋势
- **Disk预测模型**：预测磁盘使用率趋势

## 目录结构

```
.
├── train_all_models.py        # 综合模型训练器
├── generate_training_data.py  # 训练数据生成器
├── run_training.sh            # 一键训练脚本
├── data/
│   ├── models/               # 训练好的模型文件
│   │   ├── qps_prediction_model.pkl
│   │   ├── qps_prediction_scaler.pkl
│   │   ├── qps_prediction_model_metadata.json
│   │   ├── cpu_prediction_model.pkl
│   │   ├── cpu_prediction_scaler.pkl
│   │   ├── cpu_prediction_model_metadata.json
│   │   ├── memory_prediction_model.pkl
│   │   ├── memory_prediction_scaler.pkl
│   │   ├── memory_prediction_model_metadata.json
│   │   ├── disk_prediction_model.pkl
│   │   ├── disk_prediction_scaler.pkl
│   │   └── disk_prediction_model_metadata.json
│   ├── training_data/        # 训练数据
│   │   ├── qps_data.csv
│   │   ├── cpu_data.csv
│   │   ├── memory_data.csv
│   │   ├── disk_data.csv
│   │   └── data_stats.json
│   └── visualizations/       # 可视化结果
│       ├── qps_results.png
│       ├── cpu_results.png
│       ├── memory_results.png
│       └── disk_results.png
```

## 依赖要求

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

## 使用方法

### 方法1：一键训练（推荐）

```bash
# 完整训练流程（生成数据 + 训练模型）
bash run_training.sh

# 使用现有数据训练（跳过数据生成）
bash run_training.sh --skip-data
```

### 方法2：分步执行

#### 步骤1：生成训练数据

```bash
# 生成所有类型的训练数据
python3 generate_training_data.py

# 只生成特定类型的数据
python3 generate_training_data.py qps    # 只生成QPS数据
python3 generate_training_data.py cpu    # 只生成CPU数据
python3 generate_training_data.py memory # 只生成Memory数据
python3 generate_training_data.py disk   # 只生成Disk数据

# 生成测试样本数据（小数据集）
python3 generate_training_data.py test

# 可视化生成的数据
python3 generate_training_data.py visualize
```

#### 步骤2：训练模型

```bash
# 训练所有模型
python3 train_all_models.py

# 只测试已训练的模型
python3 train_all_models.py test
```

## 数据格式说明

### QPS数据格式 (qps_data.csv)

| 字段      | 类型     | 说明       |
| --------- | -------- | ---------- |
| timestamp | datetime | 时间戳     |
| QPS       | float    | 每秒查询数 |
| instances | int      | 推荐实例数 |

### CPU数据格式 (cpu_data.csv)

| 字段       | 类型     | 说明              |
| ---------- | -------- | ----------------- |
| timestamp  | datetime | 时间戳            |
| CPU        | float    | CPU使用率(%)      |
| cpu_target | float    | 下一时刻CPU预测值 |

### Memory数据格式 (memory_data.csv)

| 字段          | 类型     | 说明               |
| ------------- | -------- | ------------------ |
| timestamp     | datetime | 时间戳             |
| MEMORY        | float    | 内存使用率(%)      |
| memory_target | float    | 下一时刻内存预测值 |

### Disk数据格式 (disk_data.csv)

| 字段        | 类型     | 说明               |
| ----------- | -------- | ------------------ |
| timestamp   | datetime | 时间戳             |
| DISK        | float    | 磁盘使用率(%)      |
| disk_target | float    | 下一时刻磁盘预测值 |

## 模型配置

### 特征工程

每个模型都使用了以下基础特征：
- **时间特征**：小时、星期、月份的周期性编码
- **历史特征**：1小时前、1天前、1周前的值
- **统计特征**：移动平均、标准差、变化率等

### 模型算法

系统会自动测试多种算法并选择最佳：
- Ridge回归
- 随机森林
- 梯度提升树
- 线性回归

### 评估指标

- **R² Score**：决定系数，越接近1越好
- **RMSE**：均方根误差，越小越好
- **MAE**：平均绝对误差，越小越好
- **交叉验证得分**：5折交叉验证

## 使用自己的数据

如果要使用自己的真实数据训练：

1. 准备符合上述格式的CSV文件
2. 将文件放置在 `data/training_data/` 目录
3. 文件命名为：
   - `qps_data.csv`
   - `cpu_data.csv`
   - `memory_data.csv`
   - `disk_data.csv`
4. 运行训练脚本：`bash run_training.sh --skip-data`

## 模型集成

训练完成后，模型会自动被 `model_manager.py` 加载和管理。确保模型文件路径与 `config.yaml` 中的配置一致。

## 性能优化建议

1. **数据量**：建议至少使用3个月以上的数据进行训练
2. **特征选择**：可以根据实际业务添加更多相关特征
3. **参数调优**：可以修改 `train_all_models.py` 中的参数网格进行更细致的调优
4. **模型更新**：建议定期（如每月）使用新数据重新训练模型

## 故障排除

### 问题1：内存不足
**解决方案**：减少训练数据量或使用更小的参数网格

### 问题2：训练时间过长
**解决方案**：
- 减少参数网格搜索范围
- 使用更少的交叉验证折数
- 启用并行计算（已默认启用）

### 问题3：模型性能不佳
**解决方案**：
- 增加训练数据量
- 调整特征工程
- 尝试其他算法
- 检查数据质量

## 联系方式

- 作者：Bamboo
- 邮箱：bamboocloudops@gmail.com
- License：Apache 2.0

## 更新日志

- v2.0 (2024-01): 支持四种预测模型的统一训练框架
- v1.0 (2023-12): 初始版本，仅支持QPS预测