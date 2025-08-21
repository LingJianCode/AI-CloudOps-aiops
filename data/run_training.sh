#!/bin/bash
set -Eeuo pipefail

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# AI-CloudOps-aiops 模型训练脚本
# Author: Bamboo
# Email: bamboocloudops@gmail.com

echo "=========================================="
echo "AI-CloudOps 模型训练系统"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python环境
echo -e "${YELLOW}检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: Python3未安装${NC}"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python版本: $python_version${NC}"

# 创建必要的目录
echo -e "${YELLOW}创建目录结构...${NC}"
mkdir -p data/models
mkdir -p data/training_data
mkdir -p data/visualizations
echo -e "${GREEN}目录创建完成${NC}"

# 安装依赖（如果需要）
echo -e "${YELLOW}检查依赖...${NC}"
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo -e "${YELLOW}检测到 requirements.txt，开始安装...${NC}"
    python3 -m pip install -r "$PROJECT_ROOT/requirements.txt"
else
    pip_packages="pandas numpy scikit-learn joblib matplotlib seaborn"
    for package in $pip_packages; do
        if ! python3 -c "import $package" 2>/dev/null; then
            echo -e "${YELLOW}安装 $package...${NC}"
            python3 -m pip install "$package"
        fi
    done
fi
echo -e "${GREEN}依赖检查完成${NC}"

# 步骤1: 生成训练数据
echo ""
echo -e "${YELLOW}=========================================="
echo "步骤 1: 生成训练数据"
echo -e "==========================================${NC}"
echo ""

if [ "${1:-}" == "--skip-data" ]; then
    echo -e "${YELLOW}跳过数据生成（使用现有数据）${NC}"
else
    python3 data/generate_training_data.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}数据生成失败${NC}"
        exit 1
    fi
    echo -e "${GREEN}数据生成成功${NC}"
fi

# 步骤2: 训练模型
echo ""
echo -e "${YELLOW}=========================================="
echo "步骤 2: 训练所有模型"
echo -e "==========================================${NC}"
echo ""

python3 data/train_all_models.py
if [ $? -ne 0 ]; then
    echo -e "${RED}模型训练失败${NC}"
    exit 1
fi
echo -e "${GREEN}模型训练成功${NC}"

# 步骤3: 测试模型
echo ""
echo -e "${YELLOW}=========================================="
echo "步骤 3: 测试模型"
echo -e "==========================================${NC}"
echo ""

python3 data/train_all_models.py test
if [ $? -ne 0 ]; then
    echo -e "${RED}模型测试失败${NC}"
    exit 1
fi
echo -e "${GREEN}模型测试成功${NC}"

# 显示结果
echo ""
echo -e "${GREEN}=========================================="
echo "训练完成！"
echo -e "==========================================${NC}"
echo ""
echo "模型文件位置:"
echo "  - QPS模型: data/models/qps_prediction_model.pkl"
echo "  - CPU模型: data/models/cpu_prediction_model.pkl"
echo "  - Memory模型: data/models/memory_prediction_model.pkl"
echo "  - Disk模型: data/models/disk_prediction_model.pkl"
echo ""
echo "训练数据位置:"
echo "  - data/training_data/"
echo ""
echo "可视化结果:"
echo "  - data/visualizations/"
echo ""
echo -e "${GREEN}所有任务完成！${NC}"