#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 生成各种预测模型的训练数据
"""

import json
import os
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# 创建数据目录
os.makedirs("data/training_data", exist_ok=True)


def generate_qps_data(start_date="2024-01-01", days=365):
    """
    生成QPS和实例数的训练数据
    """
    print("生成QPS训练数据...")

    # 生成时间序列
    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h")

    data = []
    np.random.seed(42)

    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        month = ts.month

        # 基础QPS值
        base_qps = 200

        # 时间因素
        if 0 <= hour < 6:  # 凌晨
            hour_factor = 0.2 + 0.1 * np.sin(hour * np.pi / 6)
        elif 6 <= hour < 9:  # 早晨
            hour_factor = 0.3 + 0.4 * (hour - 6) / 3
        elif 9 <= hour < 12:  # 上午
            hour_factor = 0.7 + 0.3 * (hour - 9) / 3
        elif 12 <= hour < 14:  # 午休
            hour_factor = 0.9
        elif 14 <= hour < 18:  # 下午
            hour_factor = 1.0
        elif 18 <= hour < 21:  # 晚高峰
            hour_factor = 0.8 - 0.2 * (hour - 18) / 3
        else:  # 夜间
            hour_factor = 0.4 - 0.1 * (hour - 21) / 3

        # 星期因素
        if day_of_week < 5:  # 工作日
            day_factor = 1.0
        elif day_of_week == 5:  # 周六
            day_factor = 0.6
        else:  # 周日
            day_factor = 0.5

        # 月份因素（模拟季节性）
        month_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (month - 1) / 12)

        # 特殊事件（模拟促销、活动等）
        special_event = 0
        if np.random.random() < 0.02:  # 2%概率出现特殊事件
            special_event = np.random.uniform(100, 500)

        # 计算最终QPS
        qps = base_qps * hour_factor * day_factor * month_factor
        qps += special_event
        qps += np.random.normal(0, 20)  # 随机噪声

        # 确保qps是有限值
        if not np.isfinite(qps):
            qps = base_qps

        qps = max(0, qps)

        # 计算实例数（基于QPS的简单规则）
        if qps < 50:
            instances = 1
        elif qps < 150:
            instances = 2
        elif qps < 300:
            instances = 3 + int((qps - 150) / 50)
        elif qps < 500:
            instances = 6 + int((qps - 300) / 40)
        else:
            instances = 11 + int((qps - 500) / 30)

        instances = min(20, instances)  # 最大20个实例
        instances += np.random.randint(-1, 2)  # 添加一些变化
        instances = max(1, instances)

        data.append({"timestamp": ts, "QPS": round(qps, 2), "instances": instances})

    df = pd.DataFrame(data)

    # 保存数据
    df.to_csv("data/training_data/qps_data.csv", index=False)
    print(f"QPS数据已生成: {len(df)} 条记录")

    return df


def generate_cpu_data(start_date="2024-01-01", days=365):
    """
    生成CPU使用率的训练数据
    """
    print("生成CPU训练数据...")

    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h")

    data = []
    np.random.seed(43)

    # 基础CPU使用率
    base_cpu = 40
    cpu_trend = 0  # 累积趋势

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek

        # 时间模式
        if 0 <= hour < 6:
            hour_factor = 0.5
        elif 6 <= hour < 9:
            hour_factor = 0.7
        elif 9 <= hour < 12:
            hour_factor = 0.9
        elif 12 <= hour < 14:
            hour_factor = 0.85
        elif 14 <= hour < 18:
            hour_factor = 0.95
        elif 18 <= hour < 21:
            hour_factor = 0.8
        else:
            hour_factor = 0.6

        # 工作日vs周末
        if day_of_week < 5:
            day_factor = 1.0
        else:
            day_factor = 0.7

        # 计算当前CPU
        cpu = base_cpu * hour_factor * day_factor

        # 添加趋势（模拟内存泄露等）
        if np.random.random() < 0.01:  # 1%概率开始上升趋势
            cpu_trend += np.random.uniform(5, 10)
        elif np.random.random() < 0.02:  # 2%概率重置（重启）
            cpu_trend = 0

        cpu += cpu_trend

        # 添加峰值
        if np.random.random() < 0.05:  # 5%概率出现峰值
            cpu += np.random.uniform(10, 30)

        # 添加噪声
        cpu += np.random.normal(0, 5)

        # 确保cpu是有限值
        if not np.isfinite(cpu):
            cpu = base_cpu

        # 限制范围
        cpu = np.clip(cpu, 5, 95)

        # 计算预测目标（下一小时的CPU）
        if i < len(timestamps) - 1:
            next_hour = (hour + 1) % 24
            if 0 <= next_hour < 6:
                next_factor = 0.5
            elif 6 <= next_hour < 9:
                next_factor = 0.7
            elif 9 <= next_hour < 12:
                next_factor = 0.9
            elif 12 <= next_hour < 14:
                next_factor = 0.85
            elif 14 <= next_hour < 18:
                next_factor = 0.95
            elif 18 <= next_hour < 21:
                next_factor = 0.8
            else:
                next_factor = 0.6

            cpu_target = base_cpu * next_factor * day_factor + cpu_trend
            cpu_target += np.random.normal(0, 3)
            cpu_target = np.clip(cpu_target, 5, 95)
        else:
            cpu_target = cpu

        data.append(
            {"timestamp": ts, "CPU": round(cpu, 2), "cpu_target": round(cpu_target, 2)}
        )

    df = pd.DataFrame(data)
    df.to_csv("data/training_data/cpu_data.csv", index=False)
    print(f"CPU数据已生成: {len(df)} 条记录")

    return df


def generate_memory_data(start_date="2024-01-01", days=365):
    """
    生成内存使用率的训练数据
    """
    print("生成Memory训练数据...")

    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h")

    data = []
    np.random.seed(44)

    # 基础内存使用率
    base_memory = 50
    memory_leak = 0  # 内存泄露累积

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek
        day_of_month = ts.day

        # 内存使用相对稳定，但有缓慢增长趋势
        hour_factor = 0.9 + 0.1 * np.sin(2 * np.pi * hour / 24)

        # 工作日略高
        if day_of_week < 5:
            day_factor = 1.0
        else:
            day_factor = 0.9

        # 模拟内存泄露
        if np.random.random() < 0.005:  # 0.5%概率开始内存泄露
            memory_leak += np.random.uniform(0.1, 0.5)

        # 每月1号重启，清理内存
        if day_of_month == 1 and hour == 3:
            memory_leak = 0

        # 计算内存使用
        memory = base_memory + memory_leak * 10
        memory *= hour_factor * day_factor

        # 缓存影响（白天缓存更多）
        if 9 <= hour <= 18:
            memory += np.random.uniform(5, 10)

        # 添加噪声
        memory += np.random.normal(0, 2)

        # 确保memory是有限值
        if not np.isfinite(memory):
            memory = base_memory

        # 限制范围
        memory = np.clip(memory, 20, 90)

        # 预测目标（下一小时的内存）
        memory_target = memory + memory_leak * 0.1  # 继续缓慢增长
        memory_target += np.random.normal(0, 1)
        memory_target = np.clip(memory_target, 20, 90)

        data.append(
            {
                "timestamp": ts,
                "MEMORY": round(memory, 2),
                "memory_target": round(memory_target, 2),
            }
        )

    df = pd.DataFrame(data)
    df.to_csv("data/training_data/memory_data.csv", index=False)
    print(f"Memory数据已生成: {len(df)} 条记录")

    return df


def generate_disk_data(start_date="2024-01-01", days=365):
    """
    生成磁盘使用率的训练数据
    """
    print("生成Disk训练数据...")

    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h")

    data = []
    np.random.seed(45)

    # 初始磁盘使用率
    disk_usage = 30
    daily_growth = 0.1  # 每天增长0.1%

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.dayofweek
        day_of_year = ts.dayofyear

        # 磁盘持续增长
        disk_usage += daily_growth / 24

        # 日志文件影响（白天产生更多日志）
        if 8 <= hour <= 20:
            disk_usage += 0.01

        # 每周日凌晨清理
        if day_of_week == 6 and hour == 2:
            # 清理日志，减少5-10%
            cleanup = np.random.uniform(5, 10)
            disk_usage = max(20, disk_usage - cleanup)

        # 每月1号大清理
        if ts.day == 1 and hour == 3:
            # 大清理，减少10-15%
            cleanup = np.random.uniform(10, 15)
            disk_usage = max(20, disk_usage - cleanup)

        # 特殊事件（大文件上传等）
        if np.random.random() < 0.01:  # 1%概率
            disk_usage += np.random.uniform(2, 5)

        # 添加小幅波动
        disk_usage += np.random.normal(0, 0.5)

        # 确保disk_usage是有限值
        if not np.isfinite(disk_usage):
            disk_usage = 30  # 默认值

        # 限制范围
        disk_usage = np.clip(disk_usage, 15, 85)

        # 预测目标（下一小时的磁盘）
        disk_target = disk_usage + daily_growth / 24
        if hour == 1 and day_of_week == 6:  # 即将清理
            disk_target -= 5
        disk_target = np.clip(disk_target, 15, 85)

        data.append(
            {
                "timestamp": ts,
                "DISK": round(disk_usage, 2),
                "disk_target": round(disk_target, 2),
            }
        )

    df = pd.DataFrame(data)
    df.to_csv("data/training_data/disk_data.csv", index=False)
    print(f"Disk数据已生成: {len(df)} 条记录")

    return df


def generate_all_data():
    """生成所有类型的训练数据"""
    print("=" * 60)
    print("开始生成所有训练数据")
    print("=" * 60)

    # 生成各种数据
    qps_df = generate_qps_data()
    cpu_df = generate_cpu_data()
    memory_df = generate_memory_data()
    disk_df = generate_disk_data()

    # 生成数据统计
    stats = {
        "generation_date": datetime.now().isoformat(),
        "datasets": {
            "qps": {
                "records": len(qps_df),
                "columns": list(qps_df.columns),
                "qps_range": [float(qps_df["QPS"].min()), float(qps_df["QPS"].max())],
                "instances_range": [
                    int(qps_df["instances"].min()),
                    int(qps_df["instances"].max()),
                ],
            },
            "cpu": {
                "records": len(cpu_df),
                "columns": list(cpu_df.columns),
                "cpu_range": [float(cpu_df["CPU"].min()), float(cpu_df["CPU"].max())],
                "mean_cpu": float(cpu_df["CPU"].mean()),
            },
            "memory": {
                "records": len(memory_df),
                "columns": list(memory_df.columns),
                "memory_range": [
                    float(memory_df["MEMORY"].min()),
                    float(memory_df["MEMORY"].max()),
                ],
                "mean_memory": float(memory_df["MEMORY"].mean()),
            },
            "disk": {
                "records": len(disk_df),
                "columns": list(disk_df.columns),
                "disk_range": [
                    float(disk_df["DISK"].min()),
                    float(disk_df["DISK"].max()),
                ],
                "mean_disk": float(disk_df["DISK"].mean()),
            },
        },
    }

    # 保存统计信息
    with open("data/training_data/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print("\n数据统计:")
    print(f"  QPS数据: {len(qps_df)} 条")
    print(f"  CPU数据: {len(cpu_df)} 条")
    print(f"  Memory数据: {len(memory_df)} 条")
    print(f"  Disk数据: {len(disk_df)} 条")
    print("\n数据保存位置: data/training_data/")

    return {"qps": qps_df, "cpu": cpu_df, "memory": memory_df, "disk": disk_df}


def visualize_data():
    """可视化生成的数据"""
    import matplotlib.pyplot as plt

    print("\n生成数据可视化...")

    # 读取数据
    qps_df = pd.read_csv("data/training_data/qps_data.csv")
    cpu_df = pd.read_csv("data/training_data/cpu_data.csv")
    memory_df = pd.read_csv("data/training_data/memory_data.csv")
    disk_df = pd.read_csv("data/training_data/disk_data.csv")

    # 转换时间戳
    for df in [qps_df, cpu_df, memory_df, disk_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # QPS数据
    ax = axes[0, 0]
    sample_size = min(7 * 24, len(qps_df))  # 一周的数据
    ax.plot(
        range(sample_size), qps_df["QPS"][:sample_size], "b-", alpha=0.7, label="QPS"
    )
    ax2 = ax.twinx()
    ax2.plot(
        range(sample_size),
        qps_df["instances"][:sample_size],
        "r-",
        alpha=0.7,
        label="Instances",
    )
    ax.set_xlabel("Hours")
    ax.set_ylabel("QPS", color="b")
    ax2.set_ylabel("Instances", color="r")
    ax.set_title("QPS and Instances (1 Week)")
    ax.grid(True, alpha=0.3)

    # CPU数据
    ax = axes[0, 1]
    ax.plot(range(sample_size), cpu_df["CPU"][:sample_size], "g-", alpha=0.7)
    ax.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="Warning")
    ax.set_xlabel("Hours")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_title("CPU Usage (1 Week)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Memory数据
    ax = axes[1, 0]
    ax.plot(range(sample_size), memory_df["MEMORY"][:sample_size], "m-", alpha=0.7)
    ax.axhline(y=85, color="r", linestyle="--", alpha=0.5, label="Warning")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Memory Usage (%)")
    ax.set_title("Memory Usage (1 Week)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Disk数据
    ax = axes[1, 1]
    # 显示一个月的数据以看到清理效果
    month_size = min(30 * 24, len(disk_df))
    ax.plot(range(month_size), disk_df["DISK"][:month_size], "c-", alpha=0.7)
    ax.axhline(y=80, color="r", linestyle="--", alpha=0.5, label="Warning")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Disk Usage (%)")
    ax.set_title("Disk Usage (1 Month)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig("data/training_data/data_visualization.png", dpi=100)
    print("数据可视化已保存到: data/training_data/data_visualization.png")
    plt.close(fig)


def create_sample_data_for_testing():
    """创建用于测试的小样本数据"""
    print("\n生成测试用小样本数据...")

    # 生成100个样本的测试数据
    n_samples = 100

    # QPS测试数据
    qps_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
            "QPS": np.random.uniform(10, 500, n_samples),
            "instances": np.random.randint(1, 10, n_samples),
        }
    )
    qps_test.to_csv("data/training_data/qps_test_sample.csv", index=False)

    # CPU测试数据
    cpu_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
            "CPU": np.random.uniform(20, 80, n_samples),
            "cpu_target": np.random.uniform(20, 80, n_samples),
        }
    )
    cpu_test.to_csv("data/training_data/cpu_test_sample.csv", index=False)

    # Memory测试数据
    memory_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
            "MEMORY": np.random.uniform(40, 80, n_samples),
            "memory_target": np.random.uniform(40, 80, n_samples),
        }
    )
    memory_test.to_csv("data/training_data/memory_test_sample.csv", index=False)

    # Disk测试数据
    disk_test = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
            "DISK": np.random.uniform(30, 70, n_samples),
            "disk_target": np.random.uniform(30, 70, n_samples),
        }
    )
    disk_test.to_csv("data/training_data/disk_test_sample.csv", index=False)

    print(f"测试样本数据已生成（每个{n_samples}条）")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "qps":
            generate_qps_data()
        elif sys.argv[1] == "cpu":
            generate_cpu_data()
        elif sys.argv[1] == "memory":
            generate_memory_data()
        elif sys.argv[1] == "disk":
            generate_disk_data()
        elif sys.argv[1] == "test":
            create_sample_data_for_testing()
        elif sys.argv[1] == "visualize":
            visualize_data()
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("可用参数: qps, cpu, memory, disk, test, visualize")
    else:
        # 生成所有数据
        generate_all_data()

        # 生成可视化
        try:
            visualize_data()
        except Exception as e:
            print(f"可视化失败: {e}")

        # 生成测试样本
        create_sample_data_for_testing()
