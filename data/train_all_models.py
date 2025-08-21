#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 综合模型训练器 - 训练QPS、CPU、Memory、Disk四个预测模型
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 创建必要的目录
os.makedirs('data/models', exist_ok=True)
os.makedirs('data/training_data', exist_ok=True)
os.makedirs('data/visualizations', exist_ok=True)


class ModelTrainer:
    """通用模型训练器"""
    
    def __init__(self, model_type: str):
        """
        初始化训练器
        :param model_type: 模型类型 (qps, cpu, memory, disk)
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.metadata = {}
        
        # 定义模型路径
        self.model_path = f'data/models/{self.model_type}_prediction_model.pkl'
        self.scaler_path = f'data/models/{self.model_type}_prediction_scaler.pkl'
        self.metadata_path = f'data/models/{self.model_type}_prediction_model_metadata.json'
        
        # 定义不同类型的特征配置
        self.feature_configs = {
            'qps': {
                'target': 'instances',
                'main_metric': 'QPS',
                'history_features': ['1h', '1d', '1w'],
                'additional_features': ['change', 'avg_6h', 'std_6h'],
                'scale_factor': 1.0
            },
            'cpu': {
                'target': 'cpu_target',
                'main_metric': 'CPU',
                'history_features': ['1h', '1d', '1w'],
                'additional_features': ['change', 'avg_6h', 'std_6h', 'max_6h'],
                'scale_factor': 100.0
            },
            'memory': {
                'target': 'memory_target',
                'main_metric': 'MEMORY',
                'history_features': ['1h', '1d', '1w'],
                'additional_features': ['change', 'avg_6h', 'trend', 'min_6h'],
                'scale_factor': 100.0
            },
            'disk': {
                'target': 'disk_target',
                'main_metric': 'DISK',
                'history_features': ['1h', '1d', '1w'],
                'additional_features': ['change', 'avg_24h', 'growth_rate', 'max_24h'],
                'scale_factor': 100.0
            }
        }
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        提取特征
        :param df: 原始数据
        :return: 特征和目标变量
        """
        config = self.feature_configs[self.model_type]
        main_metric = config['main_metric']
        
        print(f"正在为{self.model_type}模型提取特征...")
        
        # 确保时间戳是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 基础时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                                  (df['is_weekend'] == 0)).astype(int)
        df['is_holiday'] = 0  # 可以添加节假日判断逻辑
        
        # 周期性特征
        df['sin_time'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_time'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 历史特征
        for period in config['history_features']:
            if period == '1h':
                shift = 1
            elif period == '1d':
                shift = 24
            elif period == '1w':
                shift = 24 * 7
            else:
                shift = 1
            
            col_name = f'{main_metric}_{period}_ago'
            df[col_name] = df[main_metric].shift(shift)
        
        # 统计特征
        if 'change' in config['additional_features']:
            change_values = df[main_metric].pct_change()
            # 处理无穷大和NaN值
            change_values = change_values.replace([np.inf, -np.inf], 0)
            change_values = change_values.fillna(0)
            # 限制变化率范围，避免极值
            df[f'{main_metric}_change'] = np.clip(change_values, -10, 10)
        
        if 'avg_6h' in config['additional_features']:
            df[f'{main_metric}_avg_6h'] = df[main_metric].rolling(window=6, min_periods=1).mean()
        
        if 'std_6h' in config['additional_features']:
            std_values = df[main_metric].rolling(window=6, min_periods=1).std()
            df[f'{main_metric}_std_6h'] = std_values.fillna(0)
        
        if 'max_6h' in config['additional_features']:
            df[f'{main_metric}_max_6h'] = df[main_metric].rolling(window=6, min_periods=1).max()
        
        if 'min_6h' in config['additional_features']:
            df[f'{main_metric}_min_6h'] = df[main_metric].rolling(window=6, min_periods=1).min()
        
        if 'avg_24h' in config['additional_features']:
            df[f'{main_metric}_avg_24h'] = df[main_metric].rolling(window=24, min_periods=1).mean()
        
        if 'max_24h' in config['additional_features']:
            df[f'{main_metric}_max_24h'] = df[main_metric].rolling(window=24, min_periods=1).max()
        
        if 'trend' in config['additional_features']:
            # 计算趋势（简单线性趋势）
            def safe_trend(x):
                if len(x) <= 1 or x.std() == 0:
                    return 0.0
                try:
                    slope = np.polyfit(range(len(x)), x, 1)[0]
                    return np.clip(slope, -100, 100)  # 限制趋势范围
                except:
                    return 0.0
            
            df[f'{main_metric}_trend'] = df[main_metric].rolling(window=12, min_periods=2).apply(safe_trend)
            df[f'{main_metric}_trend'] = df[f'{main_metric}_trend'].fillna(0)
        
        if 'growth_rate' in config['additional_features']:
            # 计算增长率，更安全的方式
            prev_values = df[main_metric].shift(24)
            current_values = df[main_metric]
            
            # 避免除以0或极小值
            prev_values_safe = prev_values.replace(0, np.nan)
            prev_values_safe = prev_values_safe.fillna(method='bfill').fillna(1)
            
            growth_rate = (current_values - prev_values) / prev_values_safe
            growth_rate = growth_rate.replace([np.inf, -np.inf], 0)
            growth_rate = growth_rate.fillna(0)
            # 限制增长率范围
            df[f'{main_metric}_growth_rate'] = np.clip(growth_rate, -5, 5)
        
        # 数据清理和验证
        df = self._clean_and_validate_data(df)
        
        # 构建特征列表
        feature_columns = [
            main_metric, 'sin_time', 'cos_time', 'sin_day', 'cos_day',
            'is_business_hour', 'is_weekend', 'is_holiday'
        ]
        
        # 添加历史特征
        for period in config['history_features']:
            feature_columns.append(f'{main_metric}_{period}_ago')
        
        # 添加额外特征
        for feat in config['additional_features']:
            if feat == 'change':
                feature_columns.append(f'{main_metric}_change')
            elif feat == 'avg_6h':
                feature_columns.append(f'{main_metric}_avg_6h')
            elif feat == 'std_6h':
                feature_columns.append(f'{main_metric}_std_6h')
            elif feat == 'max_6h':
                feature_columns.append(f'{main_metric}_max_6h')
            elif feat == 'min_6h':
                feature_columns.append(f'{main_metric}_min_6h')
            elif feat == 'avg_24h':
                feature_columns.append(f'{main_metric}_avg_24h')
            elif feat == 'max_24h':
                feature_columns.append(f'{main_metric}_max_24h')
            elif feat == 'trend':
                feature_columns.append(f'{main_metric}_trend')
            elif feat == 'growth_rate':
                feature_columns.append(f'{main_metric}_growth_rate')
        
        # 选择特征和目标
        features = df[feature_columns]
        target = df[config['target']]
        
        print(f"提取了 {len(features)} 条数据，{len(feature_columns)} 个特征")
        
        return features, target
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理和验证数据，确保没有无穷大值或NaN
        """
        print("正在清理和验证数据...")
        
        initial_rows = len(df)
        
        # 替换所有无穷大值为NaN，然后填充为0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 替换无穷大值
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # 检查极值并进行修正
            if col != 'timestamp':
                # 计算合理的范围
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                
                if pd.isna(q1) or pd.isna(q99):
                    continue
                    
                # 对于极值进行截断
                iqr = q99 - q1
                if iqr > 0:
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q99 + 3 * iqr
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        # 删除包含NaN的行
        df = df.dropna()
        
        # 最后检查是否还有无效值
        for col in numeric_columns:
            if col in df.columns:
                inf_count = np.isinf(df[col]).sum()
                nan_count = df[col].isna().sum()
                
                if inf_count > 0 or nan_count > 0:
                    print(f"警告: 列 {col} 仍有 {inf_count} 个无穷大值和 {nan_count} 个NaN值")
                    # 强制填充剩余的无效值
                    df[col] = df[col].fillna(0)
                    df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        
        if removed_rows > 0:
            print(f"数据清理完成，移除了 {removed_rows} 行无效数据")
        
        return df
    
    def _validate_training_data(self, X, y):
        """
        验证训练数据，移除无效样本
        """
        # 创建DataFrame以便于处理
        data = pd.concat([X, y], axis=1)
        initial_len = len(data)
        
        # 移除包含无穷大或NaN的行
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()
        
        # 检查目标变量的有效性
        y_col = y.name if hasattr(y, 'name') else data.columns[-1]
        
        # 移除目标变量为负数或异常大的样本（对于我们的应用场景）
        if self.model_type == 'qps':  # 实例数应该是正整数
            data = data[data[y_col] > 0]
            data = data[data[y_col] <= 100]  # 最大100个实例
        else:  # CPU、Memory、Disk百分比
            data = data[(data[y_col] >= 0) & (data[y_col] <= 100)]
        
        final_len = len(data)
        removed = initial_len - final_len
        
        if removed > 0:
            print(f"  移除了 {removed} 个无效样本")
        
        # 分离特征和目标
        X_clean = data.iloc[:, :-1]
        y_clean = data.iloc[:, -1]
        
        return X_clean, y_clean
    
    def train(self, X_train, y_train, X_test, y_test):
        """
        训练模型
        """
        print(f"\n开始训练{self.model_type}模型...")
        
        # 训练前的最后数据验证
        print("验证训练数据...")
        X_train, y_train = self._validate_training_data(X_train, y_train)
        X_test, y_test = self._validate_training_data(X_test, y_test)
        
        if len(X_train) == 0:
            raise ValueError("训练数据为空或全部无效")
        
        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 定义模型和参数
        models = self._get_models()
        param_grids = self._get_param_grids()
        
        best_model = None
        best_score = float('-inf')
        best_name = None
        results = {}
        
        # 训练每个模型
        tscv = TimeSeriesSplit(n_splits=5)
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            grid_params = param_grids.get(name, {})
            # 如果参数网格为空，则直接拟合模型
            if isinstance(grid_params, dict) and len(grid_params) == 0:
                model.fit(X_train_scaled, y_train)
                best_estimator = model
                best_params = {}
            else:
                # 网格搜索
                grid_search = GridSearchCV(
                    model,
                    grid_params,
                    cv=tscv,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train_scaled, y_train)
                best_estimator = grid_search.best_estimator_
                best_params = grid_search.best_params_
            
            # 预测
            y_pred_train = best_estimator.predict(X_train_scaled)
            y_pred_test = best_estimator.predict(X_test_scaled)
            
            # 评估
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # 交叉验证
            cv_scores = cross_val_score(best_estimator, X_train_scaled, y_train, cv=tscv, scoring='r2', n_jobs=-1)
            
            results[name] = {
                'model': best_estimator,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': best_params
            }
            
            print(f"  训练集 R²: {train_r2:.4f}")
            print(f"  测试集 R²: {test_r2:.4f}")
            print(f"  测试集 RMSE: {test_rmse:.4f}")
            print(f"  测试集 MAE: {test_mae:.4f}")
            print(f"  交叉验证 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # 更新最佳模型
            if test_r2 > best_score:
                best_model = best_estimator
                best_score = test_r2
                best_name = name
        
        # 保存最佳模型
        self.model = best_model
        print(f"\n最佳模型: {best_name} (测试集 R² = {best_score:.4f})")
        
        # 创建元数据
        self.metadata = {
            'model_name': f'{self.model_type}_prediction_model',
            'model_version': '2.0',
            'model_type': self.model_type,
            'algorithm': best_name,
            'created_at': datetime.now().isoformat(),
            'features': list(X_train.columns),
            'target': self.feature_configs[self.model_type]['target'],
            'performance': {
                'train_r2': results[best_name]['train_r2'],
                'test_r2': results[best_name]['test_r2'],
                'test_rmse': results[best_name]['test_rmse'],
                'test_mae': results[best_name]['test_mae'],
                'cv_mean': results[best_name]['cv_mean'],
                'cv_std': results[best_name]['cv_std']
            },
            'best_params': results[best_name]['best_params'],
            'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} 
                          for k, v in results.items()},
            'data_stats': {
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'n_features': X_train.shape[1]
            }
        }
        
        # 特征重要性（如果模型支持）
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
            self.metadata['feature_importance'] = feature_importance
        
        return results
    
    def _get_models(self):
        """获取模型配置"""
        if self.model_type == 'qps':
            return {
                "Ridge回归": Ridge(random_state=42),
                "随机森林": RandomForestRegressor(random_state=42),
                "梯度提升": GradientBoostingRegressor(random_state=42)
            }
        elif self.model_type == 'cpu':
            return {
                "随机森林": RandomForestRegressor(random_state=42),
                "梯度提升": GradientBoostingRegressor(random_state=42),
                "线性回归": LinearRegression()
            }
        elif self.model_type == 'memory':
            return {
                "梯度提升": GradientBoostingRegressor(random_state=42),
                "随机森林": RandomForestRegressor(random_state=42),
                "Ridge回归": Ridge(random_state=42)
            }
        else:  # disk
            return {
                "随机森林": RandomForestRegressor(random_state=42),
                "梯度提升": GradientBoostingRegressor(random_state=42),
                "Ridge回归": Ridge(random_state=42)
            }
    
    def _get_param_grids(self):
        """获取参数网格"""
        if self.model_type == 'qps':
            return {
                "Ridge回归": {
                    'alpha': [0.01, 0.1, 1.0, 10.0]
                },
                "随机森林": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                "梯度提升": {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        elif self.model_type == 'cpu':
            return {
                "随机森林": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5]
                },
                "梯度提升": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                },
                "线性回归": {}
            }
        elif self.model_type == 'memory':
            return {
                "梯度提升": {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7]
                },
                "随机森林": {
                    'n_estimators': [100, 150],
                    'max_depth': [15, 25, None],
                    'min_samples_split': [2, 5]
                },
                "Ridge回归": {
                    'alpha': [0.1, 1.0, 10.0]
                }
            }
        else:  # disk
            return {
                "随机森林": {
                    'n_estimators': [100, 150],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                "梯度提升": {
                    'n_estimators': [100, 150],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                },
                "Ridge回归": {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
    
    def save(self):
        """保存模型"""
        print(f"\n保存{self.model_type}模型...")
        
        # 保存模型
        joblib.dump(self.model, self.model_path)
        print(f"模型已保存到: {self.model_path}")
        
        # 保存标准化器
        joblib.dump(self.scaler, self.scaler_path)
        print(f"标准化器已保存到: {self.scaler_path}")
        
        # 保存元数据
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"元数据已保存到: {self.metadata_path}")
    
    def visualize(self, X_test, y_test, save_path=None):
        """可视化预测结果"""
        if self.model is None or self.scaler is None:
            print("模型未训练")
            return
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{self.model_type.upper()} - Actual vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 2. Residual Distribution
        ax = axes[0, 1]
        residuals = y_test - y_pred
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.model_type.upper()} - Residual Distribution')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 3. Residuals vs Predicted
        ax = axes[1, 0]
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title(f'{self.model_type.upper()} - Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 4. Feature Importance (if available)
        ax = axes[1, 1]
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            feature_names = [X_test.columns[i] for i in indices]
            feature_values = [importances[i] for i in indices]
            
            ax.barh(range(len(feature_values)), feature_values)
            ax.set_yticks(range(len(feature_values)))
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Importance')
            ax.set_title(f'{self.model_type.upper()} - Feature Importance (Top 10)')
        elif hasattr(self.model, 'coef_'):
            coefs = np.abs(np.ravel(self.model.coef_))
            indices = np.argsort(coefs)[::-1][:10]
            feature_names = [X_test.columns[i] for i in indices]
            feature_values = [coefs[i] for i in indices]

            ax.barh(range(len(feature_values)), feature_values)
            ax.set_yticks(range(len(feature_values)))
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Absolute Coefficient')
            ax.set_title(f'{self.model_type.upper()} - Feature Importance (Coefficients, Top 10)')
        else:
            ax.text(0.5, 0.5, 'Feature importance not available', 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{self.model_type.upper()} - Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")

        plt.close(fig)


def generate_synthetic_data(model_type: str, n_samples: int = 8760) -> pd.DataFrame:
    """
    生成合成训练数据
    :param model_type: 模型类型
    :param n_samples: 样本数量（默认一年的小时数）
    """
    print(f"生成{model_type}的合成数据...")
    
    # 生成时间序列
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # 基础数据
    df = pd.DataFrame({
        'timestamp': timestamps
    })
    
    # 提取时间特征
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['week'] = df['timestamp'].dt.isocalendar().week
    
    # 生成基础模式
    np.random.seed(42)
    
    if model_type == 'qps':
        # QPS模式：工作日高，周末低，有明显的日内变化
        base_qps = 100
        
        # 日内模式
        hourly_pattern = np.array([
            0.3, 0.2, 0.15, 0.1, 0.1, 0.15,  # 0-5点
            0.25, 0.4, 0.7, 0.9, 1.0, 1.0,   # 6-11点
            0.95, 1.0, 0.95, 0.9, 0.85, 0.8, # 12-17点
            0.7, 0.6, 0.5, 0.45, 0.4, 0.35   # 18-23点
        ])
        
        # 周模式
        weekly_pattern = np.array([
            1.0, 1.05, 1.1, 1.05, 0.95,  # 周一到周五
            0.6, 0.5  # 周六周日
        ])
        
        # 月度趋势
        monthly_trend = 1.0 + 0.02 * np.arange(n_samples) / n_samples
        
        qps_values = []
        for i, row in df.iterrows():
            hour_factor = hourly_pattern[row['hour']]
            week_factor = weekly_pattern[row['day_of_week']]
            month_factor = monthly_trend[i]
            
            # 添加一些随机性和异常
            noise = np.random.normal(0, 10)
            if np.random.random() < 0.01:  # 1%的异常峰值
                noise += np.random.uniform(50, 200)
            
            qps = base_qps * hour_factor * week_factor * month_factor + noise
            qps = max(0, qps)  # 确保非负
            qps_values.append(qps)
        
        df['QPS'] = qps_values
        
        # 计算实例数（基于QPS）
        df['instances'] = np.clip(
            np.round(df['QPS'] / 50 + np.random.normal(0, 0.5, n_samples)),
            1, 20
        ).astype(int)
    
    elif model_type == 'cpu':
        # CPU使用率模式
        base_cpu = 50
        
        # 日内模式
        hourly_pattern = np.array([
            0.4, 0.35, 0.3, 0.3, 0.35, 0.4,   # 0-5点
            0.5, 0.6, 0.8, 0.9, 0.95, 1.0,    # 6-11点
            1.0, 0.95, 0.9, 0.85, 0.8, 0.75,  # 12-17点
            0.7, 0.65, 0.6, 0.55, 0.5, 0.45   # 18-23点
        ])
        
        cpu_values = []
        for i, row in df.iterrows():
            hour_factor = hourly_pattern[row['hour']]
            
            # 工作日vs周末
            if row['day_of_week'] < 5:
                day_factor = 1.0
            else:
                day_factor = 0.6
            
            # 添加噪声
            noise = np.random.normal(0, 5)
            
            # 偶尔的峰值
            if np.random.random() < 0.05:
                noise += np.random.uniform(10, 30)
            
            cpu = base_cpu * hour_factor * day_factor + noise
            cpu = np.clip(cpu, 5, 95)  # 限制在5-95之间
            cpu_values.append(cpu)
        
        df['CPU'] = cpu_values
        
        # 未来CPU预测目标（简单模拟）
        df['cpu_target'] = df['CPU'].shift(-1).fillna(df['CPU'].mean())
    
    elif model_type == 'memory':
        # 内存使用率模式（更稳定，缓慢增长）
        base_memory = 60
        
        # 缓慢增长趋势
        growth_trend = np.linspace(0, 10, n_samples)
        
        memory_values = []
        for i, row in df.iterrows():
            # 基础值
            memory = base_memory + growth_trend[i]
            
            # 工作时间略高
            if 9 <= row['hour'] <= 17 and row['day_of_week'] < 5:
                memory += 5
            
            # 添加小幅波动
            noise = np.random.normal(0, 2)
            memory += noise
            
            # 偶尔的内存泄露模拟
            if np.random.random() < 0.001:
                memory += np.random.uniform(10, 20)
            
            memory = np.clip(memory, 30, 90)
            memory_values.append(memory)
        
        df['MEMORY'] = memory_values
        
        # 未来内存预测目标
        df['memory_target'] = df['MEMORY'].shift(-1).fillna(df['MEMORY'].mean())
    
    elif model_type == 'disk':
        # 磁盘使用率模式（持续缓慢增长）
        base_disk = 40
        
        # 持续增长趋势
        growth_trend = np.linspace(0, 30, n_samples)
        
        disk_values = []
        for i, row in df.iterrows():
            # 基础值 + 增长
            disk = base_disk + growth_trend[i]
            
            # 每日清理（模拟日志清理等）
            if row['hour'] == 3:  # 凌晨3点清理
                disk -= np.random.uniform(5, 10)
            
            # 工作时间略高
            if 9 <= row['hour'] <= 17 and row['day_of_week'] < 5:
                disk += np.random.uniform(1, 3)
            
            # 添加小幅波动
            noise = np.random.normal(0, 1)
            disk += noise
            
            disk = np.clip(disk, 20, 85)
            disk_values.append(disk)
        
        df['DISK'] = disk_values
        
        # 未来磁盘预测目标
        df['disk_target'] = df['DISK'].shift(-1).fillna(df['DISK'].mean())
    
    return df


def train_all_models():
    """训练所有模型的主函数"""
    print("=" * 80)
    print("开始训练所有预测模型")
    print("=" * 80)
    
    model_types = ['qps', 'cpu', 'memory', 'disk']
    results_summary = {}
    
    for model_type in model_types:
        print(f"\n{'='*40}")
        print(f"训练 {model_type.upper()} 模型")
        print(f"{'='*40}")
        
        # 生成或加载数据
        data_file = f'data/training_data/{model_type}_data.csv'
        
        if os.path.exists(data_file):
            print(f"从文件加载数据: {data_file}")
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"生成合成数据...")
            df = generate_synthetic_data(model_type)
            # 保存数据
            os.makedirs('data/training_data', exist_ok=True)
            df.to_csv(data_file, index=False)
            print(f"数据已保存到: {data_file}")
        
        # 创建训练器
        trainer = ModelTrainer(model_type)
        
        # 提取特征
        features, target = trainer.extract_features(df)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        # 训练模型
        results = trainer.train(X_train, y_train, X_test, y_test)
        
        # 保存模型
        trainer.save()
        
        # 可视化
        viz_path = f'data/visualizations/{model_type}_results.png'
        trainer.visualize(X_test, y_test, save_path=viz_path)
        
        # 记录结果
        results_summary[model_type] = trainer.metadata['performance']
    
    # 打印总结
    print("\n" + "=" * 80)
    print("训练完成 - 结果总结")
    print("=" * 80)
    
    for model_type, performance in results_summary.items():
        print(f"\n{model_type.upper()} 模型:")
        print(f"  测试集 R²: {performance['test_r2']:.4f}")
        print(f"  测试集 RMSE: {performance['test_rmse']:.4f}")
        print(f"  测试集 MAE: {performance['test_mae']:.4f}")
        print(f"  交叉验证 R²: {performance['cv_mean']:.4f} (+/- {performance['cv_std']:.4f})")
    
    # 保存总结
    summary_file = 'data/models/training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'models': results_summary
        }, f, indent=2)
    
    print(f"\n训练总结已保存到: {summary_file}")
    print("\n所有模型训练完成！")


def test_models():
    """测试所有已训练的模型"""
    print("\n" + "=" * 80)
    print("测试已训练的模型")
    print("=" * 80)
    
    model_types = ['qps', 'cpu', 'memory', 'disk']
    
    for model_type in model_types:
        print(f"\n测试 {model_type.upper()} 模型:")
        print("-" * 40)
        
        # 加载模型
        model_path = f'data/models/{model_type}_prediction_model.pkl'
        scaler_path = f'data/models/{model_type}_prediction_scaler.pkl'
        metadata_path = f'data/models/{model_type}_prediction_model_metadata.json'
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            continue
        
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # 创建测试数据
            test_cases = generate_test_cases(model_type)
            
            for case in test_cases:
                # 构建特征
                features_dict = build_features_dict(model_type, case)
                
                # 确保特征顺序与训练时一致
                feature_names = metadata['features']
                features_array = np.array([[features_dict.get(feat, 0.0) for feat in feature_names]])
                
                # 标准化
                features_scaled = scaler.transform(features_array)
                
                # 预测
                prediction = model.predict(features_scaled)[0]
                
                print(f"  {case['name']}:")
                print(f"    输入: {case['main_value']:.1f}")
                print(f"    预测: {prediction:.2f}")
                
        except Exception as e:
            print(f"测试失败: {str(e)}")


def generate_test_cases(model_type: str) -> List[Dict]:
    """生成测试用例"""
    if model_type == 'qps':
        return [
            {'name': '低流量', 'main_value': 10.0, 'hour': 3, 'day_of_week': 2},
            {'name': '中等流量', 'main_value': 100.0, 'hour': 10, 'day_of_week': 2},
            {'name': '高峰流量', 'main_value': 500.0, 'hour': 14, 'day_of_week': 3},
            {'name': '周末流量', 'main_value': 50.0, 'hour': 15, 'day_of_week': 6}
        ]
    elif model_type == 'cpu':
        return [
            {'name': '低负载', 'main_value': 20.0, 'hour': 3, 'day_of_week': 2},
            {'name': '正常负载', 'main_value': 50.0, 'hour': 10, 'day_of_week': 2},
            {'name': '高负载', 'main_value': 80.0, 'hour': 14, 'day_of_week': 3},
            {'name': '临界负载', 'main_value': 90.0, 'hour': 11, 'day_of_week': 1}
        ]
    elif model_type == 'memory':
        return [
            {'name': '低内存', 'main_value': 30.0, 'hour': 3, 'day_of_week': 2},
            {'name': '正常内存', 'main_value': 60.0, 'hour': 10, 'day_of_week': 2},
            {'name': '高内存', 'main_value': 80.0, 'hour': 14, 'day_of_week': 3},
            {'name': '内存告警', 'main_value': 85.0, 'hour': 11, 'day_of_week': 1}
        ]
    else:  # disk
        return [
            {'name': '空闲磁盘', 'main_value': 20.0, 'hour': 3, 'day_of_week': 2},
            {'name': '正常使用', 'main_value': 50.0, 'hour': 10, 'day_of_week': 2},
            {'name': '磁盘较满', 'main_value': 75.0, 'hour': 14, 'day_of_week': 3},
            {'name': '磁盘告警', 'main_value': 85.0, 'hour': 11, 'day_of_week': 1}
        ]


def build_features_dict(model_type: str, case: Dict) -> Dict:
    """构建特征字典"""
    metric_map = {
        'qps': 'QPS',
        'cpu': 'CPU',
        'memory': 'MEMORY',
        'disk': 'DISK'
    }
    
    main_metric = metric_map[model_type]
    main_value = case['main_value']
    hour = case['hour']
    day_of_week = case['day_of_week']
    
    features = {
        main_metric: main_value,
        'sin_time': np.sin(2 * np.pi * hour / 24),
        'cos_time': np.cos(2 * np.pi * hour / 24),
        'sin_day': np.sin(2 * np.pi * day_of_week / 7),
        'cos_day': np.cos(2 * np.pi * day_of_week / 7),
        'is_business_hour': 1 if 9 <= hour <= 17 and day_of_week < 5 else 0,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'is_holiday': 0,
        f'{main_metric}_1h_ago': main_value * 0.95,
        f'{main_metric}_1d_ago': main_value * 1.05,
        f'{main_metric}_1w_ago': main_value * 1.0,
        f'{main_metric}_change': 0.05,
        f'{main_metric}_avg_6h': main_value * 0.98,
    }
    
    # 添加特定特征
    if model_type in ['qps', 'cpu']:
        features[f'{main_metric}_std_6h'] = 2.0
    
    if model_type == 'cpu':
        features[f'{main_metric}_max_6h'] = main_value * 1.1
    
    if model_type == 'memory':
        features[f'{main_metric}_trend'] = 0.01
        features[f'{main_metric}_min_6h'] = main_value * 0.9
    
    if model_type == 'disk':
        features[f'{main_metric}_avg_24h'] = main_value * 0.99
        features[f'{main_metric}_growth_rate'] = 0.001
        features[f'{main_metric}_max_24h'] = main_value * 1.05
    
    return features


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 只测试已有模型
        test_models()
    else:
        # 训练所有模型
        train_all_models()
        
        # 测试模型
        print("\n" + "=" * 80)
        test_models()