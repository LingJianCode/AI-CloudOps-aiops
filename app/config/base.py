#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 配置基础类 - 基于 Pydantic Settings，支持 ENV > YAML > 默认值
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# 预加载 .env 到进程环境，确保自定义读取函数也能获取
load_dotenv()


def _get_root_dir() -> Path:
    """项目根目录（app/config/base.py -> app -> 项目根）"""
    return Path(__file__).resolve().parents[2]


def _get_env_name() -> str:
    """当前环境名，默认 development"""
    return os.getenv("ENV", "development")


def _get_yaml_path() -> Path:
    """根据 ENV 选择 YAML 配置文件路径"""
    root = _get_root_dir()
    env_name = _get_env_name()
    if env_name and env_name != "development":
        return root / "config" / f"config.{env_name}.yaml"
    return root / "config" / "config.yaml"


def _load_yaml() -> Dict[str, Any]:
    """加载 YAML 配置为字典，找不到文件返回空字典"""
    yaml_path = _get_yaml_path()
    default_path = _get_root_dir() / "config" / "config.yaml"

    try:
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    except Exception:
        # 返回空字典，避免启动失败
        return {}


class BaseAppSettings(BaseSettings):
    """应用基础配置类，统一自定义配置源顺序：
    1) 环境变量 (.env + 进程环境)
    2) YAML 配置文件 (按 ENV 选择)
    3) 初始化参数 / 默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def load_yaml_config(cls) -> Dict[str, Any]:
        """提供给外部使用的 YAML 加载方法"""
        return _load_yaml()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # 自定义 YAML 配置源
        def yaml_settings_source() -> Dict[str, Any]:
            return _load_yaml()

        # 优先级：环境变量 > .env > YAML > 初始化参数 > 文件密钥
        return (
            env_settings,
            dotenv_settings,
            yaml_settings_source,
            init_settings,
            file_secret_settings,
        )


