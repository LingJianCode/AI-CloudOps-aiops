#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Pod日志数据收集器 - 收集和分析Pod容器日志中的错误和异常
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern

from .base_collector import BaseDataCollector
from app.models.rca_models import LogData
from app.services.kubernetes import KubernetesService


class LogsCollector(BaseDataCollector):
    """
    Pod日志数据收集器

    负责从Kubernetes Pod中收集容器日志，识别错误、异常和警告信息，
    并提取关键的错误模式和堆栈跟踪信息。
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化日志收集器

        Args:
            config_dict: 收集器配置
        """
        super().__init__("logs", config_dict)
        self.k8s: Optional[KubernetesService] = None

        # 编译错误模式正则表达式
        self._compile_error_patterns()

        # 日志级别优先级
        self.log_level_priority = {
            "FATAL": 5,
            "ERROR": 4,
            "WARN": 3,
            "WARNING": 3,
            "INFO": 2,
            "DEBUG": 1,
            "TRACE": 0,
        }

    def _compile_error_patterns(self) -> None:
        """编译错误模式正则表达式"""
        # 常见错误模式
        self.error_patterns = {
            "java_exception": re.compile(
                r"(?:Exception|Error)(?:\s+in\s+thread\s+\"[^\"]*\")?\s*:\s*(.+)", re.IGNORECASE
            ),
            "python_exception": re.compile(r"(\w+Error|\w+Exception):\s*(.+)", re.IGNORECASE),
            "go_panic": re.compile(r"panic:\s*(.+)", re.IGNORECASE),
            "nodejs_error": re.compile(
                r"(?:Error|TypeError|ReferenceError):\s*(.+)", re.IGNORECASE
            ),
            "http_error": re.compile(r"HTTP\s+(\d{3,})\s*[:\-]?\s*(.+)", re.IGNORECASE),
            "database_error": re.compile(
                r"(?:SQL|Database|Connection)\s*(?:Error|Exception):\s*(.+)", re.IGNORECASE
            ),
            "timeout_error": re.compile(
                r"(?:timeout|timed?\s+out)(?:\s*:?\s*(.+))?", re.IGNORECASE
            ),
            "connection_error": re.compile(
                r"(?:connection|connect)(?:\s+(?:failed|refused|reset|timeout))(?:\s*:?\s*(.+))?",
                re.IGNORECASE,
            ),
        }

        # 日志级别模式
        self.log_level_pattern = re.compile(
            r"\b(FATAL|ERROR|WARN|WARNING|INFO|DEBUG|TRACE)\b", re.IGNORECASE
        )

        # 时间戳模式
        self.timestamp_patterns = [
            re.compile(
                r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d{3,6})?(?:Z|[+-]\d{2}:\d{2})?)"
            ),
            re.compile(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})"),
            re.compile(r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"),
        ]

    async def _do_initialize(self) -> None:
        """初始化Kubernetes服务连接"""
        self.k8s = KubernetesService()

        # 验证Kubernetes连接
        if not await self.k8s.health_check():
            raise RuntimeError("无法连接到Kubernetes集群")

    async def collect(
        self, namespace: str, start_time: datetime, end_time: datetime, **kwargs
    ) -> List[LogData]:
        """
        收集Pod日志数据

        Args:
            namespace: Kubernetes命名空间
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数，可包含：
                - pod_names: 指定要收集日志的Pod列表
                - container_names: 指定要收集日志的容器列表
                - max_lines: 每个Pod最大日志行数
                - error_only: 是否只收集错误日志

        Returns:
            List[LogData]: 日志数据列表
        """
        self._ensure_initialized()

        pod_names = kwargs.get("pod_names", [])
        container_names = kwargs.get("container_names", [])
        max_lines = kwargs.get("max_lines", 1000)
        error_only = kwargs.get("error_only", True)

        try:
            # 获取命名空间下的所有Pod
            if not pod_names:
                pods = await self.k8s.get_pods(namespace)
                pod_names = [
                    pod.get("metadata", {}).get("name")
                    for pod in pods
                    if pod.get("metadata", {}).get("name")
                ]

            collected_logs = []

            # 为每个Pod收集日志
            for pod_name in pod_names:
                try:
                    pod_logs = await self._collect_pod_logs(
                        namespace,
                        pod_name,
                        start_time,
                        end_time,
                        container_names,
                        max_lines,
                        error_only,
                    )
                    collected_logs.extend(pod_logs)

                except Exception as e:
                    self.logger.warning(f"收集Pod {pod_name} 日志失败: {str(e)}")
                    continue

            # 按时间排序
            collected_logs.sort(key=lambda x: x.timestamp, reverse=True)

            self.logger.info(f"成功收集 {len(collected_logs)} 条日志")
            return collected_logs

        except Exception as e:
            self.logger.error(f"收集Pod日志失败: {str(e)}")
            return []

    async def _collect_pod_logs(
        self,
        namespace: str,
        pod_name: str,
        start_time: datetime,
        end_time: datetime,
        container_names: List[str],
        max_lines: int,
        error_only: bool,
    ) -> List[LogData]:
        """
        收集单个Pod的日志

        Args:
            namespace: 命名空间
            pod_name: Pod名称
            start_time: 开始时间
            end_time: 结束时间
            container_names: 容器名称列表
            max_lines: 最大日志行数
            error_only: 是否只收集错误日志

        Returns:
            List[LogData]: 日志数据列表
        """
        pod_logs = []

        try:
            # 获取Pod信息以确定容器
            pod_info = await self.k8s.get_pod(namespace, pod_name)
            if not pod_info:
                return pod_logs

            # 确定要收集日志的容器
            containers = pod_info.get("spec", {}).get("containers", [])
            if not container_names:
                container_names = [c.get("name") for c in containers if c.get("name")]

            # 为每个容器收集日志
            for container_name in container_names:
                try:
                    container_logs = await self.k8s.get_pod_logs(
                        namespace=namespace,
                        pod_name=pod_name,
                        container_name=container_name,
                        since_time=start_time,
                        tail_lines=max_lines,
                    )

                    if container_logs:
                        parsed_logs = self._parse_logs(
                            container_logs,
                            pod_name,
                            container_name,
                            start_time,
                            end_time,
                            error_only,
                        )
                        pod_logs.extend(parsed_logs)

                except Exception as e:
                    self.logger.warning(f"收集容器 {container_name} 日志失败: {str(e)}")
                    continue

        except Exception as e:
            self.logger.warning(f"处理Pod {pod_name} 失败: {str(e)}")

        return pod_logs

    def _parse_logs(
        self,
        log_content: str,
        pod_name: str,
        container_name: str,
        start_time: datetime,
        end_time: datetime,
        error_only: bool,
    ) -> List[LogData]:
        """
        解析日志内容

        Args:
            log_content: 原始日志内容
            pod_name: Pod名称
            container_name: 容器名称
            start_time: 开始时间
            end_time: 结束时间
            error_only: 是否只收集错误日志

        Returns:
            List[LogData]: 解析后的日志数据
        """
        parsed_logs = []

        try:
            lines = log_content.split("\n")
            current_stack_trace = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 解析时间戳
                timestamp = self._extract_timestamp(line)
                if not timestamp:
                    timestamp = datetime.now()

                # 检查时间范围
                if not self._is_in_time_range(timestamp, start_time, end_time):
                    continue

                # 提取日志级别
                level = self._extract_log_level(line)

                # 如果只收集错误日志，跳过非错误日志
                if error_only and level not in ["ERROR", "FATAL", "WARN", "WARNING"]:
                    # 但检查是否包含错误模式
                    if not self._contains_error_pattern(line):
                        continue

                # 检测错误类型
                error_type = self._detect_error_type(line)

                # 检测堆栈跟踪
                if self._is_stack_trace_line(line):
                    current_stack_trace.append(line)
                    continue
                elif current_stack_trace:
                    # 堆栈跟踪结束，处理之前收集的堆栈
                    if parsed_logs:
                        parsed_logs[-1].stack_trace = "\n".join(current_stack_trace)
                    current_stack_trace = []

                # 创建LogData对象
                log_data = LogData(
                    timestamp=timestamp,
                    pod_name=pod_name,
                    container_name=container_name,
                    level=level,
                    message=line,
                    error_type=error_type,
                    stack_trace=None,
                )

                parsed_logs.append(log_data)

            # 处理最后的堆栈跟踪
            if current_stack_trace and parsed_logs:
                parsed_logs[-1].stack_trace = "\n".join(current_stack_trace)

        except Exception as e:
            self.logger.warning(f"解析日志失败: {str(e)}")

        return parsed_logs

    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """
        从日志行中提取时间戳

        Args:
            line: 日志行

        Returns:
            Optional[datetime]: 提取的时间戳
        """
        for pattern in self.timestamp_patterns:
            match = pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                try:
                    # 尝试多种时间格式
                    for fmt in [
                        "%Y-%m-%dT%H:%M:%S.%fZ",
                        "%Y-%m-%d %H:%M:%S.%f",
                        "%Y-%m-%d %H:%M:%S",
                        "%m/%d/%Y %H:%M:%S",
                        "%b %d %H:%M:%S",
                    ]:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
                except Exception:
                    continue

        return None

    def _extract_log_level(self, line: str) -> str:
        """
        从日志行中提取日志级别

        Args:
            line: 日志行

        Returns:
            str: 日志级别
        """
        match = self.log_level_pattern.search(line)
        if match:
            return match.group(1).upper()

        # 如果没有明确的级别，根据内容判断
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ["error", "exception", "failed", "fatal"]):
            return "ERROR"
        elif any(keyword in line_lower for keyword in ["warn", "warning"]):
            return "WARN"
        else:
            return "INFO"

    def _contains_error_pattern(self, line: str) -> bool:
        """
        检查日志行是否包含错误模式

        Args:
            line: 日志行

        Returns:
            bool: 是否包含错误模式
        """
        for pattern in self.error_patterns.values():
            if pattern.search(line):
                return True
        return False

    def _detect_error_type(self, line: str) -> Optional[str]:
        """
        检测错误类型

        Args:
            line: 日志行

        Returns:
            Optional[str]: 错误类型
        """
        for error_type, pattern in self.error_patterns.items():
            if pattern.search(line):
                return error_type.replace("_", " ").title()
        return None

    def _is_stack_trace_line(self, line: str) -> bool:
        """
        判断是否为堆栈跟踪行

        Args:
            line: 日志行

        Returns:
            bool: 是否为堆栈跟踪行
        """
        # 常见的堆栈跟踪模式
        stack_patterns = [
            r"^\s*at\s+",  # Java/Scala stack trace
            r"^\s*File\s+\"",  # Python stack trace
            r"^\s*Traceback",  # Python traceback
            r"^\s*goroutine\s+",  # Go stack trace
            r"^\s*\w+\.\w+\(",  # General function call pattern
        ]

        for pattern in stack_patterns:
            if re.search(pattern, line):
                return True

        return False

    def _is_in_time_range(
        self, log_time: datetime, start_time: datetime, end_time: datetime
    ) -> bool:
        """
        检查日志时间是否在指定范围内

        Args:
            log_time: 日志时间
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            bool: 是否在时间范围内
        """
        # 处理时区问题
        if log_time.tzinfo is None:
            log_time = log_time.replace(tzinfo=start_time.tzinfo)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=log_time.tzinfo)
            end_time = end_time.replace(tzinfo=log_time.tzinfo)

        return start_time <= log_time <= end_time

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 收集器是否健康
        """
        try:
            if not self.k8s:
                return False
            return await self.k8s.health_check()
        except Exception:
            return False
