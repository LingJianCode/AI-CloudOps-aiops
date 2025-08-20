#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: K8s事件数据收集器 - 收集和分析Kubernetes集群事件
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base_collector import BaseDataCollector
from app.models.rca_models import EventData, SeverityLevel
from app.services.kubernetes import KubernetesService


class EventsCollector(BaseDataCollector):
    """
    Kubernetes事件数据收集器

    负责从Kubernetes API收集集群事件，包括Pod状态变化、调度失败、
    资源不足等关键事件，并根据事件类型和原因进行严重程度分类。
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化事件收集器

        Args:
            config_dict: 收集器配置
        """
        super().__init__("events", config_dict)
        self.k8s: Optional[KubernetesService] = None

        # 定义严重事件类型
        self.critical_reasons = {
            "OOMKilled",
            "Killing",
            "Failed",
            "FailedScheduling",
            "FailedMount",
            "FailedCreatePodSandBox",
            "NetworkNotReady",
        }

        self.high_severity_reasons = {
            "Unhealthy",
            "BackOff",
            "ImagePullBackOff",
            "ErrImagePull",
            "CrashLoopBackOff",
            "FailedSync",
            "NodeNotReady",
        }

        self.medium_severity_reasons = {
            "Pulled",
            "Created",
            "Started",
            "Scheduled",
            "SuccessfulCreate",
            "ScalingReplicaSet",
        }

    async def _do_initialize(self) -> None:
        """初始化Kubernetes服务连接"""
        self.k8s = KubernetesService()

        # 验证Kubernetes连接
        if not await self.k8s.health_check():
            raise RuntimeError("无法连接到Kubernetes集群")

    async def collect(
        self, namespace: str, start_time: datetime, end_time: datetime, **kwargs
    ) -> List[EventData]:
        """
        收集K8s事件数据

        Args:
            namespace: Kubernetes命名空间
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数，可包含：
                - event_types: 指定要收集的事件类型
                - object_names: 指定要监控的对象名称

        Returns:
            List[EventData]: 事件数据列表
        """
        self._ensure_initialized()

        event_types = kwargs.get("event_types", ["Warning", "Normal"])
        object_names = kwargs.get("object_names", [])

        try:
            # 从Kubernetes API获取事件
            raw_events = await self.k8s.get_events(
                namespace=namespace, limit=1000  # 获取更多事件用于时间过滤
            )

            # 处理和过滤事件
            processed_events = []
            total_events = len(raw_events)
            filtered_by_time = 0
            filtered_by_type = 0
            filtered_by_name = 0
            
            self.logger.info(f"开始处理 {total_events} 个原始事件，时间范围: {start_time} - {end_time}")
            
            for i, event in enumerate(raw_events):
                try:
                    # 调试前3个事件（简化版）
                    if i < 3:
                        event_name = event.get("metadata", {}).get("name", f"event_{i+1}")
                        event_time = self._parse_event_time(event)
                        self.logger.debug(f"事件 {i+1} ({event_name}): 时间={event_time}")
                    
                    # 过滤时间范围
                    event_time = self._parse_event_time(event)
                    if not self._is_in_time_range(event_time, start_time, end_time):
                        filtered_by_time += 1
                        if i < 3:  # 只打印前3个事件的调试信息
                            self.logger.info(f"事件 {i+1} 被时间过滤: 解析后时间={event_time}, 范围={start_time} - {end_time}")
                        continue

                    # 过滤事件类型
                    if event_types and event.get("type") not in event_types:
                        filtered_by_type += 1
                        if i < 3:
                            self.logger.debug(f"事件 {i+1} 被类型过滤: type={event.get('type')}, allowed={event_types}")
                        continue

                    # 过滤对象名称（如果指定）
                    if object_names:
                        involved_object = event.get("involvedObject", {})
                        if involved_object.get("name") not in object_names:
                            filtered_by_name += 1
                            if i < 3:
                                self.logger.debug(f"事件 {i+1} 被名称过滤: name={involved_object.get('name')}, allowed={object_names}")
                            continue

                    # 转换为EventData对象
                    event_data = self._convert_to_event_data(event)
                    processed_events.append(event_data)

                except Exception as e:
                    self.logger.warning(f"处理事件失败: {str(e)}")
                    continue
            
            self.logger.info(f"事件过滤统计: 总数={total_events}, 时间过滤={filtered_by_time}, 类型过滤={filtered_by_type}, 名称过滤={filtered_by_name}, 最终保留={len(processed_events)}")

            # 按严重程度排序
            processed_events.sort(key=lambda x: self._severity_order(x.severity), reverse=True)

            self.logger.info(f"成功收集 {len(processed_events)} 个事件")
            return processed_events

        except Exception as e:
            self.logger.error(f"收集K8s事件失败: {str(e)}")
            return []

    def _parse_event_time(self, event: Dict[str, Any]) -> datetime:
        """
        解析事件时间

        Args:
            event: 原始事件数据

        Returns:
            datetime: 事件时间
        """
        # 尝试多种可能的时间戳字段名
        timestamp_fields = [
            "lastTimestamp", "last_timestamp",
            "firstTimestamp", "first_timestamp", 
            "eventTime", "event_time",
            "metadata.creationTimestamp", "metadata.creation_timestamp"
        ]
        
        timestamp_value = None
        for field in timestamp_fields:
            if "." in field:
                # 处理嵌套字段
                parts = field.split(".")
                value = event
                for part in parts:
                    value = value.get(part, {}) if isinstance(value, dict) else None
                    if value is None:
                        break
                if value:
                    timestamp_value = value
                    break
            else:
                timestamp_value = event.get(field)
                if timestamp_value:
                    break

        if timestamp_value:
            try:
                # 如果已经是datetime对象，直接返回
                if isinstance(timestamp_value, datetime):
                    return timestamp_value
                
                # 如果是字符串，进行解析
                elif isinstance(timestamp_value, str):
                    if timestamp_value.endswith("Z"):
                        timestamp_value = timestamp_value[:-1] + "+00:00"
                    return datetime.fromisoformat(timestamp_value)
                    
            except ValueError as e:
                self.logger.warning(f"无法解析时间戳 '{timestamp_value}': {e}")
                pass

        # 如果无法解析时间，使用UTC时间
        self.logger.warning(f"事件缺少时间戳，使用当前UTC时间: {event.get('metadata', {}).get('name', 'unknown')}")
        return datetime.now(timezone.utc)

    def _is_in_time_range(
        self, event_time: datetime, start_time: datetime, end_time: datetime
    ) -> bool:
        """
        检查事件是否在指定时间范围内

        Args:
            event_time: 事件时间
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            bool: 是否在时间范围内
        """
        # 处理时区问题
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=start_time.tzinfo)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=event_time.tzinfo)
            end_time = end_time.replace(tzinfo=event_time.tzinfo)

        return start_time <= event_time <= end_time

    def _convert_to_event_data(self, event: Dict[str, Any]) -> EventData:
        """
        将原始事件数据转换为EventData对象

        Args:
            event: 原始事件数据

        Returns:
            EventData: 事件数据对象
        """
        # 提取基本信息
        timestamp = self._parse_event_time(event)
        event_type = event.get("type") or "Unknown"
        reason = event.get("reason") or "Unknown"
        message = event.get("message") or ""
        count = event.get("count") or 1
        
        # 确保数据类型正确
        if not isinstance(message, str):
            message = str(message) if message is not None else ""
        if not isinstance(count, int):
            try:
                count = int(count) if count is not None else 1
            except (ValueError, TypeError):
                count = 1

        # 提取涉及的对象信息
        involved_object = event.get("involvedObject", {})
        object_info = {
            "kind": involved_object.get("kind", ""),
            "name": involved_object.get("name", ""),
            "namespace": involved_object.get("namespace", ""),
            "uid": involved_object.get("uid", ""),
        }

        # 确定严重程度
        severity = self._determine_severity(event_type, reason, message)

        return EventData(
            timestamp=timestamp,
            type=event_type,
            reason=reason,
            message=message,
            involved_object=object_info,
            severity=severity,
            count=count,
        )

    def _determine_severity(self, event_type: str, reason: str, message: str) -> SeverityLevel:
        """
        确定事件严重程度

        Args:
            event_type: 事件类型
            reason: 事件原因
            message: 事件消息

        Returns:
            SeverityLevel: 严重程度级别
        """
        # 检查原因是否在预定义的严重程度列表中
        if reason in self.critical_reasons:
            return SeverityLevel.CRITICAL
        elif reason in self.high_severity_reasons:
            return SeverityLevel.HIGH
        elif reason in self.medium_severity_reasons:
            return SeverityLevel.MEDIUM

        # 基于事件类型判断
        if event_type == "Warning":
            # 进一步分析消息内容
            message_lower = message.lower()

            if any(
                keyword in message_lower for keyword in ["failed", "error", "timeout", "killed"]
            ):
                return SeverityLevel.CRITICAL
            elif any(keyword in message_lower for keyword in ["unhealthy", "backoff", "pending"]):
                return SeverityLevel.HIGH
            else:
                return SeverityLevel.MEDIUM

        # Normal类型事件通常是低严重程度
        return SeverityLevel.LOW

    def _severity_order(self, severity: SeverityLevel) -> int:
        """
        获取严重程度的排序权重

        Args:
            severity: 严重程度

        Returns:
            int: 排序权重
        """
        severity_weights = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1,
        }
        return severity_weights.get(severity, 0)

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

    async def get_event_statistics(
        self, namespace: str, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """
        获取事件统计信息

        Args:
            namespace: Kubernetes命名空间
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            Dict[str, Any]: 事件统计信息
        """
        events = await self.collect(namespace, start_time, end_time)

        # 统计各类事件数量
        stats = {
            "total_events": len(events),
            "by_severity": {
                "critical": len([e for e in events if e.severity == SeverityLevel.CRITICAL]),
                "high": len([e for e in events if e.severity == SeverityLevel.HIGH]),
                "medium": len([e for e in events if e.severity == SeverityLevel.MEDIUM]),
                "low": len([e for e in events if e.severity == SeverityLevel.LOW]),
            },
            "by_type": {},
            "by_reason": {},
            "most_affected_objects": {},
        }

        # 统计事件类型
        for event in events:
            stats["by_type"][event.type] = stats["by_type"].get(event.type, 0) + 1
            stats["by_reason"][event.reason] = stats["by_reason"].get(event.reason, 0) + 1

            # 统计受影响的对象
            obj_key = (
                f"{event.involved_object.get('kind', '')}/{event.involved_object.get('name', '')}"
            )
            stats["most_affected_objects"][obj_key] = (
                stats["most_affected_objects"].get(obj_key, 0) + 1
            )

        return stats
