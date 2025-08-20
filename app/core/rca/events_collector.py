#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 优化的K8s事件数据收集器
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from .base_collector import BaseDataCollector
from app.models.rca_models import EventData, SeverityLevel
from app.services.kubernetes import KubernetesService


class EventsCollector(BaseDataCollector):
    """优化的Kubernetes事件数据收集器"""

    # 严重程度映射 - 简化配置
    SEVERITY_MAPPING = {
        SeverityLevel.CRITICAL: {
            "OOMKilled", "Killing", "Failed", "FailedScheduling", 
            "FailedMount", "FailedCreatePodSandBox", "NetworkNotReady"
        },
        SeverityLevel.HIGH: {
            "Unhealthy", "BackOff", "ImagePullBackOff", "ErrImagePull",
            "CrashLoopBackOff", "FailedSync", "NodeNotReady"
        },
        SeverityLevel.MEDIUM: {
            "Pulled", "Created", "Started", "Scheduled",
            "SuccessfulCreate", "ScalingReplicaSet"
        }
    }

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        super().__init__("events", config_dict)
        self.k8s: Optional[KubernetesService] = None
        self._severity_cache = self._build_severity_cache()

    def _build_severity_cache(self) -> Dict[str, SeverityLevel]:
        """构建严重程度缓存以提高查找性能"""
        cache = {}
        for severity, reasons in self.SEVERITY_MAPPING.items():
            for reason in reasons:
                cache[reason] = severity
        return cache

    async def _do_initialize(self) -> None:
        """初始化Kubernetes服务连接"""
        self.k8s = KubernetesService()
        if not await self.k8s.health_check():
            raise RuntimeError("无法连接到Kubernetes集群")

    async def collect(
        self, 
        namespace: str, 
        start_time: datetime, 
        end_time: datetime, 
        **kwargs
    ) -> List[EventData]:
        """
        收集K8s事件数据（优化版本）
        
        Args:
            namespace: Kubernetes命名空间
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数
        
        Returns:
            List[EventData]: 事件数据列表
        """
        self._ensure_initialized()
        
        # 确保时间有时区信息
        start_time = self._ensure_timezone(start_time)
        end_time = self._ensure_timezone(end_time)
        
        event_types = set(kwargs.get("event_types", ["Warning", "Normal"]))
        object_names = set(kwargs.get("object_names", []))
        
        try:
            # 获取事件
            raw_events = await self.k8s.get_events(namespace=namespace, limit=1000)
            
            # 批量处理事件
            processed_events = await self._batch_process_events(
                raw_events, start_time, end_time, event_types, object_names
            )
            
            # 按严重程度和时间排序
            processed_events.sort(
                key=lambda x: (self._severity_order(x.severity), x.timestamp),
                reverse=True
            )
            
            self.logger.info(f"成功收集 {len(processed_events)}/{len(raw_events)} 个事件")
            return processed_events
            
        except Exception as e:
            self.logger.error(f"收集K8s事件失败: {str(e)}")
            return []

    async def _batch_process_events(
        self,
        raw_events: List[Dict],
        start_time: datetime,
        end_time: datetime,
        event_types: Set[str],
        object_names: Set[str]
    ) -> List[EventData]:
        """批量处理事件以提高性能"""
        processed_events = []
        
        # 使用异步批处理
        batch_size = 100
        for i in range(0, len(raw_events), batch_size):
            batch = raw_events[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self._process_single_event(
                    event, start_time, end_time, event_types, object_names
                ) for event in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, EventData):
                    processed_events.append(result)
                elif isinstance(result, Exception):
                    self.logger.warning(f"处理事件失败: {result}")
        
        return processed_events

    async def _process_single_event(
        self,
        event: Dict,
        start_time: datetime,
        end_time: datetime,
        event_types: Set[str],
        object_names: Set[str]
    ) -> Optional[EventData]:
        """处理单个事件"""
        # 解析时间
        event_time = self._parse_event_time(event)
        
        # 时间过滤
        if not (start_time <= event_time <= end_time):
            return None
        
        # 类型过滤
        if event_types and event.get("type") not in event_types:
            return None
        
        # 对象名称过滤
        if object_names:
            involved_object = event.get("involvedObject", {})
            if involved_object.get("name") not in object_names:
                return None
        
        return self._convert_to_event_data(event)

    def _parse_event_time(self, event: Dict[str, Any]) -> datetime:
        """优化的时间解析"""
        # 按优先级尝试不同的时间字段
        for field in ["lastTimestamp", "firstTimestamp", "eventTime"]:
            timestamp = event.get(field)
            if timestamp:
                try:
                    if isinstance(timestamp, datetime):
                        return self._ensure_timezone(timestamp)
                    elif isinstance(timestamp, str):
                        # 处理ISO格式
                        if timestamp.endswith("Z"):
                            timestamp = timestamp[:-1] + "+00:00"
                        dt = datetime.fromisoformat(timestamp)
                        return self._ensure_timezone(dt)
                except (ValueError, AttributeError):
                    continue
        
        # 尝试从metadata获取
        creation_timestamp = event.get("metadata", {}).get("creationTimestamp")
        if creation_timestamp:
            try:
                if creation_timestamp.endswith("Z"):
                    creation_timestamp = creation_timestamp[:-1] + "+00:00"
                dt = datetime.fromisoformat(creation_timestamp)
                return self._ensure_timezone(dt)
            except (ValueError, AttributeError):
                pass
        
        # 返回当前UTC时间
        return datetime.now(timezone.utc)

    def _ensure_timezone(self, dt: datetime) -> datetime:
        """确保datetime对象有时区信息"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _convert_to_event_data(self, event: Dict[str, Any]) -> EventData:
        """优化的事件转换"""
        timestamp = self._parse_event_time(event)
        event_type = event.get("type", "Unknown")
        reason = event.get("reason", "Unknown")
        message = str(event.get("message", ""))
        count = int(event.get("count", 1))
        
        involved_object = event.get("involvedObject", {})
        object_info = {
            "kind": involved_object.get("kind", ""),
            "name": involved_object.get("name", ""),
            "namespace": involved_object.get("namespace", ""),
            "uid": involved_object.get("uid", ""),
        }
        
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
        """优化的严重程度判断"""
        # 使用缓存快速查找
        if reason in self._severity_cache:
            return self._severity_cache[reason]
        
        # 基于事件类型和消息内容的启发式判断
        if event_type == "Warning":
            message_lower = message.lower()
            if any(kw in message_lower for kw in ["failed", "error", "timeout", "killed"]):
                return SeverityLevel.CRITICAL
            elif any(kw in message_lower for kw in ["unhealthy", "backoff", "pending"]):
                return SeverityLevel.HIGH
            return SeverityLevel.MEDIUM
        
        return SeverityLevel.LOW

    def _severity_order(self, severity: SeverityLevel) -> int:
        """获取严重程度排序权重"""
        return {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1,
        }.get(severity, 0)

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            return self.k8s and await self.k8s.health_check()
        except Exception:
            return False

    async def get_event_patterns(
        self, 
        namespace: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """分析事件模式"""
        events = await self.collect(namespace, start_time, end_time)
        
        patterns = {
            "total_events": len(events),
            "severity_distribution": defaultdict(int),
            "reason_frequency": defaultdict(int),
            "affected_resources": defaultdict(set),
            "temporal_patterns": [],
        }
        
        for event in events:
            patterns["severity_distribution"][event.severity.value] += 1
            patterns["reason_frequency"][event.reason] += event.count
            
            obj_kind = event.involved_object.get("kind", "Unknown")
            obj_name = event.involved_object.get("name", "Unknown")
            patterns["affected_resources"][obj_kind].add(obj_name)
        
        # 转换set为list以便序列化
        patterns["affected_resources"] = {
            k: list(v) for k, v in patterns["affected_resources"].items()
        }
        
        return patterns