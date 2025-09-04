#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
License: Apache 2.0
Description: Prometheus 客户端接口定义与空实现（Core层）
"""

from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

import pandas as pd


class PrometheusClient(Protocol):
    async def health_check(self) -> bool:
        ...

    async def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "1m",
    ) -> Optional[pd.DataFrame]:
        ...

    async def query_instant(
        self, query: str, timestamp: Optional[datetime] = None
    ) -> Optional[List[Dict]]:
        ...

    async def get_available_metrics(self) -> List[str]:
        ...

    async def get_metric_metadata(self, metric_name: str) -> Optional[Dict[str, Any]]:
        ...


class NullPrometheusClient:
    async def health_check(self) -> bool:
        return False

    async def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "1m",
    ) -> Optional[pd.DataFrame]:
        return None

    async def query_instant(
        self, query: str, timestamp: Optional[datetime] = None
    ) -> Optional[List[Dict]]:
        return None

    async def get_available_metrics(self) -> List[str]:
        return []

    async def get_metric_metadata(self, metric_name: str) -> Optional[Dict[str, Any]]:
        return None


