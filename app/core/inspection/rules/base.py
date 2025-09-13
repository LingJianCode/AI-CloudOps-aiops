#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Base 规则
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class RuleContext:
    pods: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    prom: Dict[str, Any]
    namespace: Optional[str]
    nodes: List[Dict[str, Any]] = None
    services: List[Dict[str, Any]] = None
    endpoints: List[Dict[str, Any]] = None
    resource_quotas: List[Dict[str, Any]] = None
    pvcs: List[Dict[str, Any]] = None


class Rule(Protocol):
    id: str
    name: str
    category: str  # health/performance/security/reliability

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        ...


