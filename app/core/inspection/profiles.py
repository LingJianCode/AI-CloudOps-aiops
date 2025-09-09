#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Profiles 规则集
"""


from __future__ import annotations

from typing import Dict, List

from app.core.inspection.rules.health_rules import (
    PodPhaseRule,
    PodRestartsRule,
    PVCNotBoundRule,
    ServiceNoEndpointsRule,
)
from app.core.inspection.rules.performance_rules import CpuThrottlingRule, HighErrorRateRule
from app.core.inspection.rules.reliability_rules import HighRiskEventRule, NodeNotReadyRule
from app.core.inspection.rules.security_rules import PrivilegedContainerRule

PROFILES: Dict[str, List[object]] = {
    "basic": [
        PodPhaseRule(),
        PodRestartsRule(),
        HighRiskEventRule(),
        CpuThrottlingRule(),
    ],
    "extended": [
        PodPhaseRule(),
        PodRestartsRule(),
        HighRiskEventRule(),
        CpuThrottlingRule(),
        HighErrorRateRule(),
        ServiceNoEndpointsRule(),
        PVCNotBoundRule(),
        NodeNotReadyRule(),
        PrivilegedContainerRule(),
    ],
}

def get_profile_rules(profile_name: str) -> List[object]:
    return PROFILES.get(profile_name, [])

def list_rules_flat() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for profile, rules in PROFILES.items():
        for r in rules:
            items.append({"id": r.id, "category": r.category, "profile": profile})
    return items


