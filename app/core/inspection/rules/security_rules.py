#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Security 规则
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.inspection.rules.base import RuleContext


class PrivilegedContainerRule:
    id = "security_privileged_container"
    name = "容器以特权运行"
    category = "security"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for pod in ctx.pods or []:
            meta = pod.get("metadata", {})
            spec = pod.get("spec", {})
            ns = meta.get("namespace")
            pod_name = meta.get("name")
            containers = (spec.get("containers") or []) + (spec.get("init_containers") or [])
            for c in containers:
                sc = (c.get("security_context") or {})
                if sc.get("privileged") is True:
                    findings.append(
                        {
                            "rule_id": self.id,
                            "title": f"容器以特权运行: {c.get('name')}",
                            "severity": "high",
                            "resource": {"type": "pod", "namespace": ns, "name": pod_name},
                            "description": "容器启用 privileged=true，存在高风险",
                            "evidence": [{"type": "security_context", "privileged": True}],
                            "recommendations": ["移除privileged或使用更小权限capabilities"],
                        }
                    )
        return findings


