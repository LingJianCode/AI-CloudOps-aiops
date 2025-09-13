#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Reliability 规则
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.inspection.rules.base import RuleContext


class HighRiskEventRule:
    id = "event_high_risk"
    name = "高危事件"
    category = "reliability"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for ev in (ctx.events or [])[:100]:
            reason = ev.get("reason") or ""
            msg = ev.get("message") or ""
            ns = (ev.get("involved_object") or {}).get("namespace") or ev.get("metadata", {}).get("namespace")
            name = (ev.get("involved_object") or {}).get("name")
            if any(k in reason for k in ["OOMKilling", "FailedScheduling", "CrashLoopBackOff"]):
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": f"高危事件: {reason}",
                        "severity": "high",
                        "resource": {"type": "object", "name": name, "namespace": ns},
                        "description": msg[:200],
                        "evidence": [{"type": "event", "reason": reason, "message": msg[:200]}],
                        "recommendations": ["根据事件原因执行修复操作", "必要时联动 AutoFix"],
                    }
                )
        return findings


class NodeNotReadyRule:
    id = "reliability_node_not_ready"
    name = "节点NotReady"
    category = "reliability"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for node in ctx.nodes or []:
            conds = (node.get("status") or {}).get("conditions") or []
            not_ready = False
            for c in conds:
                if c.get("type") == "Ready" and c.get("status") not in {"True", True}:
                    not_ready = True
                    reason = c.get("reason")
                    message = c.get("message")
                    break
            if not_ready:
                name = (node.get("metadata") or {}).get("name")
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": "节点未就绪",
                        "severity": "high",
                        "resource": {"type": "node", "name": name},
                        "description": f"节点 {name} 处于NotReady: {reason}",
                        "evidence": [{"type": "condition", "reason": reason, "message": message}],
                        "recommendations": ["检查节点健康与kubelet服务", "确认网络与磁盘状态"],
                    }
                )
        return findings


