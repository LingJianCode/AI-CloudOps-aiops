#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Health 规则
"""


from __future__ import annotations

from typing import Any, Dict, List

from app.core.inspection.rules.base import RuleContext


class PodPhaseRule:
    id = "health_pod_phase"
    name = "Pod 状态异常"
    category = "health"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for pod in ctx.pods:
            status = (pod.get("status") or {}).get("phase")
            meta = pod.get("metadata", {})
            pod_name = meta.get("name")
            ns = meta.get("namespace")
            if status in {"Pending", "Failed"}:
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": f"Pod 处于异常状态: {status}",
                        "severity": "high" if status == "Failed" else "medium",
                        "resource": {"type": "pod", "name": pod_name, "namespace": ns},
                        "description": f"Pod 当前状态为 {status}",
                        "evidence": [{"type": "status", "phase": status}],
                        "recommendations": ["检查调度与资源是否满足", "查看事件原因并修复"],
                    }
                )
        return findings


class PodRestartsRule:
    id = "health_pod_restarts"
    name = "Pod 重启次数过多"
    category = "health"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for pod in ctx.pods:
            meta = pod.get("metadata", {})
            pod_name = meta.get("name")
            ns = meta.get("namespace")
            cs = (pod.get("status") or {}).get("container_statuses") or []
            restart_count = 0
            for c in cs:
                restart_count += int((c or {}).get("restart_count") or 0)
            if restart_count >= 3:
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": "Pod 重启次数过多",
                        "severity": "medium",
                        "resource": {"type": "pod", "name": pod_name, "namespace": ns},
                        "description": f"容器累计重启 {restart_count} 次",
                        "evidence": [{"type": "status", "restart_count": restart_count}],
                        "recommendations": ["检查容器错误日志", "评估资源限制与探针配置"],
                    }
                )
        return findings


class ServiceNoEndpointsRule:
    id = "health_service_no_endpoints"
    name = "Service 无可用端点"
    category = "health"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        # 通过 Endpoints 资源判定是否有可用地址
        svc_has_ep = {}
        for ep in ctx.endpoints or []:
            meta = ep.get("metadata", {})
            subsets = ep.get("subsets") or []
            has_ready = False
            for s in subsets:
                addrs = s.get("addresses") or []
                if addrs:
                    has_ready = True
                    break
            svc_has_ep[(meta.get("namespace"), meta.get("name"))] = has_ready

        for svc in ctx.services or []:
            m = svc.get("metadata", {})
            key = (m.get("namespace"), m.get("name"))
            if not svc_has_ep.get(key, False):
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": "Service 无可用端点",
                        "severity": "high",
                        "resource": {"type": "service", "namespace": key[0], "name": key[1]},
                        "description": "Endpoints 不包含可用地址，可能导致服务不可达",
                        "evidence": [{"type": "endpoints", "ready": False}],
                        "recommendations": ["检查后端Pod就绪与选择器", "确认探针和副本数"],
                    }
                )
        return findings


class PVCNotBoundRule:
    id = "health_pvc_not_bound"
    name = "PVC 未绑定/待处理"
    category = "health"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for pvc in ctx.pvcs or []:
            phase = (pvc.get("status") or {}).get("phase")
            if phase not in {"Bound", None}:
                m = pvc.get("metadata", {})
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": f"PVC 状态异常: {phase}",
                        "severity": "medium",
                        "resource": {"type": "pvc", "namespace": m.get("namespace"), "name": m.get("name")},
                        "description": f"PVC 状态为 {phase}",
                        "evidence": [{"type": "pvc_status", "phase": phase}],
                        "recommendations": ["检查存储类与卷供给器", "确认集群存储可用性"],
                    }
                )
        return findings


