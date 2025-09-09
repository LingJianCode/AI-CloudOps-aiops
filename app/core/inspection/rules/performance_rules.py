#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Performance 规则
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.inspection.rules.base import RuleContext


class CpuThrottlingRule:
    id = "perf_cpu_throttling"
    name = "CPU 限流提示"
    category = "performance"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        for s in (ctx.prom or {}).get("series", []):
            q = s.get("query", "")
            if "throttled_seconds_total" in q:
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": "可能存在 CPU 限流",
                        "severity": "low",
                        "resource": {"type": "namespace", "name": ctx.namespace},
                        "description": "查询到 CPU throttling 指标，建议核对 requests/limits",
                        "evidence": [{"type": "metric", "query": q}],
                        "recommendations": ["检查 requests/limits", "评估副本与调度"],
                    }
                )
        return findings


class HighErrorRateRule:
    id = "perf_http_error_rate"
    name = "HTTP 错误率偏高"
    category = "performance"

    def check(self, ctx: RuleContext) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        # 基于收集的查询进行轻度提示（无需强依赖Prom计算结果）
        for s in (ctx.prom or {}).get("series", []):
            q = s.get("query", "")
            if "http_requests_total" in q and "code=~\"5..\"" in q:
                findings.append(
                    {
                        "rule_id": self.id,
                        "title": "HTTP 5xx 错误率可能偏高",
                        "severity": "medium",
                        "resource": {"type": "namespace", "name": ctx.namespace},
                        "description": "检测到错误率相关查询，建议进一步确认服务SLO",
                        "evidence": [{"type": "metric", "query": q}],
                        "recommendations": ["检查服务错误日志", "核对依赖服务和配置"],
                    }
                )
        return findings


