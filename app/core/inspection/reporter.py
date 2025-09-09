#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Reporter 报告
"""


from __future__ import annotations

from typing import Any, Dict, List


def report_to_markdown(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {})
    findings: List[Dict[str, Any]] = report.get("findings", [])
    stats = report.get("stats", {})
    recs: List[str] = report.get("recommendations", [])

    lines: List[str] = []
    lines.append(f"# 巡检报告 {report.get('report_id', '')}")
    lines.append("")
    lines.append("## 概览")
    lines.append(f"- 范围: {summary.get('scope')} {summary.get('namespace') or ''}")
    lines.append(
        f"- 检查数: {summary.get('total_checks', 0)}，问题: {summary.get('issues_found', 0)} (H/M/L: {summary.get('high', 0)}/{summary.get('medium', 0)}/{summary.get('low', 0)})"
    )
    lines.append("")
    lines.append("## 发现列表")
    if not findings:
        lines.append("无显著问题")
    else:
        for i, f in enumerate(findings, 1):
            lines.append(f"{i}. [{f.get('severity','low').upper()}] {f.get('title')}")
            res = f.get("resource", {})
            rn = ":".join(filter(None, [res.get("type"), res.get("namespace"), res.get("name")]))
            if rn:
                lines.append(f"   - 资源: {rn}")
            desc = f.get("description")
            if desc:
                lines.append(f"   - 描述: {desc}")
    lines.append("")
    lines.append("## 建议")
    if not recs:
        lines.append("- 暂无")
    else:
        for r in recs:
            lines.append(f"- {r}")
    lines.append("")
    lines.append("## 统计")
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


