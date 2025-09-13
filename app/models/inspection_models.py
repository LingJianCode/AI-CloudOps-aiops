#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Inspection 数据模型
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InspectionRunRequest(BaseModel):
    scope: str = Field(default="namespace", description="巡检范围: cluster/namespace/workload/pod")
    namespace: Optional[str] = Field(default=None, description="命名空间")
    profiles: List[str] = Field(default_factory=lambda: ["basic"], description="规则集")
    time_window_minutes: int = Field(default=60, description="时间窗口(分钟)")
    include_events: bool = Field(default=True, description="是否包含事件")
    include_logs: bool = Field(default=False, description="是否包含日志片段")
    severity_threshold: float = Field(default=0.5, description="严重度阈值(0~1)")
    async_task: bool = Field(default=False, alias="async", description="是否异步执行")


class InspectionFinding(BaseModel):
    rule_id: str
    title: str
    severity: str
    resource: Dict[str, Any]
    description: str
    evidence: List[Dict[str, Any]] = []
    recommendations: List[str] = []


class InspectionSummary(BaseModel):
    scope: str
    namespace: Optional[str] = None
    time_window_minutes: int
    total_checks: int
    issues_found: int
    high: int
    medium: int
    low: int


class InspectionReport(BaseModel):
    report_id: str
    summary: InspectionSummary
    findings: List[InspectionFinding]
    stats: Dict[str, Any] = {}
    recommendations: List[str] = []
    timestamp: str


class InspectionTaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    report_id: Optional[str] = None
    error: Optional[str] = None


