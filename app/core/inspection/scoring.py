#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Scoring 评分
"""


from __future__ import annotations


def severity_to_score(severity: str) -> float:
    mapping = {"high": 1.0, "medium": 0.7, "low": 0.4}
    return mapping.get((severity or "").lower(), 0.0)


