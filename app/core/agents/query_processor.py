#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能查询处理器 - 查询重写和扩展功能
"""

import re
from typing import List


class QueryRewriter:
    """智能查询重写器，提升检索召回率"""

    def __init__(self):
        self.synonyms = {
            "部署": ["安装", "配置", "搭建", "建立"],
            "监控": ["观察", "跟踪", "检测", "巡检"],
            "故障": ["异常", "错误", "问题", "失败"],
            "性能": ["效率", "速度", "响应", "吞吐"],
            "日志": ["记录", "日志文件", "log", "审计"],
            "告警": ["报警", "警告", "提醒", "通知"],
            "自动化": ["自动", "自动执行", "批量"],
            "运维": ["ops", "运营", "维护", "管理"],
        }

    def expand_query(self, query: str) -> List[str]:
        """扩展查询，生成多个相关查询变体"""
        expanded_queries = [query]  # 原始查询

        # 1. 同义词替换 - 增加更多变体
        for word, synonyms in self.synonyms.items():
            if word in query:
                for synonym in synonyms:
                    new_query = query.replace(word, synonym)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)

        # 2. 关键词提取和简化
        keywords = self._extract_keywords(query)
        if len(keywords) > 1:
            # 生成关键词组合查询
            expanded_queries.extend(keywords)

        # 3. 技术术语扩展
        tech_expansions = self._expand_tech_terms(query)
        expanded_queries.extend(tech_expansions)

        # 去重并返回前10个最相关的查询
        unique_queries = list(dict.fromkeys(expanded_queries))
        return unique_queries[:10]

    def _extract_keywords(self, query: str) -> List[str]:
        """提取查询中的关键词"""
        # 移除常见停用词
        stop_words = {"如何", "怎么", "什么", "为什么", "是", "的", "了", "在", "和"}
        
        # 分词（简单的基于空格和标点符号）
        words = re.findall(r'[\w\u4e00-\u9fff]+', query)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords

    def _expand_tech_terms(self, query: str) -> List[str]:
        """扩展技术术语"""
        tech_mappings = {
            "k8s": ["kubernetes", "容器编排"],
            "kubernetes": ["k8s", "容器编排", "pod", "deployment"],
            "docker": ["容器", "镜像"],
            "prometheus": ["监控", "指标收集"],
            "grafana": ["可视化", "仪表板"],
            "cpu": ["处理器", "计算资源"],
            "memory": ["内存", "RAM"],
            "disk": ["磁盘", "存储"],
        }
        
        expansions = []
        query_lower = query.lower()
        
        for term, related in tech_mappings.items():
            if term in query_lower:
                for related_term in related:
                    new_query = query + " " + related_term
                    expansions.append(new_query)
        
        return expansions