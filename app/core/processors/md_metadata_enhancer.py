#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: MD文档元数据增强器
"""

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aiops.md_metadata_enhancer")


class ContentComplexity(Enum):
    """内容复杂度"""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TechnicalDomain(Enum):
    """技术领域"""

    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    MONITORING = "monitoring"
    NETWORKING = "networking"
    SECURITY = "security"
    DATABASE = "database"
    DEVELOPMENT = "development"
    DEVOPS = "devops"
    CLOUD = "cloud"
    GENERAL = "general"


@dataclass
class SemanticTag:
    """语义标签"""

    tag: str
    confidence: float
    category: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class TechnicalConcept:
    """技术概念"""

    concept: str
    domain: TechnicalDomain
    frequency: int
    context: List[str] = field(default_factory=list)


@dataclass
class ContentPattern:
    """内容模式"""

    pattern_type: str
    pattern_value: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedMetadata:
    """增强的元数据"""

    # 基础信息
    word_count: int = 0
    char_count: int = 0
    line_count: int = 0

    # 结构信息
    title_structure: Dict[int, List[str]] = field(default_factory=dict)
    code_languages: List[str] = field(default_factory=list)
    code_blocks_count: int = 0
    table_count: int = 0
    list_count: int = 0
    link_count: int = 0
    image_count: int = 0

    # 语义信息
    semantic_tags: List[SemanticTag] = field(default_factory=list)
    technical_concepts: List[TechnicalConcept] = field(default_factory=list)
    content_complexity: ContentComplexity = ContentComplexity.SIMPLE
    technical_domains: List[TechnicalDomain] = field(default_factory=list)

    # 内容模式
    content_patterns: List[ContentPattern] = field(default_factory=list)

    # 质量指标
    readability_score: float = 0.0
    technical_depth: float = 0.0
    completeness_score: float = 0.0

    # 关键信息
    key_commands: List[str] = field(default_factory=list)
    configuration_files: List[str] = field(default_factory=list)
    api_endpoints: List[str] = field(default_factory=list)
    error_patterns: List[str] = field(default_factory=list)


class TechnicalTermExtractor:
    """技术术语提取器"""

    def __init__(self):
        # 技术领域词典
        self.domain_keywords = {
            TechnicalDomain.KUBERNETES: {
                "primary": [
                    "kubernetes",
                    "k8s",
                    "kubectl",
                    "pod",
                    "deployment",
                    "service",
                    "namespace",
                    "configmap",
                    "secret",
                ],
                "secondary": [
                    "cluster",
                    "node",
                    "container",
                    "yaml",
                    "manifest",
                    "helm",
                    "operator",
                    "crd",
                ],
                "advanced": [
                    "admission-controller",
                    "mutating-webhook",
                    "validating-webhook",
                    "custom-resource",
                ],
            },
            TechnicalDomain.DOCKER: {
                "primary": ["docker", "dockerfile", "container", "image", "registry"],
                "secondary": [
                    "build",
                    "run",
                    "push",
                    "pull",
                    "tag",
                    "volume",
                    "network",
                ],
                "advanced": ["multi-stage", "buildkit", "layer-caching"],
            },
            TechnicalDomain.MONITORING: {
                "primary": ["prometheus", "metric", "alert", "monitor"],
                "secondary": [
                    "query",
                    "dashboard",
                    "notification",
                    "threshold",
                    "scrape",
                ],
                "advanced": ["prometheus-operator", "service-monitor", "alertmanager"],
            },
            TechnicalDomain.NETWORKING: {
                "primary": [
                    "network",
                    "ip",
                    "port",
                    "protocol",
                    "tcp",
                    "udp",
                    "http",
                    "https",
                ],
                "secondary": ["firewall", "proxy", "load-balancer", "ingress", "dns"],
                "advanced": ["network-policy", "service-mesh", "cni", "bgp"],
            },
            TechnicalDomain.SECURITY: {
                "primary": [
                    "security",
                    "auth",
                    "authorization",
                    "authentication",
                    "rbac",
                ],
                "secondary": ["certificate", "tls", "ssl", "token", "secret", "key"],
                "advanced": ["oauth", "oidc", "jwt", "pki", "mtls"],
            },
        }

        # 命令模式
        self.command_patterns = [
            r"kubectl\s+[a-zA-Z\-]+(?:\s+[a-zA-Z0-9\-\.]+)*",
            r"docker\s+[a-zA-Z\-]+(?:\s+[a-zA-Z0-9\-\.]+)*",
            r"helm\s+[a-zA-Z\-]+(?:\s+[a-zA-Z0-9\-\.]+)*",
            r"curl\s+[^\n]+",
            r"wget\s+[^\n]+",
            r"ssh\s+[^\n]+",
            r"scp\s+[^\n]+",
        ]

        # 配置文件模式
        self.config_patterns = [
            r"[a-zA-Z0-9\-_]+\.ya?ml",
            r"[a-zA-Z0-9\-_]+\.json",
            r"[a-zA-Z0-9\-_]+\.conf",
            r"[a-zA-Z0-9\-_]+\.ini",
            r"[a-zA-Z0-9\-_]+\.properties",
            r"Dockerfile",
            r"docker-compose\.ya?ml",
        ]

        # API端点模式
        self.api_patterns = [
            r"https?://[a-zA-Z0-9\-\.]+(:[0-9]+)?(/[a-zA-Z0-9\-\._/]*)?",
            r"/api/v[0-9]+/[a-zA-Z0-9\-\._/]*",
            r"/[a-zA-Z0-9\-_]+/v[0-9]+/[a-zA-Z0-9\-\._/]*",
        ]

        # 错误模式
        self.error_patterns = [
            r"Error:?\s*[^\n]+",
            r"Exception:?\s*[^\n]+",
            r"Failed:?\s*[^\n]+",
            r"\d{3}\s+(Bad Request|Unauthorized|Forbidden|Not Found|Internal Server Error)",
            r"exit code\s+\d+",
        ]

    def extract_technical_concepts(self, content: str) -> List[TechnicalConcept]:
        """提取技术概念（优化版）"""
        concepts = []
        content_lower = content.lower()

        # 预编译正则表达式，提高性能
        compiled_patterns = {}

        for domain, keywords in self.domain_keywords.items():
            for level, terms in keywords.items():
                for term in terms:
                    if term not in compiled_patterns:
                        # 使用词边界匹配，避免误匹配
                        pattern = rf"\b{re.escape(term.lower())}\b"
                        compiled_patterns[term] = re.compile(pattern)

                    matches = compiled_patterns[term].findall(content_lower)
                    count = len(matches)

                    if count > 0:
                        # 限制上下文提取，提高性能
                        contexts = self._extract_context(content, term, max_contexts=2)
                        concepts.append(
                            TechnicalConcept(
                                concept=term,
                                domain=domain,
                                frequency=count,
                                context=contexts[:2],  # 限制上下文数量
                            )
                        )

        # 限制返回的概念数量，避免过多数据
        return sorted(concepts, key=lambda x: x.frequency, reverse=True)[:20]

    def extract_key_commands(self, content: str) -> List[str]:
        """提取关键命令"""
        commands = []
        for pattern in self.command_patterns:
            matches = re.findall(pattern, content)
            commands.extend(matches)
        return list(set(commands))

    def extract_configuration_files(self, content: str) -> List[str]:
        """提取配置文件"""
        configs = []
        for pattern in self.config_patterns:
            matches = re.findall(pattern, content)
            configs.extend(matches)
        return list(set(configs))

    def extract_api_endpoints(self, content: str) -> List[str]:
        """提取API端点"""
        endpoints = []
        for pattern in self.api_patterns:
            matches = re.findall(pattern, content)
            endpoints.extend(matches)
        return list(set(endpoints))

    def extract_error_patterns(self, content: str) -> List[str]:
        """提取错误模式"""
        errors = []
        for pattern in self.error_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            errors.extend(matches)
        return list(set(errors))

    def _extract_context(
        self, content: str, term: str, max_contexts: int = 3
    ) -> List[str]:
        """提取术语上下文"""
        contexts = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if term.lower() in line.lower():
                # 提取前后各一行作为上下文
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = " ".join(lines[start:end]).strip()
                if context and len(context) > 10:  # 过滤太短的上下文
                    contexts.append(context[:200])  # 限制长度
                    if len(contexts) >= max_contexts:
                        break

        return contexts


class ComplexityAnalyzer:
    """复杂度分析器"""

    def __init__(self):
        # 复杂度指标权重
        self.complexity_weights = {
            "technical_terms": 0.3,
            "code_complexity": 0.25,
            "structure_depth": 0.2,
            "concept_density": 0.15,
            "prerequisites": 0.1,
        }

    def analyze_content_complexity(
        self, content: str, metadata: Dict[str, Any]
    ) -> ContentComplexity:
        """分析内容复杂度（简化版）"""
        # 简化计算，只计算最重要的指标
        content_lower = content.lower()
        word_count = len(content.split())

        if word_count == 0:
            return ContentComplexity.SIMPLE

        # 快速技术术语检测
        advanced_terms = [
            "operator",
            "custom-resource",
            "admission-controller",
            "service-mesh",
            "crd",
        ]
        tech_score = sum(1 for term in advanced_terms if term in content_lower) / max(
            word_count / 100, 1
        )

        # 代码复杂度
        code_blocks = metadata.get("code_blocks_count", 0)
        code_score = min(code_blocks / 5, 1.0)

        # 综合评分
        total_score = tech_score * 0.6 + code_score * 0.4

        # 简化阈值判断
        if total_score < 0.2:
            return ContentComplexity.SIMPLE
        elif total_score < 0.5:
            return ContentComplexity.MODERATE
        elif total_score < 0.8:
            return ContentComplexity.COMPLEX
        else:
            return ContentComplexity.EXPERT


class QualityScorer:
    """质量评分器"""

    def calculate_readability_score(self, content: str) -> float:
        """计算可读性分数（简化版）"""
        if not content:
            return 0.0

        words = len(content.split())
        lines = len(content.split("\n"))

        if words == 0:
            return 0.3

        # 简化计算：基于结构化元素
        structure_elements = (
            content.count("```")  # 代码块
            + content.count("- ")  # 列表
            + content.count("# ")  # 标题
        )

        # 归一化分数
        structure_score = min(structure_elements / max(lines / 10, 1), 1.0)
        length_score = min(words / 500, 1.0)  # 500词为理想长度

        return structure_score * 0.7 + length_score * 0.3

    def calculate_technical_depth(
        self, technical_concepts: List[TechnicalConcept]
    ) -> float:
        """计算技术深度"""
        if not technical_concepts:
            return 0.0

        # 基于概念数量和频率
        total_frequency = sum(concept.frequency for concept in technical_concepts)
        unique_concepts = len(technical_concepts)

        frequency_score = min(total_frequency / 20, 1.0)
        diversity_score = min(unique_concepts / 10, 1.0)

        return (frequency_score + diversity_score) / 2

    def calculate_completeness_score(self, metadata: Dict[str, Any]) -> float:
        """计算完整性分数"""
        # 检查结构完整性
        structure_indicators = [
            "title_structure",
            "code_languages",
            "key_commands",
            "configuration_files",
        ]

        present_indicators = sum(
            1 for indicator in structure_indicators if metadata.get(indicator)
        )
        structure_score = present_indicators / len(structure_indicators)

        # 检查内容完整性
        content_indicators = [
            ("code_blocks_count", lambda x: x > 0),
            ("table_count", lambda x: x > 0),
            ("list_count", lambda x: x > 0),
            ("word_count", lambda x: x > 50),
        ]

        present_content = sum(
            1
            for indicator, check in content_indicators
            if check(metadata.get(indicator, 0))
        )
        content_score = present_content / len(content_indicators)

        return (structure_score + content_score) / 2


class MDMetadataEnhancer:
    """MD元数据增强器（优化版）"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 性能配置
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_size = self.config.get("cache_size", 256)
        self.enable_batch_processing = self.config.get("enable_batch_processing", True)
        self.max_batch_size = self.config.get("max_batch_size", 10)

        # 组件初始化（延迟加载）
        self._term_extractor = None
        self._complexity_analyzer = None
        self._quality_scorer = None

        # 缓存管理
        self._metadata_cache = {} if self.enable_caching else None
        self._pattern_cache = {}

        # 统计信息
        self.stats = {
            "enhancements_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "processing_time_total": 0.0,
            "average_processing_time": 0.0,
        }

        # 预编译常用正则表达式
        self._compile_cached_patterns()

        logger.info("MD元数据增强器初始化完成")

    @property
    def term_extractor(self):
        """延迟初始化术语提取器"""
        if self._term_extractor is None:
            self._term_extractor = TechnicalTermExtractor()
        return self._term_extractor

    @property
    def complexity_analyzer(self):
        """延迟初始化复杂度分析器"""
        if self._complexity_analyzer is None:
            self._complexity_analyzer = ComplexityAnalyzer()
        return self._complexity_analyzer

    @property
    def quality_scorer(self):
        """延迟初始化质量评分器"""
        if self._quality_scorer is None:
            self._quality_scorer = QualityScorer()
        return self._quality_scorer

    def _compile_cached_patterns(self):
        """预编译缓存的正则表达式"""
        self.cached_patterns = {
            "code_blocks": re.compile(r"```", re.MULTILINE),
            "headers": re.compile(r"^#{1,6}\s+", re.MULTILINE),
            "lists": re.compile(r"^(\s*)([-*+]|\d+\.)\s+", re.MULTILINE),
            "links": re.compile(r"\]\("),
            "images": re.compile(r"!\["),
        }

    def enhance_metadata(
        self, content: str, basic_metadata: Dict[str, Any]
    ) -> EnhancedMetadata:
        """增强元数据（带缓存和性能优化）"""
        start_time = time.time()

        # 检查缓存
        if self.enable_caching and self._metadata_cache is not None:
            cache_key = self._get_cache_key(content, basic_metadata)
            if cache_key in self._metadata_cache:
                self.stats["cache_hits"] += 1
                logger.debug(f"使用缓存的元数据增强结果: {cache_key[:16]}...")
                return self._metadata_cache[cache_key]
            else:
                self.stats["cache_misses"] += 1

        enhanced = EnhancedMetadata()

        # 基础统计（使用缓存的模式匹配）
        enhanced.word_count = len(content.split())
        enhanced.char_count = len(content)
        enhanced.line_count = len(content.split("\n"))

        # 结构信息（优化的计数方式）
        enhanced.code_languages = basic_metadata.get("languages", [])
        enhanced.code_blocks_count = (
            len(self.cached_patterns["code_blocks"].findall(content)) // 2
        )
        enhanced.table_count = basic_metadata.get("table_count", 0)
        enhanced.list_count = len(self.cached_patterns["lists"].findall(content))
        enhanced.link_count = len(self.cached_patterns["links"].findall(content))
        enhanced.image_count = len(self.cached_patterns["images"].findall(content))

        # 标题结构（简化版）
        enhanced.title_structure = self._extract_title_structure_fast(content)

        # 有选择地进行重型处理
        if enhanced.word_count > 100:  # 只对有意义的文档进行详细分析
            # 技术概念提取
            enhanced.technical_concepts = (
                self.term_extractor.extract_technical_concepts(content)
            )

            # 关键信息提取（限制处理量）
            enhanced.key_commands = self.term_extractor.extract_key_commands(content)[
                :10
            ]
            enhanced.configuration_files = (
                self.term_extractor.extract_configuration_files(content)[:10]
            )
            enhanced.api_endpoints = self.term_extractor.extract_api_endpoints(content)[
                :10
            ]
            enhanced.error_patterns = self.term_extractor.extract_error_patterns(
                content
            )[:10]

            # 语义标签生成
            enhanced.semantic_tags = self._generate_semantic_tags_fast(
                content, enhanced.technical_concepts
            )

            # 技术领域识别
            enhanced.technical_domains = self._identify_technical_domains(
                enhanced.technical_concepts
            )

            # 内容模式识别
            enhanced.content_patterns = self._identify_content_patterns_fast(
                content, enhanced
            )
        else:
            # 简化处理
            enhanced.technical_concepts = []
            enhanced.key_commands = []
            enhanced.configuration_files = []
            enhanced.api_endpoints = []
            enhanced.error_patterns = []
            enhanced.semantic_tags = []
            enhanced.technical_domains = []
            enhanced.content_patterns = []

        # 复杂度分析
        enhanced.content_complexity = (
            self.complexity_analyzer.analyze_content_complexity(content, basic_metadata)
        )

        # 质量评分
        enhanced.readability_score = self.quality_scorer.calculate_readability_score(
            content
        )
        enhanced.technical_depth = self.quality_scorer.calculate_technical_depth(
            enhanced.technical_concepts
        )
        enhanced.completeness_score = self.quality_scorer.calculate_completeness_score(
            {
                **basic_metadata,
                "title_structure": enhanced.title_structure,
                "code_languages": enhanced.code_languages,
                "key_commands": enhanced.key_commands,
                "configuration_files": enhanced.configuration_files,
                "code_blocks_count": enhanced.code_blocks_count,
                "table_count": enhanced.table_count,
                "list_count": enhanced.list_count,
                "word_count": enhanced.word_count,
            }
        )

        # 更新统计信息
        processing_time = time.time() - start_time
        self.stats["enhancements_processed"] += 1
        self.stats["processing_time_total"] += processing_time
        self.stats["average_processing_time"] = (
            self.stats["processing_time_total"] / self.stats["enhancements_processed"]
        )

        # 缓存结果
        if self.enable_caching and self._metadata_cache is not None:
            cache_key = self._get_cache_key(content, basic_metadata)

            # 限制缓存大小
            if len(self._metadata_cache) >= self.cache_size:
                # 简单的LRU：移除最老的一半条目
                keys_to_remove = list(self._metadata_cache.keys())[
                    : self.cache_size // 2
                ]
                for key in keys_to_remove:
                    del self._metadata_cache[key]

            self._metadata_cache[cache_key] = enhanced

        logger.debug(
            f"元数据增强完成 - 复杂度: {enhanced.content_complexity.value}, "
            f"技术深度: {enhanced.technical_depth:.3f}, "
            f"完整性: {enhanced.completeness_score:.3f}, "
            f"耗时: {processing_time:.3f}秒"
        )

        return enhanced

    def _extract_title_structure(self, content: str) -> Dict[int, List[str]]:
        """提取标题结构"""
        structure = defaultdict(list)

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                level = 0
                for char in line:
                    if char == "#":
                        level += 1
                    else:
                        break

                title = line[level:].strip()
                if title:
                    structure[level].append(title)

        return dict(structure)

    def _extract_title_structure_fast(self, content: str) -> Dict[int, List[str]]:
        """快速提取标题结构"""
        structure = defaultdict(list)

        # 使用预编译的正则表达式
        for match in self.cached_patterns["headers"].finditer(content):
            line = content[match.start() : content.find("\n", match.start())]
            level = len(line) - len(line.lstrip("#"))
            if level > 0:
                title = line[level:].strip()
                if title:
                    structure[level].append(title)

                    # 限制每个级别的标题数量
                    if len(structure[level]) > 20:
                        break

        return dict(structure)

    def _generate_semantic_tags_fast(
        self, content: str, technical_concepts: List[TechnicalConcept]
    ) -> List[SemanticTag]:
        """快速生成语义标签"""
        tags = []

        # 基于技术概念生成标签（限制数量）
        concept_counter = Counter()
        for concept in technical_concepts[:10]:  # 只处理前10个概念
            concept_counter[concept.domain] += concept.frequency

        for domain, frequency in concept_counter.most_common(3):  # 只取前3个
            confidence = min(frequency / 5, 1.0)  # 简化计算
            tags.append(
                SemanticTag(
                    tag=domain.value,
                    confidence=confidence,
                    category="technical_domain",
                    evidence=[
                        concept.concept
                        for concept in technical_concepts[:5]  # 限制证据数量
                        if concept.domain == domain
                    ][
                        :2
                    ],  # 最多2个证据
                )
            )

        return tags

    def _identify_content_patterns_fast(
        self, content: str, enhanced: EnhancedMetadata
    ) -> List[ContentPattern]:
        """快速识别内容模式"""
        patterns = []

        # 代码密集模式
        if enhanced.code_blocks_count > 2:
            patterns.append(
                ContentPattern(
                    pattern_type="code_intensive",
                    pattern_value=f"{enhanced.code_blocks_count}_blocks",
                    confidence=min(enhanced.code_blocks_count / 8, 1.0),
                    metadata={"languages": enhanced.code_languages},
                )
            )

        # 故障排查模式（简化检查）
        troubleshoot_keywords = ["error", "problem", "issue", "failed"]
        troubleshoot_count = sum(
            content.lower().count(keyword) for keyword in troubleshoot_keywords
        )
        if troubleshoot_count > 2:
            patterns.append(
                ContentPattern(
                    pattern_type="troubleshooting",
                    pattern_value=f"{troubleshoot_count}_indicators",
                    confidence=min(troubleshoot_count / 8, 1.0),
                )
            )

        return patterns

    def _get_cache_key(self, content: str, basic_metadata: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib

        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        metadata_str = str(sorted(basic_metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()[:8]
        return f"{content_hash}_{metadata_hash}"

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if self._metadata_cache:
            stats["cache_size"] = len(self._metadata_cache)
            stats["max_cache_size"] = self.cache_size
        return stats

    def clear_cache(self):
        """清理缓存"""
        if self._metadata_cache:
            self._metadata_cache.clear()
        self._pattern_cache.clear()
        logger.info("MD元数据增强器缓存已清理")

    def _generate_semantic_tags(
        self, content: str, technical_concepts: List[TechnicalConcept]
    ) -> List[SemanticTag]:
        """生成语义标签"""
        tags = []

        # 基于技术概念生成标签
        concept_counter = Counter()
        for concept in technical_concepts:
            concept_counter[concept.domain] += concept.frequency

        for domain, frequency in concept_counter.most_common(5):
            confidence = min(frequency / 10, 1.0)
            tags.append(
                SemanticTag(
                    tag=domain.value,
                    confidence=confidence,
                    category="technical_domain",
                    evidence=[
                        concept.concept
                        for concept in technical_concepts
                        if concept.domain == domain
                    ][:3],
                )
            )

        # 基于内容模式生成标签
        content_lower = content.lower()

        pattern_tags = [
            (
                "tutorial",
                ["如何", "步骤", "教程", "guide", "tutorial", "step"],
                "content_type",
            ),
            (
                "troubleshooting",
                ["故障", "问题", "错误", "debug", "troubleshoot", "error"],
                "content_type",
            ),
            (
                "configuration",
                ["配置", "设置", "config", "setting", "parameter"],
                "content_type",
            ),
            (
                "installation",
                ["安装", "部署", "install", "deploy", "setup"],
                "content_type",
            ),
            (
                "monitoring",
                ["监控", "告警", "metric", "monitor", "alert"],
                "content_type",
            ),
        ]

        for tag_name, keywords, category in pattern_tags:
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches > 0:
                confidence = min(matches / len(keywords), 1.0)
                tags.append(
                    SemanticTag(
                        tag=tag_name,
                        confidence=confidence,
                        category=category,
                        evidence=keywords[:3],
                    )
                )

        return sorted(tags, key=lambda x: x.confidence, reverse=True)

    def _identify_technical_domains(
        self, technical_concepts: List[TechnicalConcept]
    ) -> List[TechnicalDomain]:
        """识别技术领域"""
        domain_scores = defaultdict(float)

        for concept in technical_concepts:
            domain_scores[concept.domain] += concept.frequency

        # 选择得分最高的前3个领域
        top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]
        return [domain for domain, score in top_domains if score >= 2]  # 至少出现2次

    def _identify_content_patterns(
        self, content: str, enhanced: EnhancedMetadata
    ) -> List[ContentPattern]:
        """识别内容模式"""
        patterns = []

        # 代码密集模式
        if enhanced.code_blocks_count > 3:
            patterns.append(
                ContentPattern(
                    pattern_type="code_intensive",
                    pattern_value=f"{enhanced.code_blocks_count}_blocks",
                    confidence=min(enhanced.code_blocks_count / 10, 1.0),
                    metadata={"languages": enhanced.code_languages},
                )
            )

        # 步骤导向模式
        numbered_lists = (
            content.count("1. ") + content.count("2. ") + content.count("3. ")
        )
        if numbered_lists > 5:
            patterns.append(
                ContentPattern(
                    pattern_type="step_oriented",
                    pattern_value=f"{numbered_lists}_steps",
                    confidence=min(numbered_lists / 15, 1.0),
                )
            )

        # 故障排查模式
        troubleshoot_keywords = [
            "error",
            "problem",
            "issue",
            "failed",
            "debug",
            "故障",
            "问题",
            "错误",
        ]
        troubleshoot_count = sum(
            content.lower().count(keyword) for keyword in troubleshoot_keywords
        )
        if troubleshoot_count > 3:
            patterns.append(
                ContentPattern(
                    pattern_type="troubleshooting",
                    pattern_value=f"{troubleshoot_count}_indicators",
                    confidence=min(troubleshoot_count / 10, 1.0),
                )
            )

        # 配置文档模式
        if len(enhanced.configuration_files) > 2 or enhanced.code_blocks_count > 1:
            patterns.append(
                ContentPattern(
                    pattern_type="configuration_guide",
                    pattern_value=f"{len(enhanced.configuration_files)}_configs",
                    confidence=min(
                        (len(enhanced.configuration_files) + enhanced.code_blocks_count)
                        / 8,
                        1.0,
                    ),
                    metadata={"config_files": enhanced.configuration_files},
                )
            )

        return patterns

    def to_dict(self, enhanced: EnhancedMetadata) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            # 基础统计
            "word_count": enhanced.word_count,
            "char_count": enhanced.char_count,
            "line_count": enhanced.line_count,
            # 结构信息
            "title_structure": enhanced.title_structure,
            "code_languages": enhanced.code_languages,
            "code_blocks_count": enhanced.code_blocks_count,
            "table_count": enhanced.table_count,
            "list_count": enhanced.list_count,
            "link_count": enhanced.link_count,
            "image_count": enhanced.image_count,
            # 语义信息
            "semantic_tags": [
                {
                    "tag": tag.tag,
                    "confidence": tag.confidence,
                    "category": tag.category,
                    "evidence": tag.evidence,
                }
                for tag in enhanced.semantic_tags
            ],
            "technical_concepts": [
                {
                    "concept": concept.concept,
                    "domain": concept.domain.value,
                    "frequency": concept.frequency,
                    "context": concept.context,
                }
                for concept in enhanced.technical_concepts
            ],
            "content_complexity": enhanced.content_complexity.value,
            "technical_domains": [
                domain.value for domain in enhanced.technical_domains
            ],
            # 内容模式
            "content_patterns": [
                {
                    "pattern_type": pattern.pattern_type,
                    "pattern_value": pattern.pattern_value,
                    "confidence": pattern.confidence,
                    "metadata": pattern.metadata,
                }
                for pattern in enhanced.content_patterns
            ],
            # 质量指标
            "readability_score": enhanced.readability_score,
            "technical_depth": enhanced.technical_depth,
            "completeness_score": enhanced.completeness_score,
            # 关键信息
            "key_commands": enhanced.key_commands,
            "configuration_files": enhanced.configuration_files,
            "api_endpoints": enhanced.api_endpoints,
            "error_patterns": enhanced.error_patterns,
        }
