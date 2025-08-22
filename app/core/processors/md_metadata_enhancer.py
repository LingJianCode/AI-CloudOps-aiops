#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: MD文档元数据增强器
"""

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any
from enum import Enum

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
                'primary': [
                    'kubernetes',
                    'k8s',
                    'kubectl',
                    'pod',
                    'deployment',
                    'service',
                    'namespace',
                    'configmap',
                    'secret',
                ],
                'secondary': [
                    'cluster',
                    'node',
                    'container',
                    'yaml',
                    'manifest',
                    'helm',
                    'operator',
                    'crd',
                ],
                'advanced': [
                    'admission-controller',
                    'mutating-webhook',
                    'validating-webhook',
                    'custom-resource',
                ],
            },
            TechnicalDomain.DOCKER: {
                'primary': ['docker', 'dockerfile', 'container', 'image', 'registry'],
                'secondary': [
                    'build',
                    'run',
                    'push',
                    'pull',
                    'tag',
                    'volume',
                    'network',
                ],
                'advanced': ['multi-stage', 'buildkit', 'layer-caching'],
            },
            TechnicalDomain.MONITORING: {
                'primary': ['prometheus', 'grafana', 'metric', 'alert', 'monitor'],
                'secondary': [
                    'query',
                    'dashboard',
                    'notification',
                    'threshold',
                    'scrape',
                ],
                'advanced': ['prometheus-operator', 'service-monitor', 'alertmanager'],
            },
            TechnicalDomain.NETWORKING: {
                'primary': [
                    'network',
                    'ip',
                    'port',
                    'protocol',
                    'tcp',
                    'udp',
                    'http',
                    'https',
                ],
                'secondary': ['firewall', 'proxy', 'load-balancer', 'ingress', 'dns'],
                'advanced': ['network-policy', 'service-mesh', 'cni', 'bgp'],
            },
            TechnicalDomain.SECURITY: {
                'primary': [
                    'security',
                    'auth',
                    'authorization',
                    'authentication',
                    'rbac',
                ],
                'secondary': ['certificate', 'tls', 'ssl', 'token', 'secret', 'key'],
                'advanced': ['oauth', 'oidc', 'jwt', 'pki', 'mtls'],
            },
        }

        # 命令模式
        self.command_patterns = [
            r'kubectl\s+[a-zA-Z\-]+(?:\s+[a-zA-Z0-9\-\.]+)*',
            r'docker\s+[a-zA-Z\-]+(?:\s+[a-zA-Z0-9\-\.]+)*',
            r'helm\s+[a-zA-Z\-]+(?:\s+[a-zA-Z0-9\-\.]+)*',
            r'curl\s+[^\n]+',
            r'wget\s+[^\n]+',
            r'ssh\s+[^\n]+',
            r'scp\s+[^\n]+',
        ]

        # 配置文件模式
        self.config_patterns = [
            r'[a-zA-Z0-9\-_]+\.ya?ml',
            r'[a-zA-Z0-9\-_]+\.json',
            r'[a-zA-Z0-9\-_]+\.conf',
            r'[a-zA-Z0-9\-_]+\.ini',
            r'[a-zA-Z0-9\-_]+\.properties',
            r'Dockerfile',
            r'docker-compose\.ya?ml',
        ]

        # API端点模式
        self.api_patterns = [
            r'https?://[a-zA-Z0-9\-\.]+(:[0-9]+)?(/[a-zA-Z0-9\-\._/]*)?',
            r'/api/v[0-9]+/[a-zA-Z0-9\-\._/]*',
            r'/[a-zA-Z0-9\-_]+/v[0-9]+/[a-zA-Z0-9\-\._/]*',
        ]

        # 错误模式
        self.error_patterns = [
            r'Error:?\s*[^\n]+',
            r'Exception:?\s*[^\n]+',
            r'Failed:?\s*[^\n]+',
            r'\d{3}\s+(Bad Request|Unauthorized|Forbidden|Not Found|Internal Server Error)',
            r'exit code\s+\d+',
        ]

    def extract_technical_concepts(self, content: str) -> List[TechnicalConcept]:
        """提取技术概念"""
        concepts = []
        content_lower = content.lower()

        for domain, keywords in self.domain_keywords.items():
            for level, terms in keywords.items():
                for term in terms:
                    count = content_lower.count(term.lower())
                    if count > 0:
                        contexts = self._extract_context(content, term, max_contexts=3)
                        concepts.append(
                            TechnicalConcept(
                                concept=term,
                                domain=domain,
                                frequency=count,
                                context=contexts,
                            )
                        )

        return sorted(concepts, key=lambda x: x.frequency, reverse=True)

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
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if term.lower() in line.lower():
                # 提取前后各一行作为上下文
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = ' '.join(lines[start:end]).strip()
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
            'technical_terms': 0.3,
            'code_complexity': 0.25,
            'structure_depth': 0.2,
            'concept_density': 0.15,
            'prerequisites': 0.1,
        }

    def analyze_content_complexity(
        self, content: str, metadata: Dict[str, Any]
    ) -> ContentComplexity:
        """分析内容复杂度"""
        scores = {}

        # 技术术语复杂度
        scores['technical_terms'] = self._analyze_technical_terms(content)

        # 代码复杂度
        scores['code_complexity'] = self._analyze_code_complexity(content, metadata)

        # 结构深度
        scores['structure_depth'] = self._analyze_structure_depth(metadata)

        # 概念密度
        scores['concept_density'] = self._analyze_concept_density(content)

        # 前置条件
        scores['prerequisites'] = self._analyze_prerequisites(content)

        # 计算加权分数
        total_score = sum(scores[key] * self.complexity_weights[key] for key in scores)

        # 根据分数确定复杂度
        if total_score < 0.3:
            return ContentComplexity.SIMPLE
        elif total_score < 0.6:
            return ContentComplexity.MODERATE
        elif total_score < 0.8:
            return ContentComplexity.COMPLEX
        else:
            return ContentComplexity.EXPERT

    def _analyze_technical_terms(self, content: str) -> float:
        """分析技术术语复杂度"""
        advanced_terms = [
            'kubernetes-operator',
            'custom-resource',
            'admission-controller',
            'service-mesh',
            'istio',
            'envoy',
            'prometheus-operator',
            'helm-chart',
            'operator-sdk',
            'crd',
            'rbac',
            'network-policy',
            'multi-cluster',
            'federation',
            'cross-cluster',
        ]

        content_lower = content.lower()
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        total_words = len(content.split())

        if total_words == 0:
            return 0.0

        return min(advanced_count / max(total_words / 100, 1), 1.0)

    def _analyze_code_complexity(self, content: str, metadata: Dict[str, Any]) -> float:
        """分析代码复杂度"""
        code_blocks = metadata.get('code_blocks_count', 0)
        languages = metadata.get('code_languages', [])

        if code_blocks == 0:
            return 0.0

        # 复杂编程语言权重
        complex_languages = {'go', 'rust', 'scala', 'haskell'}
        moderate_languages = {'python', 'java', 'javascript', 'typescript'}

        language_complexity = 0.0
        for lang in languages:
            if lang.lower() in complex_languages:
                language_complexity += 0.8
            elif lang.lower() in moderate_languages:
                language_complexity += 0.5
            else:
                language_complexity += 0.3

        # 归一化
        language_score = min(
            language_complexity / len(languages) if languages else 0, 1.0
        )
        block_score = min(code_blocks / 10, 1.0)

        return (language_score + block_score) / 2

    def _analyze_structure_depth(self, metadata: Dict[str, Any]) -> float:
        """分析结构深度"""
        title_structure = metadata.get('title_structure', {})

        if not title_structure:
            return 0.0

        max_depth = max(title_structure.keys()) if title_structure else 0
        total_titles = sum(len(titles) for titles in title_structure.values())

        # 深度权重
        depth_score = min(max_depth / 6, 1.0)  # 最大6级标题
        density_score = min(total_titles / 20, 1.0)  # 每20个标题为满分

        return (depth_score + density_score) / 2

    def _analyze_concept_density(self, content: str) -> float:
        """分析概念密度"""
        # 计算技术概念密度
        technical_indicators = [
            'deployment',
            'service',
            'configmap',
            'secret',
            'ingress',
            'persistentvolume',
            'statefulset',
            'daemonset',
            'job',
            'cronjob',
            'networkpolicy',
            'servicemonitor',
            'prometheus',
            'grafana',
        ]

        content_lower = content.lower()
        concept_count = sum(
            1 for indicator in technical_indicators if indicator in content_lower
        )
        total_words = len(content.split())

        if total_words == 0:
            return 0.0

        return min(concept_count / max(total_words / 50, 1), 1.0)

    def _analyze_prerequisites(self, content: str) -> float:
        """分析前置条件复杂度"""
        prerequisite_indicators = [
            '前置条件',
            '先决条件',
            'prerequisite',
            'requirement',
            '需要安装',
            '必须',
            '依赖',
            'dependency',
            '假设',
            'assume',
            '确保',
            'ensure',
        ]

        content_lower = content.lower()
        prereq_count = sum(
            1 for indicator in prerequisite_indicators if indicator in content_lower
        )

        return min(prereq_count / 5, 1.0)


class QualityScorer:
    """质量评分器"""

    def calculate_readability_score(self, content: str) -> float:
        """计算可读性分数"""
        if not content:
            return 0.0

        # 基础指标
        sentences = content.count('.') + content.count('!') + content.count('?')
        words = len(content.split())

        if sentences == 0 or words == 0:
            return 0.3

        # 平均句长
        avg_sentence_length = words / sentences

        # 代码块和列表提升可读性
        code_blocks = content.count('```')
        lists = content.count('- ') + content.count('* ') + content.count('1. ')

        # 计算分数
        length_score = 1.0 - min(
            abs(avg_sentence_length - 15) / 15, 0.5
        )  # 理想句长15词
        structure_score = min((code_blocks + lists) / 10, 1.0)

        return (length_score + structure_score) / 2

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
            'title_structure',
            'code_languages',
            'key_commands',
            'configuration_files',
        ]

        present_indicators = sum(
            1 for indicator in structure_indicators if metadata.get(indicator)
        )
        structure_score = present_indicators / len(structure_indicators)

        # 检查内容完整性
        content_indicators = [
            ('code_blocks_count', lambda x: x > 0),
            ('table_count', lambda x: x > 0),
            ('list_count', lambda x: x > 0),
            ('word_count', lambda x: x > 50),
        ]

        present_content = sum(
            1
            for indicator, check in content_indicators
            if check(metadata.get(indicator, 0))
        )
        content_score = present_content / len(content_indicators)

        return (structure_score + content_score) / 2


class MDMetadataEnhancer:
    """MD元数据增强器"""

    def __init__(self):
        self.term_extractor = TechnicalTermExtractor()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.quality_scorer = QualityScorer()

        logger.info("MD元数据增强器初始化完成")

    def enhance_metadata(
        self, content: str, basic_metadata: Dict[str, Any]
    ) -> EnhancedMetadata:
        """增强元数据"""
        enhanced = EnhancedMetadata()

        # 基础统计
        enhanced.word_count = len(content.split())
        enhanced.char_count = len(content)
        enhanced.line_count = len(content.split('\n'))

        # 结构信息（从基础元数据获取）
        enhanced.code_languages = basic_metadata.get('languages', [])
        enhanced.code_blocks_count = content.count('```') // 2
        enhanced.table_count = basic_metadata.get('table_count', 0)
        enhanced.list_count = (
            content.count('- ') + content.count('* ') + content.count('1. ')
        )
        enhanced.link_count = content.count('](')
        enhanced.image_count = content.count('![')

        # 标题结构
        enhanced.title_structure = self._extract_title_structure(content)

        # 技术概念提取
        enhanced.technical_concepts = self.term_extractor.extract_technical_concepts(
            content
        )

        # 关键信息提取
        enhanced.key_commands = self.term_extractor.extract_key_commands(content)
        enhanced.configuration_files = self.term_extractor.extract_configuration_files(
            content
        )
        enhanced.api_endpoints = self.term_extractor.extract_api_endpoints(content)
        enhanced.error_patterns = self.term_extractor.extract_error_patterns(content)

        # 语义标签生成
        enhanced.semantic_tags = self._generate_semantic_tags(
            content, enhanced.technical_concepts
        )

        # 技术领域识别
        enhanced.technical_domains = self._identify_technical_domains(
            enhanced.technical_concepts
        )

        # 内容模式识别
        enhanced.content_patterns = self._identify_content_patterns(content, enhanced)

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
                'title_structure': enhanced.title_structure,
                'code_languages': enhanced.code_languages,
                'key_commands': enhanced.key_commands,
                'configuration_files': enhanced.configuration_files,
                'code_blocks_count': enhanced.code_blocks_count,
                'table_count': enhanced.table_count,
                'list_count': enhanced.list_count,
                'word_count': enhanced.word_count,
            }
        )

        logger.debug(
            f"元数据增强完成 - 复杂度: {enhanced.content_complexity.value}, "
            f"技术深度: {enhanced.technical_depth:.3f}, "
            f"完整性: {enhanced.completeness_score:.3f}"
        )

        return enhanced

    def _extract_title_structure(self, content: str) -> Dict[int, List[str]]:
        """提取标题结构"""
        structure = defaultdict(list)

        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break

                title = line[level:].strip()
                if title:
                    structure[level].append(title)

        return dict(structure)

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
            content.count('1. ') + content.count('2. ') + content.count('3. ')
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
            'error',
            'problem',
            'issue',
            'failed',
            'debug',
            '故障',
            '问题',
            '错误',
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
