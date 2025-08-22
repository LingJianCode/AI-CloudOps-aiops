#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI-CloudOp小助手
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

logger = logging.getLogger("aiops.rag_assistant")


class QueryType(Enum):
    """查询类型枚举"""

    FACTUAL = "factual"
    TROUBLESHOOTING = "troubleshooting"
    TUTORIAL = "tutorial"
    CONCEPTUAL = "conceptual"
    GENERAL = "general"


@dataclass
class RetrievalStrategy:
    """检索策略配置"""

    # 核心参数
    base_similarity: float = 0.5
    min_similarity: float = 0.3
    initial_k: int = 20
    final_k: int = 5

    # 缓存参数
    enable_cache: bool = True
    cache_ttl: int = 3600

    # 过滤参数
    diversity_threshold: float = 0.7


@dataclass
class EnhancedRAGState:
    """增强的RAG工作流状态"""

    question: str = ""
    session_id: Optional[str] = None

    # 查询分析结果
    query_type: Optional[QueryType] = None
    expanded_queries: List[str] = field(default_factory=list)
    query_intent: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0

    # 检索结果
    documents: List[Document] = field(default_factory=list)
    semantic_scores: List[float] = field(default_factory=list)
    contextual_scores: List[float] = field(default_factory=list)

    # 生成结果
    answer: str = ""
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)

    # 上下文信息
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    user_feedback: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "question": self.question,
            "session_id": self.session_id,
            "query_type": self.query_type,
            "expanded_queries": self.expanded_queries,
            "weight": self.weight,
            "documents": self.documents,
            "answer": self.answer,
            "confidence": self.confidence,
            "sources": self.sources,
            "conversation_history": self.conversation_history,
            "processing_metadata": self.processing_metadata,
        }


class QueryProcessor:
    """智能查询处理器"""

    def __init__(self, config=None):
        from app.config.settings import config as app_config

        self.config = config or app_config

        # 从配置文件读取查询模式和权重
        self.query_patterns = self._load_query_patterns()

        # 查询历史和意图分类
        self.query_history = deque(maxlen=100)
        self.intent_classifier = IntentClassifier()

    def _load_query_patterns(self) -> Dict[QueryType, Dict[str, Any]]:
        """从配置加载查询模式"""
        # 默认模式
        default_patterns = {
            QueryType.TROUBLESHOOTING: {
                "keywords": {
                    "错误",
                    "故障",
                    "失败",
                    "不能",
                    "error",
                    "fail",
                    "issue",
                    "problem",
                },
                "weight": 1.2,
            },
            QueryType.TUTORIAL: {
                "keywords": {
                    "如何",
                    "怎么",
                    "步骤",
                    "教程",
                    "how to",
                    "tutorial",
                    "guide",
                },
                "weight": 1.0,
            },
            QueryType.CONCEPTUAL: {
                "keywords": {
                    "什么是",
                    "概念",
                    "原理",
                    "what is",
                    "concept",
                    "principle",
                },
                "weight": 0.9,
            },
            QueryType.FACTUAL: {
                "keywords": {
                    "多少",
                    "几个",
                    "数量",
                    "统计",
                    "count",
                    "number",
                    "statistics",
                },
                "weight": 0.8,
            },
        }

        return default_patterns

    def analyze_and_expand(self, query: str, context: Optional[Dict] = None):
        """智能查询分析和扩展，支持上下文感知"""
        start_time = time.time()

        # 预处理查询
        processed_query = self._preprocess_query(query)

        # 多维度意图分析
        intent_scores = self.intent_classifier.classify_intent(processed_query, context)
        query_type = max(intent_scores.items(), key=lambda x: x[1])[0]
        confidence = intent_scores[query_type]

        # 计算动态权重
        weight = self._calculate_dynamic_weight(query_type, confidence, context)

        # 智能查询扩展
        expanded = self._generate_query_expansions(processed_query, query_type, context)

        # 记录查询历史用于学习
        self._record_query_for_learning(query, query_type, weight)

        # 移除性能统计（简化）

        return query_type, expanded, weight, intent_scores

    def _preprocess_query(self, query: str) -> str:
        """查询预处理"""
        import re

        query = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", query)
        query = " ".join(query.split())
        return query.strip()

    def _calculate_dynamic_weight(
        self, query_type: QueryType, confidence: float, context: Optional[Dict]
    ) -> float:
        """计算动态权重"""
        base_weight = self.query_patterns.get(query_type, {}).get("weight", 1.0)

        # 根据置信度调整
        confidence_factor = 1.0 + (confidence - 0.5) * 0.4

        # 根据上下文调整
        context_factor = 1.0
        if context and context.get("recent_failures"):
            if query_type == QueryType.TROUBLESHOOTING:
                context_factor = 1.2

        return base_weight * confidence_factor * context_factor

    def _generate_query_expansions(
        self, query: str, query_type: QueryType, context: Optional[Dict]
    ) -> List[str]:
        """生成智能查询扩展"""
        expansions = [query]

        # 基于查询类型的扩展
        if query_type == QueryType.TROUBLESHOOTING:
            expansions.extend(
                [f"{query} 解决方案", f"{query} 原因分析", f"{query} 故障排查"]
            )
        elif query_type == QueryType.TUTORIAL:
            expansions.extend(
                [f"{query} 详细步骤", f"{query} 操作指南", f"如何 {query}"]
            )
        elif query_type == QueryType.CONCEPTUAL:
            expansions.extend([f"{query} 概念解释", f"{query} 工作原理"])

        # 基于上下文的扩展
        if context and context.get("domain"):
            domain = context["domain"]
            expansions.append(f"{domain} {query}")

        # 限制扩展数量为5个
        return expansions[:5]

    def _record_query_for_learning(
        self, query: str, query_type: QueryType, weight: float
    ):
        """记录查询用于学习"""
        self.query_history.append(
            {
                "query": query,
                "type": query_type,
                "weight": weight,
                "timestamp": datetime.now(),
            }
        )


class IntentClassifier:
    """查询意图分类器"""

    def __init__(self):
        self.intent_patterns = {
            QueryType.TROUBLESHOOTING: [
                "错误",
                "故障",
                "失败",
                "问题",
                "error",
                "fail",
                "problem",
            ],
            QueryType.TUTORIAL: ["如何", "怎么", "步骤", "how", "tutorial", "guide"],
            QueryType.CONCEPTUAL: [
                "什么是",
                "概念",
                "原理",
                "what",
                "concept",
                "principle",
            ],
            QueryType.FACTUAL: [
                "多少",
                "数量",
                "统计",
                "count",
                "number",
                "statistics",
            ],
            QueryType.GENERAL: [],
        }

    def classify_intent(
        self, query: str, context: Optional[Dict] = None
    ) -> Dict[QueryType, float]:
        """分类查询意图"""
        query_lower = query.lower()
        scores = {}

        for intent, keywords in self.intent_patterns.items():
            score = 0.0

            # 关键词匹配得分
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1.0 / len(keywords) if keywords else 0

            # 上下文增强得分
            if context:
                score = self._enhance_with_context(score, intent, context)

            scores[intent] = min(score, 1.0)

        # 确保总分为1
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            scores[QueryType.GENERAL] = 1.0

        return scores

    def _enhance_with_context(
        self, base_score: float, intent: QueryType, context: Dict
    ) -> float:
        """基于上下文增强得分"""
        if context.get("recent_failures") and intent == QueryType.TROUBLESHOOTING:
            return base_score * 1.3
        if context.get("learning_mode") and intent == QueryType.TUTORIAL:
            return base_score * 1.2
        return base_score


class DocumentRetriever:
    """简化的文档检索器"""

    def __init__(self, vector_store, strategy: RetrievalStrategy, config=None):
        from app.config.settings import config as app_config

        self.vector_store = vector_store
        self.strategy = strategy
        self.config = config or app_config

        # 简单缓存系统 - 使用OrderedDict实现LRU
        self._cache = OrderedDict()
        self._max_cache_size = 100
        self._cache_stats = {"hits": 0, "misses": 0}

    async def retrieve(
        self, queries: List[str], weight: float = 1.0, context: Optional[Dict] = None
    ) -> List[Document]:
        """简化的文档检索"""
        start_time = time.time()

        # 缓存检查
        cache_key = self._generate_cache_key(queries, weight)
        if self.strategy.enable_cache and cache_key in self._cache:
            cached_time, cached_docs = self._cache[cache_key]
            if time.time() - cached_time < self.strategy.cache_ttl:
                logger.debug(f"缓存命中: {cache_key[:8]}")
                self._cache_stats["hits"] += 1
                self._cache.move_to_end(cache_key)
                return cached_docs

        self._cache_stats["misses"] += 1

        try:
            # 并行检索所有查询
            all_docs = await self._parallel_retrieval(queries, weight)

            # 多样性过滤
            final_docs = self._apply_diversity_filter(all_docs)

            # 更新缓存
            if self.strategy.enable_cache:
                self._update_cache(cache_key, final_docs)

            logger.debug(
                f"检索完成，耗时: {time.time() - start_time:.3f}s，文档数: {len(final_docs)}"
            )
            return final_docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    async def _parallel_retrieval(
        self, queries: List[str], weight: float
    ) -> List[Document]:
        """并行检索所有查询"""
        tasks = [self._retrieve_single(q, weight) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 去重和合并
        all_docs = []
        seen_hashes = set()

        for docs in results:
            if isinstance(docs, Exception):
                logger.warning(f"检索失败: {docs}")
                continue

            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    all_docs.append(doc)

        # 按分数排序并取前K个
        all_docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        return all_docs[: self.strategy.initial_k]

    def _apply_diversity_filter(self, documents: List[Document]) -> List[Document]:
        """第四阶段：多样性过滤"""
        if len(documents) <= self.strategy.final_k:
            return documents

        # 基于内容多样性选择文档
        selected = []
        for doc in documents:
            is_diverse = True
            for selected_doc in selected:
                similarity = self._calculate_document_similarity(doc, selected_doc)
                if similarity > self.strategy.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(doc)
                if len(selected) >= self.strategy.final_k:
                    break

        return selected

    def _calculate_document_similarity(self, doc1: Document, doc2: Document) -> float:
        """计算文档相似度（简化版）"""
        # 只比较前50个词，提高性能
        content1 = set(doc1.page_content.lower().split()[:50])
        content2 = set(doc2.page_content.lower().split()[:50])

        if not content1 or not content2:
            return 0.0

        intersection = len(content1 & content2)
        union = len(content1 | content2)
        return intersection / union if union > 0 else 0.0

    def _generate_cache_key(self, queries: List[str], weight: float) -> str:
        """生成缓存键"""
        cache_input = f"{'|'.join(queries)}:{weight}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _update_cache(self, cache_key: str, documents: List[Document]):
        """更新缓存"""
        self._cache[cache_key] = (time.time(), documents)

        # 缓存大小控制
        if len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)

    async def _retrieve_single(self, query: str, weight: float) -> List[Document]:
        """单查询检索"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []

            logger.info(f"执行向量检索 - 查询: {query[:50]}, 权重: {weight}")
            logger.info(
                f"检索配置 - initial_k: {self.strategy.initial_k}, min_similarity: {self.strategy.min_similarity}"
            )

            # 使用向量存储检索
            from app.core.vector.redis_vector_store import SearchConfig

            search_config = SearchConfig(
                similarity_threshold=self.strategy.min_similarity,
                use_cache=True,
                use_mmr=True,
            )

            results = await self.vector_store.similarity_search(
                query=query,
                k=self.strategy.initial_k,
                config=search_config,
            )

            logger.info(f"向量存储返回结果数量: {len(results) if results else 0}")

            # 将分数存储在元数据中
            docs = []
            for doc, score in results:
                doc.metadata = doc.metadata or {}
                doc.metadata["score"] = score
                docs.append(doc)
                logger.debug(
                    f"文档得分: {score:.3f}, 内容预览: {doc.page_content[:50]}"
                )

            return docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            import traceback

            logger.error(f"错误详情: {traceback.format_exc()}")
            return []


class RAGAssistant:
    """RAG助手主类"""

    def __init__(self, vector_store, llm_service, cache_manager=None, config=None):
        from app.config.settings import config as app_config

        self.config = config or app_config
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.cache_manager = cache_manager

        # 初始化新策略和组件
        self.strategy = self._init_strategy()
        self.query_processor = QueryProcessor(config=self.config)
        self.retriever = DocumentRetriever(
            vector_store, self.strategy, config=self.config
        )
        self.reranker = DocumentReranker(llm_service, config=self.config)
        self.generator = AnswerGenerator(llm_service, config=self.config)

        # 上下文管理器
        self.context_manager = ContextManager(config=self.config)

        # 构建工作流
        self.graph = self._build_graph()

        # 性能统计
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "successful_queries": 0,
            "failed_queries": 0,
        }

    def _init_strategy(self) -> RetrievalStrategy:
        """从配置初始化检索策略"""
        # 临时降低相似度阈值以确保能检索到文档
        min_similarity = getattr(self.config.rag, "min_similarity", 0.3)
        if min_similarity > 0.2:
            logger.info(f"降低相似度阈值从 {min_similarity} 到 0.1 以提高召回率")
            min_similarity = 0.1  # 临时降低阈值

        return RetrievalStrategy(
            base_similarity=getattr(self.config.rag, "base_similarity", 0.5),
            min_similarity=min_similarity,
            initial_k=getattr(self.config.rag, "initial_k", 20),
            final_k=getattr(self.config.rag, "final_k", self.config.rag.top_k),
            enable_cache=getattr(self.config.rag, "enable_cache", True),
            cache_ttl=getattr(
                self.config.rag, "cache_ttl", self.config.rag.cache_expiry
            ),
            diversity_threshold=getattr(self.config.rag, "diversity_threshold", 0.7),
        )

    def _build_graph(self) -> StateGraph:
        """构建优化的工作流"""
        workflow = StateGraph(EnhancedRAGState)

        # 简化的线性流程
        workflow.add_node("process", self._process)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)

        workflow.set_entry_point("process")
        workflow.add_edge("process", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile(checkpointer=MemorySaver())

    async def _process(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """处理查询"""
        query_type, expanded, weight, intent_scores = (
            self.query_processor.analyze_and_expand(state.question)
        )

        state.query_type = query_type
        state.expanded_queries = expanded
        state.weight = weight
        state.query_intent = intent_scores

        return state

    async def _retrieve(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """检索和重排序"""
        logger.info(f"开始检索阶段 - 问题: {state.question}")
        logger.info(f"扩展查询: {state.expanded_queries}")

        # 获取上下文
        context = self.context_manager.get_context(state.session_id)

        # 检索
        docs = await self.retriever.retrieve(
            state.expanded_queries, state.weight or 1.0, context
        )

        logger.info(f"检索到文档数量: {len(docs) if docs else 0}")
        if docs:
            logger.info(f"文档预览: {[doc.page_content[:100] for doc in docs[:2]]}")

        # 重排序
        if docs:
            docs = await self.reranker.rerank(
                state.question, docs, context=context, top_k=self.strategy.final_k
            )
            logger.info(f"重排序后文档数量: {len(docs)}")

        state.documents = docs

        return state

    async def _generate(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """生成答案"""
        logger.info(
            f"生成阶段 - 接收到文档数量: {len(state.documents) if state.documents else 0}"
        )

        # 获取上下文
        context = self.context_manager.get_context(state.session_id)

        result = await self.generator.generate(
            state.question,
            state.documents or [],
            state.query_type or QueryType.GENERAL,
            context,
        )

        state.answer = result["answer"]
        state.confidence = result["confidence"]

        logger.info(
            f"生成结果 - 答案长度: {len(result['answer'])}, 置信度: {result['confidence']}"
        )

        # 提取源信息
        sources = []
        for doc in (state.documents or [])[:3]:
            sources.append(
                {
                    "content": doc.page_content[:200] + "...",
                    "score": doc.metadata.get("score", 0),
                }
            )
        state.sources = sources

        return state

    async def get_answer(
        self, question: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取答案 - 主接口"""
        start_time = time.time()
        self._stats["total_queries"] += 1

        try:
            # 缓存检查
            if self.cache_manager:
                cache_key = hashlib.md5(question.encode()).hexdigest()
                cached = await asyncio.to_thread(self.cache_manager.get, cache_key)
                if cached:
                    self._stats["cache_hits"] += 1
                    cached["cache_hit"] = True
                    cached["processing_time"] = time.time() - start_time
                    return cached

            # 更新上下文
            if session_id:
                self.context_manager.update_context(
                    session_id, {"recent_queries": [question]}
                )

            # 运行工作流
            initial_state = EnhancedRAGState(question=question, session_id=session_id)
            config = {"thread_id": session_id or "default"}

            result = await self.graph.ainvoke(initial_state, config=config)

            # 构建响应
            if isinstance(result, dict):
                # 如果返回的是字典类型
                answer = result.get("answer", "")
                confidence = result.get("confidence", 0.0)
                sources = result.get("sources", [])
            else:
                # 如果返回的是EnhancedRAGState对象
                answer = getattr(result, "answer", "") or ""
                confidence = getattr(result, "confidence", 0.0) or 0.0
                sources = getattr(result, "sources", []) or []

            response = {
                "answer": answer,
                "confidence_score": confidence,
                "source_documents": sources,
                "cache_hit": False,
                "processing_time": time.time() - start_time,
                "success": True,
            }

            # 缓存高质量答案
            if self.cache_manager and response["confidence_score"] > 0.6:
                await asyncio.to_thread(
                    self.cache_manager.set,
                    cache_key,
                    response,
                    ttl=self.strategy.cache_ttl,
                )

            # 更新统计
            self._stats["successful_queries"] += 1

            return response

        except Exception as e:
            logger.error(f"处理失败: {e}")
            self._stats["failed_queries"] += 1
            return {
                "answer": f"处理出错: {str(e)}",
                "confidence_score": 0.0,
                "source_documents": [],
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "components": {
                "vector_store": bool(self.vector_store),
                "llm_service": bool(self.llm_service),
                "cache": bool(self.cache_manager),
            },
            "stats": self._stats,
            "timestamp": datetime.now().isoformat(),
        }

    async def clear_cache(self) -> Dict[str, Any]:
        """清理检索/重排/查询缓存"""
        try:
            cleared_items = 0
            # 检索缓存
            if hasattr(self.retriever, "_cache"):
                cleared_items += len(getattr(self.retriever, "_cache", {}))
                self.retriever._cache.clear()

            # 重排缓存
            if hasattr(self.reranker, "_rerank_cache"):
                cleared_items += len(getattr(self.reranker, "_rerank_cache", {}))
                self.reranker._rerank_cache.clear()

            # 向量存储查询缓存
            if getattr(self, "vector_store", None) and hasattr(
                self.vector_store, "query_cache"
            ):
                try:
                    # 尝试估算将要清理的键数量
                    # 不严格统计，尽可能减少开销
                    try:
                        cursor = 0
                        count = 0
                        prefix = getattr(
                            self.vector_store.query_cache, "prefix", "query_cache:"
                        )
                        while True:
                            cursor, keys = self.vector_store.client.scan(
                                cursor, match=prefix + "*", count=100
                            )
                            count += len(keys or [])
                            if cursor == 0:
                                break
                        cleared_items += count
                    except Exception:
                        pass

                    self.vector_store.query_cache.invalidate_pattern("*")
                except Exception:
                    pass

            return {
                "success": True,
                "message": "缓存已清理",
                "cleared_items": cleared_items,
            }
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return {"success": False, "message": str(e), "cleared_items": 0}

    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """刷新知识库（清理缓存，避免重建索引）"""
        result = await self.clear_cache()
        if not result.get("success"):
            return {"success": False, "message": result.get("message", "失败")}
        return {"success": True, "message": "知识库缓存已刷新"}

    async def create_session(
        self, session_id: str, info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建/初始化会话上下文"""
        self.context_manager.update_context(session_id, info or {})
        return {"success": True, "session_id": session_id}

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话上下文信息"""
        context = self.context_manager.get_context(session_id)
        return {"success": True, "session_id": session_id, "context": context}

    async def upload_knowledge(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """上传结构化知识（写入向量库）"""
        try:
            from langchain_core.documents import Document

            content = document_data.get("content", "")
            if not content:
                return {"success": False, "message": "内容为空"}

            metadata = {
                "title": document_data.get("title"),
                "source": document_data.get("source", "user_upload"),
                "category": document_data.get("category", "general"),
                "tags": document_data.get("tags", []),
                **document_data.get("metadata", {}),
            }

            docs = [Document(page_content=content, metadata=metadata)]
            ids = await self.vector_store.add_documents(docs)
            return {
                "success": True,
                "document_id": ids[0] if ids else None,
                "message": "知识上传成功",
            }
        except Exception as e:
            logger.error(f"上传知识失败: {e}")
            return {"success": False, "message": str(e)}

    async def upload_knowledge_file(
        self, document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """上传文件型知识（写入向量库）"""
        return await self.upload_knowledge(document_data)

    async def add_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加文档（API上传的轻量入口）"""
        return await self.upload_knowledge(document_data)


class ContextManager:
    """上下文管理器"""

    def __init__(self, config=None):
        from app.config.settings import config as app_config

        self.config = config or app_config
        self.session_contexts = {}
        self.global_context = {
            "system_status": "normal",
            "recent_patterns": deque(maxlen=100),
            "domain_preferences": defaultdict(float),
        }

    def get_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取会话上下文"""
        context = self.global_context.copy()

        if session_id and session_id in self.session_contexts:
            session_context = self.session_contexts[session_id]
            context.update(session_context)

        return context

    def update_context(self, session_id: Optional[str], updates: Dict[str, Any]):
        """更新上下文"""
        if session_id:
            if session_id not in self.session_contexts:
                self.session_contexts[session_id] = {
                    "created_at": datetime.now(),
                    "query_count": 0,
                    "recent_queries": deque(maxlen=10),
                    "user_preferences": [],
                    "domain": None,
                }

            # 处理recent_queries更新
            if "recent_queries" in updates:
                for query in updates["recent_queries"]:
                    self.session_contexts[session_id]["recent_queries"].append(query)
                updates.pop("recent_queries")

            self.session_contexts[session_id].update(updates)
            self.session_contexts[session_id]["query_count"] += 1


class DocumentReranker:
    """文档重排序器"""

    def __init__(self, llm_service=None, config=None):
        from app.config.settings import config as app_config

        self.llm_service = llm_service
        self.config = config or app_config
        self._rerank_cache = {}
        self.rerank_threshold = getattr(
            self.config.rag, "rerank_threshold", 0.1
        )  # 降低阈值

    async def rerank(
        self,
        query: str,
        docs: List[Document],
        context: Optional[Dict] = None,
        top_k: int = 5,
    ) -> List[Document]:
        """智能重排序"""
        if not docs:
            return []

        # 缓存检查
        cache_key = f"{query[:50]}_{len(docs)}_{top_k}"
        if cache_key in self._rerank_cache:
            cached_ids = self._rerank_cache[cache_key]
            id_to_doc = {id(doc): doc for doc in docs}
            return [id_to_doc[doc_id] for doc_id in cached_ids if doc_id in id_to_doc]

        # 多维度评分
        scored_docs = await self._multi_dimensional_scoring(query, docs, context)

        # 排序并去除过于相似的文档
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 多样性过滤
        final_docs = []
        for doc, score in scored_docs:
            if score > self.rerank_threshold and self._is_diverse(doc, final_docs):
                final_docs.append(doc)
                if len(final_docs) >= top_k:
                    break

        # 缓存结果
        self._rerank_cache[cache_key] = [id(doc) for doc in final_docs]
        if len(self._rerank_cache) > 50:
            self._rerank_cache.pop(next(iter(self._rerank_cache)))

        return final_docs

    async def _multi_dimensional_scoring(
        self, query: str, docs: List[Document], context: Optional[Dict]
    ):
        """简化文档评分"""
        scored = []
        query_terms = set(query.lower().split())

        for doc in docs:
            # 基础语义分数
            semantic_score = doc.metadata.get("score", 0.5)

            # 关键词匹配分数
            content_terms = set(
                doc.page_content.lower().split()[:100]
            )  # 只取前100个词提高性能
            keyword_score = len(query_terms & content_terms) / max(len(query_terms), 1)

            # 简化的综合分数
            final_score = semantic_score * 0.6 + keyword_score * 0.4

            scored.append((doc, final_score))

        return scored

    def _is_diverse(
        self, doc: Document, selected: List[Document], threshold: float = 0.7
    ) -> bool:
        """检查文档多样性"""
        if not selected:
            return True

        doc_words = set(doc.page_content.lower().split()[:50])

        for selected_doc in selected:
            selected_words = set(selected_doc.page_content.lower().split()[:50])
            if doc_words and selected_words:
                similarity = len(doc_words & selected_words) / len(
                    doc_words | selected_words
                )
                if similarity > threshold:
                    return False

        return True


class AnswerGenerator:
    """答案生成器"""

    def __init__(self, llm_service, config=None):
        from app.config.settings import config as app_config

        self.llm_service = llm_service
        self.config = config or app_config

        # 从配置加载模板
        self.base_prompt = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """从配置加载提示模板"""
        default_prompt = """基于以下文档回答问题，提供准确、简洁的答案。

文档内容:
{context}

问题: {question}

请根据上下文信息提供专业的回答。如果是故障排查，请包含解决步骤。
如果涉及操作步骤，请提供详细的指导。

答案:"""

        # 尝试从配置加载自定义提示模板
        return getattr(self.config.rag, "prompt_template", default_prompt)

    async def generate(
        self,
        question: str,
        docs: List[Document],
        query_type: QueryType,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """生成上下文感知的答案"""
        if not docs:
            logger.warning(f"未找到相关文档进行回答：{question}")
            # 提供更友好的回复，包含一些通用建议
            return {
                "answer": f"""针对您的问题「{question}」，我暂时没有找到完全匹配的资料。

**可能的解决方案：**

1. **尝试更具体的关键词**：使用更准确的技术术语
2. **分步提问**：将复杂问题拆分为多个简单问题
3. **检查文档**：查阅 AI-CloudOps 官方文档
4. **联系支持**：如需人工协助，请联系技术支持团队

**常见操作建议：**
- 检查系统状态和日志
- 确认配置参数设置
- 验证网络连接和权限

如果问题持续存在，建议提供更多上下文信息以便更好地帮助您。""",
                "confidence": 0.2,  # 提供基础建议，置信度较低
            }

        # 准备增强的上下文
        enhanced_context = self._prepare_enhanced_context(docs, context)

        # 根据查询类型和上下文调整生成参数
        generation_params = self._get_generation_params(query_type, context)

        try:
            prompt = self.base_prompt.format(
                context=enhanced_context, question=question
            )

            # 添加上下文提示
            if context and context.get("domain"):
                prompt = f"领域: {context['domain']}\n\n{prompt}"

            response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}], **generation_params
            )

            # 计算增强的置信度
            confidence = self._calculate_enhanced_confidence(
                docs, len(response), context
            )

            return {"answer": response, "confidence": confidence}

        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return {
                "answer": f"生成答案时出错: {str(e)}，请稍后重试。",
                "confidence": 0.0,
            }

    def _prepare_enhanced_context(
        self, docs: List[Document], context: Optional[Dict]
    ) -> str:
        """准备增强的上下文"""
        max_length = getattr(self.config.rag, "max_context_length", 4000)
        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs):
            content = doc.page_content

            # 添加文档元信息
            doc_info = f"[文档{i+1}"
            if doc.metadata.get("source"):
                doc_info += f" - 来源: {doc.metadata['source']}"
            if doc.metadata.get("score"):
                doc_info += f" - 相关性: {doc.metadata['score']:.2f}"
            doc_info += "]"

            # 智能截断
            available_length = max_length - current_length - len(doc_info) - 10
            if available_length <= 200:
                break

            if len(content) > available_length:
                content = content[:available_length] + "..."

            full_content = f"{doc_info} {content}"
            context_parts.append(full_content)
            current_length += len(full_content)

        # 添加上下文信息
        if context and context.get("recent_queries"):
            recent_queries = list(context["recent_queries"])[-3:]
            recent_info = f"\n\n[最近相关查询]: {', '.join(recent_queries)}"
            if current_length + len(recent_info) < max_length:
                context_parts.append(recent_info)

        return "\n\n".join(context_parts)

    def _get_generation_params(
        self, query_type: QueryType, context: Optional[Dict]
    ) -> Dict[str, Any]:
        """获取生成参数"""
        # 基础参数
        params = {
            "temperature": getattr(self.config.rag, "temperature", 0.1),
            "max_tokens": getattr(self.config.rag, "max_tokens", 1500),
        }

        # 根据查询类型调整
        if query_type == QueryType.FACTUAL:
            params["temperature"] = 0.1
        elif query_type == QueryType.TUTORIAL:
            params["temperature"] = 0.3
            params["max_tokens"] = 2000
        elif query_type == QueryType.TROUBLESHOOTING:
            params["temperature"] = 0.2
            params["max_tokens"] = 1800

        # 根据上下文调整
        if context and context.get("urgent"):
            params["max_tokens"] = min(params["max_tokens"], 1000)

        return params

    def _calculate_enhanced_confidence(
        self, docs: List[Document], answer_length: int, context: Optional[Dict]
    ) -> float:
        """计算增强的置信度"""
        if not docs or answer_length < 10:
            return 0.0

        # 基于文档数量和质量
        doc_confidence = min(len(docs) / 3, 1.0) * 0.3
        avg_score = sum(d.metadata.get("score", 0.5) for d in docs[:3]) / min(
            3, len(docs)
        )
        quality_confidence = avg_score * 0.4

        # 基于答案长度
        length_confidence = min(answer_length / 500, 1.0) * 0.15

        # 基于上下文匹配度
        context_confidence = 0.0
        if context:
            if context.get("domain") and any(
                context["domain"].lower() in doc.page_content.lower() for doc in docs
            ):
                context_confidence += 0.1
            if context.get("user_preferences"):
                matching_prefs = sum(
                    1
                    for pref in context["user_preferences"]
                    if any(pref.lower() in doc.page_content.lower() for doc in docs)
                )
                context_confidence += min(matching_prefs * 0.05, 0.15)

        return min(
            doc_confidence
            + quality_confidence
            + length_confidence
            + context_confidence,
            1.0,
        )


# 全局实例管理
_assistant_instance = None
_lock = asyncio.Lock()


async def get_enterprise_assistant() -> RAGAssistant:
    """获取全局助手实例"""
    global _assistant_instance

    if _assistant_instance is None:
        async with _lock:
            if _assistant_instance is None:
                try:
                    from app.config.settings import config
                    from app.core.cache.redis_cache_manager import RedisCacheManager
                    from app.services.llm import LLMService
                    from app.core.vector.redis_vector_store import (
                        EnhancedRedisVectorStore,
                    )

                    logger.info("初始化优化的RAG助手...")

                    # 创建服务
                    llm_service = LLMService()

                    # 缓存配置
                    cache_config = {
                        "host": config.redis.host,
                        "port": config.redis.port,
                        "db": config.redis.db + 1,
                        "password": config.redis.password,
                        "decode_responses": True,
                    }
                    cache_manager = RedisCacheManager(
                        redis_config=cache_config, cache_prefix="rag:", default_ttl=3600
                    )

                    # 向量存储配置
                    embedding_model = await _create_embedding_model()
                    vector_dim = await _get_embedding_dimension(embedding_model)

                    vector_config = {
                        "host": config.redis.host,
                        "port": config.redis.port,
                        "db": config.redis.db + 2,
                        "password": config.redis.password,
                        "decode_responses": False,
                    }

                    vector_store = EnhancedRedisVectorStore(
                        redis_config=vector_config,
                        collection_name="aiops_knowledge",
                        embedding_model=embedding_model,
                        vector_dim=vector_dim,
                        index_type="HNSW",
                    )

                    # 创建增强助手实例
                    _assistant_instance = RAGAssistant(
                        vector_store=vector_store,
                        llm_service=llm_service,
                        cache_manager=cache_manager,
                        config=config,
                    )

                    logger.info("优化的RAG助手初始化完成")

                except Exception as e:
                    logger.error(f"RAG助手初始化失败: {e}")
                    _assistant_instance = None
                    raise

    return _assistant_instance


async def _create_embedding_model():
    """创建嵌入模型"""
    try:
        from app.config.settings import config

        provider = config.llm.provider.lower()
        model_name = config.rag.effective_embedding_model

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=config.llm.effective_api_key,
                openai_api_base=config.llm.effective_base_url,
            )
        elif provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                model=model_name, base_url=config.llm.ollama_base_url.replace("/v1", "")
            )
        else:
            # 使用备用嵌入
            from app.core.agents.fallback_models import FallbackEmbeddings

            return FallbackEmbeddings()

    except Exception as e:
        logger.error(f"创建嵌入模型失败: {e}")
        from app.core.agents.fallback_models import FallbackEmbeddings

        return FallbackEmbeddings()


async def _get_embedding_dimension(embedding_model) -> int:
    """获取嵌入维度"""
    try:
        test_embedding = embedding_model.embed_query("test")
        return len(test_embedding)
    except:
        # 默认维度
        return 1024


def reset_assistant():
    """重置助手实例"""
    global _assistant_instance
    _assistant_instance = None
