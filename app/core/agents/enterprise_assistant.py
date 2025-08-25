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
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

logger = logging.getLogger("aiops.rag_assistant")


class QueryType(Enum):

    FACTUAL = "factual"
    TROUBLESHOOTING = "troubleshooting"
    TUTORIAL = "tutorial"
    CONCEPTUAL = "conceptual"
    GENERAL = "general"


@dataclass
class RetrievalStrategy:
    base_similarity: float = 0.7
    min_similarity: float = 0.15  # 降低最小相似度阈值，提高召回率
    initial_k: int = 30  # 增加初始检索数量
    final_k: int = 8     # 增加最终返回数量
    enable_cache: bool = True
    cache_ttl: int = 3600
    diversity_threshold: float = 0.6  # 降低多样性阈值，允许更多相关文档


@dataclass
class EnhancedRAGState:

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

    def __init__(self, config=None):
        from app.config.settings import config as app_config

        self.config = config or app_config
        self.query_patterns = self._load_query_patterns()
        query_history_limit = getattr(self.config, "query_history_limit", 100)
        self.query_history = deque(maxlen=query_history_limit)
        self.intent_classifier = IntentClassifier()

    def _load_query_patterns(self) -> Dict[QueryType, Dict[str, Any]]:
        """加载查询模式"""
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
                    "什么是", "概念", "原理",
                    "what is", "concept", "principle",
                },
                "weight": 0.9,
            },
            QueryType.FACTUAL: {
                "keywords": {
                    "多少", "几个", "数量", "统计",
                    "count", "number", "statistics",
                },
                "weight": 0.8,
            },
        }

        return default_patterns

    def analyze_and_expand(self, query: str, context: Optional[Dict] = None):
        """查询分析和扩展"""
        processed_query = self._preprocess_query(query)
        intent_scores = self.intent_classifier.classify_intent(processed_query, context)
        query_type = max(intent_scores.items(), key=lambda x: x[1])[0]
        confidence = intent_scores[query_type]
        weight = self._calculate_dynamic_weight(query_type, confidence, context)
        expanded = self._generate_query_expansions(processed_query, query_type, context)
        self._record_query_for_learning(query, query_type, weight)
        return query_type, expanded, weight, intent_scores

    def _preprocess_query(self, query: str) -> str:
        import re
        query = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", query)
        return " ".join(query.split()).strip()

    def _calculate_dynamic_weight(
        self, query_type: QueryType, confidence: float, context: Optional[Dict]
    ) -> float:
        """计算动态权重"""
        base_weight = self.query_patterns.get(query_type, {}).get("weight", 1.0)

        confidence_factor = 1.0 + (confidence - 0.5) * 0.4
        context_factor = 1.0
        if context and context.get("recent_failures") and query_type == QueryType.TROUBLESHOOTING:
            context_factor = 1.2

        return base_weight * confidence_factor * context_factor

    def _generate_query_expansions(
        self, query: str, query_type: QueryType, context: Optional[Dict]
    ) -> List[str]:
        """生成查询扩展"""
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

        max_expansions = getattr(self.config, "max_query_expansions", 5)
        return expansions[:max_expansions]

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
            if keywords:  # 检查keywords不为空
                for keyword in keywords:
                    if keyword in query_lower:
                        score += 1.0 / len(keywords)

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

    def __init__(self, vector_store, strategy: RetrievalStrategy, config=None):
        from app.config.settings import config as app_config

        self.vector_store = vector_store
        self.strategy = strategy
        self.config = config or app_config

        self._cache = OrderedDict()
        max_cache_size = getattr(self.config, "retrieval_cache_size", 100)
        self._max_cache_size = max_cache_size
        self._cache_stats = {"hits": 0, "misses": 0}

    async def retrieve(
        self, queries: List[str], weight: float = 1.0, context: Optional[Dict] = None
    ) -> List[Document]:
        """文档检索"""
        start_time = time.time()

        # 简化的缓存检查
        cache_key = None
        if self.strategy.enable_cache and len(queries) == 1:  # 只为单查询缓存
            cache_key = hashlib.md5(f"{queries[0]}:{weight}".encode()).hexdigest()
            if cache_key in self._cache:
                cached_time, cached_docs = self._cache[cache_key]
                if time.time() - cached_time < self.strategy.cache_ttl:
                    logger.debug(f"缓存命中")
                    self._cache_stats["hits"] += 1
                    return cached_docs

        self._cache_stats["misses"] += 1

        try:
            all_docs = await self._parallel_retrieval(queries, weight)
            final_docs = self._apply_diversity_filter(all_docs)

            # 更新缓存（简化版）
            if self.strategy.enable_cache and cache_key:
                self._cache[cache_key] = (time.time(), final_docs)
                # 简单的缓存大小控制
                if len(self._cache) > self._max_cache_size:
                    # 删除最旧的条目
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            logger.debug(
                f"检索完成，耗时: {time.time() - start_time:.3f}s，文档数: {len(final_docs)}"
            )
            return final_docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            # 提供fallback机制
            return (
                []
                if not hasattr(self, "_fallback_docs")
                else getattr(self, "_fallback_docs", [])
            )

    async def _parallel_retrieval(
        self, queries: List[str], weight: float
    ) -> List[Document]:
        """并行检索所有查询"""
        tasks = [self._retrieve_single(q, weight) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_docs = []
        seen_hashes = set()

        for docs in results:
            if isinstance(docs, Exception):
                logger.warning(f"检索失败: {docs}")
                continue

            for doc in docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    all_docs.append(doc)

        all_docs.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)
        return all_docs[: self.strategy.initial_k]

    def _apply_diversity_filter(self, documents: List[Document]) -> List[Document]:
        """多样性过滤"""
        if len(documents) <= self.strategy.final_k:
            return documents

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
        word_limit = getattr(self.config, "similarity_word_limit", 50)
        content1 = set(doc1.page_content.lower().split()[:word_limit])
        content2 = set(doc2.page_content.lower().split()[:word_limit])

        if not content1 or not content2:
            return 0.0

        intersection = len(content1 & content2)
        union = len(content1 | content2)
        return intersection / union if union > 0 else 0.0

    async def _retrieve_single(self, query: str, weight: float) -> List[Document]:
        """单查询检索"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []

            query_preview_length = 50
            logger.info(f"向量检索: {query[:query_preview_length]}, 权重: {weight}")
            logger.info(f"配置: k={self.strategy.initial_k}, sim={self.strategy.min_similarity}")

            from app.core.vector.redis_vector_store import SearchConfig

            search_config = SearchConfig(
                similarity_threshold=self.strategy.min_similarity,
                use_cache=True,
                use_mmr=True,
                use_hierarchical_retrieval=True,
                hierarchical_threshold=50,
                auto_switch_retrieval=True,
                prefer_structured_content=True,
                boost_code_blocks=True,
                boost_titles=True,
            )

            results = await self.vector_store.similarity_search(
                query=query,
                k=self.strategy.initial_k,
                config=search_config,
            )

            logger.info(f"返回结果: {len(results) if results else 0}")

            docs = []
            for doc, score in results:
                doc.metadata = doc.metadata or {}
                doc.metadata["score"] = score
                docs.append(doc)
                logger.debug(f"文档得分: {score:.3f}, 内容: {doc.page_content[:50]}")

            return docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            logger.debug(f"错误详情: {str(e)}")
            return []


class RAGAssistant:

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
        min_similarity = getattr(self.config.rag, "min_similarity", 0.15)  # 降低默认阈值
        similarity_threshold = getattr(self.config.rag, "similarity_threshold", 0.2)
        if min_similarity > similarity_threshold:
            logger.info(f"调整相似度阈值: {min_similarity} -> 0.1")
            min_similarity = 0.1

        return RetrievalStrategy(
            base_similarity=getattr(self.config.rag, "base_similarity", 0.7),  # 提高基础相似度要求
            min_similarity=min_similarity,
            initial_k=getattr(self.config.rag, "initial_k", 30),  # 增加初始检索数量
            final_k=getattr(self.config.rag, "final_k", min(8, self.config.rag.top_k * 2)),  # 动态调整最终数量
            enable_cache=getattr(self.config.rag, "enable_cache", True),
            cache_ttl=getattr(
                self.config.rag, "cache_ttl", self.config.rag.cache_expiry
            ),
            diversity_threshold=getattr(self.config.rag, "diversity_threshold", 0.6),  # 降低多样性阈值
        )

    def _build_graph(self) -> StateGraph:
        """构建工作流"""
        workflow = StateGraph(EnhancedRAGState)

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
        logger.info(f"检索阶段 - 问题: {state.question}")
        logger.info(f"扩展查询: {state.expanded_queries}")

        context = self.context_manager.get_context(state.session_id)
        docs = await self.retriever.retrieve(
            state.expanded_queries, state.weight or 1.0, context
        )

        logger.info(f"检索文档: {len(docs) if docs else 0}")
        if docs:
            logger.info(f"文档预览: {[doc.page_content[:100] for doc in docs[:2]]}")
            docs = await self.reranker.rerank(
                state.question, docs, context=context, top_k=self.strategy.final_k
            )
            logger.info(f"重排序后: {len(docs)}")

        state.documents = docs
        return state

    async def _generate(self, state: EnhancedRAGState) -> EnhancedRAGState:
        logger.info(f"生成阶段 - 文档: {len(state.documents) if state.documents else 0}")

        context = self.context_manager.get_context(state.session_id)
        result = await self.generator.generate(
            state.question,
            state.documents or [],
            state.query_type or QueryType.GENERAL,
            context,
        )

        state.answer = result["answer"]
        state.confidence = result["confidence"]
        logger.info(f"结果 - 长度: {len(result['answer'])}, 置信度: {result['confidence']}")

        sources = []
        max_source_docs = getattr(self.config, "max_source_docs", 3)
        max_source_content = getattr(self.config, "max_source_content", 200)
        for doc in (state.documents or [])[:max_source_docs]:
            sources.append({
                "content": doc.page_content[:max_source_content] + "...",
                "score": doc.metadata.get("score", 0),
            })
        state.sources = sources
        return state

    async def get_answer(
        self, question: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取答案"""
        start_time = time.time()
        self._stats["total_queries"] += 1

        try:
            if self.cache_manager:
                cache_key = hashlib.md5(question.encode()).hexdigest()
                cached = await asyncio.to_thread(self.cache_manager.get, cache_key)
                if cached:
                    self._stats["cache_hits"] += 1
                    cached["cache_hit"] = True
                    cached["processing_time"] = time.time() - start_time
                    return cached

            if session_id:
                self.context_manager.update_context(
                    session_id, {"recent_queries": [question]}
                )

            initial_state = EnhancedRAGState(question=question, session_id=session_id)
            config = {"thread_id": session_id or "default"}

            result = await self.graph.ainvoke(initial_state, config=config)

            if isinstance(result, dict):
                answer = result.get("answer", "")
                confidence = result.get("confidence", 0.0)
                sources = result.get("sources", [])
            else:
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

            cache_threshold = getattr(self.config, "cache_confidence_threshold", 0.6)
            if self.cache_manager and response["confidence_score"] > cache_threshold:
                await asyncio.to_thread(
                    self.cache_manager.set,
                    cache_key,
                    response,
                    ttl=self.strategy.cache_ttl,
                )

            if session_id:
                self.context_manager.update_context(
                    session_id, {"assistant_response": answer}
                )

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
        try:
            cleared_items = 0
            if hasattr(self.retriever, "_cache"):
                cleared_items += len(getattr(self.retriever, "_cache", {}))
                self.retriever._cache.clear()

            if hasattr(self.reranker, "_rerank_cache"):
                cleared_items += len(getattr(self.reranker, "_rerank_cache", {}))
                self.reranker._rerank_cache.clear()

            if getattr(self, "vector_store", None) and hasattr(self.vector_store, "query_cache"):
                try:
                    prefix = getattr(self.vector_store.query_cache, "prefix", "query_cache:")
                    cursor, count = 0, 0
                    while True:
                        cursor, keys = self.vector_store.client.scan(
                            cursor, match=prefix + "*", count=100
                        )
                        count += len(keys or [])
                        if cursor == 0:
                            break
                    cleared_items += count
                    await self.vector_store.query_cache.invalidate_pattern("*")
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
        result = await self.clear_cache()
        if not result.get("success"):
            return {"success": False, "message": result.get("message", "失败")}
        return {"success": True, "message": "知识库缓存已刷新"}

    async def create_session(
        self, session_id: str, info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        self.context_manager.update_context(session_id, info or {})
        return {"success": True, "session_id": session_id}

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        context = self.context_manager.get_context(session_id)
        session_details = self.context_manager.get_session_details(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "created_time": session_details.get("created_time", ""),
            "last_activity": session_details.get("last_activity", ""),
            "message_count": session_details.get("message_count", 0),
            "mode": session_details.get("mode", 1),
            "status": session_details.get("status", "active"),
            "conversation_history": session_details.get("conversation_history", []),
            "context": context
        }

    async def upload_knowledge(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
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

            # 检测是否是MD文档并使用专门的处理方法
            if self._is_markdown_content(content):
                logger.info("检测到MD文档，使用专门处理流程")
                ids = await self.vector_store.add_md_documents([content], [metadata])
                return {
                    "success": True,
                    "document_ids": ids,
                    "chunks_count": len(ids),
                    "message": f"MD知识上传成功，生成 {len(ids)} 个结构化块",
                    "document_type": "markdown",
                }
            else:
                # 普通文档处理
                from langchain_core.documents import Document

                docs = [Document(page_content=content, metadata=metadata)]
                ids = await self.vector_store.add_documents(docs)
                return {
                    "success": True,
                    "document_id": ids[0] if ids else None,
                    "message": "知识上传成功",
                    "document_type": "plain",
                }
        except Exception as e:
            logger.error(f"上传知识失败: {e}")
            return {"success": False, "message": str(e)}

    def _is_markdown_content(self, content: str) -> bool:
        """检测内容是否是Markdown格式"""
        md_indicators = [
            "# ",
            "## ",
            "### ",  # 标题
            "```",  # 代码块
            "- ",
            "* ",
            "1. ",  # 列表
            "| ",  # 表格
            "> ",  # 引用
            "[",
            "](",  # 链接
        ]

        content_lower = content.lower()
        indicator_count = sum(
            1 for indicator in md_indicators if indicator in content_lower
        )

        # 如果包含2个以上MD标记，认为是MD文档
        return indicator_count >= 2

    async def upload_knowledge_file(
        self, document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self.upload_knowledge(document_data)

    async def add_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.upload_knowledge(document_data)


class ContextManager:

    def __init__(self, config=None):
        from app.config.settings import config as app_config

        self.config = config or app_config
        self.session_contexts = {}
        max_recent_patterns = 100
        self.global_context = {
            "system_status": "normal",
            "recent_patterns": deque(maxlen=max_recent_patterns),
            "domain_preferences": defaultdict(float),
        }

    def get_context(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        context = self.global_context.copy()
        if session_id and session_id in self.session_contexts:
            context.update(self.session_contexts[session_id])
        return context

    def update_context(self, session_id: Optional[str], updates: Dict[str, Any]):
        if not session_id:
            return
            
        if session_id not in self.session_contexts:
            current_time = datetime.now()
            self.session_contexts[session_id] = {
                "created_at": current_time,
                "last_activity": current_time,
                "query_count": 0,
                "message_count": 0,
                "recent_queries": deque(maxlen=10),
                "conversation_history": [],
                "user_preferences": [],
                "domain": None,
                "mode": updates.get("mode", 1),
                "status": "active",
            }

        self.session_contexts[session_id]["last_activity"] = datetime.now()

        if "recent_queries" in updates:
            for query in updates["recent_queries"]:
                self.session_contexts[session_id]["recent_queries"].append(query)
                self.session_contexts[session_id]["query_count"] += 1
                self.session_contexts[session_id]["message_count"] += 1
                
                conversation_entry = {
                    "type": "user",
                    "content": query,
                    "timestamp": datetime.now().isoformat()
                }
                self.session_contexts[session_id]["conversation_history"].append(conversation_entry)
                
                if len(self.session_contexts[session_id]["conversation_history"]) > 50:
                    self.session_contexts[session_id]["conversation_history"] = \
                        self.session_contexts[session_id]["conversation_history"][-50:]
            
            updates.pop("recent_queries")
        
        if "assistant_response" in updates:
            response = updates["assistant_response"]
            conversation_entry = {
                "type": "assistant", 
                "content": response,
                "timestamp": datetime.now().isoformat()
            }
            self.session_contexts[session_id]["conversation_history"].append(conversation_entry)
            self.session_contexts[session_id]["message_count"] += 1
            updates.pop("assistant_response")

        for key, value in updates.items():
            if key not in ["recent_queries", "assistant_response"]:
                self.session_contexts[session_id][key] = value

    def get_session_details(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.session_contexts:
            return {
                "created_time": "",
                "last_activity": "",
                "message_count": 0,
                "mode": 1,
                "status": "unknown",
                "conversation_history": []
            }
        
        session = self.session_contexts[session_id]
        return {
            "created_time": session.get("created_at", datetime.now()).isoformat(),
            "last_activity": session.get("last_activity", datetime.now()).isoformat(),
            "message_count": session.get("message_count", 0),
            "mode": session.get("mode", 1),
            "status": session.get("status", "active"),
            "conversation_history": session.get("conversation_history", [])
        }


class DocumentReranker:

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
        query_key_length = 50
        cache_key = f"{query[:query_key_length]}_{len(docs)}_{top_k}"
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

        # 缓存结果（避免内存泄漏）
        max_cache_entries = 50
        if len(self._rerank_cache) >= max_cache_entries:
            # 删除最旧的缓存条目
            oldest_key = next(iter(self._rerank_cache))
            del self._rerank_cache[oldest_key]

        self._rerank_cache[cache_key] = [id(doc) for doc in final_docs]

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
            content_word_limit = 100  # 性能优化
            content_terms = set(doc.page_content.lower().split()[:content_word_limit])
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

        diversity_word_limit = 50
        doc_words = set(doc.page_content.lower().split()[:diversity_word_limit])

        for selected_doc in selected:
            selected_words = set(
                selected_doc.page_content.lower().split()[:diversity_word_limit]
            )
            if doc_words and selected_words:
                similarity = len(doc_words & selected_words) / len(
                    doc_words | selected_words
                )
                if similarity > threshold:
                    return False

        return True


class AnswerGenerator:

    def __init__(self, llm_service, config=None):
        from app.config.settings import config as app_config

        self.llm_service = llm_service
        self.config = config or app_config

        # 从配置加载模板
        self.base_prompt = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """从配置加载提示模板"""
        default_prompt = """基于以下文档回答问题：

文档内容:
{context}

问题: {question}

答案:"""
        return getattr(self.config.rag, "prompt_template", default_prompt)

    def _load_md_prompt_template(self) -> str:
        return """基于以下结构化文档回答问题，保持原有格式：

文档内容:
{context}

问题: {question}

答案:"""

    async def generate(
        self,
        question: str,
        docs: List[Document],
        query_type: QueryType,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """生成答案"""
        if not docs:
            logger.warning(f"未找到相关文档进行回答：{question}")
            return {
                "answer": f"针对您的问题「{question}」，我暂时没有找到匹配的资料。建议使用更具体的关键词重新提问。",
                "confidence": self.config.rag.similarity_threshold * 0.3,
            }

        is_md_content = self._has_markdown_documents(docs)
        enhanced_context = self._prepare_enhanced_context(docs, context, is_md_content)
        generation_params = self._get_generation_params(query_type, context)

        try:
            prompt_template = self._load_md_prompt_template() if is_md_content else self.base_prompt

            prompt = prompt_template.format(context=enhanced_context, question=question)

            if context and context.get("domain"):
                prompt = f"领域: {context['domain']}\n\n{prompt}"

            response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}], **generation_params
            )

            if is_md_content:
                response = self._post_process_md_response(response, docs)

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

    def _has_markdown_documents(self, docs: List[Document]) -> bool:
        """检测文档列表是否包含MD文档"""
        if not docs:
            return False

        md_doc_count = 0
        for doc in docs:
            if (
                doc.metadata.get("document_type") == "markdown"
                or doc.metadata.get("is_structured")
                or doc.metadata.get("has_code")
                or doc.metadata.get("title_hierarchy")
            ):
                md_doc_count += 1

        return md_doc_count > len(docs) / 2

    def _post_process_md_response(self, response: str, docs: List[Document]) -> str:
        """MD响应后处理"""
        response = self._ensure_code_block_formatting(response)
        response = self._ensure_list_formatting(response)
        if not response.startswith("#"):
            response = self._add_structure_headers(response, docs)
        return response

    def _ensure_code_block_formatting(self, text: str) -> str:
        """确保代码块格式正确"""
        import re

        text = re.sub(r"```\n([^`]+)\n```", r"```\n\1\n```", text)
        text = re.sub(r"([^\n])\n```", r"\1\n\n```", text)
        text = re.sub(r"```\n([^\n])", r"```\n\n\1", text)
        return text

    def _ensure_list_formatting(self, text: str) -> str:
        """确保列表格式正确"""
        import re

        text = re.sub(r"([^\n])\n(\d+\. )", r"\1\n\n\2", text)
        text = re.sub(r"([^\n])\n(- )", r"\1\n\n\2", text)
        return text

    def _add_structure_headers(self, response: str, docs: List[Document]) -> str:
        """添加结构化标题"""
        has_code = any(doc.metadata.get("has_code", False) for doc in docs)
        has_config = any(
            "配置" in doc.page_content or "config" in doc.page_content.lower()
            for doc in docs
        )

        if has_code and has_config:
            return f"## 配置和代码示例\n\n{response}"
        elif has_code:
            return f"## 代码示例\n\n{response}"
        elif has_config:
            return f"## 配置说明\n\n{response}"
        else:
            return response

    def _prepare_enhanced_context(
        self, docs: List[Document], context: Optional[Dict], is_md_content: bool = False
    ) -> str:
        """准备增强的上下文"""
        max_length = getattr(self.config.rag, "max_context_length", 4000)
        context_parts = []
        current_length = 0

        for i, doc in enumerate(docs):
            content = doc.page_content

            # 为MD文档添加更详细的元信息
            if is_md_content:
                doc_info = f"[文档{i+1}"
                if doc.metadata.get("title_hierarchy"):
                    hierarchy = " > ".join(doc.metadata["title_hierarchy"])
                    doc_info += f" - 章节: {hierarchy}"
                if doc.metadata.get("has_code"):
                    languages = doc.metadata.get("languages", [])
                    if languages:
                        doc_info += f" - 代码: {', '.join(languages)}"
                    else:
                        doc_info += f" - 包含代码"
                if doc.metadata.get("has_table"):
                    doc_info += f" - 包含表格"
                if doc.metadata.get("source"):
                    doc_info += f" - 来源: {doc.metadata['source']}"
                doc_info += "]"
            else:
                # 普通文档信息
                doc_info = f"[文档{i+1}"
                if doc.metadata.get("source"):
                    doc_info += f" - 来源: {doc.metadata['source']}"
                if doc.metadata.get("score"):
                    doc_info += f" - 相关性: {doc.metadata['score']:.2f}"
                doc_info += "]"

            # 智能截断，对MD文档保持结构完整性
            available_length = max_length - current_length - len(doc_info) - 10
            if available_length <= 200:
                break

            if len(content) > available_length:
                if is_md_content:
                    # 对MD文档进行结构化截断
                    content = self._smart_truncate_md_content(content, available_length)
                else:
                    content = content[:available_length] + "..."

            full_content = f"{doc_info}\n{content}"
            context_parts.append(full_content)
            current_length += len(full_content)

        # 添加上下文信息
        if context and context.get("recent_queries"):
            recent_queries = list(context["recent_queries"])[-3:]
            recent_info = f"\n\n[最近相关查询]: {', '.join(recent_queries)}"
            if current_length + len(recent_info) < max_length:
                context_parts.append(recent_info)

        return "\n\n".join(context_parts)

    def _smart_truncate_md_content(self, content: str, max_length: int) -> str:
        """智能截断MD内容，保持结构完整性"""
        if len(content) <= max_length:
            return content

        lines = content.split("\n")
        truncated_lines = []
        current_length = 0

        for line in lines:
            # 为重要结构预留空间
            if line.strip().startswith("#"):  # 标题
                if current_length + len(line) + 50 > max_length:
                    break
                truncated_lines.append(line)
                current_length += len(line) + 1
            elif line.strip().startswith("```"):  # 代码块
                # 尝试包含完整的代码块
                remaining_lines = lines[len(truncated_lines) :]
                code_block = []

                for remaining_line in remaining_lines:
                    code_block.append(remaining_line)
                    if remaining_line.strip() == "```" and len(code_block) > 1:
                        break

                code_block_text = "\n".join(code_block)
                if current_length + len(code_block_text) <= max_length:
                    truncated_lines.extend(code_block)
                    current_length += len(code_block_text)
                    # 跳过已处理的行
                    for _ in range(len(code_block) - 1):
                        if len(truncated_lines) < len(lines):
                            lines.pop(len(truncated_lines))
                else:
                    break
            else:
                if current_length + len(line) > max_length:
                    # 添加截断标记
                    if current_length + 10 <= max_length:
                        truncated_lines.append("...")
                    break
                truncated_lines.append(line)
                current_length += len(line) + 1

        return "\n".join(truncated_lines)

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
                    from app.core.vector.redis_vector_store import (
                        EnhancedRedisVectorStore,
                    )
                    from app.services.llm import LLMService

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
