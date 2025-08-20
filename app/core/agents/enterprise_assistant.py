#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 基于LangGraph的企业级智能助手
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


logger = logging.getLogger("aiops.rag_assistant")


class QueryType(Enum):
    """查询类型枚举"""

    FACTUAL = "factual"  # 事实性查询
    TROUBLESHOOTING = "troubleshooting"  # 故障排查
    TUTORIAL = "tutorial"  # 教程指导
    CONCEPTUAL = "conceptual"  # 概念解释
    GENERAL = "general"  # 通用查询


@dataclass
class RetrievalStrategy:
    """检索策略配置"""

    semantic_weight: float = 0.6
    lexical_weight: float = 0.4
    similarity_threshold: float = 0.5  # 提高阈值，过滤低质量结果
    max_candidates: int = 20  # 候选文档数
    final_top_k: int = 5  # 最终返回数
    enable_rerank: bool = True  # 启用重排序
    enable_query_expansion: bool = True  # 查询扩展


class RAGState(TypedDict):
    """优化的RAG工作流状态"""

    # 输入
    question: str
    session_id: Optional[str]

    # 处理阶段
    query_type: QueryType
    expanded_queries: List[str]  # 扩展查询
    retrieved_docs: List[Document]
    reranked_docs: List[Document]

    # 输出
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]

    # 元数据
    latency_breakdown: Dict[str, float]
    cache_hit: bool
    error: Optional[str]


# ============= 核心组件 =============


class QueryProcessor:
    """查询处理器 - 负责查询分析、重写和扩展"""

    def __init__(self):
        self.query_templates = {
            QueryType.TROUBLESHOOTING: [
                "{query} 故障原因",
                "{query} 解决方案",
                "{query} 排查步骤",
            ],
            QueryType.TUTORIAL: [
                "{query} 操作步骤",
                "{query} 配置方法",
                "{query} 最佳实践",
            ],
            QueryType.CONCEPTUAL: ["{query} 概念", "{query} 原理", "{query} 定义"],
        }

    async def analyze_query(self, query: str) -> QueryType:
        """分析查询类型"""
        query_lower = query.lower()

        # 基于关键词的快速分类
        if any(
            kw in query_lower
            for kw in ["错误", "故障", "失败", "不能", "error", "fail"]
        ):
            return QueryType.TROUBLESHOOTING
        elif any(
            kw in query_lower for kw in ["如何", "怎么", "步骤", "教程", "how to"]
        ):
            return QueryType.TUTORIAL
        elif any(kw in query_lower for kw in ["什么是", "概念", "原理", "what is"]):
            return QueryType.CONCEPTUAL
        elif any(kw in query_lower for kw in ["多少", "几个", "数量", "统计"]):
            return QueryType.FACTUAL
        else:
            return QueryType.GENERAL

    async def expand_query(self, query: str, query_type: QueryType) -> List[str]:
        """查询扩展 - 生成多个相关查询"""
        expanded = [query]  # 保留原始查询

        # 根据查询类型添加扩展
        templates = self.query_templates.get(query_type, [])
        for template in templates[:2]:  # 限制扩展数量
            expanded.append(template.format(query=query))

        return expanded


class DocumentRetriever:
    """文档检索器 - 负责多路检索和融合"""

    def __init__(self, vector_store, strategy: RetrievalStrategy):
        self.vector_store = vector_store
        self.strategy = strategy

    async def retrieve(self, queries: List[str]) -> List[Document]:
        """并行检索多个查询"""
        tasks = []
        for query in queries:
            task = asyncio.create_task(self._retrieve_single(query))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并和去重
        all_docs = []
        seen_contents = set()

        for docs in results:
            if isinstance(docs, Exception):
                logger.warning(f"检索失败: {docs}")
                continue

            for doc in docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)

        return all_docs[: self.strategy.max_candidates]

    async def _retrieve_single(self, query: str) -> List[Document]:
        """单个查询的检索"""
        try:
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
            
            # 使用SearchConfig配置检索参数
            from app.core.vector.redis_vector_store import SearchConfig
            search_config = SearchConfig(
                semantic_weight=self.strategy.semantic_weight,
                lexical_weight=self.strategy.lexical_weight,
                similarity_threshold=self.strategy.similarity_threshold,
                use_cache=True
            )
            
            # 调用similarity_search方法
            results = await self.vector_store.similarity_search(
                query=query,
                k=self.strategy.max_candidates,
                config=search_config
            )
            return [doc for doc, _ in results]
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []


class DocumentReranker:
    """文档重排序器 - 提高检索精度"""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    async def rerank(
        self, query: str, docs: List[Document], top_k: int = 5
    ) -> List[Document]:
        """重排序文档"""
        if not docs:
            return []

        if not self.llm_service:
            # 简单的基于长度和关键词的重排序
            return self._simple_rerank(query, docs, top_k)

        # 使用LLM进行语义重排序
        try:
            scored_docs = []
            for doc in docs[:10]:  # 限制LLM处理数量
                score = await self._calculate_relevance_score(query, doc)
                scored_docs.append((doc, score))

            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")
            return self._simple_rerank(query, docs, top_k)

    def _simple_rerank(
        self, query: str, docs: List[Document], top_k: int
    ) -> List[Document]:
        """简单的基于关键词的重排序"""
        query_terms = set(query.lower().split())

        scored_docs = []
        for doc in docs:
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)
            scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

    async def _calculate_relevance_score(self, query: str, doc: Document) -> float:
        """使用LLM计算相关性分数"""
        prompt = f"""
        Query: {query}
        Document: {doc.page_content[:500]}
        
        Rate relevance from 0 to 1:
        """

        try:
            response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            # 解析分数
            import re

            match = re.search(r"(\d+\.?\d*)", response)
            if match:
                return min(float(match.group(1)), 1.0)
        except:
            pass

        return 0.5  # 默认分数


class AnswerGenerator:
    """答案生成器 - 负责生成高质量答案"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.prompt_templates = {
            QueryType.TROUBLESHOOTING: """你是运维专家。基于文档解决问题。

相关文档:
{context}

问题: {question}

请提供:
1. 问题原因分析
2. 具体解决步骤
3. 预防措施

答案:""",
            QueryType.TUTORIAL: """你是技术导师。基于文档提供操作指导。

相关文档:
{context}

问题: {question}

请提供清晰的步骤说明，包括命令示例。

答案:""",
            QueryType.GENERAL: """你是智能助手。基于文档回答问题。

相关文档:
{context}

问题: {question}

请提供准确、专业的回答。

答案:""",
        }

    async def generate(
        self, question: str, docs: List[Document], query_type: QueryType
    ) -> Dict[str, Any]:
        """生成答案"""
        if not docs:
            return {
                "answer": "抱歉，没有找到相关信息。请尝试换个问法或联系技术支持。",
                "confidence": 0.0,
            }

        # 准备上下文
        context = self._prepare_context(docs)

        # 选择提示模板
        template = self.prompt_templates.get(
            query_type, self.prompt_templates[QueryType.GENERAL]
        )
        prompt = template.format(context=context, question=question)

        try:
            # 生成答案
            response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 降低温度提高准确性
                max_tokens=1500,
            )

            return {"answer": response, "confidence": self._calculate_confidence(docs)}
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            # 使用备用响应生成器
            try:
                from app.core.agents.fallback_models import generate_fallback_answer, ResponseContext, SessionData
                
                # 创建响应上下文
                context = ResponseContext(
                    user_input=question,
                    session=None,  # 可以后续添加会话支持
                    additional_context={"docs": docs} if docs else None
                )
                
                fallback_answer = generate_fallback_answer(context)
                return {"answer": fallback_answer, "confidence": 0.3}  # 降低置信度
            except Exception as fallback_e:
                logger.error(f"备用答案生成也失败: {fallback_e}")
                return {"answer": f"生成答案时出错: {str(e)}", "confidence": 0.0}

    def _prepare_context(self, docs: List[Document], max_length: int = 3000) -> str:
        """准备上下文"""
        context_parts = []
        total_length = 0

        for i, doc in enumerate(docs):
            content = doc.page_content
            source = doc.metadata.get("source", f"文档{i+1}")

            # 截断过长内容
            if total_length + len(content) > max_length:
                remaining = max_length - total_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(f"[{source}]\n{content}")
            total_length += len(content)

        return "\n\n".join(context_parts)

    def _calculate_confidence(self, docs: List[Document]) -> float:
        """计算答案置信度"""
        if not docs:
            return 0.0

        # 基于文档数量和质量
        doc_score = min(len(docs) / 5, 1.0) * 0.5

        # 基于文档相关性（假设有分数）
        avg_relevance = 0.5  # 默认值
        if hasattr(docs[0], "metadata") and "score" in docs[0].metadata:
            scores = [d.metadata.get("score", 0.5) for d in docs[:5]]
            avg_relevance = sum(scores) / len(scores)

        return doc_score + avg_relevance * 0.5


# ============= 优化的LangGraph工作流 =============


class OptimizedRAGAssistant:
    """优化的RAG助手主类"""

    def __init__(self, vector_store, llm_service, cache_manager=None):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.cache_manager = cache_manager

        # 初始化组件
        self.query_processor = QueryProcessor()
        self.retriever = DocumentRetriever(vector_store, RetrievalStrategy())
        self.reranker = DocumentReranker(llm_service)
        self.generator = AnswerGenerator(llm_service)

        # 初始化会话管理器
        from app.core.agents.fallback_models import SessionManager
        self.session_manager = SessionManager()

        # 初始化checkpointer
        self.checkpointer = MemorySaver()
        
        # 构建工作流
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建简化的线性工作流"""
        workflow = StateGraph(RAGState)

        # 定义节点
        workflow.add_node("check_cache", self._check_cache)
        workflow.add_node("process_query", self._process_query)
        workflow.add_node("retrieve_docs", self._retrieve_docs)
        workflow.add_node("rerank_docs", self._rerank_docs)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("store_cache", self._store_cache)

        # 定义边 - 简单线性流
        workflow.set_entry_point("check_cache")

        workflow.add_conditional_edges(
            "check_cache",
            lambda x: "cached" if x.get("cache_hit") else "process",
            {"cached": END, "process": "process_query"},
        )

        workflow.add_edge("process_query", "retrieve_docs")
        workflow.add_edge("retrieve_docs", "rerank_docs")
        workflow.add_edge("rerank_docs", "generate_answer")
        workflow.add_edge("generate_answer", "store_cache")
        workflow.add_edge("store_cache", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def _check_cache(self, state: RAGState) -> Dict[str, Any]:
        """缓存检查节点"""
        start = time.time()

        if not self.cache_manager:
            return {"cache_hit": False}

        cache_key = self._get_cache_key(state["question"])
        cached = await asyncio.to_thread(self.cache_manager.get, cache_key)

        if cached:
            logger.info(f"缓存命中: {state['question'][:50]}...")
            return {
                "cache_hit": True,
                "answer": cached["answer"],
                "confidence": cached["confidence"],
                "sources": cached["sources"],
                "latency_breakdown": {"cache_check": time.time() - start},
            }

        return {
            "cache_hit": False,
            "latency_breakdown": {"cache_check": time.time() - start},
        }

    async def _process_query(self, state: RAGState) -> Dict[str, Any]:
        """查询处理节点"""
        start = time.time()

        # 分析查询类型
        query_type = await self.query_processor.analyze_query(state["question"])

        # 查询扩展
        expanded = await self.query_processor.expand_query(
            state["question"], query_type
        )

        latency = state.get("latency_breakdown", {})
        latency["query_process"] = time.time() - start

        return {
            "query_type": query_type,
            "expanded_queries": expanded,
            "latency_breakdown": latency,
        }

    async def _retrieve_docs(self, state: RAGState) -> Dict[str, Any]:
        """文档检索节点"""
        start = time.time()

        # 并行检索扩展查询
        docs = await self.retriever.retrieve(state["expanded_queries"])

        latency = state["latency_breakdown"]
        latency["retrieval"] = time.time() - start

        logger.info(f"检索到 {len(docs)} 个候选文档")

        return {"retrieved_docs": docs, "latency_breakdown": latency}

    async def _rerank_docs(self, state: RAGState) -> Dict[str, Any]:
        """文档重排序节点"""
        start = time.time()

        # 重排序提高精度
        reranked = await self.reranker.rerank(
            state["question"], state["retrieved_docs"], top_k=5
        )

        latency = state["latency_breakdown"]
        latency["rerank"] = time.time() - start

        logger.info(f"重排序后保留 {len(reranked)} 个文档")

        return {"reranked_docs": reranked, "latency_breakdown": latency}

    async def _generate_answer(self, state: RAGState) -> Dict[str, Any]:
        """答案生成节点"""
        start = time.time()

        # 生成答案
        result = await self.generator.generate(
            state["question"], state["reranked_docs"], state["query_type"]
        )

        # 提取源文档
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in state["reranked_docs"][:3]
        ]

        latency = state["latency_breakdown"]
        latency["generation"] = time.time() - start
        latency["total"] = sum(latency.values())

        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": sources,
            "latency_breakdown": latency,
        }

    async def _store_cache(self, state: RAGState) -> Dict[str, Any]:
        """缓存存储节点"""
        if not self.cache_manager or state.get("confidence", 0) < 0.6:
            return {}

        cache_key = self._get_cache_key(state["question"])
        cache_data = {
            "answer": state["answer"],
            "confidence": state["confidence"],
            "sources": state["sources"],
            "cached_at": datetime.now().isoformat(),
        }

        await asyncio.to_thread(
            self.cache_manager.set, cache_key, cache_data, ttl=3600  # 1小时缓存
        )

        return {}

    def _get_cache_key(self, question: str) -> str:
        """生成缓存键"""
        return hashlib.md5(question.encode()).hexdigest()

    async def get_answer(
        self, question: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取答案 - 主接口"""
        try:
            # 管理会话
            session = None
            if session_id:
                session = self.session_manager.get_session(session_id)
                if not session:
                    session = self.session_manager.create_session(session_id)
                # 更新会话活动时间
                session.update_activity()

            # 初始状态
            initial_state = {
                "question": question,
                "session_id": session_id,
                "cache_hit": False,
            }

            # 运行工作流
            config = {"thread_id": session_id or "default"}
            result = await self.graph.ainvoke(initial_state, config=config)

            # 更新会话历史
            if session:
                self.session_manager.update_session(session_id, question)

            # 构建响应
            response = {
                "answer": result.get("answer", ""),
                "confidence_score": result.get("confidence", 0.0),
                "source_documents": result.get("sources", []),
                "cache_hit": result.get("cache_hit", False),
                "processing_time": result.get("latency_breakdown", {}).get("total", 0),
                "session_id": session_id,
                "success": True,
            }

            # 日志性能指标
            if "latency_breakdown" in result:
                logger.info(f"性能分析: {result['latency_breakdown']}")

            return response

        except Exception as e:
            logger.error(f"处理失败: {e}")
            
            # 使用备用响应生成
            try:
                from app.core.agents.fallback_models import generate_fallback_answer, ResponseContext
                
                # 获取会话信息
                session = None
                if session_id:
                    session = self.session_manager.get_session(session_id)
                
                # 创建响应上下文
                context = ResponseContext(
                    user_input=question,
                    session=session,
                    additional_context={"error": str(e)}
                )
                
                fallback_answer = generate_fallback_answer(context)
                
                return {
                    "answer": fallback_answer,
                    "confidence_score": 0.2,  # 低置信度
                    "source_documents": [],
                    "success": False,
                    "error": str(e),
                    "fallback_used": True,
                    "session_id": session_id,
                }
            except Exception as fallback_e:
                logger.error(f"备用答案生成也失败: {fallback_e}")
                return {
                    "answer": f"处理出错: {str(e)}",
                    "confidence_score": 0.0,
                    "source_documents": [],
                    "success": False,
                    "error": str(e),
                    "session_id": session_id,
                }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        # 检查备用实现的可用性
        fallback_available = False
        try:
            from app.core.agents.fallback_models import FallbackChatModel, FallbackEmbeddings
            fallback_available = True
        except Exception:
            pass
            
        return {
            "status": "healthy",
            "components": {
                "vector_store": bool(self.vector_store),
                "llm_service": bool(self.llm_service),
                "cache": bool(self.cache_manager),
                "graph": bool(self.graph),
                "session_manager": bool(self.session_manager),
                "fallback_models": fallback_available,
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """刷新知识库"""
        try:
            # 清空缓存
            if self.cache_manager:
                await asyncio.to_thread(self.cache_manager.clear_pattern, "*")

            return {
                "success": True,
                "message": "知识库刷新完成",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        # 简化实现
        return {
            "session_id": session_id,
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }


# ============= 工厂函数 =============

_assistant_instance = None
_lock = asyncio.Lock()


async def get_enterprise_assistant() -> OptimizedRAGAssistant:
    """获取全局助手实例"""
    global _assistant_instance

    if _assistant_instance is None:
        async with _lock:
            if _assistant_instance is None:
                try:
                    # 初始化依赖
                    from app.services.llm import LLMService
                    from app.core.cache.redis_cache_manager import RedisCacheManager
                    from app.config.settings import config

                    logger.info("正在初始化企业级智能助手...")

                    # 创建LLM服务
                    logger.debug("创建LLM服务...")
                    llm_service = LLMService()

                    # 创建缓存管理器
                    logger.debug("创建Redis缓存管理器...")
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

                    # 初始化向量存储
                    logger.debug("初始化向量存储...")
                    from app.core.vector.redis_vector_store import EnhancedRedisVectorStore
                    
                    # 根据配置创建嵌入模型
                    embedding_model = await _create_embedding_model()
                    
                    # 创建向量存储配置
                    vector_config = {
                        "host": config.redis.host,
                        "port": config.redis.port,
                        "db": config.redis.db + 2,  # 使用独立的数据库
                        "password": config.redis.password,
                        "decode_responses": False,
                    }
                    
                    # 获取向量维度
                    vector_dim = _get_embedding_dimension(embedding_model)
                    
                    # 初始化向量存储
                    vector_store = EnhancedRedisVectorStore(
                        redis_config=vector_config,
                        collection_name="aiops_knowledge",
                        embedding_model=embedding_model,
                        vector_dim=vector_dim,
                        index_type="HNSW"
                    )
                    
                    # 加载知识库
                    await _load_knowledge_base(vector_store)

                    logger.debug("创建RAG助手实例...")
                    _assistant_instance = OptimizedRAGAssistant(
                        vector_store=vector_store,
                        llm_service=llm_service,
                        cache_manager=cache_manager,
                    )

                    logger.info("企业级RAG助手初始化完成")
                    
                except Exception as e:
                    logger.error(f"企业级RAG助手初始化失败: {str(e)}")
                    # 确保失败时实例为None，以便重试
                    _assistant_instance = None
                    raise

    return _assistant_instance


async def _create_embedding_model():
    """根据配置创建嵌入模型"""
    try:
        from app.config.settings import config
        
        # 获取有效的嵌入模型名称
        embedding_model_name = config.rag.effective_embedding_model
        provider = config.llm.provider.lower()
        api_key = config.llm.effective_api_key
        
        logger.info(f"正在初始化嵌入模型: {embedding_model_name} (provider: {provider})")
        
        # 检查API密钥有效性
        if provider == "openai" and (not api_key or api_key in ["sk-xxx", "", "your-api-key"]):
            logger.warning("检测到无效的 OpenAI API 密钥，使用备用嵌入模型")
            from app.core.agents.fallback_models import FallbackEmbeddings
            return FallbackEmbeddings()
        
        if provider == "openai":
            # 使用 OpenAI 嵌入模型
            try:
                from langchain_openai import OpenAIEmbeddings
                embedding_model = OpenAIEmbeddings(
                    model=embedding_model_name,
                    openai_api_key=config.llm.effective_api_key,
                    openai_api_base=config.llm.effective_base_url
                )
                logger.info("OpenAI嵌入模型创建成功")
                return embedding_model
            except ImportError:
                logger.warning("OpenAI embeddings 包不可用，尝试使用通用实现")
                # 尝试使用通用的 OpenAI 客户端
                return _create_openai_embeddings(embedding_model_name)
            except Exception as e:
                logger.error(f"OpenAI嵌入模型创建失败: {e}")
                # 不立即返回备用模型，继续尝试自定义实现
                try:
                    return _create_openai_embeddings(embedding_model_name)
                except Exception as e2:
                    logger.error(f"自定义OpenAI嵌入模型也创建失败: {e2}")
        
        elif provider == "ollama":
            # 使用 Ollama 嵌入模型
            try:
                from langchain_community.embeddings import OllamaEmbeddings
                embedding_model = OllamaEmbeddings(
                    model=embedding_model_name,
                    base_url=config.llm.ollama_base_url.replace("/v1", "")
                )
                logger.info("Ollama嵌入模型创建成功")
                return embedding_model
            except ImportError:
                logger.warning("Ollama embeddings 包不可用，尝试使用自定义实现")
                return _create_ollama_embeddings(embedding_model_name)
            except Exception as e:
                logger.error(f"Ollama嵌入模型创建失败: {e}")
                try:
                    return _create_ollama_embeddings(embedding_model_name)
                except Exception as e2:
                    logger.error(f"自定义Ollama嵌入模型也创建失败: {e2}")
        
        else:
            logger.warning(f"不支持的 provider: {provider}")
            
    except Exception as e:
        logger.error(f"创建嵌入模型时发生异常: {e}")
    
    # 最终备用方案：使用 FallbackEmbeddings
    logger.warning("所有嵌入模型创建失败，使用备用嵌入模型")
    from app.core.agents.fallback_models import FallbackEmbeddings
    return FallbackEmbeddings()


def _create_openai_embeddings(model_name: str):
    """创建自定义 OpenAI 嵌入模型"""
    from langchain_core.embeddings import Embeddings
    from app.config.settings import config
    
    # 提前检查API密钥
    api_key = config.llm.effective_api_key
    if not api_key or api_key in ["sk-xxx", "", "your-api-key"]:
        logger.warning("API密钥无效，直接使用备用嵌入模型")
        from app.core.agents.fallback_models import FallbackEmbeddings
        return FallbackEmbeddings()
    
    class CustomOpenAIEmbeddings(Embeddings):
        def __init__(self, model: str):
            self.model = model
            self.fallback_used = False
            
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=config.llm.effective_api_key,
                    base_url=config.llm.effective_base_url,
                    timeout=config.llm.request_timeout  # 设置超时避免长时间等待
                )
                logger.debug(f"OpenAI客户端初始化成功，模型: {model}")
            except Exception as e:
                logger.error(f"OpenAI客户端初始化失败: {e}")
                self.client = None
                self.fallback_used = True
        
        def embed_documents(self, texts):
            if self.fallback_used or not self.client:
                return self._use_fallback().embed_documents(texts)
                
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"OpenAI embeddings 调用失败: {e}")
                self.fallback_used = True
                return self._use_fallback().embed_documents(texts)
        
        def embed_query(self, text):
            if self.fallback_used or not self.client:
                return self._use_fallback().embed_query(text)
                
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text]
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"OpenAI embeddings 查询失败: {e}")
                self.fallback_used = True
                return self._use_fallback().embed_query(text)
        
        def _use_fallback(self):
            """使用备用嵌入模型"""
            if not hasattr(self, '_fallback'):
                from app.core.agents.fallback_models import FallbackEmbeddings
                self._fallback = FallbackEmbeddings()
                if not self.fallback_used:
                    logger.warning("切换到备用嵌入模型")
            return self._fallback
    
    return CustomOpenAIEmbeddings(model_name)


def _create_ollama_embeddings(model_name: str):
    """创建自定义 Ollama 嵌入模型"""
    from langchain_core.embeddings import Embeddings
    import ollama
    import os
    from app.config.settings import config
    
    class CustomOllamaEmbeddings(Embeddings):
        def __init__(self, model: str):
            self.model = model
            # 设置 Ollama 主机
            os.environ["OLLAMA_HOST"] = config.llm.ollama_base_url.replace("/v1", "")
        
        def embed_documents(self, texts):
            try:
                embeddings = []
                for text in texts:
                    response = ollama.embeddings(model=self.model, prompt=text)
                    embeddings.append(response['embedding'])
                return embeddings
            except Exception as e:
                logger.error(f"Ollama embeddings 调用失败: {e}")
                # 使用备用方案
                from app.core.agents.fallback_models import FallbackEmbeddings
                fallback = FallbackEmbeddings()
                return fallback.embed_documents(texts)
        
        def embed_query(self, text):
            try:
                response = ollama.embeddings(model=self.model, prompt=text)
                return response['embedding']
            except Exception as e:
                logger.error(f"Ollama embeddings 查询失败: {e}")
                # 使用备用方案
                from app.core.agents.fallback_models import FallbackEmbeddings
                fallback = FallbackEmbeddings()
                return fallback.embed_query(text)
    
    return CustomOllamaEmbeddings(model_name)


def _get_embedding_dimension(embedding_model):
    """获取嵌入模型的维度"""
    try:
        # 尝试通过测试嵌入获取维度
        logger.debug("正在测试嵌入模型以获取维度...")
        test_embedding = embedding_model.embed_query("test")
        dimension = len(test_embedding)
        logger.info(f"成功获取嵌入维度: {dimension}")
        return dimension
    except Exception as e:
        logger.warning(f"无法获取嵌入维度: {e}")
        
        # 根据模型名称猜测维度
        from app.config.settings import config
        model_name = config.rag.effective_embedding_model.lower()
        
        # 更准确的维度映射
        dimension_map = {
            "bge-m3": 1024,
            "bge-large": 1024,
            "bge-base": 768,
            "bge-small": 512,
            "nomic-embed": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
        }
        
        # 尝试找到匹配的模型
        for model_key, dimension in dimension_map.items():
            if model_key in model_name:
                logger.info(f"根据模型名称 '{model_name}' 推断维度: {dimension}")
                return dimension
        
        # 如果是 FallbackEmbeddings，返回其默认维度
        try:
            from app.core.agents.fallback_models import FallbackEmbeddings, DEFAULT_EMBEDDING_DIMENSION
            if isinstance(embedding_model, FallbackEmbeddings):
                dimension = getattr(embedding_model, 'dimension', DEFAULT_EMBEDDING_DIMENSION)
                logger.info(f"使用备用嵌入模型，维度: {dimension}")
                return dimension
        except Exception as e:
            logger.warning(f"检查备用嵌入模型时出错: {e}")
            
        # 默认维度
        default_dimension = 1024
        logger.warning(f"无法确定嵌入维度，使用默认值: {default_dimension}")
        return default_dimension


async def _load_knowledge_base(vector_store) -> None:
    """加载知识库到向量存储"""
    import os
    
    try:
        knowledge_base_path = "data/knowledge_base"
        if not os.path.exists(knowledge_base_path):
            logger.warning("知识库目录不存在，跳过加载")
            return
        
        documents = []
        supported_extensions = ['.md', '.txt']
        
        # 扫描知识库文件
        for root, dirs, files in os.walk(knowledge_base_path):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                # 创建文档对象
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": file,
                                        "path": file_path,
                                        "type": "knowledge_base"
                                    }
                                )
                                documents.append(doc)
                    except Exception as e:
                        logger.warning(f"读取文件失败 {file_path}: {e}")
        
        if documents:
            logger.info(f"开始加载 {len(documents)} 个知识库文档...")
            # 批量添加文档到向量存储
            doc_ids = await vector_store.add_documents(documents)
            logger.info(f"知识库加载完成，共添加 {len(doc_ids)} 个文档")
        else:
            logger.warning("未找到有效的知识库文档")
            
    except Exception as e:
        logger.error(f"知识库加载失败: {e}")
        # 不抛出异常，允许系统继续运行


def reset_assistant():
    """重置助手实例"""
    global _assistant_instance
    _assistant_instance = None
