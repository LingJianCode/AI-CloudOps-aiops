#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .answer_generator import ReliableAnswerGenerator
from .document_loader import DocumentLoader
from .document_processor import ContextAwareRetriever
from .fallback_models import (
    FallbackEmbeddings,
    SessionData,
    create_session_id,
    sanitize_input,
    validate_session_id,
)
from .streaming_answer import ContinuousAnswerGenerator
from .query_processor import QueryRewriter

from app.core.cache.redis_cache_manager import RedisCacheManager
from app.core.vector.redis_vector_store import (
    OptimizedRedisVectorStore,
    RedisVectorStoreManager,
)
from app.config.settings import config
from app.services.llm import LLMService

logger = logging.getLogger("aiops.assistant")


@dataclass
class PerformanceMetrics:
    response_times: List[float] = field(default_factory=list)
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    success_rate: float = 0.0
    
    def update_response_time(self, time_taken: float):
        self.response_times.append(time_taken)
        self.avg_response_time = sum(self.response_times) / len(self.response_times)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def update_cache_stats(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        total = self.cache_hits + self.cache_misses
        self.cache_hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def update_request_stats(self, success: bool):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        self.success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
    
    def get_stats_dict(self) -> Dict[str, Any]:
        return {
            "response_time": {
                "avg": round(self.avg_response_time, 3),
                "total_samples": len(self.response_times)
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round(self.cache_hit_rate, 2)
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "success_rate": round(self.success_rate, 2)
            }
        }


class VectorStoreManager:
    def __init__(self, cache_manager=None, performance_metrics=None):
        self.vector_store = None
        self.embeddings = None
        self.redis_manager = None
        self.cache_manager = cache_manager
        self.performance_metrics = performance_metrics
        self.initialization_lock = threading.Lock()
        self.is_initialized = False
        self.initialization_error = None

    async def get_vector_store(self) -> Optional[OptimizedRedisVectorStore]:
        if self.vector_store is not None:
            return self.vector_store

        with self.initialization_lock:
            if self.vector_store is not None:
                return self.vector_store

            try:
                await self._initialize_vector_store()
                return self.vector_store
            except Exception as e:
                logger.error(f"向量存储初始化失败: {str(e)}")
                self.initialization_error = str(e)
                return None

    async def _initialize_vector_store(self):
        try:
            self.embeddings = await self._create_embeddings()
            if not self.embeddings:
                raise Exception("嵌入模型创建失败")

            redis_config = {
                "host": config.redis.host,
                "port": config.redis.port,
                "db": config.redis.db,
                "password": config.redis.password,
                "connection_timeout": config.redis.connection_timeout,
                "socket_timeout": config.redis.socket_timeout,
                "max_connections": config.redis.max_connections,
                "decode_responses": config.redis.decode_responses,
            }
            
            try:
                test_embedding = self.embeddings.embed_query("测试")
                vector_dim = len(test_embedding)
            except Exception as e:
                logger.warning(f"无法检测嵌入维度，使用默认值1536: {e}")
                vector_dim = 1536
            
            self.redis_manager = RedisVectorStoreManager(
                redis_config=redis_config,
                collection_name="aiops_knowledge_base",
                embedding_dimensions=vector_dim,
                local_storage_path=str(Path(__file__).parent.parent.parent.parent / "data" / "vector_db"),
            )

            self.vector_store = OptimizedRedisVectorStore(
                redis_config=redis_config,
                collection_name="aiops_knowledge_base",
                embedding_model=self.embeddings,
                vector_dim=vector_dim,
                local_storage_path=str(Path(__file__).parent.parent.parent.parent / "data" / "vector_db"),
                use_faiss=True,
                faiss_index_type="Flat"
            )

            self.is_initialized = True

        except Exception as e:
            logger.error(f"向量存储初始化失败: {str(e)}")
            self.initialization_error = str(e)
            raise

    async def _create_embeddings(self):
        try:
            provider = config.llm.provider.lower()

            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                    model=config.llm.embedding_model
                )
            elif provider == "ollama":
                from langchain_ollama import OllamaEmbeddings
                return OllamaEmbeddings(
                    base_url=config.llm.ollama_base_url,
                    model=config.llm.ollama_embedding_model
                )
            else:
                return FallbackEmbeddings()

        except Exception as e:
            logger.error(f"创建嵌入模型失败: {str(e)}")
            return FallbackEmbeddings()

    async def add_documents(self, documents: List[Document]) -> bool:
        try:
            vector_store = await self.get_vector_store()
            if not vector_store:
                return False

            vector_store.add_documents(documents)
            return True

        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            return False

    async def search_documents(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        try:
            vector_store = await self.get_vector_store()
            if not vector_store:
                return []

            return vector_store.similarity_search_with_score(query, k=k)

        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}")
            return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.is_initialized,
            "vector_store_available": self.vector_store is not None,
            "embeddings_available": self.embeddings is not None,
            "initialization_error": self.initialization_error
        }


class AssistantAgent:
    def __init__(self):
        self.llm_service = None
        self.vector_store_manager = None
        self.document_loader = DocumentLoader()
        self.query_rewriter = QueryRewriter()
        self.context_retriever = None
        self.answer_generator = None
        
        self.knowledge_loaded = False
        self.sessions = {}
        self.response_cache = {}
        self.cache_manager = None
        
        self.performance_metrics = PerformanceMetrics()
        
        self.init_lock = threading.Lock()
        self.is_initializing = False
        self.initialization_complete = False
        
        self.fast_mode = True

    async def initialize(self) -> bool:
        if self.initialization_complete:
            return True

        init_start_time = time.time()
        
        with self.init_lock:
            if self.initialization_complete:
                return True

            if self.is_initializing:
                while self.is_initializing:
                    time.sleep(0.1)
                return self.initialization_complete

            self.is_initializing = True

        try:
            if not await self._initialize_llm_service():
                logger.error("LLM服务初始化失败")
                return False

            await self._initialize_cache_manager()

            self.vector_store_manager = VectorStoreManager(
                cache_manager=self.cache_manager,
                performance_metrics=self.performance_metrics
            )

            self.context_retriever = ContextAwareRetriever(self.vector_store_manager)
            self.answer_generator = ReliableAnswerGenerator(self.llm_service)
            self.continuous_generator = ContinuousAnswerGenerator(self.answer_generator)

            await self._load_knowledge_base()

            self.initialization_complete = True
            return True

        except Exception as e:
            logger.error(f"智能助手初始化失败: {str(e)}")
            return False
        finally:
            self.is_initializing = False

    async def _initialize_llm_service(self) -> bool:
        try:
            self.llm_service = LLMService()
            if hasattr(self.llm_service, 'is_healthy') and callable(self.llm_service.is_healthy):
                if not self.llm_service.is_healthy():
                    logger.warning("LLM服务健康检查失败，但继续初始化")
            return True
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {str(e)}")
            return False

    async def _initialize_cache_manager(self):
        try:
            redis_config = {
                "host": config.redis.host,
                "port": config.redis.port,
                "db": config.redis.db + 1,
                "password": config.redis.password,
                "connection_timeout": config.redis.connection_timeout,
                "socket_timeout": config.redis.socket_timeout,
                "max_connections": config.redis.max_connections,
                "decode_responses": config.redis.decode_responses,
            }
            
            self.cache_manager = RedisCacheManager(
                redis_config=redis_config,
                cache_prefix="aiops_assistant_optimized_cache:",
                default_ttl=3600,
                max_cache_size=1000,
                enable_compression=True,
            )
        except Exception as e:
            logger.warning(f"缓存管理器初始化失败: {str(e)}")
            self.cache_manager = None

    async def _load_knowledge_base(self):
        try:
            documents = await self.document_loader.load_all_documents()
            
            if not documents:
                logger.warning("没有找到任何文档，知识库为空")
                self.knowledge_loaded = False
                return

            success = await self.vector_store_manager.add_documents(documents)
            
            if success:
                self.knowledge_loaded = True
                stats = self.document_loader.get_document_stats(documents)
                logger.info(f"知识库加载完成: {stats}")
            else:
                logger.error("文档添加到向量存储失败")
                self.knowledge_loaded = False

        except Exception as e:
            logger.error(f"知识库加载失败: {str(e)}")
            self.knowledge_loaded = False

    async def get_answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        max_context_docs: int = 1  # 从2减少到1
    ) -> Dict[str, Any]:
        """
        获取问题答案
        
        Args:
            question: 用户问题
            session_id: 会话ID
            max_context_docs: 最大上下文文档数
            
        Returns:
            包含答案和相关信息的字典
        """
        start_time = time.time()
        success = False
        
        try:
            # 确保初始化完成
            if not await self.initialize():
                self.performance_metrics.update_request_stats(False)
                return self._create_error_response("助手初始化失败")

            # 输入验证和清理
            clean_question = sanitize_input(question)
            if not clean_question:
                self.performance_metrics.update_request_stats(False)
                return self._create_error_response("问题不能为空")

            # 检查缓存
            cache_key = f"answer:{hash(clean_question)}"
            cached_answer = None
            if self.cache_manager:
                cached_answer = self.cache_manager.get(clean_question, session_id)
                if cached_answer:
                    logger.info("使用缓存的答案")
                    self.performance_metrics.update_cache_stats(True)
                    self.performance_metrics.update_request_stats(True)
                    response_time = time.time() - start_time
                    self.performance_metrics.update_response_time(response_time)
                    return cached_answer
                else:
                    self.performance_metrics.update_cache_stats(False)

            # 获取会话信息
            session = self._get_or_create_session(session_id)

            # 扩展查询 - 简化为只生成一个变体
            expanded_queries = [clean_question]  # 直接使用原始问题，节省时间
            logger.debug(f"查询扩展: {len(expanded_queries)} 个变体")

            # 检索相关文档 - 记录检索时间
            retrieval_start_time = time.time()
            context_docs, recall_rate = await self.context_retriever.retrieve_with_context(
                clean_question,
                session.history if session else None,
                max_context_docs
            )
            retrieval_time = time.time() - retrieval_start_time
            self.performance_metrics.update_document_retrieval_time(retrieval_time)

            # 生成答案 - 使用连续生成器确保完整性
            llm_start_time = time.time()
            answer_result = await self.continuous_generator.generate_continuous_answer(
                clean_question,
                context_docs,
                session.history if session else None
            )
            llm_time = time.time() - llm_start_time
            self.performance_metrics.update_llm_call_time(llm_time)

            # 更新会话历史
            if session:
                session.history.append(f"Q: {clean_question}")
                session.history.append(f"A: {answer_result['answer'][:50]}...")  # 从100减少剀50
                session.last_activity = datetime.now()
                
                # 限制历史长度，从10减少到4
                if len(session.history) > 4:
                    session.history = session.history[-4:]

            # 添加召回率信息
            answer_result["recall_rate"] = recall_rate

            # 缓存结果，提高阈值至0.7
            if self.cache_manager and answer_result.get("relevance_score", 0) > 0.7:
                self.cache_manager.set(clean_question, answer_result, session_id, ttl=1800)  # 减少TTL到1800秒

            success = True
            logger.info(f"问答完成: 召回率={recall_rate:.2f}, 相关性={answer_result.get('relevance_score', 0):.2f}, 检索时间={retrieval_time:.2f}s, LLM时间={llm_time:.2f}s")
            return answer_result

        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return self._create_error_response(f"处理问题时出错: {str(e)}")
        finally:
            # 记录性能指标
            response_time = time.time() - start_time
            self.performance_metrics.update_response_time(response_time)
            self.performance_metrics.update_request_stats(success)

    def _get_or_create_session(self, session_id: Optional[str]) -> Optional[SessionData]:
        """获取或创建会话"""
        if not session_id:
            return None

        if not validate_session_id(session_id):
            logger.warning(f"无效的会话ID: {session_id}")
            return None

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(
                session_id=session_id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                history=[]
            )

        return self.sessions[session_id]

    def create_session(self) -> str:
        """创建新会话"""
        session_id = create_session_id()
        self.sessions[session_id] = SessionData(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            history=[]
        )
        logger.info(f"创建新会话: {session_id}")
        return session_id

    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """刷新知识库"""
        try:
            logger.info("开始刷新知识库...")
            
            # 清空当前向量存储
            self.vector_store_manager = VectorStoreManager()
            
            # 重新加载知识库
            await self._load_knowledge_base()
            
            # 清空缓存
            if self.cache_manager:
                self.cache_manager.clear_pattern("answer:*")
            
            self.response_cache.clear()
            
            result = {
                "success": self.knowledge_loaded,
                "documents_count": 0,  # 这里应该从实际统计中获取
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("知识库刷新完成")
            return result

        except Exception as e:
            logger.error(f"刷新知识库失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def add_document(self, content: str, metadata: dict = None) -> bool:
        """添加文档到知识库"""
        try:
            document = self.document_loader.add_document_from_content(content, metadata)
            
            if not self.document_loader.validate_document(document):
                logger.warning("文档验证失败")
                return False

            # 同步包装异步调用
            from app.core.agents.assistant_utils import safe_async_run
            success = safe_async_run(self.vector_store_manager.add_documents([document]))
            
            if success:
                logger.info("文档添加成功")
            else:
                logger.warning("文档添加到向量存储失败")
            
            return success

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False

    def clear_cache(self) -> Dict[str, Any]:
        """清除缓存"""
        try:
            cleared_items = len(self.response_cache)
            self.response_cache.clear()
            
            # 清除Redis缓存
            if self.cache_manager:
                self.cache_manager.clear_pattern("answer:*")
            
            logger.info(f"清除了 {cleared_items} 个缓存项")
            return {
                "cleared_items": cleared_items,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"清除缓存失败: {str(e)}")
            return {
                "cleared_items": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "answer": f"抱歉，处理您的问题时遇到了错误：{error_message}",
            "source_documents": [],
            "follow_up_questions": ["可以尝试重新提问吗？", "需要其他帮助吗？"],
            "relevance_score": 0.0,
            "recall_rate": 0.0,
            "error": True,
            "timestamp": datetime.now().isoformat()
        }

    def get_status(self) -> Dict[str, Any]:
        """获取助手状态"""
        vector_status = self.vector_store_manager.get_status() if self.vector_store_manager else {}
        
        return {
            "initialized": self.initialization_complete,
            "knowledge_loaded": self.knowledge_loaded,
            "llm_available": self.llm_service is not None,
            "cache_available": self.cache_manager is not None,
            "active_sessions": len(self.sessions),
            "cached_responses": len(self.response_cache),
            "vector_store": vector_status,
            "performance_metrics": self.performance_metrics.get_stats_dict(),
            "timestamp": datetime.now().isoformat()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能监控指标"""
        return self.performance_metrics.get_stats_dict()

    def reset_performance_metrics(self):
        """重置性能监控指标"""
        self.performance_metrics = PerformanceMetrics()
        logger.info("性能监控指标已重置")