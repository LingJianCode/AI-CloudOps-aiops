#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 基于LangGraph的企业级智能助手 - 提供高可用、高性能、可扩展的AI问答服务
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from app.config.settings import config
from app.constants import (
    ASSISTANT_CACHE_TTL_SECONDS,
    ASSISTANT_DEFAULT_MAX_CONTEXT_DOCS,
    ASSISTANT_CACHE_DEFAULT_TTL,
    ASSISTANT_RELEVANCE_CACHE_THRESHOLD
)
from app.services.llm import LLMService
from app.core.cache.redis_cache_manager import RedisCacheManager
from app.core.vector.redis_vector_store import OptimizedRedisVectorStore, RedisVectorStoreManager
from .fallback_models import FallbackEmbeddings, sanitize_input, validate_session_id

logger = logging.getLogger("aiops.enterprise_assistant")


class AssistantState(TypedDict):
    """企业级助手状态定义"""
    # 输入信息
    question: str
    session_id: Optional[str]
    max_context_docs: int
    
    # 处理过程
    cleaned_question: str
    intent_type: Optional[str]
    context_docs: List[Dict[str, Any]]
    chat_history: Annotated[List, add_messages]
    
    # 生成结果
    answer: str
    confidence_score: float
    source_documents: List[Dict[str, Any]]
    
    # 元数据
    cache_hit: bool
    processing_time: float
    error: Optional[str]
    retry_count: int
    
    # 质量评估
    quality_score: float
    needs_regeneration: bool
    
    # 性能指标
    retrieval_time: float
    generation_time: float
    total_time: float


class QualityMetrics(BaseModel):
    """质量评估指标"""
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EnterpriseAssistant:
    """企业级LangGraph智能助手"""
    
    def __init__(self):
        self.llm_service = None
        self.vector_store = None
        self.cache_manager = None
        self.embeddings = None
        self.graph = None
        self.memory_saver = MemorySaver()
        
        # 性能配置
        self.max_retries = 2
        self.quality_threshold = 0.7
        self.cache_threshold = ASSISTANT_RELEVANCE_CACHE_THRESHOLD
        
        # 状态管理
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        
        logger.info("企业级助手初始化开始")
    
    async def initialize(self) -> bool:
        """初始化企业级助手"""
        if self._initialized:
            return True
            
        async with self._initialization_lock:
            if self._initialized:  # 双重检查
                return True
                
            try:
                logger.info("开始初始化企业级助手组件...")
                
                # 初始化核心组件
                await self._initialize_llm_service()
                await self._initialize_cache_manager()
                await self._initialize_vector_store()
                
                # 构建LangGraph工作流
                self._build_graph()
                
                self._initialized = True
                logger.info("企业级助手初始化完成")
                return True
                
            except Exception as e:
                logger.error(f"企业级助手初始化失败: {str(e)}")
                return False
    
    async def _initialize_llm_service(self):
        """初始化LLM服务"""
        try:
            self.llm_service = LLMService()
            logger.info("LLM服务初始化完成")
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {str(e)}")
            raise
    
    async def _initialize_cache_manager(self):
        """初始化缓存管理器"""
        try:
            redis_config = {
                "host": config.redis.host,
                "port": config.redis.port,
                "db": config.redis.db + 1,  # 使用不同的数据库
                "password": config.redis.password,
                "connection_timeout": config.redis.connection_timeout,
                "socket_timeout": config.redis.socket_timeout,
                "max_connections": config.redis.max_connections,
                "decode_responses": config.redis.decode_responses,
            }
            
            self.cache_manager = RedisCacheManager(
                redis_config=redis_config,
                cache_prefix="enterprise_assistant:",
                default_ttl=ASSISTANT_CACHE_DEFAULT_TTL,
                enable_compression=True,
            )
            logger.info("缓存管理器初始化完成")
        except Exception as e:
            logger.warning(f"缓存管理器初始化失败，将使用内存缓存: {str(e)}")
            self.cache_manager = None
    
    async def _initialize_vector_store(self):
        """初始化向量存储"""
        try:
            # 初始化嵌入模型
            await self._initialize_embeddings()
            
            # 初始化Redis向量存储
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
            
            # 获取嵌入维度
            test_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, "测试"
            )
            vector_dim = len(test_embedding)
            
            self.vector_store = OptimizedRedisVectorStore(
                redis_config=redis_config,
                collection_name="enterprise_knowledge_base",
                embedding_model=self.embeddings,
                vector_dim=vector_dim,
                local_storage_path="/tmp/enterprise_vector_db",
                use_faiss=True,
                faiss_index_type="Flat"
            )
            
            logger.info("向量存储初始化完成")
        except Exception as e:
            logger.error(f"向量存储初始化失败: {str(e)}")
            raise
    
    async def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            provider = config.llm.provider.lower()
            
            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                    model=getattr(config.rag, "openai_embedding_model", "text-embedding-ada-002")
                )
            elif provider == "ollama":
                from langchain_ollama import OllamaEmbeddings
                self.embeddings = OllamaEmbeddings(
                    base_url=config.llm.ollama_base_url,
                    model=getattr(config.rag, "ollama_embedding_model", "nomic-embed-text")
                )
            else:
                self.embeddings = FallbackEmbeddings()
                
            logger.info(f"嵌入模型初始化完成: {provider}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {str(e)}")
            self.embeddings = FallbackEmbeddings()
    
    def _build_graph(self):
        """构建简化的LangGraph工作流 - 避免复杂循环"""
        # 创建状态图
        workflow = StateGraph(AssistantState)
        
        # 添加节点
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("check_cache", self._check_cache_node)
        workflow.add_node("recognize_intent", self._recognize_intent_node)
        workflow.add_node("retrieve_knowledge", self._retrieve_knowledge_node)
        workflow.add_node("enhance_context", self._enhance_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("post_process", self._post_process_node)
        workflow.add_node("store_cache", self._store_cache_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # 设置入口点
        workflow.set_entry_point("validate_input")
        
        # 简化的线性边 - 减少复杂的条件判断
        workflow.add_edge("validate_input", "check_cache")
        
        # 缓存检查的条件边
        workflow.add_conditional_edges(
            "check_cache",
            self._should_use_cache,
            {
                "cache_hit": END,
                "cache_miss": "recognize_intent",
                "error": "handle_error"
            }
        )
        
        # 简化意图识别 - 直接进入知识检索
        workflow.add_edge("recognize_intent", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "enhance_context")
        workflow.add_edge("enhance_context", "generate_answer")
        
        # 简化答案生成后的流程 - 直接进入后处理，避免复杂的重试逻辑
        workflow.add_conditional_edges(
            "generate_answer",
            self._route_after_generation,
            {
                "success": "post_process",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("post_process", "store_cache")
        workflow.add_edge("store_cache", END)
        workflow.add_edge("handle_error", END)
        
        # 编译图
        self.graph = workflow.compile(checkpointer=self.memory_saver)
        
        logger.info("简化LangGraph工作流构建完成")
    
    # ========== 工作流节点实现 ==========
    
    async def _validate_input_node(self, state: AssistantState) -> Dict[str, Any]:
        """输入验证节点"""
        try:
            question = state.get("question", "")
            session_id = state.get("session_id")
            max_context_docs = state.get("max_context_docs", ASSISTANT_DEFAULT_MAX_CONTEXT_DOCS)
            
            # 验证问题
            if not question or not isinstance(question, str):
                return {"error": "问题不能为空"}
            
            if len(question.strip()) == 0:
                return {"error": "问题内容不能为空"}
            
            if len(question) > 1000:
                return {"error": "问题长度不能超过1000字符"}
            
            # 验证会话ID
            if session_id and not validate_session_id(session_id):
                return {"error": "无效的会话ID"}
            
            # 验证上下文文档数量
            if not (1 <= max_context_docs <= 10):
                return {"error": "上下文文档数量必须在1-10之间"}
            
            # 清理输入
            cleaned_question = sanitize_input(question)
            
            return {
                "cleaned_question": cleaned_question,
                "retry_count": 0,
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"输入验证失败: {str(e)}")
            return {"error": f"输入验证失败: {str(e)}"}
    
    async def _check_cache_node(self, state: AssistantState) -> Dict[str, Any]:
        """缓存检查节点"""
        try:
            if not self.cache_manager:
                return {"cache_hit": False}
            
            cleaned_question = state.get("cleaned_question", "")
            session_id = state.get("session_id")
            
            # 生成缓存键
            cache_key = self._generate_cache_key(cleaned_question, session_id)
            
            # 检查缓存
            cached_result = await asyncio.to_thread(
                self.cache_manager.get, cache_key
            )
            
            if cached_result:
                logger.info(f"缓存命中: {cleaned_question[:50]}...")
                return {
                    "cache_hit": True,
                    "answer": cached_result.get("answer", ""),
                    "confidence_score": cached_result.get("confidence_score", 0.9),
                    "source_documents": cached_result.get("source_documents", []),
                    "total_time": time.time() - state.get("processing_time", time.time())
                }
            
            return {"cache_hit": False}
            
        except Exception as e:
            logger.error(f"缓存检查失败: {str(e)}")
            return {"cache_hit": False, "error": f"缓存检查失败: {str(e)}"}
    
    async def _recognize_intent_node(self, state: AssistantState) -> Dict[str, Any]:
        """意图识别节点"""
        try:
            cleaned_question = state.get("cleaned_question", "")
            
            # 简单的基于关键词的意图识别
            intent_type = self._classify_intent(cleaned_question)
            
            logger.debug(f"识别意图: {intent_type} for question: {cleaned_question[:50]}...")
            
            return {"intent_type": intent_type}
            
        except Exception as e:
            logger.error(f"意图识别失败: {str(e)}")
            return {"intent_type": "general", "error": f"意图识别失败: {str(e)}"}
    
    async def _retrieve_knowledge_node(self, state: AssistantState) -> Dict[str, Any]:
        """知识检索节点"""
        try:
            start_time = time.time()
            cleaned_question = state.get("cleaned_question", "")
            max_context_docs = state.get("max_context_docs", ASSISTANT_DEFAULT_MAX_CONTEXT_DOCS)
            intent_type = state.get("intent_type", "general")
            
            if not self.vector_store:
                logger.warning("向量存储不可用")
                return {
                    "context_docs": [],
                    "retrieval_time": time.time() - start_time
                }
            
            # 根据意图调整检索策略
            search_query = self._enhance_query_by_intent(cleaned_question, intent_type)
            
            # 执行向量搜索
            results = await asyncio.to_thread(
                self.vector_store.hybrid_similarity_search,
                query=search_query,
                k=max_context_docs * 2,  # 检索更多候选
                similarity_threshold=0.1
            )
            
            # 转换为标准格式
            context_docs = []
            for doc, score in results[:max_context_docs]:
                context_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            retrieval_time = time.time() - start_time
            
            logger.debug(f"检索到 {len(context_docs)} 个相关文档，耗时 {retrieval_time:.3f}s")
            
            return {
                "context_docs": context_docs,
                "retrieval_time": retrieval_time
            }
            
        except Exception as e:
            logger.error(f"知识检索失败: {str(e)}")
            return {
                "context_docs": [],
                "retrieval_time": time.time() - start_time,
                "error": f"知识检索失败: {str(e)}"
            }
    
    async def _enhance_context_node(self, state: AssistantState) -> Dict[str, Any]:
        """上下文增强节点"""
        try:
            context_docs = state.get("context_docs", [])
            session_id = state.get("session_id")
            
            # 构建上下文
            enhanced_context = self._build_enhanced_context(context_docs)
            
            # 获取会话历史（如果需要）
            chat_history = []
            if session_id and self.cache_manager:
                history_key = f"session_history:{session_id}"
                cached_history = await asyncio.to_thread(
                    self.cache_manager.get, history_key
                )
                if cached_history:
                    chat_history = cached_history.get("messages", [])[-6:]  # 最近3轮对话
            
            return {
                "chat_history": chat_history,
                "enhanced_context": enhanced_context
            }
            
        except Exception as e:
            logger.error(f"上下文增强失败: {str(e)}")
            return {"error": f"上下文增强失败: {str(e)}"}
    
    async def _generate_answer_node(self, state: AssistantState) -> Dict[str, Any]:
        """简化的答案生成节点 - 避免过于复杂的逻辑"""
        try:
            start_time = time.time()
            
            cleaned_question = state.get("cleaned_question", "")
            context_docs = state.get("context_docs", [])
            chat_history = state.get("chat_history", [])
            intent_type = state.get("intent_type", "general")
            
            if not self.llm_service:
                return {"error": "LLM服务不可用"}
            
            # 构建提示词
            prompt = self._build_prompt(cleaned_question, context_docs, chat_history, intent_type)
            
            # 调用LLM生成答案
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_service.generate_response(
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            answer = response if isinstance(response, str) else response.get("content", "抱歉，暂时无法生成回答")
            
            # 计算置信度
            confidence_score = self._calculate_confidence(context_docs, answer)
            
            generation_time = time.time() - start_time
            
            logger.debug(f"答案生成完成，耗时 {generation_time:.3f}s，置信度 {confidence_score:.3f}")
            
            # 简化返回，直接包含最终结果
            return {
                "answer": answer,
                "confidence_score": confidence_score,
                "generation_time": generation_time,
                "quality_score": confidence_score,  # 简化：直接使用置信度作为质量分数
                "source_documents": [
                    {
                        "content": doc.get("page_content", "")[:200] + "...",
                        "source": doc.get("metadata", {}).get("source", "unknown"),
                        "score": doc.get("score", 0.0)
                    } for doc in context_docs
                ]
            }
            
        except Exception as e:
            logger.error(f"答案生成失败: {str(e)}")
            return {
                "answer": f"抱歉，答案生成时出现错误: {str(e)}",
                "confidence_score": 0.0,
                "generation_time": time.time() - start_time,
                "error": f"答案生成失败: {str(e)}"
            }
    
    async def _assess_quality_node(self, state: AssistantState) -> Dict[str, Any]:
        """质量评估节点"""
        try:
            answer = state.get("answer", "")
            context_docs = state.get("context_docs", [])
            confidence_score = state.get("confidence_score", 0.0)
            retry_count = state.get("retry_count", 0)
            
            # 评估答案质量
            quality_metrics = self._evaluate_answer_quality(answer, context_docs, confidence_score)
            
            quality_score = quality_metrics.overall_score
            
            # 检查是否需要重新生成（质量不达标且未超过重试次数）
            needs_regeneration = (
                quality_score < self.quality_threshold and 
                retry_count < self.max_retries
            )
            
            # 如果需要重新生成，增加重试计数
            updated_retry_count = retry_count + 1 if needs_regeneration else retry_count
            
            logger.debug(f"质量评估: {quality_score:.3f}, 重试次数: {retry_count}/{self.max_retries}, 需要重新生成: {needs_regeneration}")
            
            return {
                "quality_score": quality_score,
                "needs_regeneration": needs_regeneration,
                "retry_count": updated_retry_count
            }
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            return {
                "quality_score": 0.5,  # 默认中等质量
                "needs_regeneration": False,
                "error": f"质量评估失败: {str(e)}"
            }
    
    async def _post_process_node(self, state: AssistantState) -> Dict[str, Any]:
        """简化的后处理节点"""
        try:
            answer = state.get("answer", "")
            
            # 格式化答案
            formatted_answer = self._format_answer(answer)
            
            # 计算总耗时
            total_time = time.time() - state.get("processing_time", time.time())
            
            # 确保source_documents存在
            source_documents = state.get("source_documents", [])
            
            return {
                "answer": formatted_answer,
                "source_documents": source_documents,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"后处理失败: {str(e)}")
            return {"error": f"后处理失败: {str(e)}"}
    
    async def _store_cache_node(self, state: AssistantState) -> Dict[str, Any]:
        """缓存存储节点"""
        try:
            if not self.cache_manager:
                return {}
            
            cleaned_question = state.get("cleaned_question", "")
            session_id = state.get("session_id")
            answer = state.get("answer", "")
            confidence_score = state.get("confidence_score", 0.0)
            source_documents = state.get("source_documents", [])
            
            # 只缓存高质量答案
            if confidence_score >= self.cache_threshold:
                cache_key = self._generate_cache_key(cleaned_question, session_id)
                cache_data = {
                    "answer": answer,
                    "confidence_score": confidence_score,
                    "source_documents": source_documents,
                    "cached_at": datetime.now().isoformat()
                }
                
                await asyncio.to_thread(
                    self.cache_manager.set,
                    cache_key, 
                    cache_data,
                    ttl=ASSISTANT_CACHE_TTL_SECONDS
                )
                
                logger.debug(f"答案已缓存: {cleaned_question[:50]}...")
            
            # 更新会话历史
            if session_id:
                await self._update_session_history(session_id, cleaned_question, answer)
            
            return {}
            
        except Exception as e:
            logger.error(f"缓存存储失败: {str(e)}")
            return {"error": f"缓存存储失败: {str(e)}"}
    
    async def _handle_error_node(self, state: AssistantState) -> Dict[str, Any]:
        """错误处理节点"""
        try:
            error = state.get("error", "未知错误")
            
            logger.error(f"工作流错误: {error}")
            
            # 返回友好的错误响应
            return {
                "answer": f"抱歉，处理您的问题时遇到了问题：{error}",
                "confidence_score": 0.0,
                "source_documents": [],
                "error": error,
                "total_time": time.time() - state.get("processing_time", time.time())
            }
            
        except Exception as e:
            logger.error(f"错误处理失败: {str(e)}")
            return {
                "answer": "抱歉，系统遇到了未知错误",
                "error": str(e)
            }
    
    # ========== 条件判断函数 ==========
    
    def _should_use_cache(self, state: AssistantState) -> Literal["cache_hit", "cache_miss", "error"]:
        """判断是否使用缓存"""
        if state.get("error"):
            return "error"
        elif state.get("cache_hit"):
            return "cache_hit"
        else:
            return "cache_miss"
    
    def _route_after_generation(self, state: AssistantState) -> Literal["success", "error"]:
        """答案生成后的简单路由 - 避免复杂循环"""
        if state.get("error"):
            return "error"
        else:
            return "success"
    
    # ========== 辅助方法 ==========
    
    def _classify_intent(self, question: str) -> str:
        """简单的意图分类"""
        question_lower = question.lower()
        
        # 故障排查相关关键词
        troubleshooting_keywords = [
            "错误", "异常", "故障", "问题", "失败", "无法", "不能", 
            "报错", "crash", "error", "exception", "fault", "fail"
        ]
        
        # 操作指导相关关键词
        operation_keywords = [
            "如何", "怎么", "怎样", "怎么样", "步骤", "方法", "教程",
            "how", "step", "guide", "tutorial", "instruction"
        ]
        
        # Kubernetes/运维相关关键词
        k8s_keywords = [
            "kubernetes", "k8s", "pod", "deployment", "service", "namespace",
            "docker", "容器", "集群", "节点", "监控", "日志"
        ]
        
        if any(keyword in question_lower for keyword in troubleshooting_keywords):
            return "troubleshooting"
        elif any(keyword in question_lower for keyword in operation_keywords):
            return "operation_guide"
        elif any(keyword in question_lower for keyword in k8s_keywords):
            return "knowledge_qa"
        else:
            return "general"
    
    def _enhance_query_by_intent(self, question: str, intent_type: str) -> str:
        """根据意图增强查询"""
        if intent_type == "troubleshooting":
            return f"故障排查 问题解决 {question}"
        elif intent_type == "operation_guide":
            return f"操作指南 步骤教程 {question}"
        elif intent_type == "knowledge_qa":
            return f"知识问答 技术文档 {question}"
        else:
            return question
    
    def _build_enhanced_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """构建增强的上下文"""
        if not context_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(context_docs):
            content = doc.get("page_content", "")
            source = doc.get("metadata", {}).get("source", f"文档{i+1}")
            score = doc.get("score", 0.0)
            
            context_parts.append(f"[文档{i+1} - {source} (相关度: {score:.2f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context_docs: List[Dict[str, Any]], 
                     chat_history: List, intent_type: str) -> str:
        """构建提示词"""
        context = self._build_enhanced_context(context_docs)
        
        # 根据意图定制提示词
        if intent_type == "troubleshooting":
            prompt_prefix = "你是一个专业的运维故障排查专家。请基于提供的文档内容，帮助用户分析和解决问题。"
        elif intent_type == "operation_guide":
            prompt_prefix = "你是一个经验丰富的运维工程师。请基于提供的文档内容，为用户提供详细的操作指导。"
        elif intent_type == "knowledge_qa":
            prompt_prefix = "你是一个知识渊博的技术专家。请基于提供的文档内容，准确回答用户的技术问题。"
        else:
            prompt_prefix = "你是一个智能助手。请基于提供的文档内容，帮助用户解答问题。"
        
        # 构建历史对话上下文
        history_context = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-4:]:  # 最近2轮对话
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    history_parts.append(f"{role}: {content}")
            if history_parts:
                history_context = f"\n\n历史对话:\n{chr(10).join(history_parts)}"
        
        prompt = f"""{prompt_prefix}

{history_context}

参考文档:
{context}

用户问题: {question}

请基于以上文档内容回答用户问题。要求:
1. 回答要准确、专业、有条理
2. 如果文档中没有相关信息，请诚实说明
3. 适当引用文档来源以增加可信度
4. 保持回答的简洁性和实用性

回答:"""
        
        return prompt
    
    def _calculate_confidence(self, context_docs: List[Dict[str, Any]], answer: str) -> float:
        """计算置信度"""
        if not context_docs or not answer:
            return 0.0
        
        # 基于文档相关度计算置信度
        avg_score = sum(doc.get("score", 0.0) for doc in context_docs) / len(context_docs)
        
        # 基于答案长度调整（太短或太长的答案置信度较低）
        answer_length = len(answer)
        length_factor = 1.0
        if answer_length < 50:
            length_factor = 0.8
        elif answer_length > 2000:
            length_factor = 0.9
        
        # 综合计算置信度
        confidence = min(avg_score * 2 * length_factor, 1.0)
        
        return max(confidence, 0.1)  # 最低置信度为0.1
    
    def _evaluate_answer_quality(self, answer: str, context_docs: List[Dict[str, Any]], 
                                confidence_score: float) -> QualityMetrics:
        """评估答案质量"""
        metrics = QualityMetrics()
        
        # 相关性评分（基于上下文文档）
        if context_docs:
            metrics.relevance_score = min(
                sum(doc.get("score", 0.0) for doc in context_docs) / len(context_docs) * 2,
                1.0
            )
        else:
            metrics.relevance_score = 0.3  # 无上下文时的默认分数
        
        # 连贯性评分（基于答案结构）
        if answer and len(answer.strip()) > 0:
            # 简单检查：句子完整性、长度合理性
            sentences = answer.split('。')
            if len(sentences) >= 2 and len(answer) >= 30:
                metrics.coherence_score = 0.8
            elif len(answer) >= 10:
                metrics.coherence_score = 0.6
            else:
                metrics.coherence_score = 0.3
        else:
            metrics.coherence_score = 0.0
        
        # 完整性评分（基于答案长度和结构）
        if len(answer) >= 100:
            metrics.completeness_score = 0.9
        elif len(answer) >= 50:
            metrics.completeness_score = 0.7
        elif len(answer) >= 20:
            metrics.completeness_score = 0.5
        else:
            metrics.completeness_score = 0.3
        
        # 置信度评分
        metrics.confidence_score = confidence_score
        
        # 综合评分
        metrics.overall_score = (
            metrics.relevance_score * 0.3 +
            metrics.coherence_score * 0.25 +
            metrics.completeness_score * 0.25 +
            metrics.confidence_score * 0.2
        )
        
        return metrics
    
    def _format_answer(self, answer: str) -> str:
        """格式化答案"""
        if not answer:
            return "抱歉，暂时无法提供回答。"
        
        # 简单的格式化处理
        formatted = answer.strip()
        
        # 确保答案以句号结尾
        if formatted and not formatted.endswith(('。', '.', '!', '?', '！', '？')):
            formatted += '。'
        
        return formatted
    
    def _generate_cache_key(self, question: str, session_id: Optional[str]) -> str:
        """生成缓存键"""
        # 使用问题内容和会话ID生成唯一键
        content = f"{question}:{session_id or 'anonymous'}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def _update_session_history(self, session_id: str, question: str, answer: str):
        """更新会话历史"""
        if not self.cache_manager:
            return
        
        try:
            history_key = f"session_history:{session_id}"
            
            # 获取现有历史
            existing_history = await asyncio.to_thread(
                self.cache_manager.get, history_key
            ) or {"messages": []}
            
            # 添加新的问答对
            messages = existing_history.get("messages", [])
            messages.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer[:500]}  # 限制长度
            ])
            
            # 保持历史记录数量
            if len(messages) > 20:  # 最多保留10轮对话
                messages = messages[-20:]
            
            # 更新缓存
            await asyncio.to_thread(
                self.cache_manager.set,
                history_key,
                {"messages": messages, "updated_at": datetime.now().isoformat()},
                ttl=3600  # 1小时过期
            )
            
        except Exception as e:
            logger.error(f"更新会话历史失败: {str(e)}")
    
    # ========== 公共接口 ==========
    
    async def get_answer(self, question: str, session_id: Optional[str] = None, 
                        max_context_docs: int = ASSISTANT_DEFAULT_MAX_CONTEXT_DOCS) -> Dict[str, Any]:
        """获取问题答案 - 主要接口"""
        if not self._initialized:
            if not await self.initialize():
                return {
                    "answer": "智能助手暂时不可用，请稍后重试",
                    "error": "初始化失败",
                    "success": False
                }
        
        try:
            # 构建初始状态
            initial_state = {
                "question": question,
                "session_id": session_id,
                "max_context_docs": max_context_docs,
                "processing_time": time.time()
            }
            
            # 运行LangGraph工作流，设置递归限制
            config = {
                "thread_id": session_id or "anonymous",
                "recursion_limit": 50  # 增加递归限制
            }
            result = await self.graph.ainvoke(initial_state, config=config)
            
            # 构建响应
            response = {
                "answer": result.get("answer", "抱歉，暂时无法提供回答"),
                "confidence_score": result.get("confidence_score", 0.0),
                "source_documents": result.get("source_documents", []),
                "processing_time": result.get("total_time", 0.0),
                "cache_hit": result.get("cache_hit", False),
                "success": not bool(result.get("error")),
                "timestamp": datetime.now().isoformat()
            }
            
            if result.get("error"):
                response["error"] = result["error"]
            
            logger.info(f"问答完成: {question[:50]}... -> {len(response['answer'])}字符")
            
            return response
            
        except Exception as e:
            logger.error(f"获取答案失败: {str(e)}")
            return {
                "answer": f"抱歉，处理问题时出现错误: {str(e)}",
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy" if self._initialized else "initializing",
            "initialized": self._initialized,
            "components": {
                "llm_service": bool(self.llm_service),
                "vector_store": bool(self.vector_store),
                "cache_manager": bool(self.cache_manager),
                "graph": bool(self.graph)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """刷新知识库"""
        try:
            # 重新初始化向量存储
            await self._initialize_vector_store()
            
            # 清空相关缓存
            if self.cache_manager:
                await asyncio.to_thread(
                    self.cache_manager.clear_pattern, "enterprise_assistant:*"
                )
            
            return {
                "success": True,
                "message": "知识库刷新完成",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"刷新知识库失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        if not self.cache_manager:
            return {
                "session_id": session_id,
                "status": "cache_unavailable",
                "message": "缓存服务不可用"
            }
        
        try:
            history_key = f"session_history:{session_id}"
            session_data = await asyncio.to_thread(
                self.cache_manager.get, history_key
            )
            
            if session_data:
                return {
                    "session_id": session_id,
                    "status": "active",
                    "message_count": len(session_data.get("messages", [])),
                    "last_updated": session_data.get("updated_at"),
                    "messages": session_data.get("messages", [])[-10:]  # 最近5轮对话
                }
            else:
                return {
                    "session_id": session_id,
                    "status": "not_found",
                    "message": "会话未找到或已过期"
                }
                
        except Exception as e:
            logger.error(f"获取会话信息失败: {str(e)}")
            return {
                "session_id": session_id,
                "status": "error",
                "error": str(e)
            }


# ========== 全局实例管理 ==========

_enterprise_assistant_instance = None
_instance_lock = asyncio.Lock()


async def get_enterprise_assistant() -> EnterpriseAssistant:
    """获取企业级助手全局实例"""
    global _enterprise_assistant_instance
    
    if _enterprise_assistant_instance is None:
        async with _instance_lock:
            if _enterprise_assistant_instance is None:
                _enterprise_assistant_instance = EnterpriseAssistant()
                await _enterprise_assistant_instance.initialize()
    
    return _enterprise_assistant_instance


def reset_enterprise_assistant():
    """重置企业级助手实例（用于测试或重新初始化）"""
    global _enterprise_assistant_instance
    _enterprise_assistant_instance = None
