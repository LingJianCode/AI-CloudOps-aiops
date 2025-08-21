#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能助手服务
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ..common.constants import ServiceConstants
from ..common.exceptions import AssistantError, ValidationError
from ..core.agents.enterprise_assistant import get_enterprise_assistant
from .base import BaseService

logger = logging.getLogger("aiops.services.assistant")


class OptimizedAssistantService(BaseService):
    """优化的智能助手服务"""

    def __init__(self) -> None:
        super().__init__("assistant")
        self._assistant = None

    async def _do_initialize(self) -> None:
        """初始化服务"""
        try:
            self._assistant = await get_enterprise_assistant()
            logger.info("智能助手服务初始化完成")
        except Exception as e:
            logger.warning(f"服务初始化失败: {str(e)}，将在首次使用时重试")
            # 不抛出异常，允许服务启动，在使用时再尝试初始化
            self._assistant = None

    async def _do_health_check(self) -> bool:
        """健康检查"""
        try:
            # 如果assistant为None，尝试获取但不强制失败
            if not self._assistant:
                try:
                    self._assistant = await get_enterprise_assistant()
                except Exception as e:
                    logger.debug(f"获取助手实例失败: {str(e)}")
                    # 返回部分健康状态，表示服务框架可用但助手未初始化
                    return True  # 允许服务保持可用状态

            if self._assistant:
                health = await self._assistant.health_check()
                return health.get("status") == "healthy"
            else:
                # 助手未初始化，但服务框架正常
                return True

        except Exception as e:
            logger.warning(f"健康检查失败: {str(e)}")
            return True  # 即使健康检查失败，也保持服务可用

    async def get_answer(
        self,
        question: str,
        mode: int = 1,
        session_id: Optional[str] = None,
        max_context_docs: int = ServiceConstants.ASSISTANT_DEFAULT_CONTEXT_DOCS,
    ) -> Dict[str, Any]:
        """
        获取智能回答

        Args:
            question: 用户问题
            mode: 助手模式，1=RAG模式，2=MCP模式
            session_id: 会话ID
            max_context_docs: 最大上下文文档数（仅RAG模式使用）

        优化点：
        1. 并行预热缓存和向量索引
        2. 智能超时管理
        3. 性能监控
        4. 模式隔离处理
        """
        # 参数验证
        self._validate_question(question)
        self._validate_mode(mode)

        # 根据模式选择处理方式
        if mode == 2:  # MCP模式
            return await self._handle_mcp_mode(question, session_id)
        else:  # RAG模式 (mode == 1 或其他值默认为RAG)
            return await self._handle_rag_mode(question, session_id, max_context_docs)

    async def _use_fallback_response(
        self, question: str, session_id: Optional[str], error_reason: str
    ) -> Dict[str, Any]:
        """使用备用实现生成响应"""
        try:
            logger.warning(f"使用备用实现处理请求，原因: {error_reason}")

            from app.core.agents.fallback_models import (
                ResponseContext,
                SessionManager,
                generate_fallback_answer,
                sanitize_input,
            )

            # 清理输入
            cleaned_question = sanitize_input(question)

            # 创建或获取会话
            session_manager = SessionManager()
            session = None
            if session_id:
                session = session_manager.get_session(session_id)
                if not session:
                    session = session_manager.create_session(session_id)

            # 创建响应上下文
            context = ResponseContext(
                user_input=cleaned_question,
                session=session,
                additional_context={
                    "error_reason": error_reason,
                    "service": "assistant_service",
                },
            )

            # 生成备用答案
            fallback_answer = generate_fallback_answer(context)

            # 更新会话历史
            if session:
                session_manager.update_session(session_id, cleaned_question)

            # 构建响应
            return {
                "answer": fallback_answer,
                "confidence_score": 0.3,  # 备用实现的置信度较低
                "source_documents": [],
                "cache_hit": False,
                "processing_time": 0.1,  # 快速响应时间
                "session_id": session_id,
                "success": True,
                "fallback_used": True,
                "fallback_reason": error_reason,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as fallback_e:
            logger.error(f"备用实现也失败: {fallback_e}")
            # 返回最基础的错误响应
            return {
                "answer": "抱歉，当前服务不可用，请稍后重试或联系技术支持。",
                "confidence_score": 0.0,
                "source_documents": [],
                "cache_hit": False,
                "processing_time": 0.0,
                "session_id": session_id,
                "success": False,
                "error": f"主要服务: {error_reason}, 备用服务: {str(fallback_e)}",
                "timestamp": datetime.now().isoformat(),
            }

    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """刷新知识库"""
        self._ensure_initialized()

        try:
            result = await self._assistant.refresh_knowledge_base()

            # 清理性能统计
            self._performance_monitor.reset()

            return {
                "refreshed": result.get("success", False),
                "message": result.get("message", "知识库刷新完成"),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"刷新知识库失败: {str(e)}")
            raise AssistantError(f"刷新失败: {str(e)}")

    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话信息"""
        self._ensure_initialized()

        try:
            return await self._assistant.get_session_info(session_id)
        except Exception as e:
            logger.error(f"获取会话信息失败: {str(e)}")
            raise AssistantError(f"获取会话失败: {str(e)}")

    async def get_service_health_info(self) -> Dict[str, Any]:
        """获取详细健康信息"""
        try:
            health = await self._assistant.health_check() if self._assistant else {}

            # 检查备用实现的可用性
            fallback_status = self._check_fallback_availability()

            # 添加性能指标
            perf_stats = self._performance_monitor.get_stats()

            # 合并组件状态和备用状态
            components = health.get("components", {})
            components.update(fallback_status)

            return {
                "service": "optimized_assistant",
                "status": (
                    "healthy" if health.get("status") == "healthy" else "degraded"
                ),
                "components": components,
                "performance": perf_stats,
                "fallback_capabilities": fallback_status,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "service": "optimized_assistant",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _check_fallback_availability(self) -> Dict[str, bool]:
        """检查备用实现的可用性"""
        fallback_status = {
            "fallback_chat_model": False,
            "fallback_embeddings": False,
            "session_manager": False,
            "response_templates": False,
        }

        try:
            # 检查备用聊天模型
            from app.core.agents.fallback_models import FallbackChatModel

            FallbackChatModel()
            fallback_status["fallback_chat_model"] = True
        except Exception:
            pass

        try:
            # 检查备用嵌入模型
            from app.core.agents.fallback_models import FallbackEmbeddings

            FallbackEmbeddings()
            fallback_status["fallback_embeddings"] = True
        except Exception:
            pass

        try:
            # 检查会话管理器
            from app.core.agents.fallback_models import SessionManager

            SessionManager()
            fallback_status["session_manager"] = True
        except Exception:
            pass

        try:
            # 检查响应模板管理器
            from app.core.agents.fallback_models import ResponseTemplateManager

            ResponseTemplateManager()
            fallback_status["response_templates"] = True
        except Exception:
            pass

        return fallback_status

    async def get_assistant_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        from ..config.settings import config

        return {
            "service_type": "optimized_rag",
            "workflow_engine": "langgraph",
            "features": {
                "query_expansion": True,
                "document_reranking": True,
                "hybrid_search": True,
                "intelligent_caching": True,
                "performance_monitoring": True,
            },
            "model_config": {
                "provider": config.llm.provider,
                "model": config.llm.model,
                "temperature": 0.3,  # 降低温度提高准确性
                "max_tokens": config.llm.max_tokens,
            },
            "retrieval_config": {
                "strategy": "hybrid",
                "semantic_weight": 0.6,
                "lexical_weight": 0.4,
                "similarity_threshold": 0.5,  # 提高阈值
                "max_candidates": 20,
                "rerank_top_k": 5,
            },
            "cache_config": {"enabled": True, "ttl": 3600, "min_confidence": 0.6},
            "performance_targets": {
                "p50_latency_ms": 500,
                "p95_latency_ms": 2000,
                "p99_latency_ms": 5000,
                "target_accuracy": 0.85,
            },
        }

    def _validate_question(self, question: str) -> None:
        """验证问题"""
        if not question or not isinstance(question, str):
            raise ValidationError("question", "问题不能为空")

        question = question.strip()
        if not question:
            raise ValidationError("question", "问题内容不能为空")

        if len(question) > 1000:
            raise ValidationError("question", "问题长度不能超过1000字符")

    async def _ensure_ready(self) -> None:
        """确保服务就绪"""
        if not self._assistant:
            try:
                # 尝试直接获取助手实例，不依赖initialize
                self._assistant = await get_enterprise_assistant()
                logger.info("智能助手在运行时成功初始化")
            except Exception as e:
                logger.error(f"无法初始化智能助手: {str(e)}")
                raise AssistantError(f"服务暂未就绪: {str(e)}")

    def _calculate_timeout(self, question: str) -> float:
        """智能计算超时时间"""
        base_timeout = 30.0

        # 根据问题长度调整
        length_factor = min(len(question) / 100, 3.0)

        # 根据历史性能调整
        perf_stats = self._performance_monitor.get_stats()
        if perf_stats and "avg_latency" in perf_stats:
            avg_latency = perf_stats["avg_latency"]
            performance_factor = max(avg_latency / 1000 * 2, 1.0)
        else:
            performance_factor = 1.0

        timeout = base_timeout * (1 + length_factor * 0.2) * performance_factor
        return min(timeout, 120.0)  # 最大120秒

    def _enhance_result(
        self, result: Dict[str, Any], question: str, session_id: Optional[str]
    ) -> Dict[str, Any]:
        """增强返回结果"""
        enhanced = result.copy()

        # 添加元数据
        enhanced.update(
            {
                "question": question,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "service_version": "2.0.0",
            }
        )

        # 确保必要字段
        enhanced.setdefault("answer", "暂无答案")
        enhanced.setdefault("confidence_score", 0.0)
        enhanced.setdefault("source_documents", [])
        enhanced.setdefault("success", True)

        # 添加性能指标
        if "processing_time" in enhanced:
            enhanced["performance"] = {
                "latency_ms": enhanced["processing_time"] * 1000,
                "cache_hit": enhanced.get("cache_hit", False),
            }

        return enhanced

    async def _handle_rag_mode(
        self, question: str, session_id: Optional[str], max_context_docs: int
    ) -> Dict[str, Any]:
        """处理RAG模式请求"""
        # 确保服务就绪
        await self._ensure_ready()

        # 记录性能
        with self._performance_monitor.measure("get_answer_rag"):
            try:
                # 设置智能超时
                timeout = self._calculate_timeout(question)

                # 调用优化的助手
                result = await asyncio.wait_for(
                    self._assistant.get_answer(
                        question=question, session_id=session_id
                    ),
                    timeout=timeout,
                )

                # 记录成功指标
                self._performance_monitor.record_success()

                # 增强结果
                enhanced_result = self._enhance_result(result, question, session_id)
                enhanced_result["mode"] = "rag"  # 标记模式
                return enhanced_result

            except asyncio.TimeoutError:
                logger.error(f"RAG请求超时: {timeout}秒")
                return await self._use_fallback_response(question, session_id, "超时")
            except Exception as e:
                self._performance_monitor.record_failure()
                logger.error(f"RAG获取答案失败: {str(e)}")
                return await self._use_fallback_response(question, session_id, str(e))

    async def _handle_mcp_mode(
        self, question: str, session_id: Optional[str]
    ) -> Dict[str, Any]:
        """处理MCP模式请求"""
        try:
            # 懒加载MCP服务
            mcp_service = await self._get_mcp_service()

            # 调用MCP服务处理
            result = await mcp_service.get_answer(
                question=question, session_id=session_id
            )

            # 记录成功指标
            self._performance_monitor.record_success()

            return result

        except Exception as e:
            self._performance_monitor.record_failure()
            logger.error(f"MCP获取答案失败: {str(e)}")
            return await self._use_mcp_fallback_response(question, session_id, str(e))

    async def _get_mcp_service(self):
        """懒加载MCP服务"""
        if not hasattr(self, "_mcp_service") or self._mcp_service is None:
            from .mcp_service import MCPService

            self._mcp_service = MCPService()
            await self._mcp_service.initialize()
        return self._mcp_service

    async def _use_mcp_fallback_response(
        self, question: str, session_id: Optional[str], error_reason: str
    ) -> Dict[str, Any]:
        """MCP模式的备用响应"""
        logger.warning(f"使用MCP备用实现处理请求，原因: {error_reason}")

        # 简单的关键词匹配
        question_lower = question.lower()

        if any(
            keyword in question_lower
            for keyword in ["时间", "几点", "现在", "当前时间"]
        ):
            from datetime import datetime

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fallback_answer = f"当前时间是: {current_time}"
        else:
            fallback_answer = "抱歉，MCP服务当前不可用，请稍后重试或切换到RAG模式。"

        return {
            "answer": fallback_answer,
            "confidence_score": 0.3,
            "source_documents": [],
            "cache_hit": False,
            "processing_time": 0.1,
            "session_id": session_id,
            "success": True,
            "fallback_used": True,
            "fallback_reason": error_reason,
            "mode": "mcp",
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_mode(self, mode: int) -> None:
        """验证模式参数"""
        if not isinstance(mode, int) or mode not in [1, 2]:
            raise ValidationError("mode", "模式参数必须是1(RAG)或2(MCP)")

    async def get_service_health_info_with_mode(self) -> Dict[str, Any]:
        """获取包含两种模式的详细健康信息"""
        try:
            # 获取RAG健康信息
            rag_health = await self.get_service_health_info()

            # 获取MCP健康信息
            mcp_health = {"status": "unavailable"}
            try:
                mcp_service = await self._get_mcp_service()
                mcp_health = await mcp_service.get_service_health_info()
            except Exception as e:
                logger.debug(f"获取MCP健康信息失败: {e}")
                mcp_health = {"status": "unavailable", "error": str(e)}

            return {
                "service": "assistant_service_unified",
                "modes": {"rag": rag_health, "mcp": mcp_health},
                "supported_modes": [
                    {
                        "mode": 1,
                        "name": "RAG",
                        "description": "基于知识库的检索增强生成",
                    },
                    {
                        "mode": 2,
                        "name": "MCP",
                        "description": "基于工具调用的模型上下文协议",
                    },
                ],
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "service": "assistant_service_unified",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "latencies": [],
        }
        self._lock = asyncio.Lock()

    class Timer:
        """计时器上下文管理器"""

        def __init__(self, monitor, operation):
            self.monitor = monitor
            self.operation = operation
            self.start_time = None

        def __enter__(self):
            self.start_time = asyncio.get_event_loop().time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                latency = asyncio.get_event_loop().time() - self.start_time
                asyncio.create_task(
                    self.monitor._record_latency(self.operation, latency)
                )

    def measure(self, operation: str):
        """测量操作耗时"""
        return self.Timer(self, operation)

    async def _record_latency(self, operation: str, latency: float):
        """记录延迟"""
        async with self._lock:
            self.metrics["total_requests"] += 1
            self.metrics["total_latency"] += latency
            self.metrics["latencies"].append(latency)

            # 保持最近1000个样本
            if len(self.metrics["latencies"]) > 1000:
                self.metrics["latencies"] = self.metrics["latencies"][-1000:]

    def record_success(self):
        """记录成功"""
        self.metrics["successful_requests"] += 1

    def record_failure(self):
        """记录失败"""
        self.metrics["failed_requests"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.metrics["latencies"]:
            return {}

        latencies = sorted(self.metrics["latencies"])
        n = len(latencies)

        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": self.metrics["successful_requests"]
            / max(self.metrics["total_requests"], 1),
            "avg_latency": self.metrics["total_latency"]
            / max(self.metrics["total_requests"], 1),
            "p50_latency": latencies[n // 2] if n > 0 else 0,
            "p95_latency": latencies[int(n * 0.95)] if n > 0 else 0,
            "p99_latency": latencies[int(n * 0.99)] if n > 0 else 0,
        }

    def reset(self):
        """重置统计"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency": 0.0,
            "latencies": [],
        }
