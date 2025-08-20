#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 企业级智能助手服务 - 基于LangGraph的高可用智能问答服务
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .base import BaseService
from ..common.constants import ServiceConstants
from ..common.exceptions import AssistantError, ValidationError
from ..core.agents.enterprise_assistant import get_enterprise_assistant

logger = logging.getLogger("aiops.services.assistant")


class AssistantService(BaseService):
    """
    企业级智能助手服务 - 基于LangGraph的智能问答和对话管理
    """
    
    def __init__(self) -> None:
        super().__init__("assistant")
        self._enterprise_assistant = None
    
    async def _do_initialize(self) -> None:
        """初始化企业级智能助手服务"""
        try:
            # 获取企业级助手实例
            self._enterprise_assistant = await get_enterprise_assistant()
            self.logger.info("企业级智能助手服务初始化完成")
        except Exception as e:
            self.logger.error(f"企业级智能助手服务初始化失败: {str(e)}")
            self._enterprise_assistant = None
            raise
    
    async def _do_health_check(self) -> bool:
        """企业级智能助手服务健康检查"""
        try:
            if not self._enterprise_assistant:
                self._enterprise_assistant = await get_enterprise_assistant()
            
            if not self._enterprise_assistant:
                return False
            
            # 检查企业级助手的健康状态
            health_result = await self.execute_with_timeout(
                self._enterprise_assistant.health_check(),
                timeout=30.0,
                operation_name="enterprise_assistant_health_check"
            )
            
            return health_result.get("status") == "healthy"
            
        except Exception as e:
            self.logger.warning(f"企业级智能助手健康检查失败: {str(e)}")
            return False
    
    async def get_answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        max_context_docs: int = ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS
    ) -> Dict[str, Any]:
        """
        获取企业级智能助手的回答 - 基于LangGraph工作流
        
        Args:
            question: 用户问题
            session_id: 会话ID
            max_context_docs: 最大上下文文档数量
            
        Returns:
            回答结果字典
            
        Raises:
            ValidationError: 参数验证失败
            AssistantError: 助手服务失败
        """
        # 验证输入参数
        self._validate_query_params(question, max_context_docs)
        
        # 确保企业级助手已初始化
        if not self._enterprise_assistant:
            await self.initialize()
            if not self._enterprise_assistant:
                raise AssistantError("企业级智能助手暂未就绪，请稍后重试")
        
        try:
            # 调用企业级助手获取答案，使用360秒超时
            result = await self.execute_with_timeout(
                self._enterprise_assistant.get_answer(
                    question=question,
                    session_id=session_id,
                    max_context_docs=max_context_docs
                ),
                timeout=360,  # 使用360秒超时，符合配置要求
                operation_name="enterprise_get_answer"
            )
            
            # 包装结果
            return self._wrap_answer_result(result, question, session_id)
            
        except Exception as e:
            self.logger.error(f"获取企业级智能助手回答失败: {str(e)}")
            if isinstance(e, (ValidationError, AssistantError)):
                raise e
            raise AssistantError(f"获取回答失败: {str(e)}")
    
    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """
        刷新企业级智能助手知识库
        
        Returns:
            刷新结果字典
            
        Raises:
            AssistantError: 刷新失败
        """
        self._ensure_initialized()
        
        # 确保企业级助手可用
        if not self._enterprise_assistant:
            await self.initialize()
            if not self._enterprise_assistant:
                raise AssistantError("企业级智能助手暂未就绪，无法刷新知识库")
        
        try:
            # 调用企业级助手刷新知识库
            refresh_result = await self.execute_with_timeout(
                self._enterprise_assistant.refresh_knowledge_base(),
                timeout=120.0,  # 2分钟超时
                operation_name="enterprise_refresh_knowledge"
            )
            
            return {
                "refreshed": refresh_result.get("success", True),
                "result": refresh_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"刷新知识库失败: {str(e)}")
            raise AssistantError(f"刷新知识库失败: {str(e)}")
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        获取企业级智能助手会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话信息字典
            
        Raises:
            AssistantError: 获取会话信息失败
        """
        self._ensure_initialized()
        
        # 确保企业级助手可用
        if not self._enterprise_assistant:
            await self.initialize()
            if not self._enterprise_assistant:
                raise AssistantError("企业级智能助手暂未就绪")
        
        try:
            # 获取会话信息
            session_info = await self.execute_with_timeout(
                self._enterprise_assistant.get_session_info(session_id),
                timeout=30.0,
                operation_name="enterprise_get_session_info"
            )
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"获取会话信息失败: {str(e)}")
            raise AssistantError(f"获取会话信息失败: {str(e)}")
    
    async def get_assistant_config(self) -> Dict[str, Any]:
        """
        获取企业级智能助手配置信息
        
        Returns:
            配置信息字典
        """
        from ..config.settings import config
        
        config_info = {
            "service_type": "enterprise_assistant",
            "workflow_engine": "langgraph",
            "model_config": {
                "provider": getattr(config.llm, 'provider', 'openai'),
                "model": getattr(config.llm, 'model', 'gpt-3.5-turbo'),
                "temperature": getattr(config.llm, 'temperature', 0.7),
                "max_tokens": getattr(config.llm, 'max_tokens', 2000)
            },
            "retrieval_config": {
                "max_context_docs": ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS,
                "default_context_docs": ServiceConstants.ASSISTANT_DEFAULT_CONTEXT_DOCS,
                "vector_store": "redis",
                "similarity_threshold": getattr(config.rag, 'similarity_threshold', 0.7),
                "hybrid_search": True
            },
            "session_config": {
                "timeout": ServiceConstants.ASSISTANT_SESSION_TIMEOUT,
                "max_history": getattr(config.rag, 'max_history', 10),
                "auto_cleanup": True,
                "cache_enabled": True
            },
            "workflow_config": {
                "intent_recognition": True,
                "quality_assessment": True,
                "max_retries": 2,
                "quality_threshold": 0.7,
                "cache_threshold": 0.75
            },
            "constraints": {
                "max_question_length": getattr(config, 'max_question_length', 1000),
                "timeout": ServiceConstants.ASSISTANT_TIMEOUT,
                "max_context_docs": ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS
            },
            "knowledge_base": {
                "enabled": True,
                "auto_refresh": getattr(config.rag, 'auto_refresh', False),
                "sources": ["documents", "faqs", "troubleshooting", "operational_guides"]
            },
            "performance_features": {
                "caching": True,
                "parallel_processing": True,
                "error_recovery": True,
                "metrics_collection": True
            }
        }
        
        return config_info
    
    async def get_service_health_info(self) -> Dict[str, Any]:
        """
        获取企业级智能助手服务详细健康信息
        
        Returns:
            健康信息字典
        """
        try:
            health_status = {
                "service": "enterprise_assistant",
                "service_type": "langgraph_workflow",
                "status": ServiceConstants.STATUS_HEALTHY if await self.health_check() else ServiceConstants.STATUS_UNHEALTHY,
                "timestamp": datetime.now().isoformat(),
                "assistant_available": False,
                "assistant_initialized": False,
                "components": {
                    "llm_service": "unknown",
                    "vector_store": "unknown",
                    "cache_manager": "unknown",
                    "langgraph_workflow": "unknown"
                }
            }

            # 检查企业级助手状态
            if not self._enterprise_assistant:
                try:
                    self._enterprise_assistant = await get_enterprise_assistant()
                except Exception:
                    pass
            
            if self._enterprise_assistant:
                health_status["assistant_available"] = True
                health_status["assistant_initialized"] = self._enterprise_assistant._initialized
                
                # 获取详细健康信息
                try:
                    detailed_health = await self._enterprise_assistant.health_check()
                    health_status.update({
                        "detailed_status": detailed_health,
                        "components": detailed_health.get("components", {})
                    })
                    
                    # 检查LangGraph工作流
                    if self._enterprise_assistant.graph:
                        health_status["components"]["langgraph_workflow"] = ServiceConstants.STATUS_HEALTHY
                    else:
                        health_status["components"]["langgraph_workflow"] = ServiceConstants.STATUS_UNHEALTHY
                        
                except Exception as e:
                    self.logger.warning(f"获取详细健康信息失败: {str(e)}")
                    health_status["components"]["langgraph_workflow"] = ServiceConstants.STATUS_UNHEALTHY

            return health_status
            
        except Exception as e:
            self.logger.error(f"获取企业级智能助手服务健康信息失败: {str(e)}")
            return {
                "service": "enterprise_assistant",
                "status": ServiceConstants.STATUS_UNHEALTHY,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_query_params(self, question: str, max_context_docs: int) -> None:
        """
        验证查询参数
        
        Args:
            question: 问题内容
            max_context_docs: 最大上下文文档数
            
        Raises:
            ValidationError: 参数验证失败
        """
        if not question or not isinstance(question, str):
            raise ValidationError("question", "问题内容不能为空")
        
        if len(question.strip()) == 0:
            raise ValidationError("question", "问题内容不能为空")
        
        if len(question) > 1000:  # 可以从配置获取
            raise ValidationError("question", "问题内容长度不能超过1000字符")
        
        if not (1 <= max_context_docs <= ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS):
            raise ValidationError(
                "max_context_docs",
                f"上下文文档数量必须在 1-{ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS} 之间"
            )
    
    def _wrap_answer_result(
        self,
        result: Any,
        question: str,
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        包装企业级助手回答结果
        
        Args:
            result: 企业级助手原始结果
            question: 原始问题
            session_id: 会话ID
            
        Returns:
            标准化的回答结果
        """
        if isinstance(result, dict):
            wrapped_result = result.copy()
        else:
            wrapped_result = {"answer": str(result)}
        
        # 添加元数据
        wrapped_result.update({
            "question": question,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "service": "enterprise_assistant",
            "workflow_engine": "langgraph"
        })
        
        # 确保包含必要字段
        if "answer" not in wrapped_result:
            wrapped_result["answer"] = "暂无回答"
        
        # 添加企业级特性标识
        wrapped_result.setdefault("success", True)
        wrapped_result.setdefault("confidence_score", 0.0)
        wrapped_result.setdefault("source_documents", [])
        wrapped_result.setdefault("cache_hit", False)
        wrapped_result.setdefault("processing_time", 0.0)
        
        return wrapped_result
