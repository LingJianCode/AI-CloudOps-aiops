#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能助手FastAPI路由模块
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.decorators import api_response, log_api_call
from app.common.constants import ServiceConstants, AppConstants, ApiEndpoints
from app.common.response import ResponseWrapper
from app.services.assistant_service import OptimizedAssistantService

logger = logging.getLogger("aiops.api.assistant")

router = APIRouter(tags=["assistant"])
assistant_service = OptimizedAssistantService()

class AssistantQueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    session_id: Optional[str] = Field(None, description="会话ID")
    max_context_docs: int = Field(ServiceConstants.ASSISTANT_DEFAULT_CONTEXT_DOCS, description="最大上下文文档数", ge=1, le=ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS)

class SessionRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")


@router.post("/query", summary="智能助手查询")
@api_response("智能助手查询")
@log_api_call(log_request=True)
async def assistant_query(request: AssistantQueryRequest) -> Dict[str, Any]:
    await assistant_service.initialize()

    answer_result = await assistant_service.get_answer(
        question=request.question,
        session_id=request.session_id,
        max_context_docs=request.max_context_docs
    )

    return ResponseWrapper.success(
        data=answer_result,
        message="success"
    )


@router.get("/session/{session_id}", summary="获取会话信息")
@api_response("获取会话信息")
async def get_session_info(session_id: str) -> Dict[str, Any]:
    await assistant_service.initialize()
    
    session_info = await assistant_service.get_session_info(session_id)

    return ResponseWrapper.success(
        data=session_info,
        message="success"
    )


@router.post("/refresh", summary="刷新知识库")
@api_response("刷新知识库")
async def refresh_knowledge_base() -> Dict[str, Any]:
    await assistant_service.initialize()
    
    refresh_result = await assistant_service.refresh_knowledge_base()

    return ResponseWrapper.success(
        data=refresh_result,
        message="success"
    )


@router.get("/health", summary="智能助手健康检查")
@api_response("智能助手健康检查")
async def assistant_health() -> Dict[str, Any]:
    await assistant_service.initialize()
    
    health_status = await assistant_service.get_service_health_info()

    return ResponseWrapper.success(
        data=health_status,
        message="success"
    )


@router.get("/config", summary="获取智能助手配置")
@api_response("获取智能助手配置")
async def get_assistant_config() -> Dict[str, Any]:
    await assistant_service.initialize()
    
    config_info = await assistant_service.get_assistant_config()

    return ResponseWrapper.success(
        data=config_info,
        message="success"
    )


@router.get("/info", summary="企业级智能助手服务信息")
@api_response("企业级智能助手服务信息")
async def assistant_info() -> Dict[str, Any]:
    info = {
        "service": "企业级智能助手",
        "service_type": "enterprise_assistant",
        "workflow_engine": "langgraph",
        "version": AppConstants.APP_VERSION,
        "description": "基于LangGraph的企业级智能运维助手，提供高可用、高性能的问答服务",
        "capabilities": [
            "智能意图识别",
            "知识库问答",
            "故障排查指导",
            "操作步骤指导",
            "上下文对话",
            "文档检索",
            "智能推荐",
            "质量评估",
            "自动重试"
        ],
        "enterprise_features": [
            "LangGraph工作流引擎",
            "多级缓存机制",
            "智能质量评估",
            "自动错误恢复",
            "性能监控",
            "并行处理",
            "意图路由"
        ],
        "endpoints": {
            "query": ApiEndpoints.ASSISTANT_QUERY,
            "session": ApiEndpoints.ASSISTANT_SESSION,
            "refresh": ApiEndpoints.ASSISTANT_REFRESH,
            "health": ApiEndpoints.ASSISTANT_HEALTH,
            "config": "/assistant/config",
            "info": "/assistant/info"
        },
        "workflow_nodes": [
            "输入验证",
            "缓存检查",
            "意图识别",
            "知识检索",
            "上下文增强",
            "答案生成",
            "质量评估",
            "后处理",
            "缓存存储"
        ],
        "model_info": {
            "provider": "OpenAI/Ollama",
            "architecture": "LangGraph + LLM",
            "capabilities": ["text_generation", "conversation", "reasoning", "workflow_management"]
        },
        "constraints": {
            "max_question_length": 1000,
            "max_context_docs": ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS,
            "default_context_docs": ServiceConstants.ASSISTANT_DEFAULT_CONTEXT_DOCS,
            "timeout": ServiceConstants.ASSISTANT_TIMEOUT,
            "session_timeout": ServiceConstants.ASSISTANT_SESSION_TIMEOUT,
            "max_retries": 2,
            "quality_threshold": 0.7
        },
        "performance": {
            "caching_enabled": True,
            "parallel_processing": True,
            "error_recovery": True,
            "metrics_collection": True
        },
        "status": "available" if assistant_service else "unavailable",
        "initialized": assistant_service.is_initialized() if assistant_service else False
    }

    return ResponseWrapper.success(
        data=info,
        message="success"
    )


__all__ = ["router"]
