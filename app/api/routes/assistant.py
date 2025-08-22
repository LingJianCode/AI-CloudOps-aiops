#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能助手API接口
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.api.decorators import api_response, log_api_call
from app.common.constants import (
    ApiEndpoints,
    AppConstants,
    ErrorMessages,
    HttpStatusCodes,
    ServiceConstants,
)
from app.common.response import ResponseWrapper
from app.models import AssistantRequest, AssistantResponse, SessionInfoResponse
from app.services.assistant_service import OptimizedAssistantService

logger = logging.getLogger("aiops.api.assistant")

router = APIRouter(tags=["assistant"])
assistant_service = OptimizedAssistantService()


@router.post("/query", summary="智能助手查询")
@api_response("智能助手查询")
@log_api_call(log_request=True)
async def assistant_query(request: AssistantRequest) -> Dict[str, Any]:
    await assistant_service.initialize()

    answer_result = await assistant_service.get_answer(
        question=request.question,
        mode=request.mode,
        session_id=request.session_id,
        max_context_docs=request.max_context_docs,
    )

    # 使用统一的响应模型
    response = AssistantResponse(
        answer=answer_result.get("answer", ""),
        source_documents=answer_result.get("source_documents"),
        relevance_score=answer_result.get("relevance_score"),
        recall_rate=answer_result.get("recall_rate"),
        follow_up_questions=answer_result.get("follow_up_questions"),
        session_id=answer_result.get("session_id"),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/session/{session_id}", summary="获取会话信息")
@api_response("获取会话信息")
async def get_session_info(session_id: str) -> Dict[str, Any]:
    await assistant_service.initialize()

    session_info = await assistant_service.get_session_info(session_id)

    # 使用统一的响应模型
    response = SessionInfoResponse(
        session_id=session_id,
        created_time=session_info.get("created_time", ""),
        last_activity=session_info.get("last_activity", ""),
        message_count=session_info.get("message_count", 0),
        mode=session_info.get("mode", 1),
        status=session_info.get("status", "active"),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.post("/refresh", summary="刷新知识库")
@api_response("刷新知识库")
async def refresh_knowledge_base() -> Dict[str, Any]:
    await assistant_service.initialize()

    refresh_result = await assistant_service.refresh_knowledge_base()

    return ResponseWrapper.success(data=refresh_result, message="success")


@router.get("/health", summary="智能助手健康检查")
@api_response("智能助手健康检查")
async def assistant_health() -> Dict[str, Any]:
    await assistant_service.initialize()

    # 使用新的包含两种模式的健康检查
    health_status = await assistant_service.get_service_health_info_with_mode()

    return ResponseWrapper.success(data=health_status, message="success")


@router.get("/ready", summary="智能助手就绪检查")
@api_response("智能助手就绪检查")
async def assistant_ready() -> Dict[str, Any]:
    """检查智能助手是否已就绪可以提供服务"""
    try:
        await assistant_service.initialize()
        health_status = await assistant_service.get_service_health_info_with_mode()

        # 检查服务是否健康
        is_ready = (
            health_status.get("modes", {}).get("rag", {}).get("status") == "healthy"
            or health_status.get("modes", {}).get("mcp", {}).get("status") == "healthy"
        )

        if not is_ready:
            raise HTTPException(
                status_code=HttpStatusCodes.SERVICE_UNAVAILABLE,
                detail=ErrorMessages.SERVICE_UNAVAILABLE,
            )

        return ResponseWrapper.success(
            data={
                "ready": True,
                "service": "assistant",
                "timestamp": datetime.now().isoformat(),
            },
            message="服务就绪",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"就绪检查失败: {str(e)}")
        raise HTTPException(
            status_code=HttpStatusCodes.SERVICE_UNAVAILABLE,
            detail=ErrorMessages.SERVICE_UNAVAILABLE,
        )


@router.get("/config", summary="获取智能助手配置")
@api_response("获取智能助手配置")
async def get_assistant_config() -> Dict[str, Any]:
    await assistant_service.initialize()

    config_info = await assistant_service.get_assistant_config()

    return ResponseWrapper.success(data=config_info, message="success")


@router.get("/info", summary="AI-CloudOps智能助手服务信息")
@api_response("AI-CloudOps智能助手服务信息")
async def assistant_info() -> Dict[str, Any]:
    info = {
        "service": "AI-CloudOps智能助手",
        "service_type": "enterprise_assistant",
        "workflow_engine": "langgraph + mcp",
        "version": AppConstants.APP_VERSION,
        "description": "AI-CloudOps智能运维助手，支持RAG和MCP双模式，提供高可用、高性能的问答服务",
        "supported_modes": [
            {"mode": 1, "name": "RAG", "description": "基于知识库的检索增强生成"},
            {"mode": 2, "name": "MCP", "description": "基于工具调用的模型上下文协议"},
        ],
        "capabilities": [
            "智能意图识别",
            "知识库问答",
            "工具调用",
            "故障排查指导",
            "操作步骤指导",
            "上下文对话",
            "文档检索",
            "智能推荐",
            "质量评估",
            "自动重试",
            "Kubernetes操作",
            "系统信息查询",
        ],
        "enterprise_features": [
            "双模式架构(RAG+MCP)",
            "LangGraph工作流引擎",
            "MCP工具调用协议",
            "多级缓存机制",
            "智能质量评估",
            "自动错误恢复",
            "性能监控",
            "并行处理",
            "意图路由",
            "模式隔离",
        ],
        "endpoints": {
            "query": ApiEndpoints.ASSISTANT_QUERY,
            "session": ApiEndpoints.ASSISTANT_SESSION,
            "refresh": ApiEndpoints.ASSISTANT_REFRESH,
            "health": ApiEndpoints.ASSISTANT_HEALTH,
            "config": "/assistant/config",
            "info": "/assistant/info",
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
            "缓存存储",
        ],
        "model_info": {
            "provider": "OpenAI/Ollama",
            "architecture": "LangGraph + LLM + MCP",
            "capabilities": [
                "text_generation",
                "conversation",
                "reasoning",
                "workflow_management",
                "tool_calling",
                "kubernetes_operations",
            ],
        },
        "constraints": {
            "max_question_length": 1000,
            "max_context_docs": ServiceConstants.ASSISTANT_MAX_CONTEXT_DOCS,
            "default_context_docs": ServiceConstants.ASSISTANT_DEFAULT_CONTEXT_DOCS,
            "timeout": ServiceConstants.ASSISTANT_TIMEOUT,
            "session_timeout": ServiceConstants.ASSISTANT_SESSION_TIMEOUT,
            "max_retries": 2,
            "quality_threshold": 0.7,
        },
        "performance": {
            "caching_enabled": True,
            "parallel_processing": True,
            "error_recovery": True,
            "metrics_collection": True,
        },
        "status": "available" if assistant_service else "unavailable",
        "initialized": (
            assistant_service.is_initialized() if assistant_service else False
        ),
    }

    return ResponseWrapper.success(data=info, message="success")


__all__ = ["router"]
