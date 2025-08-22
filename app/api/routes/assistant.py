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

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.api.decorators import api_response, log_api_call
from app.common.constants import (
    ApiEndpoints,
    AppConstants,
    ErrorMessages,
    HttpStatusCodes,
)
from app.common.response import ResponseWrapper
from app.models import (
    AddDocumentRequest,
    AddDocumentResponse,
    AssistantRequest,
    AssistantResponse,
    ClearCacheResponse,
    CreateSessionResponse,
    RefreshKnowledgeResponse,
    ServiceConfigResponse,
    ServiceHealthResponse,
    ServiceInfoResponse,
    ServiceReadyResponse,
    SessionInfoResponse,
    UploadKnowledgeRequest,
    UploadKnowledgeResponse,
)
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
    )

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

    response = RefreshKnowledgeResponse(
        refreshed=refresh_result.get("refreshed", True),
        documents_count=refresh_result.get("documents_count", 0),
        vector_count=refresh_result.get("vector_count", 0),
        timestamp=datetime.now().isoformat(),
        message=refresh_result.get("message", "知识库刷新成功"),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/health", summary="智能助手健康检查")
@api_response("智能助手健康检查")
async def assistant_health() -> Dict[str, Any]:
    await assistant_service.initialize()

    health_status = await assistant_service.get_service_health_info_with_mode()

    response = ServiceHealthResponse(
        status=health_status.get("status", "healthy"),
        service="assistant",
        version=health_status.get("version"),
        dependencies=health_status.get("dependencies"),
        last_check_time=datetime.now().isoformat(),
        uptime=health_status.get("uptime"),
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/ready", summary="智能助手就绪检查")
@api_response("智能助手就绪检查")
async def assistant_ready() -> Dict[str, Any]:
    """检查智能助手就绪状态"""
    try:
        await assistant_service.initialize()
        health_status = await assistant_service.get_service_health_info_with_mode()

        is_ready = (
            health_status.get("modes", {}).get("rag", {}).get("status") == "healthy"
            or health_status.get("modes", {}).get("mcp", {}).get("status") == "healthy"
        )

        if not is_ready:
            raise HTTPException(
                status_code=HttpStatusCodes.SERVICE_UNAVAILABLE,
                detail=ErrorMessages.SERVICE_UNAVAILABLE,
            )

        response = ServiceReadyResponse(
            ready=True,
            service="assistant",
            timestamp=datetime.now().isoformat(),
            message="服务就绪",
        )
        return ResponseWrapper.success(
            data=response.dict(),
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


@router.post("/clear-cache", summary="清除缓存")
@api_response("清除缓存")
async def clear_cache() -> Dict[str, Any]:
    await assistant_service.initialize()
    clear_result = await assistant_service.clear_cache()

    response = ClearCacheResponse(
        cleared=clear_result.get("cleared", True),
        cache_keys_cleared=clear_result.get("cache_keys_cleared", 0),
        timestamp=datetime.now().isoformat(),
        message=clear_result.get("message", "缓存清除成功"),
    )
    return ResponseWrapper.success(data=response.dict(), message="success")


@router.post("/session", summary="创建会话")
@api_response("创建会话")
async def create_session(request: AssistantRequest) -> Dict[str, Any]:
    await assistant_service.initialize()
    session_result = await assistant_service.create_session(request)

    response = CreateSessionResponse(
        session_id=(
            session_result
            if isinstance(session_result, str)
            else session_result.get("session_id", "")
        ),
        mode=request.mode,
        created_time=datetime.now().isoformat(),
        status="active",
    )
    return ResponseWrapper.success(data=response.dict(), message="success")


@router.post("/upload_knowledge", summary="上传知识库(JSON格式)")
@api_response("上传知识库")
async def upload_knowledge(request: UploadKnowledgeRequest) -> Dict[str, Any]:
    """上传结构化知识库数据"""
    await assistant_service.initialize()

    upload_result = await assistant_service.upload_knowledge(request)

    response = UploadKnowledgeResponse(
        uploaded=upload_result.get("uploaded", True),
        document_id=upload_result.get("document_id"),
        filename=None,
        file_size=None,
        message=upload_result.get("message", "知识库上传成功"),
        timestamp=datetime.now().isoformat(),
    )
    return ResponseWrapper.success(data=response.dict(), message="success")


@router.post("/upload_knowledge_file", summary="上传知识库文件")
@api_response("上传知识库文件")
async def upload_knowledge_file(
    file: UploadFile = File(...),
    title: str = Form(None),
    source: str = Form(None),
) -> Dict[str, Any]:
    """上传知识库文件"""
    await assistant_service.initialize()

    upload_result = await assistant_service.upload_knowledge_file(
        file=file, title=title, source=source
    )

    response = UploadKnowledgeResponse(
        uploaded=upload_result.get("uploaded", True),
        document_id=upload_result.get("document_id"),
        filename=upload_result.get("filename"),
        file_size=upload_result.get("file_size"),
        message=upload_result.get("message", "知识库上传成功"),
        timestamp=datetime.now().isoformat(),
    )
    return ResponseWrapper.success(data=response.dict(), message="success")


@router.post("/add-document", summary="添加知识库文档")
@api_response("添加知识库文档")
async def add_document(request: AddDocumentRequest) -> Dict[str, Any]:
    await assistant_service.initialize()
    result = await assistant_service.add_document(request.dict())

    response = AddDocumentResponse(
        added=result.get("added", True),
        document_id=result.get("document_id", ""),
        message=result.get("message", "文档添加成功"),
        timestamp=datetime.now().isoformat(),
    )
    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/config", summary="获取智能助手配置")
@api_response("获取智能助手配置")
async def get_assistant_config() -> Dict[str, Any]:
    await assistant_service.initialize()

    config_info = await assistant_service.get_assistant_config()

    response = ServiceConfigResponse(
        service="assistant", config=config_info, timestamp=datetime.now().isoformat()
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


@router.get("/info", summary="AI-CloudOps智能助手服务信息")
@api_response("AI-CloudOps智能助手服务信息")
async def assistant_info() -> Dict[str, Any]:
    info = {
        "service": "AI-CloudOps智能助手",
        "service_type": "enterprise_assistant",
        "workflow_engine": "langgraph + mcp",
        "version": AppConstants.APP_VERSION,
        "description": "智能运维助手，支持RAG和MCP双模式",
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
            "max_context_docs": 10,
            "default_context_docs": 3,
            "timeout": 360,
            "session_timeout": 3600,
            "max_retries": 2,
            "quality_threshold": 0.7,
        },
        "performance": {"caching_enabled": True, "parallel_processing": True},
        "status": "available" if assistant_service else "unavailable",
        "initialized": (
            assistant_service.is_initialized() if assistant_service else False
        ),
    }

    response = ServiceInfoResponse(
        service=info["service"],
        version=info["version"],
        description=info["description"],
        capabilities=info["capabilities"],
        endpoints=info["endpoints"],
        constraints=info["constraints"],
        status=info["status"],
    )

    return ResponseWrapper.success(data=response.dict(), message="success")


__all__ = ["router"]
