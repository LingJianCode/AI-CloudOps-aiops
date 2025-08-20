#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能助手FastAPI路由模块
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config.settings import config
from app.core.agents.assistant_manager import get_assistant_agent, init_assistant_in_background
from app.core.agents.assistant_utils import sanitize_result_data

logger = logging.getLogger("aiops.api.assistant")

# 创建路由器
router = APIRouter(tags=["assistant"])

# 初始化助手
init_assistant_in_background()

# 请求模型
class AssistantQueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    session_id: str = Field(None, description="会话ID")
    max_context_docs: int = Field(1, description="最大上下文文档数", ge=1, le=10)

class SessionRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")

# 响应模型
class AssistantResponse(BaseModel):
    code: int
    message: str
    data: Dict[str, Any]

@router.post("/query", response_model=AssistantResponse, summary="智能助手查询")
async def assistant_query(request: AssistantQueryRequest) -> AssistantResponse:
    """智能小助手查询API"""
    try:
        logger.info(f"收到查询请求 - 问题: {request.question[:100]}...")

        agent = get_assistant_agent()
        if not agent:
            raise HTTPException(status_code=500, detail="智能小助手服务未正确初始化")

        # 调用助手代理获取回答，使用配置的超时时间
        try:
            result = await asyncio.wait_for(
                agent.get_answer(
                    question=request.question, 
                    session_id=request.session_id, 
                    max_context_docs=request.max_context_docs
                ),
                timeout=config.rag.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"智能助手查询超时 ({config.rag.timeout}秒)")
            raise HTTPException(status_code=504, detail=f"智能助手查询超时，请稍后重试")
        except Exception as e:
            logger.error(f"获取回答失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"获取回答时出错: {str(e)}")

        # 生成会话ID（如果不存在）
        session_id = request.session_id
        if not session_id:
            session_id = agent.create_session()

        # 清理结果数据，确保JSON安全
        clean_result = sanitize_result_data(result)

        return AssistantResponse(
            code=0,
            message="查询成功",
            data={
                "answer": clean_result["answer"],
                "session_id": session_id,
                "relevance_score": clean_result.get("relevance_score"),
                "recall_rate": clean_result.get("recall_rate", 0.0),
                "sources": clean_result.get("source_documents", []),
                "follow_up_questions": clean_result.get("follow_up_questions", []),
                "timestamp": datetime.now().isoformat(),
                "performance": clean_result.get("performance", {}),
                "debug_info": clean_result.get("debug_info", {}) if logger.isEnabledFor(logging.DEBUG) else None,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"助手查询异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"助手查询异常: {str(e)}")


@router.delete("/session/{session_id}", response_model=AssistantResponse, summary="删除会话")
async def delete_session(session_id: str) -> AssistantResponse:
    """删除指定会话"""
    try:
        agent = get_assistant_agent()
        if not agent:
            raise HTTPException(status_code=500, detail="智能小助手服务未正确初始化")

        # 删除会话
        success = agent.delete_session(session_id)
        
        if success:
            return AssistantResponse(
                code=0,
                message="会话删除成功",
                data={"session_id": session_id}
            )
        else:
            raise HTTPException(status_code=404, detail="会话不存在或删除失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除会话异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除会话异常: {str(e)}")


@router.post("/session/clear", response_model=AssistantResponse, summary="清空会话历史")
async def clear_session(request: SessionRequest) -> AssistantResponse:
    """清空指定会话的历史记录"""
    try:
        agent = get_assistant_agent()
        if not agent:
            raise HTTPException(status_code=500, detail="智能小助手服务未正确初始化")

        # 清空会话历史
        success = agent.clear_session_history(request.session_id)
        
        if success:
            return AssistantResponse(
                code=0,
                message="会话历史清空成功",
                data={"session_id": request.session_id}
            )
        else:
            raise HTTPException(status_code=404, detail="会话不存在或清空失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空会话历史异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空会话历史异常: {str(e)}")


@router.get("/session/{session_id}/history", response_model=AssistantResponse, summary="获取会话历史")
async def get_session_history(session_id: str) -> AssistantResponse:
    """获取指定会话的历史记录"""
    try:
        agent = get_assistant_agent()
        if not agent:
            raise HTTPException(status_code=500, detail="智能小助手服务未正确初始化")

        # 获取会话历史
        history = agent.get_session_history(session_id)
        
        if history is not None:
            return AssistantResponse(
                code=0,
                message="获取会话历史成功",
                data={
                    "session_id": session_id,
                    "history": history,
                    "count": len(history)
                }
            )
        else:
            raise HTTPException(status_code=404, detail="会话不存在")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话历史异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话历史异常: {str(e)}")


@router.post("/refresh", response_model=AssistantResponse, summary="刷新助手")
async def refresh_assistant() -> AssistantResponse:
    """强制刷新智能助手，重新加载知识库"""
    try:
        from app.core.agents.assistant_manager import reinitialize_assistant
        
        # 执行重新初始化
        new_agent = reinitialize_assistant()
        
        if new_agent:
            return AssistantResponse(
                code=0,
                message="智能助手刷新成功",
                data={
                    "status": "刷新成功",
                    "timestamp": datetime.now().isoformat(),
                    "agent_id": id(new_agent)
                }
            )
        else:
            raise HTTPException(status_code=500, detail="助手重新初始化失败")

    except Exception as e:
        logger.error(f"刷新助手异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刷新助手异常: {str(e)}")


@router.get("/health", response_model=AssistantResponse, summary="助手健康检查")
async def health_check() -> AssistantResponse:
    """智能助手健康检查"""
    try:
        agent = get_assistant_agent()
        if not agent:
            raise HTTPException(status_code=503, detail="智能小助手服务未初始化")

        # 获取助手状态
        status = {
            "service": "assistant",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agent_initialized": True,
            "vector_store_status": "unknown",
            "llm_service_status": "unknown"
        }

        # 检查向量存储状态
        try:
            if hasattr(agent, 'vector_store') and agent.vector_store:
                # 简单的连接测试
                test_result = agent.vector_store.similarity_search("test", k=1)
                status["vector_store_status"] = "healthy"
                status["vector_store_docs"] = len(test_result) if test_result else 0
        except Exception as e:
            status["vector_store_status"] = f"error: {str(e)}"

        # 检查LLM服务状态
        try:
            if hasattr(agent, 'llm_service') and agent.llm_service:
                llm_health = agent.llm_service.health_check()
                status["llm_service_status"] = "healthy" if llm_health else "unhealthy"
        except Exception as e:
            status["llm_service_status"] = f"error: {str(e)}"

        return AssistantResponse(
            code=0,
            message="健康检查完成",
            data=status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"健康检查异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查异常: {str(e)}")


@router.get("/ready", response_model=AssistantResponse, summary="助手就绪检查")
async def readiness_check() -> AssistantResponse:
    """智能助手就绪检查"""
    try:
        agent = get_assistant_agent()
        is_ready = agent is not None
        
        return AssistantResponse(
            code=0 if is_ready else 1,
            message="就绪" if is_ready else "未就绪",
            data={
                "ready": is_ready,
                "timestamp": datetime.now().isoformat()
            }
        )

    except Exception as e:
        logger.error(f"就绪检查异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"就绪检查异常: {str(e)}")


@router.get("/info", response_model=AssistantResponse, summary="助手信息")
async def get_info() -> AssistantResponse:
    """获取智能助手信息"""
    try:
        agent = get_assistant_agent()
        
        info = {
            "service": "智能助手",
            "version": "1.0.0",
            "description": "基于大语言模型的智能运维助手",
            "capabilities": [
                "知识库问答",
                "上下文对话",
                "文档检索",
                "智能推荐"
            ],
            "endpoints": {
                "query": "/api/v1/assistant/query",
                "session": "/api/v1/assistant/session",
                "refresh": "/api/v1/assistant/refresh",
                "health": "/api/v1/assistant/health"
            },
            "status": "available" if agent else "unavailable",
            "timestamp": datetime.now().isoformat()
        }

        return AssistantResponse(
            code=0,
            message="获取信息成功",
            data=info
        )

    except Exception as e:
        logger.error(f"获取助手信息异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取助手信息异常: {str(e)}")


# 导出
__all__ = ["router", "get_assistant_agent"]