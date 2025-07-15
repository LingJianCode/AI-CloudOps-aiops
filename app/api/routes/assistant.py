import logging
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from app.core.agents.assistant_manager import get_assistant_agent, init_assistant_in_background
from app.core.agents.assistant_utils import safe_async_run, sanitize_result_data

logger = logging.getLogger("aiops.api.assistant")
assistant_bp = Blueprint("assistant", __name__, url_prefix="")

init_assistant_in_background()

__all__ = ["assistant_bp", "get_assistant_agent"]


@assistant_bp.route("/query", methods=["POST"])
def assistant_query() -> tuple:
    """智能小助手查询API"""
    try:
        # 详细的请求调试信息
        logger.info(f"收到查询请求 - Content-Type: {request.content_type}")
        logger.info(f"请求头: {dict(request.headers)}")

        # 验证请求内容类型 - 增强检测与兼容性
        content_type = request.headers.get("Content-Type", "")
        is_json_type = "json" in content_type.lower()

        if not is_json_type and not request.is_json:
            logger.error(f"请求不是JSON格式 - Content-Type: {request.content_type}")
            return (
                jsonify(
                    {
                        "code": 400,
                        "message": "请求必须是JSON格式，请设置Content-Type为application/json",
                        "data": {},
                    }
                ),
                400,
            )

        # 尝试获取原始数据用于调试
        try:
            raw_data = request.get_data(as_text=True)
            logger.info(f"原始请求数据: {raw_data[:200]}...")
        except Exception as e:
            logger.warning(f"无法获取原始请求数据: {e}")

        data = request.json
        if data is None:
            logger.error("请求体解析为None")
            return (
                jsonify({"code": 400, "message": "请求体必须包含有效的JSON数据", "data": {}}),
                400,
            )

        logger.info(f"解析的JSON数据: {data}")

        question = data.get("question", "")
        session_id = data.get("session_id")
        max_context_docs = data.get("max_context_docs", 1)  # 从2减少到1

        if not question:
            logger.error("问题字段为空")
            return jsonify({"code": 400, "message": "问题不能为空", "data": {}}), 400

        agent = get_assistant_agent()
        if not agent:
            return jsonify({"code": 500, "message": "智能小助手服务未正确初始化", "data": {}}), 500

        # 调用助手代理获取回答
        try:
            result = safe_async_run(
                agent.get_answer(
                    question=question, session_id=session_id, max_context_docs=max_context_docs
                )
            )
        except Exception as e:
            logger.error(f"获取回答失败: {str(e)}")
            return jsonify({"code": 500, "message": f"获取回答时出错: {str(e)}", "data": {}}), 500

        # 生成会话ID（如果不存在）
        if not session_id:
            session_id = agent.create_session()

        # 清理结果数据，确保JSON安全
        clean_result = sanitize_result_data(result)

        return jsonify(
            {
                "code": 0,
                "message": "查询成功",
                "data": {
                    "answer": clean_result["answer"],
                    "session_id": session_id,
                    "relevance_score": clean_result.get("relevance_score"),
                    "recall_rate": clean_result.get("recall_rate", 0.0),
                    "sources": clean_result.get("source_documents", []),
                    "follow_up_questions": clean_result.get("follow_up_questions", []),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )
    except Exception as e:
        # 更详细的错误日志
        logger.error(f"查询处理失败 - 异常类型: {type(e).__name__}")
        logger.error(f"查询处理失败 - 异常信息: {str(e)}")
        logger.error(f"查询处理失败 - 请求方法: {request.method}")
        logger.error(f"查询处理失败 - 请求URL: {request.url}")
        logger.error(f"查询处理失败 - Content-Type: {request.content_type}")

        # 尝试获取请求体信息用于调试
        try:
            if hasattr(request, "data") and request.data:
                logger.error(f"请求体数据: {request.data[:200]}...")
        except Exception as debug_e:
            logger.error(f"无法获取请求体数据: {debug_e}")

        return jsonify({"code": 500, "message": f"处理查询时出错: {str(e)}", "data": {}}), 500


@assistant_bp.route("/session", methods=["POST"])
def create_session() -> tuple:
    """
    创建新会话 - 同步包装异步函数

    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        agent = get_assistant_agent()
        if not agent:
            return jsonify({"code": 500, "message": "智能小助手服务未正确初始化", "data": {}}), 500

        session_id = agent.create_session()

        return jsonify(
            {
                "code": 0,
                "message": "会话创建成功",
                "data": {"session_id": session_id, "timestamp": datetime.now().isoformat()},
            }
        )
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        return jsonify({"code": 500, "message": f"创建会话时出错: {str(e)}", "data": {}}), 500


@assistant_bp.route("/refresh", methods=["POST"])
def refresh_knowledge_base() -> tuple:
    """
    刷新知识库 - 同步包装异步函数

    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        agent = get_assistant_agent()
        if not agent:
            return jsonify({"code": 500, "message": "智能小助手服务未正确初始化", "data": {}}), 500

        try:
            # 强制清理缓存
            agent.response_cache = {}
            logger.info("API层强制清理了响应缓存")

            # 刷新知识库
            result = safe_async_run(agent.refresh_knowledge_base())

            # 为确保向量数据库完全初始化，添加小延迟
            import time

            time.sleep(1)  # 等待1秒钟

        except Exception as e:
            logger.error(f"刷新知识库失败: {str(e)}")
            return jsonify({"code": 500, "message": f"刷新知识库时出错: {str(e)}", "data": {}}), 500

        return jsonify(
            {
                "code": 0,
                "message": "知识库刷新成功",
                "data": {
                    "documents_count": result.get("documents_count", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )
    except Exception as e:
        logger.error(f"刷新知识库失败: {str(e)}")
        return jsonify({"code": 500, "message": f"刷新知识库时出错: {str(e)}", "data": {}}), 500


@assistant_bp.route("/add-document", methods=["POST"])
def add_document() -> tuple:
    """
    添加文档到知识库 - 同步包装异步函数

    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        data = request.json
        content = data.get("content", "")
        metadata = data.get("metadata", {})

        if not content:
            return jsonify({"code": 400, "message": "文档内容不能为空", "data": {}}), 400

        agent = get_assistant_agent()
        if not agent:
            return jsonify({"code": 500, "message": "智能小助手服务未正确初始化", "data": {}}), 500

        # 添加文档
        success = agent.add_document(content=content, metadata=metadata)

        if not success:
            return jsonify({"code": 500, "message": "添加文档失败", "data": {}}), 500

        return jsonify(
            {
                "code": 0,
                "message": "文档添加成功",
                "data": {"timestamp": datetime.now().isoformat()},
            }
        )
    except Exception as e:
        logger.error(f"添加文档失败: {str(e)}")
        return jsonify({"code": 500, "message": f"添加文档时出错: {str(e)}", "data": {}}), 500


@assistant_bp.route("/clear-cache", methods=["POST"])
def clear_cache() -> tuple:
    """
    清除缓存 - 同步包装异步函数

    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        agent = get_assistant_agent()
        if not agent:
            return jsonify({"code": 500, "message": "智能小助手服务未正确初始化", "data": {}}), 500

        # 清除缓存
        result = agent.clear_cache()

        return jsonify(
            {
                "code": 0,
                "message": "缓存清理成功",
                "data": {
                    "cleared_items": result.get("cleared_items", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )
    except Exception as e:
        logger.error(f"清除缓存失败: {str(e)}")
        return jsonify({"code": 500, "message": f"清除缓存时出错: {str(e)}", "data": {}}), 500


@assistant_bp.route("/health", methods=["GET"])
def assistant_health() -> tuple:
    """
    智能小助手健康检查
    
    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        agent = get_assistant_agent()
        
        if not agent:
            return jsonify({
                "code": 500,
                "message": "健康检查失败",
                "data": {
                    "status": "unhealthy",
                    "healthy": False,
                    "agent_initialized": False,
                    "timestamp": datetime.now().isoformat(),
                }
            }), 500
        
        # 检查各组件状态
        vector_store_healthy = True
        llm_healthy = True
        knowledge_base_healthy = True
        
        try:
            # 检查向量存储
            if hasattr(agent, 'vector_store') and agent.vector_store:
                # 简单的存在检查
                vector_store_healthy = True
            else:
                vector_store_healthy = False
                
            # 检查LLM服务
            if hasattr(agent, 'llm_service') and agent.llm_service:
                llm_healthy = agent.llm_service.is_healthy()
            else:
                llm_healthy = False
                
            # 检查知识库
            if hasattr(agent, 'knowledge_loaded') and agent.knowledge_loaded:
                knowledge_base_healthy = True
            else:
                knowledge_base_healthy = False
                
        except Exception as e:
            logger.warning(f"组件健康检查出错: {str(e)}")
        
        # 判断整体健康状态
        overall_healthy = llm_healthy and knowledge_base_healthy
        
        health_status = {
            "status": "healthy" if overall_healthy else "degraded",
            "healthy": overall_healthy,
            "components": {
                "agent": True,
                "vector_store": vector_store_healthy,
                "llm": llm_healthy,
                "knowledge_base": knowledge_base_healthy,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        return jsonify({
            "code": 0,
            "message": "健康检查完成",
            "data": health_status
        })
        
    except Exception as e:
        logger.error(f"智能小助手健康检查失败: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"健康检查失败: {str(e)}",
            "data": {
                "status": "error",
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
            }
        }), 500


@assistant_bp.route("/ready", methods=["GET"])
def assistant_ready() -> tuple:
    """
    智能小助手就绪性检查
    
    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        agent = get_assistant_agent()
        
        if not agent:
            return jsonify({
                "code": 503,
                "message": "智能小助手服务未就绪",
                "data": {
                    "status": "not ready",
                    "ready": False,
                    "agent_initialized": False,
                    "timestamp": datetime.now().isoformat(),
                }
            }), 503
        
        # 检查就绪性条件
        ready = True
        issues = []
        
        try:
            # 检查LLM服务
            if hasattr(agent, 'llm_service') and agent.llm_service:
                if hasattr(agent.llm_service, 'is_healthy') and callable(agent.llm_service.is_healthy):
                    try:
                        if not agent.llm_service.is_healthy():
                            ready = False
                            issues.append("LLM服务不健康")
                    except Exception as e:
                        logger.warning(f"LLM健康检查失败: {str(e)}")
                        ready = False
                        issues.append(f"LLM健康检查失败: {str(e)}")
                else:
                    # LLM服务存在但没有健康检查方法，认为是健康的
                    logger.debug("LLM服务没有健康检查方法，假定健康")
            else:
                # 检查是否仍在初始化中
                if hasattr(agent, 'is_initializing') and agent.is_initializing:
                    ready = False
                    issues.append("LLM服务正在初始化中")
                else:
                    ready = False
                    issues.append("LLM服务未初始化")
                
            # 检查知识库
            if hasattr(agent, 'knowledge_loaded'):
                if not agent.knowledge_loaded:
                    # 检查是否仍在初始化中
                    if hasattr(agent, 'is_initializing') and agent.is_initializing:
                        ready = False
                        issues.append("知识库正在初始化中")
                    else:
                        ready = False
                        issues.append("知识库未加载")
            else:
                # 如果没有knowledge_loaded属性，尝试检查其他相关属性
                if hasattr(agent, 'vector_store_manager') and agent.vector_store_manager:
                    logger.debug("通过向量存储管理器检查知识库状态")
                else:
                    ready = False
                    issues.append("知识库状态未知")
                
        except Exception as e:
            logger.warning(f"就绪性检查出错: {str(e)}")
            ready = False
            issues.append(f"检查出错: {str(e)}")
        
        if ready:
            return jsonify({
                "code": 0,
                "message": "智能小助手服务就绪",
                "data": {
                    "status": "ready",
                    "ready": True,
                    "timestamp": datetime.now().isoformat(),
                }
            })
        else:
            return jsonify({
                "code": 503,
                "message": "智能小助手服务未就绪",
                "data": {
                    "status": "not ready",
                    "ready": False,
                    "issues": issues,
                    "timestamp": datetime.now().isoformat(),
                }
            }), 503
        
    except Exception as e:
        logger.error(f"智能小助手就绪性检查失败: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"就绪性检查失败: {str(e)}",
            "data": {
                "status": "error",
                "ready": False,
                "timestamp": datetime.now().isoformat(),
            }
        }), 500


@assistant_bp.route("/info", methods=["GET"])
def assistant_info() -> tuple:
    """
    获取智能小助手服务信息
    
    Returns:
        tuple: 包含JSON响应和HTTP状态码的元组
    """
    try:
        agent = get_assistant_agent()
        
        if not agent:
            return jsonify({
                "code": 500,
                "message": "智能小助手服务未初始化",
                "data": {}
            }), 500
        
        # 收集服务信息
        info = {
            "service": "智能小助手",
            "version": "1.0.0",
            "agent_type": type(agent).__name__,
            "initialized": True,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 添加组件信息
        try:
            if hasattr(agent, 'knowledge_loaded'):
                info["knowledge_loaded"] = agent.knowledge_loaded
            if hasattr(agent, 'vector_store'):
                info["vector_store_available"] = agent.vector_store is not None
            if hasattr(agent, 'llm_service'):
                info["llm_service_available"] = agent.llm_service is not None
            if hasattr(agent, 'response_cache'):
                info["cache_size"] = len(agent.response_cache)
        except Exception as e:
            logger.warning(f"获取组件信息失败: {str(e)}")
        
        return jsonify({
            "code": 0,
            "message": "获取服务信息成功",
            "data": info
        })
        
    except Exception as e:
        logger.error(f"获取智能小助手信息失败: {str(e)}")
        return jsonify({
            "code": 500,
            "message": f"获取服务信息失败: {str(e)}",
            "data": {}
        }), 500
