#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: AI助手API路由模块，提供智能问答和流式对话功能
"""

import asyncio
import logging
import threading
import re
from datetime import datetime
from flask import Blueprint, request, jsonify

# 创建日志器
logger = logging.getLogger("aiops.api.assistant")

def sanitize_for_json(text):
    """
    清理文本中的控制字符，确保JSON安全
    """
    if not isinstance(text, str):
        return text
    
    # 替换换行符为空格，而不是转义序列，避免在JSON响应中出现真实换行符
    text = text.replace('\n', ' ').replace('\r', ' ')
    # 替换多个连续空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 移除其他控制字符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text.strip()

def sanitize_result_data(data):
    """
    递归清理结果数据中的所有字符串字段
    """
    if isinstance(data, dict):
        return {k: sanitize_result_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_result_data(item) for item in data]
    elif isinstance(data, str):
        return sanitize_for_json(data)
    else:
        return data

# 创建蓝图
assistant_bp = Blueprint('assistant', __name__, url_prefix='')


# 创建助手代理全局实例
_assistant_agent = None
_init_lock = threading.Lock()
_is_initializing = False

def get_assistant_agent():
    """获取助手代理单例实例，采用懒加载+锁机制优化初始化性能"""
    global _assistant_agent, _is_initializing
    
    if _assistant_agent is not None:
        return _assistant_agent
        
    # 使用锁避免多线程重复初始化
    with _init_lock:
        if _is_initializing:
            # 如果正在初始化中，等待一小段时间后再检查
            logger.info("另一个线程正在初始化小助手，等待...")
            import time
            for _ in range(10):  # 最多等待5秒
                time.sleep(0.5)
                if _assistant_agent is not None:
                    return _assistant_agent
        
        # 标记为正在初始化
        _is_initializing = True
        
        try:
            logger.info("初始化智能小助手代理...")
            from app.core.agents.assistant import AssistantAgent
            _assistant_agent = AssistantAgent()
            logger.info("智能小助手代理初始化完成")
        except Exception as e:
            logger.error(f"初始化智能小助手代理失败: {str(e)}")
            _is_initializing = False
            return None
            
        _is_initializing = False
        return _assistant_agent

# 提前在后台线程初始化小助手
def init_assistant_in_background():
    """在后台线程中初始化小助手，避免首次调用时的延迟"""
    def _init_thread():
        try:
            logger.info("开始在后台预初始化智能小助手...")
            get_assistant_agent()
            logger.info("小助手后台预初始化完成")
        except Exception as e:
            logger.error(f"后台初始化小助手失败: {str(e)}")
    
    threading.Thread(target=_init_thread, daemon=True).start()

# 应用启动时自动初始化
init_assistant_in_background()


def safe_async_run(coroutine):
    """安全地运行异步函数，处理不同环境下的运行方式"""
    try:
        # 检查是否已有事件循环在运行
        try:
            loop = asyncio.get_running_loop()
            # 如果有运行中的事件循环，使用 asyncio.create_task 或直接同步调用
            # 对于Flask同步环境，我们需要在新线程中运行事件循环
            import threading
            import concurrent.futures
            
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        # 确保所有协程都被等待
                        if asyncio.iscoroutine(coroutine):
                            result = new_loop.run_until_complete(coroutine)
                        else:
                            # 如果不是协程对象，直接返回
                            result = coroutine
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
            
        except RuntimeError:
            # 没有运行中的事件循环，可以安全创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 确保所有协程都被等待
                if asyncio.iscoroutine(coroutine):
                    return loop.run_until_complete(coroutine)
                else:
                    # 如果不是协程对象，直接返回
                    return coroutine
            finally:
                loop.close()
                
    except Exception as e:
        logger.error(f"执行异步函数失败: {str(e)}")
        raise e


@assistant_bp.route('/query', methods=['POST'])
def assistant_query():
    """智能小助手查询API - 同步包装异步函数"""
    try:
        # 详细的请求调试信息
        logger.info(f"收到查询请求 - Content-Type: {request.content_type}")
        logger.info(f"请求头: {dict(request.headers)}")
        
        # 验证请求内容类型 - 增强检测与兼容性
        content_type = request.headers.get('Content-Type', '')
        is_json_type = 'json' in content_type.lower()
        
        if not is_json_type and not request.is_json:
            logger.error(f"请求不是JSON格式 - Content-Type: {request.content_type}")
            return jsonify({
                'code': 400,
                'message': '请求必须是JSON格式，请设置Content-Type为application/json',
                'data': {}
            }), 400
        
        # 尝试获取原始数据用于调试
        try:
            raw_data = request.get_data(as_text=True)
            logger.info(f"原始请求数据: {raw_data[:200]}...")
        except Exception as e:
            logger.warning(f"无法获取原始请求数据: {e}")
            
        data = request.json
        if data is None:
            logger.error("请求体解析为None")
            return jsonify({
                'code': 400,
                'message': '请求体必须包含有效的JSON数据',
                'data': {}
            }), 400
        
        logger.info(f"解析的JSON数据: {data}")
        
        question = data.get('question', '')
        session_id = data.get('session_id')
        max_context_docs = data.get('max_context_docs', 4)
        mode = data.get('mode', 'rag')
        
        if not question:
            logger.error("问题字段为空")
            return jsonify({
                'code': 400,
                'message': '问题不能为空',
                'data': {}
            }), 400
        
        # 根据模式处理查询
        if mode == 'mcp':
            logger.info("使用MCP模式处理查询")
            try:
                from app.mcp.mcp_client import MCPAssistant
                mcp_assistant = MCPAssistant()
                
                # 检查MCP服务是否可用
                is_available = safe_async_run(mcp_assistant.is_available())
                if not is_available:
                    return jsonify({
                        'code': 503,
                        'message': 'MCP服务暂时不可用',
                        'data': {}
                    }), 503
                
                # 使用MCP处理查询
                answer = safe_async_run(mcp_assistant.process_query(question))
                
                return jsonify({
                    'code': 0,
                    'message': '查询成功',
                    'data': {
                        'answer': answer,
                        'session_id': session_id,
                        'mode': 'mcp',
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                logger.error(f"MCP模式处理失败: {str(e)}")
                return jsonify({
                    'code': 500,
                    'message': f'MCP模式处理失败: {str(e)}',
                    'data': {}
                }), 500
        
        # 默认使用RAG模式
        agent = get_assistant_agent()
        if not agent:
            return jsonify({
                'code': 500,
                'message': '智能小助手服务未正确初始化',
                'data': {}
            }), 500
        
        # 调用助手代理获取回答
        try:
            result = safe_async_run(agent.get_answer(
                question=question,
                session_id=session_id,
                max_context_docs=max_context_docs
            ))
        except Exception as e:
            logger.error(f"获取回答失败: {str(e)}")
            return jsonify({
                'code': 500,
                'message': f'获取回答时出错: {str(e)}',
                'data': {}
            }), 500
        
        # 生成会话ID（如果不存在）
        if not session_id:
            session_id = agent.create_session()
        
        # 清理结果数据，确保JSON安全
        clean_result = sanitize_result_data(result)
        
        return jsonify({
            'code': 0,
            'message': '查询成功',
            'data': {
                'answer': clean_result['answer'],
                'session_id': session_id,
                'relevance_score': clean_result.get('relevance_score'),
                'recall_rate': clean_result.get('recall_rate', 0.0),
                'sources': clean_result.get('source_documents', []),
                'follow_up_questions': clean_result.get('follow_up_questions', []),
                'mode': 'rag',
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        # 更详细的错误日志
        logger.error(f"查询处理失败 - 异常类型: {type(e).__name__}")
        logger.error(f"查询处理失败 - 异常信息: {str(e)}")
        logger.error(f"查询处理失败 - 请求方法: {request.method}")
        logger.error(f"查询处理失败 - 请求URL: {request.url}")
        logger.error(f"查询处理失败 - Content-Type: {request.content_type}")
        
        # 尝试获取请求体信息用于调试
        try:
            if hasattr(request, 'data') and request.data:
                logger.error(f"请求体数据: {request.data[:200]}...")
        except Exception as debug_e:
            logger.error(f"无法获取请求体数据: {debug_e}")
            
        return jsonify({
            'code': 500,
            'message': f'处理查询时出错: {str(e)}',
            'data': {}
        }), 500


@assistant_bp.route('/session', methods=['POST'])
def create_session():
    """创建新会话 - 同步包装异步函数"""
    try:
        agent = get_assistant_agent()
        if not agent:
            return jsonify({
                'code': 500,
                'message': '智能小助手服务未正确初始化',
                'data': {}
            }), 500
        
        session_id = agent.create_session()
        
        return jsonify({
            'code': 0,
            'message': '会话创建成功',
            'data': {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f'创建会话时出错: {str(e)}',
            'data': {}
        }), 500


@assistant_bp.route('/refresh', methods=['POST'])
def refresh_knowledge_base():
    """刷新知识库 - 同步包装异步函数"""
    try:
        agent = get_assistant_agent()
        if not agent:
            return jsonify({
                'code': 500,
                'message': '智能小助手服务未正确初始化',
                'data': {}
            }), 500
        
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
            return jsonify({
                'code': 500,
                'message': f'刷新知识库时出错: {str(e)}',
                'data': {}
            }), 500
        
        return jsonify({
            'code': 0,
            'message': '知识库刷新成功',
            'data': {
                'documents_count': result.get('documents_count', 0),
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"刷新知识库失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f'刷新知识库时出错: {str(e)}',
            'data': {}
        }), 500


@assistant_bp.route('/add-document', methods=['POST'])
def add_document():
    """添加文档到知识库 - 同步包装异步函数"""
    try:
        data = request.json
        content = data.get('content', '')
        metadata = data.get('metadata', {})
        
        if not content:
            return jsonify({
                'code': 400,
                'message': '文档内容不能为空',
                'data': {}
            }), 400
        
        agent = get_assistant_agent()
        if not agent:
            return jsonify({
                'code': 500,
                'message': '智能小助手服务未正确初始化',
                'data': {}
            }), 500
        
        # 添加文档到知识库
        success = agent.add_document(content, metadata)
        
        if success:
            # 刷新知识库
            try:
                # 强制清理缓存
                agent.response_cache = {}
                logger.info("API层强制清理了响应缓存")
                
                # 刷新知识库
                result = safe_async_run(agent.refresh_knowledge_base())
                
                # 为确保向量数据库完全初始化，添加小延迟
                import time
                time.sleep(1)  # 等待1秒钟
                
                documents_count = result.get('documents_count', 0)
            except Exception as e:
                logger.error(f"添加文档后刷新知识库失败: {str(e)}")
                documents_count = 0
            
            return jsonify({
                'code': 0,
                'message': '文档添加成功',
                'data': {
                    'success': True,
                    'documents_count': documents_count,
                    'timestamp': datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                'code': 500,
                'message': '文档添加失败',
                'data': {
                    'success': False
                }
            }), 500
            
    except Exception as e:
        logger.error(f"添加文档失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f'添加文档时出错: {str(e)}',
            'data': {}
        }), 500


@assistant_bp.route('/clear-cache', methods=['POST'])
def clear_cache():
    """清除智能小助手的缓存"""
    try:
        agent = get_assistant_agent()
        if not agent:
            return jsonify({
                'code': 500,
                'message': '智能小助手服务未正确初始化',
                'data': {}
            }), 500
        
        # 使用新的Redis缓存清空功能
        try:
            result = agent.clear_cache()
            return jsonify({
                'code': 0,
                'message': result.get('message', '缓存清除成功'),
                'data': {
                    'cleared_items': result.get('cleared_count', 0),
                    'success': result.get('success', True)
                }
            })
        except Exception as e:
            logger.error(f"清空缓存失败: {str(e)}")
            return jsonify({
                'code': 500,
                'message': f'清空缓存失败: {str(e)}',
                'data': {}
            }), 500
    except Exception as e:
        logger.error(f"清除缓存失败: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f'清除缓存时出错: {str(e)}',
            'data': {}
        }), 500


@assistant_bp.route('/reinitialize', methods=['POST'])
def reinitialize_assistant():
    """手动重新初始化智能小助手 - 完全重置助手状态，重建向量数据库，重新载入知识库"""
    try:
        global _assistant_agent, _is_initializing
        
        logger.info("开始手动重新初始化智能小助手...")
        
        # 使用锁保证线程安全
        with _init_lock:
            # 如果正在初始化，等待初始化完成
            if _is_initializing:
                logger.info("检测到正在进行初始化，等待初始化完成...")
                import time
                for _ in range(20):  # 增加等待时间到10秒
                    time.sleep(0.5)
                    if not _is_initializing:
                        break
                else:
                    logger.error("等待初始化完成超时")
                    return jsonify({
                        'code': 500,
                        'message': '等待初始化完成超时，请稍后重试',
                        'data': {'status': 'timeout'}
                    }), 500
            
            # 如果存在当前实例，先进行强制重新初始化
            if _assistant_agent is not None:
                try:
                    logger.info("对现有小助手实例进行强制重新初始化...")
                    
                    # 调用强制重新初始化方法 - 增加超时时间
                    result = safe_async_run(_assistant_agent.force_reinitialize())
                    
                    if result.get('success'):
                        logger.info("小助手强制重新初始化成功")
                        return jsonify({
                            'code': 0,
                            'message': '智能小助手重新初始化成功',
                            'data': {
                                'status': 'success',
                                'documents_count': result.get('documents_count', 0),
                                'message': result.get('message', ''),
                                'processing_time': result.get('processing_time', 0),
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                    else:
                        logger.error(f"强制重新初始化失败: {result.get('error')}")
                        # 如果强制重新初始化失败，继续尝试完全重建
                        
                except Exception as e:
                    logger.error(f"强制重新初始化时出错: {str(e)}")
                    # 如果出错，继续尝试完全重建
            
            # 如果强制重新初始化失败或没有现有实例，进行完全重建
            logger.info("进行完全重建小助手...")
            
            # 关闭当前实例（如果存在）
            if _assistant_agent is not None:
                try:
                    logger.info("关闭当前小助手实例...")
                    safe_async_run(_assistant_agent.shutdown())
                    logger.info("当前小助手实例已关闭")
                except Exception as e:
                    logger.error(f"关闭当前小助手实例时出错: {str(e)}")
            
            # 重置实例
            _assistant_agent = None
            _is_initializing = False
            
            # 清理缓存和向量存储（防止旧数据影响）
            try:
                logger.info("清理残留的缓存和向量数据...")
                from app.core.cache.redis_cache_manager import RedisCacheManager
                from app.config.settings import config
                
                # 创建临时缓存管理器来清理数据
                redis_config = {
                    'host': config.redis.host,
                    'port': config.redis.port,
                    'db': config.redis.db + 1,
                    'password': config.redis.password,
                    'connection_timeout': config.redis.connection_timeout,
                    'socket_timeout': config.redis.socket_timeout,
                    'max_connections': config.redis.max_connections,
                    'decode_responses': config.redis.decode_responses
                }
                
                temp_cache_manager = RedisCacheManager(
                    redis_config=redis_config,
                    cache_prefix="aiops_assistant_cache:",
                    default_ttl=3600,
                    max_cache_size=1000,
                    enable_compression=True
                )
                temp_cache_manager.clear_all()
                temp_cache_manager.shutdown()
                logger.info("缓存数据清理完成")
                
            except Exception as e:
                logger.warning(f"清理缓存数据时出现警告: {str(e)}")
            
            # 重新初始化小助手
            logger.info("开始创建新的小助手实例...")
            agent = get_assistant_agent()
            
            if agent is not None:
                # 创建成功后，再次进行强制重新初始化以确保所有数据都是最新的
                try:
                    logger.info("对新创建的小助手实例进行强制重新初始化...")
                    result = safe_async_run(agent.force_reinitialize())
                    
                    if result.get('success'):
                        logger.info("智能小助手完全重新初始化成功")
                        return jsonify({
                            'code': 0,
                            'message': '智能小助手完全重新初始化成功',
                            'data': {
                                'status': 'success',
                                'documents_count': result.get('documents_count', 0),
                                'message': '已完全重建向量数据库和重新载入知识库',
                                'processing_time': result.get('processing_time', 0),
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                    else:
                        logger.warning(f"新实例强制重新初始化失败: {result.get('error')}")
                        # 即使失败，也返回成功，因为基本的实例已经创建
                        return jsonify({
                            'code': 0,
                            'message': '智能小助手重新初始化完成（部分功能可能需要时间生效）',
                            'data': {
                                'status': 'partial_success',
                                'warning': result.get('error'),
                                'timestamp': datetime.now().isoformat()
                            }
                        })
                        
                except Exception as e:
                    logger.warning(f"新实例强制重新初始化时出错: {str(e)}")
                    # 即使出错，也返回成功，因为基本的实例已经创建
                    return jsonify({
                        'code': 0,
                        'message': '智能小助手重新初始化完成（部分功能可能需要时间生效）',
                        'data': {
                            'status': 'partial_success',
                            'warning': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
            else:
                logger.error("智能小助手重新初始化失败")
                return jsonify({
                    'code': 500,
                    'message': '智能小助手重新初始化失败',
                    'data': {
                        'status': 'failed',
                        'timestamp': datetime.now().isoformat()
                    }
                }), 500
    
    except Exception as e:
        logger.error(f"重新初始化小助手时发生错误: {str(e)}")
        return jsonify({
            'code': 500,
            'message': f'重新初始化小助手时发生错误: {str(e)}',
            'data': {
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
        }), 500
