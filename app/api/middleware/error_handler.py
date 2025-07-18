#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 错误处理中间件 - 提供统一的HTTP错误响应格式和异常处理机制
"""

import logging
import traceback
from flask import jsonify, request
from datetime import datetime, timezone
from app.models.response_models import APIResponse

logger = logging.getLogger("aiops.error_handler")


def _safe_get_request_info():
  """安全地获取请求信息，避免在错误处理过程中再次出错"""
  try:
    return {
      "url": request.url if request else "unknown",
      "path": request.path if request else "unknown",
      "method": request.method if request else "unknown",
      "content_type": request.content_type if request else "unknown"
    }
  except Exception as e:
    logger.error(f"无法获取请求信息: {e}")
    return {
      "url": "error",
      "path": "error",
      "method": "error",
      "content_type": "error"
    }


def _create_error_response(code, message, extra_data=None):
  """创建统一的错误响应"""
  try:
    request_info = _safe_get_request_info()

    data = {
      "timestamp": datetime.now(timezone.utc).isoformat(),
      "path": request_info["path"]
    }

    if extra_data:
      data.update(extra_data)

    return jsonify(APIResponse(
      code=code,
      message=message,
      data=data
    ).model_dump()), code

  except Exception as handler_error:
    logger.error(f"创建错误响应时出错: {handler_error}")
    return jsonify({
      "code": code,
      "message": message,
      "data": {"timestamp": datetime.now(timezone.utc).isoformat()}
    }), code


def setup_error_handlers(app):
  """设置错误处理器"""

  @app.errorhandler(400)
  def bad_request(error):
    """处理400错误"""
    try:
      request_info = _safe_get_request_info()

      logger.error(f"400错误处理器被触发 - 错误: {str(error)}")
      logger.error(f"错误类型: {type(error).__name__}")
      logger.error(
        f"请求信息 - URL: {request_info['url']}, Method: {request_info['method']}, Content-Type: {request_info['content_type']}")

      error_description = str(error.description) if hasattr(error, 'description') else "请求参数错误"
      logger.warning(f"Bad request: {request_info['url']} - {error_description}")

      return _create_error_response(400, error_description)

    except Exception as handler_error:
      logger.error(f"400错误处理器本身出错: {handler_error}")
      return _create_error_response(400, "请求错误")

  @app.errorhandler(401)
  def unauthorized(error):
    logger.warning(f"未授权访问:{error}")
    """处理401错误"""
    try:
      request_info = _safe_get_request_info()
      logger.warning(f"Unauthorized access: {request_info['url']}")
      return _create_error_response(401, "未授权访问")
    except Exception as handler_error:
      logger.error(f"401错误处理器出错: {handler_error}")
      return _create_error_response(401, "未授权访问")

  @app.errorhandler(403)
  def forbidden(error):
    logger.warning(f"访问被禁止: {error}")
    """处理403错误"""
    try:
      request_info = _safe_get_request_info()
      logger.warning(f"Forbidden access: {request_info['url']}")
      return _create_error_response(403, "访问被禁止")
    except Exception as handler_error:
      logger.error(f"403错误处理器出错: {handler_error}")
      return _create_error_response(403, "访问被禁止")

  @app.errorhandler(404)
  def not_found(error):
    logger.warning(f"未找到资源: {error}")
    """处理404错误"""
    try:
      request_info = _safe_get_request_info()
      logger.warning(f"Not found: {request_info['url']}")
      message = f"请求的资源 {request_info['path']} 不存在"
      return _create_error_response(404, message)
    except Exception as handler_error:
      logger.error(f"404错误处理器出错: {handler_error}")
      return _create_error_response(404, "请求的资源不存在")

  @app.errorhandler(405)
  def method_not_allowed(error):
    logger.warning(f"405错误: {error}")
    """处理405错误"""
    try:
      request_info = _safe_get_request_info()
      logger.warning(f"Method not allowed: {request_info['method']} {request_info['url']}")
      message = f"方法 {request_info['method']} 不被允许用于 {request_info['path']}"
      extra_data = {"method": request_info['method']}
      return _create_error_response(405, message, extra_data=extra_data)
    except Exception as handler_error:
      logger.error(f"405错误处理器出错: {handler_error}")
      return _create_error_response(405, "请求方法不被允许")

  @app.errorhandler(422)
  def unprocessable_entity(error):
    """处理422错误"""
    try:
      request_info = _safe_get_request_info()
      logger.warning(f"Unprocessable entity: {request_info['url']} - {str(error)}")
      return _create_error_response(422, "请求格式正确但语义错误")
    except Exception as handler_error:
      logger.error(f"422错误处理器出错: {handler_error}")
      return _create_error_response(422, "请求格式正确但语义错误")

  @app.errorhandler(429)
  def rate_limit_exceeded(error):
    logger.warning(f"Rate limit exceeded: {error}")
    """处理429错误"""
    try:
      request_info = _safe_get_request_info()
      logger.warning(f"Rate limit exceeded: {request_info['url']}")
      return _create_error_response(429, "请求过于频繁，请稍后再试")
    except Exception as handler_error:
      logger.error(f"429错误处理器出错: {handler_error}")
      return _create_error_response(429, "请求过于频繁，请稍后再试")

  @app.errorhandler(500)
  def internal_server_error(error):
    """处理500错误"""
    try:
      error_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
      request_info = _safe_get_request_info()

      logger.error(f"Internal server error [{error_id}]: {request_info['url']}")
      logger.error(f"Error details: {str(error)}")
      logger.error(f"Traceback: {traceback.format_exc()}")

      extra_data = {"error_id": error_id}
      return _create_error_response(500, "服务器遇到意外错误", extra_data=extra_data)

    except Exception as handler_error:
      logger.error(f"500错误处理器出错: {handler_error}")
      error_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
      return _create_error_response(500, "服务器遇到意外错误", extra_data={"error_id": error_id})

  @app.errorhandler(502)
  def bad_gateway(error):
    logger.warning(f"Bad gateway: {error}")
    """处理502错误"""
    try:
      request_info = _safe_get_request_info()
      logger.error(f"Bad gateway: {request_info['url']}")
      return _create_error_response(502, "上游服务器返回无效响应")
    except Exception as handler_error:
      logger.error(f"502错误处理器出错: {handler_error}")
      return _create_error_response(502, "上游服务器返回无效响应")

  @app.errorhandler(503)
  def service_unavailable(error):
    logger.warning(f"503错误: {error}")
    """处理503错误"""
    try:
      request_info = _safe_get_request_info()
      logger.error(f"Service unavailable: {request_info['url']}")
      return _create_error_response(503, "服务暂时不可用，请稍后重试")
    except Exception as handler_error:
      logger.error(f"503错误处理器出错: {handler_error}")
      return _create_error_response(503, "服务暂时不可用，请稍后重试")

  @app.errorhandler(Exception)
  def handle_unexpected_error(error):
    """处理未预期的错误"""
    try:
      error_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
      request_info = _safe_get_request_info()

      logger.error(f"Unexpected error [{error_id}]: {request_info['url']}")
      logger.error(f"Error type: {type(error).__name__}")
      logger.error(f"Error message: {str(error)}")
      logger.error(f"Traceback: {traceback.format_exc()}")

      # 获取配置信息，避免在错误处理中再次出错
      try:
        from app.config.settings import config
        is_debug = config.debug
      except Exception as config_error:
        logger.error(f"无法获取配置信息: {config_error}")
        is_debug = False

      # 在开发模式下返回详细错误信息
      if is_debug:
        extra_data = {
          "type": type(error).__name__,
          "error_id": error_id,
          "traceback": traceback.format_exc().split('\n')
        }
        return _create_error_response(500, f"意外错误（调试模式）: {str(error)}",
                                      extra_data=extra_data)
      else:
        extra_data = {"error_id": error_id}
        return _create_error_response(500, "服务器遇到意外错误，请联系管理员", extra_data=extra_data)

    except Exception as handler_error:
      logger.error(f"通用错误处理器出错: {handler_error}")
      error_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
      return _create_error_response(500, "服务器遇到严重错误", extra_data={"error_id": error_id})

  logger.info("错误处理器设置完成")
