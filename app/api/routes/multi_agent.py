#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
多Agent修复API路由
Author: AI Assistant
License: Apache 2.0
Description: 提供多Agent协作的K8s修复API接口
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import asyncio
import logging
from app.core.agents.coordinator import K8sCoordinatorAgent
from app.models.response_models import APIResponse
from app.utils.validators import validate_deployment_name, validate_namespace, sanitize_input

logger = logging.getLogger("aiops.multi_agent")

multi_agent_bp = Blueprint('multi_agent', __name__)

# 初始化协调器
coordinator = K8sCoordinatorAgent()

@multi_agent_bp.route('/multi-agent/repair', methods=['POST'])
def repair_deployment():
    """修复单个部署"""
    try:
        data = request.get_json() or {}
        
        deployment = data.get('deployment')
        namespace = data.get('namespace', 'default')
        
        # 验证参数
        if not deployment:
            return jsonify(APIResponse(
                code=400,
                message="必须提供部署名称",
                data={}
            ).model_dump()), 400
            
        if not validate_deployment_name(deployment):
            return jsonify(APIResponse(
                code=400,
                message="无效的部署名称",
                data={}
            ).model_dump()), 400
            
        if not validate_namespace(namespace):
            return jsonify(APIResponse(
                code=400,
                message="无效的命名空间名称",
                data={}
            ).model_dump()), 400
        
        deployment = sanitize_input(deployment)
        namespace = sanitize_input(namespace)
        
        logger.info(f"开始多Agent修复: {deployment}/{namespace}")
        
        # 执行修复工作流
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                coordinator.run_full_workflow(deployment, namespace)
            )
        finally:
            loop.close()
        
        # 构建响应
        if result.get('success'):
            message = "多Agent修复完成"
            code = 0
        else:
            message = f"修复失败: {result.get('error', '未知错误')}"
            code = 500
        
        return jsonify(APIResponse(
            code=code,
            message=message,
            data=result
        ).model_dump())
        
    except Exception as e:
        logger.error(f"修复请求失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"修复请求失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/repair/batch', methods=['POST'])
def repair_namespace():
    """批量修复命名空间"""
    try:
        data = request.get_json() or {}
        namespace = data.get('namespace', 'default')
        
        if not validate_namespace(namespace):
            return jsonify(APIResponse(
                code=400,
                message="无效的命名空间名称",
                data={}
            ).model_dump()), 400
        
        namespace = sanitize_input(namespace)
        
        logger.info(f"开始批量修复: {namespace}")
        
        # 执行批量工作流
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                coordinator.run_batch_workflow(namespace)
            )
        finally:
            loop.close()
        
        if result.get('success'):
            message = "批量修复完成"
            code = 0
        else:
            message = f"批量修复失败: {result.get('error', '未知错误')}"
            code = 500
        
        return jsonify(APIResponse(
            code=code,
            message=message,
            data=result
        ).model_dump())
        
    except Exception as e:
        logger.error(f"批量修复请求失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"批量修复请求失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/detect', methods=['POST'])
def detect_issues():
    """检测问题"""
    try:
        data = request.get_json() or {}
        namespace = data.get('namespace', 'default')
        deployment = data.get('deployment')
        
        if not validate_namespace(namespace):
            return jsonify(APIResponse(
                code=400,
                message="无效的命名空间名称",
                data={}
            ).model_dump()), 400
        
        namespace = sanitize_input(namespace)
        
        logger.info(f"开始检测问题: {namespace}")
        
        # 执行检测
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if deployment:
                result = loop.run_until_complete(
                    coordinator.detector.detect_deployment_issues(deployment, namespace)
                )
            else:
                result = loop.run_until_complete(
                    coordinator.detector.detect_all_issues(namespace)
                )
        finally:
            loop.close()
        
        return jsonify(APIResponse(
            code=0,
            message="问题检测完成",
            data=result
        ).model_dump())
        
    except Exception as e:
        logger.error(f"检测请求失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"检测请求失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/strategy', methods=['POST'])
def create_strategy():
    """创建修复策略"""
    try:
        data = request.get_json() or {}
        issues = data.get('issues')
        
        if not issues:
            return jsonify(APIResponse(
                code=400,
                message="必须提供问题数据",
                data={}
            ).model_dump()), 400
        
        logger.info("开始制定修复策略")
        
        # 制定策略
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                coordinator.strategist.analyze_issues(issues)
            )
        finally:
            loop.close()
        
        if 'error' in result:
            return jsonify(APIResponse(
                code=500,
                message=f"策略制定失败: {result['error']}",
                data={}
            ).model_dump()), 500
        
        return jsonify(APIResponse(
            code=0,
            message="策略制定完成",
            data=result
        ).model_dump())
        
    except Exception as e:
        logger.error(f"策略制定请求失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"策略制定请求失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/execute', methods=['POST'])
def execute_strategy():
    """执行修复策略"""
    try:
        data = request.get_json() or {}
        strategy = data.get('strategy')
        
        if not strategy:
            return jsonify(APIResponse(
                code=400,
                message="必须提供策略数据",
                data={}
            ).model_dump()), 400
        
        logger.info("开始执行修复策略")
        
        # 执行策略
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                coordinator.executor.execute_strategy(strategy)
            )
        finally:
            loop.close()
        
        return jsonify(APIResponse(
            code=0 if result.get('success') else 500,
            message="策略执行完成" if result.get('success') else "策略执行失败",
            data=result
        ).model_dump())
        
    except Exception as e:
        logger.error(f"策略执行请求失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"策略执行请求失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/history', methods=['GET'])
def get_history():
    """获取工作流历史"""
    try:
        history = coordinator.get_workflow_history()
        return jsonify(APIResponse(
            code=0,
            message="获取历史成功",
            data={'history': history}
        ).model_dump())
        
    except Exception as e:
        logger.error(f"获取历史失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"获取历史失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        # 执行健康检查
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            health = loop.run_until_complete(coordinator.health_check())
        finally:
            loop.close()
        
        return jsonify(APIResponse(
            code=0,
            message="健康检查完成",
            data=health
        ).model_dump())
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"健康检查失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500

@multi_agent_bp.route('/multi-agent/reset', methods=['POST'])
def reset_workflow():
    """重置工作流"""
    try:
        data = request.get_json() or {}
        deployment = data.get('deployment')
        namespace = data.get('namespace', 'default')
        
        if not deployment:
            return jsonify(APIResponse(
                code=400,
                message="必须提供部署名称",
                data={}
            ).model_dump()), 400
        
        # 重置工作流
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                coordinator.reset_workflow(deployment, namespace)
            )
        finally:
            loop.close()
        
        return jsonify(APIResponse(
            code=0 if result.get('success') else 500,
            message=result.get('message', '重置完成'),
            data=result
        ).model_dump())
        
    except Exception as e:
        logger.error(f"重置请求失败: {str(e)}")
        return jsonify(APIResponse(
            code=500,
            message=f"重置请求失败: {str(e)}",
            data={"timestamp": datetime.now(timezone.utc).isoformat()}
        ).model_dump()), 500