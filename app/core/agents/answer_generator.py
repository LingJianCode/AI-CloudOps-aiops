#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 答案生成器 - 可靠的答案生成和任务管理
"""

import asyncio
import logging
import threading
import time
import uuid
from asyncio import CancelledError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger("aiops.answer_generator")


class ReliableAnswerGenerator:
    """可靠的答案生成器，集成多种策略确保高质量答案"""

    def __init__(self, llm_service):
        self.llm_service = llm_service
        self.fallback_attempts = 0
        self.max_fallback_attempts = 3

    async def generate_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        session_history: Optional[List[str]] = None,
        max_retries: int = 1  # 减少重试次数
    ) -> Dict[str, Any]:
        """
        生成高质量答案
        
        Args:
            question: 用户问题
            context_docs: 上下文文档
            session_history: 会话历史
            max_retries: 最大重试次数
            
        Returns:
            包含答案和元数据的字典
        """
        for attempt in range(max_retries + 1):
            try:
                # 1. 构建系统提示词
                system_prompt = self._build_system_prompt()
                
                # 2. 构建用户提示词
                user_prompt = self._build_user_prompt(question, context_docs, session_history)
                
                # 3. 生成答案
                response = await self._call_llm_with_timeout(system_prompt, user_prompt)
                
                # 4. 验证和增强答案
                validated_answer = self._validate_and_enhance_answer(response, question, context_docs)
                
                # 5. 生成后续问题建议
                follow_up_questions = self._generate_follow_up_questions(question, validated_answer)
                
                return {
                    "answer": validated_answer,
                    "source_documents": self._format_source_documents(context_docs),
                    "follow_up_questions": follow_up_questions,
                    "relevance_score": self._calculate_relevance_score(validated_answer, question),
                    "generation_time": time.time(),
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                logger.warning(f"答案生成尝试 {attempt + 1} 失败: {str(e)}")
                if attempt == max_retries:
                    # 最后一次尝试失败，返回降级答案
                    return self._generate_fallback_answer(question, context_docs)
                
                # 短暂等待后重试，减少等待时间
                await asyncio.sleep(0.5)  # 从1秒减少到0.5秒
        
        # 不应该到达这里，但为了安全返回降级答案
        return self._generate_fallback_answer(question, context_docs)

    def _build_system_prompt(self) -> str:
        """构建优化的系统提示词"""
        return """你是AI-CloudOps智能运维平台的专业助手。

要求：
1. **极简**：回答控制在100-150字
2. **要点**：使用1-2-3数字列表
3. **核心**：只提供关键操作

格式：
- 重要信息用**粗体**
- 代码用`backticks`
- 直接给出方案，不要解释

语言：中文，技术术语保持准确。150字以内。"""

    def _build_user_prompt(
        self, 
        question: str, 
        context_docs: List[Dict[str, Any]], 
        session_history: Optional[List[str]] = None
    ) -> str:
        """构建优化的用户提示词"""
        prompt_parts = []
        
        # 只使用第一个最相关的文档，减少内容
        if context_docs:
            doc = context_docs[0]
            content = doc.get("page_content", "")
            
            # 限制文档内容为200字符
            if len(content) > 200:
                content = content[:200] + "..."
            
            prompt_parts.append(f"**文档**: {content}")
        
        # 简化指导语
        prompt_parts.append(f"**问题**: {question}")
        prompt_parts.append("**要求**: 基于文档，1-2-3步骤回答（100-150字）")
        
        return "\n".join(prompt_parts)

    async def _call_llm_with_timeout(self, system_prompt: str, user_prompt: str, timeout: int = 20) -> str:
        """带超时的LLM调用"""
        try:
            # 创建超时任务
            task = asyncio.create_task(
                self._async_llm_call(system_prompt, user_prompt)
            )
            
            # 等待任务完成或超时
            response = await asyncio.wait_for(task, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"LLM调用超时 ({timeout}秒)")
            raise Exception("LLM调用超时")
        except Exception as e:
            logger.error(f"LLM调用失败: {str(e)}")
            raise

    async def _async_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """异步LLM调用"""
        try:
            # 使用LLMService的标准接口
            if hasattr(self.llm_service, 'generate_response') and callable(self.llm_service.generate_response):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                response = await self.llm_service.generate_response(messages)
                return response if isinstance(response, str) else str(response)
            else:
                # 降级到同步调用
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._sync_llm_call,
                    system_prompt,
                    user_prompt
                )
                return response
                
        except Exception as e:
            logger.error(f"异步LLM调用失败: {str(e)}")
            raise

    def _sync_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """同步LLM调用"""
        try:
            # 首先尝试使用新的LLMService接口
            if hasattr(self.llm_service, 'generate_response'):
                # 创建简单的同步包装
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                # 使用asyncio运行异步方法
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    response = loop.run_until_complete(self.llm_service.generate_response(messages))
                    return response if isinstance(response, str) else str(response)
                except RuntimeError:
                    # 没有事件循环，创建新的
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        response = loop.run_until_complete(self.llm_service.generate_response(messages))
                        return response if isinstance(response, str) else str(response)
                    finally:
                        loop.close()
            
            # 备用：尝试使用langchain接口（如果llm_service有llm属性）
            elif hasattr(self.llm_service, 'llm') and self.llm_service.llm:
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                result = self.llm_service.llm.invoke(messages)
                if isinstance(result, AIMessage):
                    return result.content
                else:
                    return str(result)
            else:
                raise Exception("LLM服务不可用")
                
        except Exception as e:
            logger.error(f"同步LLM调用失败: {str(e)}")
            raise

    def _validate_and_enhance_answer(self, answer: str, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """验证和增强答案"""
        if not answer or len(answer.strip()) < 10:
            logger.warning("答案太短，使用降级策略")
            return self._generate_simple_answer(question, context_docs)
        
        # 基本清理和长度控制
        cleaned_answer = answer.strip()
        
        # 如果答案过长，进行智能截断
        if len(cleaned_answer) > 200:
            # 找到最后一个完整句子的位置
            sentences = cleaned_answer.split('。')
            truncated_answer = ""
            for sentence in sentences:
                if len(truncated_answer + sentence + '。') <= 200:
                    truncated_answer += sentence + '。'
                else:
                    break
            
            if truncated_answer:
                cleaned_answer = truncated_answer
            else:
                # 如果没有完整句子，直接截断
                cleaned_answer = cleaned_answer[:200] + "..."
        
        # 检查是否是有效的回答（不是拒绝回答）
        refusal_patterns = [
            "文档中未提供", "文档不足", "无法回答", "抱歉，我无法", "没有相关信息", 
            "请查阅", "建议联系", "我不知道", "无法确定", "文档中没有"
        ]
        
        is_refusal = any(pattern in cleaned_answer for pattern in refusal_patterns)
        
        # 如果是拒绝回答，但实际上有相关文档，重新生成答案
        if is_refusal and context_docs:
            logger.warning("检测到拒绝回答，但有相关文档，重新生成答案")
            return self._generate_positive_answer(question, context_docs)
        
        # 返回精简的答案
        return cleaned_answer

    def _generate_positive_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """生成积极的答案，基于文档内容"""
        if not context_docs:
            return f"关于「{question}」的问题，我需要更多具体信息才能提供准确的回答。"
        
        # 构建积极的回答
        answer_parts = []
        
        # 分析问题类型
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["什么", "定义", "介绍", "概述"]):
            answer_parts.append("**定义**：")
        elif any(keyword in question_lower for keyword in ["如何", "怎么", "步骤", "方法"]):
            answer_parts.append("**操作步骤**：")
        elif any(keyword in question_lower for keyword in ["功能", "特性", "能力"]):
            answer_parts.append("**功能特性**：")
        else:
            answer_parts.append("**关键信息**：")
        
        # 从文档中提取关键信息，只使用第一个文档
        doc = context_docs[0]
        content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        
        if content:
            # 提取前120个字符的关键内容
            key_content = content[:120]
            if len(content) > 120:
                key_content += "..."
            
            # 简化来源信息
            source_info = ""
            if metadata.get("source"):
                source_name = metadata['source'].split('/')[-1]  # 只取文件名
                source_info = f"（参考：{source_name}）"
            
            answer_parts.append(f"\n{key_content} {source_info}")
        
        return "\n".join(answer_parts)

    def _generate_simple_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """生成简单答案"""
        if not context_docs:
            return f"关于「{question}」的问题，我需要更多相关的文档信息。"
        
        # 基于文档内容生成简洁回答
        doc = context_docs[0]
        content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        
        if content:
            # 取前100个字符作为精简摘要
            snippet = content[:100]
            if len(content) > 100:
                snippet += "..."
            
            # 简化来源信息
            source_info = ""
            if metadata.get("source"):
                source_name = metadata['source'].split('/')[-1]  # 只取文件名
                source_info = f"（参考：{source_name}）"
            
            return f"**要点**：\n{snippet} {source_info}"
        else:
            return f"找到了相关文档，但提取信息时遇到困难。建议您重新描述问题。"

    def _generate_follow_up_questions(self, original_question: str, answer: str) -> List[str]:
        """生成后续问题建议"""
        follow_ups = []
        
        # 基于问题类型生成相关问题
        question_lower = original_question.lower()
        
        if any(keyword in question_lower for keyword in ["部署", "安装", "配置"]):
            follow_ups.extend([
                "部署常见问题？",
                "如何验证部署？"
            ])
        elif any(keyword in question_lower for keyword in ["监控", "告警", "指标"]):
            follow_ups.extend([
                "告警规则设置？",
                "监控数据分析？"
            ])
        elif any(keyword in question_lower for keyword in ["故障", "问题", "错误"]):
            follow_ups.extend([
                "如何预防？",
                "诊断工具？"
            ])
        else:
            # 通用后续问题
            follow_ups.extend([
                "最佳实践？",
                "注意事项？"
            ])
        
        return follow_ups[:2]  # 只返回2个建议

    def _format_source_documents(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化源文档信息"""
        formatted_docs = []
        
        # 只格式化第一个文档，节省空间
        if context_docs:
            doc = context_docs[0]
            metadata = doc.get("metadata", {})
            
            # 简化源信息
            source_path = metadata.get("source", "未知来源")
            source_name = source_path.split('/')[-1] if source_path else "未知来源"
            
            formatted_doc = {
                "source": source_name,
                "title": metadata.get("title", "")[:30] if metadata.get("title") else "",  # 标题限制30字符
                "content_preview": doc.get("page_content", "")[:60] + "..." 
                if len(doc.get("page_content", "")) > 60 else doc.get("page_content", "")
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs

    def _calculate_relevance_score(self, answer: str, question: str) -> float:
        """计算答案相关性分数"""
        try:
            # 简单的相关性评分：基于问题关键词在答案中的出现
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            
            # 去除常见停用词
            stop_words = {"如何", "什么", "为什么", "怎么", "是", "的", "了", "在", "和", "a", "an", "the", "and", "or"}
            question_keywords = question_words - stop_words
            
            if not question_keywords:
                return 0.7  # 默认分数
            
            # 计算关键词覆盖率
            matched_keywords = question_keywords.intersection(answer_words)
            coverage = len(matched_keywords) / len(question_keywords)
            
            # 基础分数 + 覆盖率奖励
            base_score = 0.6
            coverage_bonus = coverage * 0.4
            
            return min(base_score + coverage_bonus, 1.0)
            
        except Exception:
            return 0.7  # 默认分数

    def _generate_fallback_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成降级答案"""
        self.fallback_attempts += 1
        
        fallback_answer = f"抱歉，我在处理您的问题时遇到了技术困难。\n\n问题：{question}\n\n"
        
        if context_docs:
            fallback_answer += "虽然找到了一些相关信息，但无法生成完整的回答。建议您：\n"
            fallback_answer += "1. 查阅相关技术文档\n"
            fallback_answer += "2. 联系技术支持团队\n"
            fallback_answer += "3. 尝试重新描述问题"
        else:
            fallback_answer += "未找到相关的文档信息。建议您：\n"
            fallback_answer += "1. 检查问题描述是否准确\n"
            fallback_answer += "2. 尝试使用不同的关键词\n"
            fallback_answer += "3. 查阅官方文档"
        
        return {
            "answer": fallback_answer,
            "source_documents": self._format_source_documents(context_docs),
            "follow_up_questions": ["如何获得更多技术支持？", "有其他相关问题吗？"],
            "relevance_score": 0.3,
            "generation_time": time.time(),
            "is_fallback": True
        }


# ==================== 任务管理器 ====================

class TaskManager:
    """任务管理器 - 管理异步任务的生命周期"""
    
    def __init__(self):
        self.tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def create_task(self, coro, task_id: str = None, description: str = "未命名任务"):
        """创建新任务"""
        if not task_id:
            task_id = str(uuid.uuid4())
            
        task = asyncio.create_task(coro)
        self.tasks[task_id] = {
            'task': task,
            'description': description,
            'created_at': datetime.now(),
            'status': 'running'
        }
        
        # 添加完成回调
        task.add_done_callback(lambda t: self._task_done_callback(task_id, t))
        
        logger.debug(f"创建任务: {task_id} - {description}")
        return task_id, task
    
    def _task_done_callback(self, task_id: str, task):
        """任务完成回调"""
        if task_id in self.tasks:
            if task.cancelled():
                self.tasks[task_id]['status'] = 'cancelled'
            elif task.exception():
                self.tasks[task_id]['status'] = 'failed'
                self.tasks[task_id]['error'] = str(task.exception())
            else:
                self.tasks[task_id]['status'] = 'completed'
            
            self.tasks[task_id]['finished_at'] = datetime.now()
            logger.debug(f"任务完成: {task_id} - {self.tasks[task_id]['status']}")
    
    def cancel_task(self, task_id: str):
        """取消任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]['task']
            if not task.done():
                task.cancel()
                logger.debug(f"取消任务: {task_id}")
    
    def cleanup_finished_tasks(self):
        """清理已完成的任务"""
        finished_tasks = [
            task_id for task_id, info in self.tasks.items()
            if info['status'] in ['completed', 'failed', 'cancelled']
        ]
        
        for task_id in finished_tasks:
            del self.tasks[task_id]
        
        if finished_tasks:
            logger.debug(f"清理了 {len(finished_tasks)} 个已完成任务")


# 全局任务管理器实例
_task_manager = None
_task_manager_lock = threading.Lock()


def get_task_manager():
    """获取全局任务管理器实例"""
    global _task_manager
    if _task_manager is None:
        with _task_manager_lock:
            if _task_manager is None:
                _task_manager = TaskManager()
    return _task_manager


def create_safe_task(coro, description="未命名任务"):
    """安全地创建异步任务"""
    task_manager = get_task_manager()
    return task_manager.create_task(coro, description=description)