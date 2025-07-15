#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 连续回答生成器 - 解决回答断层问题
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

logger = logging.getLogger("aiops.streaming_answer")


class ContinuousAnswerGenerator:
    """连续回答生成器 - 确保回答完整性"""

    def __init__(self, base_generator):
        self.base_generator = base_generator
        self.connection_timeout = 30
        self.retry_count = 2

    async def generate_continuous_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        session_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        生成连续回答，确保完整性

        Args:
            question: 用户问题
            context_docs: 上下文文档
            session_history: 会话历史

        Returns:
            完整的回答字典
        """
        for attempt in range(self.retry_count + 1):
            try:
                # 设置超时控制
                result = await asyncio.wait_for(
                    self.base_generator.generate_answer(
                        question, context_docs, session_history
                    ),
                    timeout=self.connection_timeout
                )

                # 验证回答完整性
                if self._validate_answer_completeness(result):
                    return result
                else:
                    logger.warning(f"回答不完整，尝试重新生成 (尝试 {attempt + 1})")
                    if attempt < self.retry_count:
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        # 使用降级策略
                        return self._create_fallback_answer(question, context_docs)

            except asyncio.TimeoutError:
                logger.error(f"回答生成超时 (尝试 {attempt + 1})")
                if attempt < self.retry_count:
                    await asyncio.sleep(1)
                    continue
                else:
                    return self._create_timeout_answer(question)

            except Exception as e:
                logger.error(f"回答生成失败 (尝试 {attempt + 1}): {str(e)}")
                if attempt < self.retry_count:
                    await asyncio.sleep(1)
                    continue
                else:
                    return self._create_error_answer(question, str(e))

        # 不应该到达这里
        return self._create_fallback_answer(question, context_docs)

    def _validate_answer_completeness(self, result: Dict[str, Any]) -> bool:
        """验证回答完整性"""
        answer = result.get("answer", "")

        # 检查基本要求
        if len(answer) < 10:
            return False

        # 检查是否有明显的截断
        truncation_indicators = [
            "...", "（未完）", "待续", "继续", "更多内容"
        ]

        # 检查是否意外截断
        if any(indicator in answer[-20:] for indicator in truncation_indicators):
            return False

        # 检查是否有基本的结构
        if "**" in answer or "1." in answer or "2." in answer:
            return True

        # 简单长度检查
        return len(answer) >= 30

    def _create_fallback_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建降级回答"""
        if context_docs:
            doc = context_docs[0]
            content = doc.get("page_content", "")

            if content:
                # 提取关键信息
                key_info = content[:100]
                if len(content) > 100:
                    key_info += "..."

                answer = f"**关键信息**：\n{key_info}\n\n如需更详细信息，请查阅相关文档。"
            else:
                answer = f"关于「{question}」的问题，已找到相关文档，但需要更多时间处理。请稍后重试。"
        else:
            answer = f"关于「{question}」的问题，正在查找相关信息。请稍后重试或尝试更具体的问题。"

        return {
            "answer": answer,
            "source_documents": self._format_source_documents(context_docs),
            "follow_up_questions": ["可以重新提问吗？", "需要更具体的信息吗？"],
            "relevance_score": 0.5,
            "is_fallback": True,
            "timestamp": datetime.now().isoformat()
        }

    def _create_timeout_answer(self, question: str) -> Dict[str, Any]:
        """创建超时回答"""
        return {
            "answer": f"处理「{question}」时响应超时。系统正在优化中，请稍后重试。",
            "source_documents": [],
            "follow_up_questions": ["可以重新提问吗？", "需要简化问题吗？"],
            "relevance_score": 0.0,
            "is_timeout": True,
            "timestamp": datetime.now().isoformat()
        }

    def _create_error_answer(self, question: str, error_msg: str) -> Dict[str, Any]:
        """创建错误回答"""
        return {
            "answer": f"处理「{question}」时出现技术问题。请稍后重试。",
            "source_documents": [],
            "follow_up_questions": ["可以重新提问吗？", "需要技术支持吗？"],
            "relevance_score": 0.0,
            "is_error": True,
            "error_details": error_msg,
            "timestamp": datetime.now().isoformat()
        }

    def _format_source_documents(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化源文档"""
        if not context_docs:
            return []

        doc = context_docs[0]
        metadata = doc.get("metadata", {})

        source_path = metadata.get("source", "未知来源")
        source_name = source_path.split('/')[-1] if source_path else "未知来源"

        return [{
            "source": source_name,
            "title": metadata.get("title", "")[:30] if metadata.get("title") else "",
            "content_preview": doc.get("page_content", "")[:60] + "..."
            if len(doc.get("page_content", "")) > 60 else doc.get("page_content", "")
        }]
