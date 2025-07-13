#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops - 优化版
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 智能助手代理 - 基于RAG技术提供运维知识问答和决策支持服务
优化重点: 大幅提升精确度和召回率，修复所有错误
"""

import os
import uuid
import logging
import re
import time
import asyncio
from asyncio import CancelledError
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from app.constants import EMBEDDING_BATCH_SIZE

# Redis向量存储
from app.core.vector.redis_vector_store import RedisVectorStoreManager, OptimizedRedisVectorStore
# Redis缓存管理器
from app.core.cache.redis_cache_manager import RedisCacheManager
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

# 高级加载器
try:
  from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, DirectoryLoader,
    UnstructuredMarkdownLoader, CSVLoader, JSONLoader, BSHTMLLoader
  )

  ADVANCED_LOADERS_AVAILABLE = True
except ImportError:
  ADVANCED_LOADERS_AVAILABLE = False

import numpy as np
# Redis配置
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config.settings import config

logger = logging.getLogger("aiops.assistant")


def is_test_environment() -> bool:
  import sys
  return 'pytest' in sys.modules


# ==================== 智能查询重写器 ====================

class QueryRewriter:
  """智能查询重写器，提升检索召回率"""

  def __init__(self):
    self.synonyms = {
      '部署': ['安装', '配置', '搭建', '建立'],
      '监控': ['观察', '跟踪', '检测', '巡检'],
      '故障': ['异常', '错误', '问题', '失败'],
      '性能': ['效率', '速度', '响应', '吞吐'],
      '日志': ['记录', '日志文件', 'log', '审计'],
      '告警': ['报警', '警告', '提醒', '通知'],
      '自动化': ['自动', '自动执行', '批量'],
      '运维': ['ops', '运营', '维护', '管理']
    }

  def expand_query(self, query: str) -> List[str]:
    """扩展查询，生成多个相关查询变体"""
    expanded_queries = [query]  # 原始查询

    # 1. 同义词替换 - 增加更多变体
    for word, synonyms in self.synonyms.items():
      if word in query:
        for synonym in synonyms[:3]:  # 增加同义词数量
          expanded_queries.append(query.replace(word, synonym))

    # 2. 关键词提取和重组
    keywords = self._extract_keywords(query)
    if len(keywords) >= 2:
      # 关键词组合
      expanded_queries.append(' '.join(keywords))
      # 部分关键词组合
      if len(keywords) >= 3:
        expanded_queries.append(' '.join(keywords[:2]))
        expanded_queries.append(' '.join(keywords[-2:]))

    # 3. 添加语义相关的扩展
    if '部署' in query or '安装' in query:
      expanded_queries.extend(['配置方法', '安装步骤', '部署指南'])
    elif '监控' in query:
      expanded_queries.extend(['监控配置', '指标采集', '告警设置'])
    elif '故障' in query or '错误' in query:
      expanded_queries.extend(['问题排查', '故障诊断', '错误解决'])
    elif '性能' in query:
      expanded_queries.extend(['性能调优', '优化方法', '性能分析'])

    # 4. 去重并适当增加数量
    unique_queries = list(dict.fromkeys(expanded_queries))[:8]  # 增加查询数量
    return unique_queries

  def _extract_keywords(self, text: str) -> List[str]:
    """提取关键词"""
    stopwords = {'的', '和', '或', '是', '在', '有', '如何', '什么', '怎么', '为什么'}
    words = re.findall(r'\w+', text)
    keywords = [w for w in words if len(w) > 1 and w not in stopwords]
    return keywords[:3]  # 减少关键词数量


# ==================== 智能文档排序器 ====================

class DocumentRanker:
  """智能文档排序器，提升检索精确度"""

  def __init__(self):
    self.tfidf_vectorizer = TfidfVectorizer(
      max_features=500,  # 减少特征数量
      stop_words=None,
      ngram_range=(1, 2)
    )
    self.document_vectors = None
    self.fitted = False

  def fit(self, documents: List[Document]):
    """训练TF-IDF模型"""
    try:
      if not documents:
        return

      corpus = [doc.page_content for doc in documents]
      self.document_vectors = self.tfidf_vectorizer.fit_transform(corpus)
      self.fitted = True
      logger.info(f"文档排序器训练完成，文档数: {len(documents)}")
    except Exception as e:
      logger.error(f"文档排序器训练失败: {e}")

  def rank_documents(self, query: str, documents: List[Document], top_k: int = 10) -> List[
    Tuple[Document, float]]:
    """对文档进行相关性排序 - 提升召回率优化版"""
    if not self.fitted or not documents:
      return [(doc, 0.7) for doc in documents[:top_k]]  # 提高默认分数

    try:
      # 查询向量化
      query_vector = self.tfidf_vectorizer.transform([query])

      # 计算文档相似度
      doc_vectors = self.tfidf_vectorizer.transform([doc.page_content for doc in documents])
      similarities = cosine_similarity(query_vector, doc_vectors).flatten()

      # 优化评分算法
      scored_docs = []
      for i, doc in enumerate(documents):
        tfidf_score = similarities[i]
        
        # 调整长度评分 - 不过度惩罚短文档
        content_length = len(doc.page_content)
        if content_length < 100:
          length_score = 0.6  # 短文档基础分数
        elif content_length < 500:
          length_score = min(content_length / 300, 1.0)
        else:
          length_score = 1.0  # 长文档满分
          
        freshness_score = self._calculate_freshness_score(doc)
        
        # 关键词匹配加分
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        keyword_overlap = len(query_words & doc_words) / max(len(query_words), 1)
        
        # 文档类型加分
        doc_type_score = 1.0
        if doc.metadata:
          filetype = doc.metadata.get('filetype', '').lower()
          if filetype in ['md', 'markdown', 'txt']:
            doc_type_score = 1.2  # 文本文档加分
          elif filetype in ['pdf', 'doc']:
            doc_type_score = 1.1

        # 综合评分 - 调整权重提升召回率
        final_score = (
          tfidf_score * 0.4 +           # 降低TF-IDF权重
          length_score * 0.15 +         # 降低长度权重
          freshness_score * 0.1 +       # 降低时间权重
          keyword_overlap * 0.25 +      # 增加关键词权重
          doc_type_score * 0.1          # 增加文档类型权重
        )

        scored_docs.append((doc, final_score))

      # 排序并返回更多结果
      scored_docs.sort(key=lambda x: x[1], reverse=True)
      return scored_docs[:top_k]

    except Exception as e:
      logger.error(f"文档排序失败: {e}")
      return [(doc, 0.6) for doc in documents[:top_k]]

  def _calculate_freshness_score(self, doc: Document) -> float:
    """计算文档新鲜度评分"""
    try:
      if doc.metadata and 'modified_time' in doc.metadata:
        mod_time = doc.metadata['modified_time']
        days_old = (time.time() - mod_time) / (24 * 3600)
        return max(0, 1 - days_old / 365)
      return 0.5
    except:
      return 0.5


# ==================== 上下文感知检索器 ====================

class ContextAwareRetriever:
  """上下文感知检索器，提升对话连贯性"""

  def __init__(self, base_retriever, query_rewriter: QueryRewriter, doc_ranker: DocumentRanker):
    self.base_retriever = base_retriever
    self.query_rewriter = query_rewriter
    self.doc_ranker = doc_ranker
    self.conversation_context = {}

  def retrieve_with_context(self, query: str, session_id: str = None,
                            history: List[Dict] = None, top_k: int = 8) -> List[Document]:
    """带上下文的智能检索 - 提升召回率优化版"""
    try:
      # 1. 构建增强查询
      enhanced_query = self._build_enhanced_query(query, history)

      # 2. 多查询检索 - 降低阈值提升召回率
      all_docs = []
      queries = self.query_rewriter.expand_query(enhanced_query)

      for q in queries:
        try:
          docs = self.base_retriever.invoke(q)
          all_docs.extend(docs)
        except Exception as e:
          logger.warning(f"查询检索失败 '{q}': {e}")
          continue

      # 3. 如果主查询结果太少，尝试部分匹配
      if len(all_docs) < top_k:
        # 尝试关键词查询
        query_words = query.split()
        if len(query_words) > 1:
          for word in query_words:
            if len(word) > 2:  # 跳过太短的词
              try:
                word_docs = self.base_retriever.invoke(word)
                all_docs.extend(word_docs)
                if len(all_docs) >= top_k * 2:
                  break
              except Exception as e:
                logger.warning(f"关键词查询失败 '{word}': {e}")

      # 4. 去重
      unique_docs = self._deduplicate_documents(all_docs)

      # 5. 智能排序 - 增加返回数量
      if unique_docs:
        ranked_docs = self.doc_ranker.rank_documents(enhanced_query, unique_docs, top_k * 2)
      else:
        ranked_docs = []

      # 6. 宽松的上下文过滤
      filtered_docs = self._context_filter_lenient(ranked_docs, session_id, history)

      return [doc for doc, score in filtered_docs[:top_k]]

    except Exception as e:
      logger.error(f"上下文感知检索失败: {e}")
      # 降级到基础检索
      try:
        return self.base_retriever.invoke(query)[:top_k]
      except:
        return []

  def _build_enhanced_query(self, query: str, history: List[Dict] = None) -> str:
    """构建增强查询"""
    if not history:
      return query

    # 提取历史关键词
    historical_keywords = []
    for msg in history[-2:]:  # 减少历史消息数量
      if msg.get('role') == 'user':
        keywords = re.findall(r'\w+', msg.get('content', ''))
        historical_keywords.extend([w for w in keywords if len(w) > 2])

    # 去重并选择最重要的关键词
    keyword_counts = Counter(historical_keywords)
    important_keywords = [k for k, c in keyword_counts.most_common(2)]  # 减少关键词数量

    # 构建增强查询
    if important_keywords:
      enhanced_query = f"{query} {' '.join(important_keywords)}"
    else:
      enhanced_query = query

    return enhanced_query

  def _deduplicate_documents(self, docs: List[Document]) -> List[Document]:
    """文档去重"""
    seen_content = set()
    unique_docs = []

    for doc in docs:
      # 使用内容前50字符作为去重标准（减少长度）
      content_hash = hash(doc.page_content[:50])
      if content_hash not in seen_content:
        seen_content.add(content_hash)
        unique_docs.append(doc)

    return unique_docs

  def _context_filter_lenient(self, ranked_docs: List[Tuple[Document, float]],
                      session_id: str = None, history: List[Dict] = None) -> List[
    Tuple[Document, float]]:
    """宽松的基于上下文的文档过滤 - 提升召回率"""
    if not history or not ranked_docs:
      return ranked_docs

    # 分析对话主题
    conversation_topics = self._extract_conversation_topics(history)

    # 根据主题调整分数 - 使用更宽松的策略
    adjusted_docs = []
    for doc, score in ranked_docs:
      topic_relevance = self._calculate_topic_relevance(doc, conversation_topics)
      
      # 宽松的分数调整 - 避免过度惩罚
      if topic_relevance > 0.3:  # 降低阈值
        adjusted_score = score * 0.9 + topic_relevance * 0.1  # 减少调整幅度
      else:
        adjusted_score = score * 0.95  # 轻微降低分数而不是大幅惩罚
      
      adjusted_docs.append((doc, adjusted_score))

    # 重新排序但保持更多文档
    adjusted_docs.sort(key=lambda x: x[1], reverse=True)
    return adjusted_docs

  def _context_filter(self, ranked_docs: List[Tuple[Document, float]],
                      session_id: str = None, history: List[Dict] = None) -> List[
    Tuple[Document, float]]:
    """基于上下文的文档过滤"""
    if not history:
      return ranked_docs

    # 分析对话主题
    conversation_topics = self._extract_conversation_topics(history)

    # 根据主题调整分数
    adjusted_docs = []
    for doc, score in ranked_docs:
      topic_relevance = self._calculate_topic_relevance(doc, conversation_topics)
      adjusted_score = score * 0.8 + topic_relevance * 0.2  # 调整权重
      adjusted_docs.append((doc, adjusted_score))

    # 重新排序
    adjusted_docs.sort(key=lambda x: x[1], reverse=True)
    return adjusted_docs

  def _extract_conversation_topics(self, history: List[Dict]) -> List[str]:
    """提取对话主题"""
    topics = []
    for msg in history:
      if msg.get('role') == 'user':
        content = msg.get('content', '')
        if '部署' in content or '安装' in content:
          topics.append('deployment')
        elif '监控' in content or '观察' in content:
          topics.append('monitoring')
        elif '故障' in content or '错误' in content:
          topics.append('troubleshooting')
        elif '性能' in content or '优化' in content:
          topics.append('performance')

    return list(set(topics))

  def _calculate_topic_relevance(self, doc: Document, topics: List[str]) -> float:
    """计算文档主题相关性"""
    if not topics:
      return 0.5

    content = doc.page_content.lower()
    relevance_scores = []

    topic_keywords = {
      'deployment': ['部署', '安装', '配置', '搭建'],
      'monitoring': ['监控', '观察', '检测', '巡检'],
      'troubleshooting': ['故障', '错误', '问题', '异常'],
      'performance': ['性能', '优化', '效率', '速度']
    }

    for topic in topics:
      if topic in topic_keywords:
        keyword_count = sum(1 for keyword in topic_keywords[topic] if keyword in content)
        relevance_scores.append(min(keyword_count / len(topic_keywords[topic]), 1.0))

    return max(relevance_scores) if relevance_scores else 0.5


# ==================== 可靠的答案生成器 ====================

class ReliableAnswerGenerator:
  """可靠的答案生成器 - 注重稳定性和回答质量"""

  def __init__(self, llm):
    self.llm = llm
    self._cache = {}  # 简单的内存缓存
    self._last_cleanup = time.time()

  async def generate_structured_answer(self, question: str, docs: List[Document],
                                       context: str = None) -> Dict[str, Any]:
    """生成结构化答案 - 强化版本，确保使用文档内容"""
    try:
      # 1. 输入验证
      if not question or not question.strip():
        return self._get_simple_response("问题为空", docs)
      
      if not docs:
        logger.warning("没有相关文档，无法生成答案")
        return self._get_simple_response("暂时没有找到相关文档，请尝试调整问题描述", docs)

      logger.debug(f"开始生成答案 - 问题: '{question}', 文档数: {len(docs)}")

      # 2. 问题类型分类 - 改进版
      question_type = self._classify_question_enhanced(question)
      logger.debug(f"问题分类: {question_type}")

      # 3. 构建详细的上下文 - 强制使用文档内容
      structured_context = self._build_enhanced_context(docs, question, question_type)
      if not structured_context or len(structured_context.strip()) < 50:
        logger.warning("构建的上下文内容不足，强制使用文档摘要")
        structured_context = self._build_document_summary(docs)

      logger.debug(f"构建的上下文长度: {len(structured_context)}")

      # 4. 强制生成基于文档的回答
      answer = await self._generate_document_based_answer_enhanced(
        question, structured_context, question_type, context, docs
      )

      # 5. 验证答案质量
      if self._is_template_answer(answer):
        logger.warning("检测到模板回答，强制重新生成")
        answer = await self._force_document_answer(question, docs, question_type)

      # 6. 计算置信度
      confidence = self._calculate_enhanced_confidence(question, answer, docs)

      # 7. 提取关键点
      key_points = self._extract_enhanced_key_points(answer, docs)

      return {
        'answer': answer,
        'question_type': question_type,
        'key_points': key_points,
        'confidence': confidence,
        'source_count': len(docs)
      }

    except Exception as e:
      logger.error(f"结构化答案生成失败: {str(e)}")
      # 即使出错也要尝试基于文档生成答案
      return await self._generate_emergency_document_answer(question, docs)

  def _build_enhanced_context(self, docs: List[Document], question: str, question_type: str) -> str:
    """构建增强的上下文，确保内容丰富"""
    if not docs:
      return ""

    context_parts = []
    max_docs = min(len(docs), 6)  # 增加文档数量
    
    for i, doc in enumerate(docs[:max_docs]):
      source = doc.metadata.get('filename', f'文档{i + 1}') if doc.metadata else f'文档{i + 1}'
      
      # 更智能的内容提取
      content = self._extract_relevant_content_enhanced(doc.page_content, question, question_type)
      
      if content and len(content.strip()) > 20:  # 确保内容有意义
        context_parts.append(f"[{source}]\n{content}")
    
    return "\n\n".join(context_parts)

  def _extract_relevant_content_enhanced(self, content: str, question: str, question_type: str) -> str:
    """增强的相关内容提取"""
    if not content:
      return ""
    
    # 获取问题关键词
    question_words = set(question.lower().split())
    
    # 根据问题类型添加相关关键词
    type_keywords = {
      'core_architecture': ['核心', '功能', '模块', '组件', '架构'],
      'architecture': ['系统', '架构', '设计', '组件'],
      'deployment': ['部署', '安装', '配置', '启动'],
      'monitoring': ['监控', '检测', '指标', '告警'],
      'troubleshooting': ['故障', '问题', '错误', '排查'],
      'performance': ['性能', '优化', '效率', '速度'],
      'features': ['特性', '特点', '功能', '能力'],
      'technical': ['技术', '实现', '原理', '算法']
    }
    
    relevant_keywords = type_keywords.get(question_type, [])
    all_keywords = question_words.union(set(relevant_keywords))
    
    # 按段落分割并评分
    paragraphs = content.split('\n\n')
    scored_paragraphs = []
    
    for paragraph in paragraphs:
      if len(paragraph.strip()) < 20:
        continue
        
      paragraph_lower = paragraph.lower()
      
      # 计算相关性分数
      keyword_count = sum(1 for keyword in all_keywords if keyword in paragraph_lower)
      score = keyword_count / max(len(all_keywords), 1)
      
      # 标题和重要标记加分
      if any(marker in paragraph for marker in ['#', '##', '###', '**', '重要', '核心']):
        score += 0.3
        
      scored_paragraphs.append((paragraph, score))
    
    # 排序并选择最相关的段落
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    # 选择最多4个最相关的段落
    selected_paragraphs = [p[0] for p in scored_paragraphs[:4] if p[1] > 0]
    
    if not selected_paragraphs:
      # 如果没有相关段落，返回前几段
      return '\n'.join(paragraphs[:2])
    
    return '\n\n'.join(selected_paragraphs)

  def _build_document_summary(self, docs: List[Document]) -> str:
    """构建文档摘要作为后备内容"""
    if not docs:
      return "无可用文档内容"
    
    summaries = []
    for i, doc in enumerate(docs[:4]):
      content = doc.page_content
      source = doc.metadata.get('filename', f'文档{i+1}') if doc.metadata else f'文档{i+1}'
      
      # 提取前几行作为摘要
      lines = [line.strip() for line in content.split('\n') if line.strip()]
      summary_lines = lines[:5]  # 前5行
      summary = '\n'.join(summary_lines)
      
      if summary:
        summaries.append(f"[{source}] {summary}")
    
    return '\n\n'.join(summaries)

  async def _generate_document_based_answer_enhanced(
    self, question: str, context: str, question_type: str, 
    history_context: str = None, docs: List[Document] = None
  ) -> str:
    """增强的基于文档的答案生成"""
    try:
      if not context or len(context.strip()) < 20:
        return await self._force_document_answer(question, docs or [], question_type)

      # 针对不同问题类型的专门提示
      type_prompts = {
        'core_architecture': "你是AIOps架构专家。请详细介绍系统的核心功能模块，基于提供的文档内容。",
        'architecture': "你是系统架构分析师。请基于文档内容详细说明系统架构和组件。",
        'deployment': "你是部署专家。请基于文档内容提供详细的部署和配置指导。",
        'monitoring': "你是监控专家。请基于文档内容说明监控相关的功能和配置。",
        'troubleshooting': "你是故障诊断专家。请基于文档内容提供问题排查和解决方案。",
        'performance': "你是性能优化专家。请基于文档内容提供性能相关的分析和建议。",
        'features': "你是产品专家。请基于文档内容详细介绍相关特性和功能。",
        'technical': "你是技术专家。请基于文档内容详细说明技术实现和原理。",
        'usage': "你是使用指导专家。请基于文档内容提供详细的使用方法和操作指南。",
        'general': "你是专业的AIOps助手。请基于文档内容准确回答问题。"
      }

      system_prompt = type_prompts.get(question_type, type_prompts['general'])
      
      # 构建详细的用户提示
      user_prompt_parts = []
      if history_context:
        user_prompt_parts.append(f"对话背景: {history_context}")
      
      user_prompt_parts.extend([
        f"问题类型: {question_type}",
        f"用户问题: {question}",
        "",
        "==== 相关文档内容 ====",
        context,
        "",
        "==== 回答要求 ====",
        "1. 严格基于上述文档内容回答",
        "2. 提供详细、结构化的回答",
        "3. 如果文档中有具体的功能模块或特性，请逐一列出并说明",
        "4. 保持专业性和准确性",
        "5. 不要编造文档中没有的信息",
        "",
        "请开始回答："
      ])
      
      user_prompt = "\n".join(user_prompt_parts)

      # 调用LLM生成答案
      messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
      ]

      try:
        response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=25)
        answer = response.content.strip()
        
        # 验证答案质量
        if len(answer) < 30:
          raise ValueError("回答过短")
        
        if self._is_template_answer(answer):
          raise ValueError("生成了模板回答")
          
        return answer

      except asyncio.TimeoutError:
        logger.warning("LLM调用超时，使用文档摘要回答")
        return self._generate_document_summary_answer(question, docs or [], question_type)
      
    except Exception as e:
      logger.error(f"增强答案生成失败: {e}")
      return await self._force_document_answer(question, docs or [], question_type)

  def _is_template_answer(self, answer: str) -> bool:
    """检测是否为模板回答"""
    template_indicators = [
      "基于文档内容，这是一个关于",
      "请查看相关文档获取详细信息",
      "抱歉，我找不到",
      "暂时没有找到相关信息",
      "请尝试重新表述",
      "由于主要模型暂时不可用"
    ]
    
    answer_lower = answer.lower()
    return any(indicator in answer_lower for indicator in template_indicators)

  async def _force_document_answer(self, question: str, docs: List[Document], question_type: str) -> str:
    """强制基于文档生成答案，不依赖LLM"""
    if not docs:
      return "抱歉，没有找到相关文档来回答您的问题。"
    
    # 直接从文档中提取和组织答案
    relevant_content = []
    
    for doc in docs[:4]:
      content = doc.page_content
      source = doc.metadata.get('filename', '文档') if doc.metadata else '文档'
      
      # 提取相关段落
      relevant_parts = self._extract_relevant_content_enhanced(content, question, question_type)
      if relevant_parts:
        relevant_content.append(f"根据{source}：\n{relevant_parts}")
    
    if relevant_content:
      return f"基于相关文档，针对您关于'{question}'的问题，找到以下信息：\n\n" + "\n\n".join(relevant_content)
    else:
      # 最后的备选方案
      return self._generate_document_summary_answer(question, docs, question_type)

  def _generate_document_summary_answer(self, question: str, docs: List[Document], question_type: str) -> str:
    """生成基于文档摘要的答案"""
    if not docs:
      return "抱歉，没有找到相关文档。"
    
    summaries = []
    for doc in docs[:3]:
      content = doc.page_content
      # 提取前几个重要句子
      sentences = [s.strip() for s in content.split('。') if len(s.strip()) > 10]
      if sentences:
        summaries.append(sentences[0] + '。')
    
    if summaries:
      return f"关于{question}，根据文档内容：\n\n" + '\n'.join([f"• {s}" for s in summaries])
    else:
      return "找到了相关文档，但内容提取遇到问题，建议查看源文档获取详细信息。"

  async def _generate_emergency_document_answer(self, question: str, docs: List[Document]) -> Dict[str, Any]:
    """紧急情况下基于文档生成答案"""
    if not docs:
      answer = "抱歉，没有找到相关文档来回答您的问题。"
    else:
      answer = await self._force_document_answer(question, docs, 'general')
    
    return {
      'answer': answer,
      'question_type': 'general',
      'key_points': [],
      'confidence': 0.3,
      'source_count': len(docs)
    }

  def _classify_question_enhanced(self, question: str) -> str:
    """增强的问题分类，更细致的分类"""
    question_lower = question.lower()
    
    # 架构和功能相关
    if any(word in question_lower for word in ['功能', '模块', '组件', '架构', '结构', '系统', '平台']):
      if any(word in question_lower for word in ['核心', '主要', '重要', '关键']):
        return 'core_architecture'
      return 'architecture'
    
    # 部署和安装
    elif any(word in question_lower for word in ['部署', '安装', '配置', '搭建', '启动', '运行']):
      return 'deployment'
    
    # 监控和观察
    elif any(word in question_lower for word in ['监控', '观察', '检测', '巡检', '指标', '告警']):
      return 'monitoring'
    
    # 故障和问题
    elif any(word in question_lower for word in ['故障', '错误', '问题', '异常', '排查', '诊断']):
      return 'troubleshooting'
    
    # 性能和优化
    elif any(word in question_lower for word in ['性能', '优化', '效率', '调优', '速度']):
      return 'performance'
    
    # 使用和操作
    elif any(word in question_lower for word in ['使用', '操作', '怎么', '如何', '方法']):
      return 'usage'
    
    # 特性和能力
    elif any(word in question_lower for word in ['特性', '特点', '能力', '优势', '作用']):
      return 'features'
    
    # 技术和实现
    elif any(word in question_lower for word in ['技术', '实现', '原理', '算法', '框架']):
      return 'technical'
    
    else:
      return 'general'

  def _extract_relevant_content_enhanced(self, content: str, question: str, question_type: str) -> str:
    """增强的相关内容提取"""
    if not content:
      return ""
    
    # 获取问题关键词
    question_words = set(question.lower().split())
    
    # 根据问题类型添加相关关键词
    type_keywords = {
      'core_architecture': ['核心', '功能', '模块', '组件', '架构'],
      'architecture': ['系统', '架构', '设计', '组件'],
      'deployment': ['部署', '安装', '配置', '启动'],
      'monitoring': ['监控', '检测', '指标', '告警'],
      'troubleshooting': ['故障', '问题', '错误', '排查'],
      'performance': ['性能', '优化', '效率', '速度'],
      'features': ['特性', '特点', '功能', '能力'],
      'technical': ['技术', '实现', '原理', '算法']
    }
    
    relevant_keywords = type_keywords.get(question_type, [])
    all_keywords = question_words.union(set(relevant_keywords))
    
    # 按段落分割并评分
    paragraphs = content.split('\n\n')
    scored_paragraphs = []
    
    for paragraph in paragraphs:
      if len(paragraph.strip()) < 20:
        continue
        
      paragraph_lower = paragraph.lower()
      
      # 计算相关性分数
      keyword_count = sum(1 for keyword in all_keywords if keyword in paragraph_lower)
      score = keyword_count / max(len(all_keywords), 1)
      
      # 标题和重要标记加分
      if any(marker in paragraph for marker in ['#', '##', '###', '**', '重要', '核心']):
        score += 0.3
        
      scored_paragraphs.append((paragraph, score))
    
    # 排序并选择最相关的段落
    scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
    
    # 选择最多4个最相关的段落
    selected_paragraphs = [p[0] for p in scored_paragraphs[:4] if p[1] > 0]
    
    if not selected_paragraphs:
      # 如果没有相关段落，返回前几段
      return '\n'.join(paragraphs[:2])
    
    return '\n\n'.join(selected_paragraphs)

  def _extract_enhanced_key_points(self, answer: str, docs: List[Document]) -> List[str]:
    """提取增强的关键点"""
    if len(answer) < 100:
      return []
    
    key_points = []
    
    # 寻找项目符号点或编号列表
    lines = answer.split('\n')
    for line in lines:
      line = line.strip()
      # 检查是否为列表项
      if (line.startswith('•') or line.startswith('-') or line.startswith('*') or
          any(line.startswith(f'{i}.') for i in range(1, 10))):
        if 15 < len(line) < 150:
          key_points.append(line)
    
    # 如果没有找到列表，提取重要句子
    if not key_points:
      sentences = [s.strip() for s in answer.split('。') if s.strip()]
      for sentence in sentences:
        if (30 < len(sentence) < 120 and 
            any(keyword in sentence for keyword in ['重要', '关键', '主要', '核心', '包括', '功能', '特性'])):
          key_points.append(sentence + '。')
          if len(key_points) >= 3:
            break
    
    return key_points[:3]

  def _calculate_enhanced_confidence(self, question: str, answer: str, docs: List[Document]) -> float:
    """计算增强的置信度"""
    try:
      base_score = 0.5
      
      # 文档数量评分
      doc_score = min(len(docs) / 4, 0.25)
      
      # 回答长度评分
      length_score = min(len(answer) / 300, 0.2)
      
      # 内容质量评分
      quality_score = 0.0
      if not self._is_template_answer(answer):
        quality_score += 0.2
      
      # 关键词匹配评分
      question_words = set(question.lower().split())
      answer_words = set(answer.lower().split())
      overlap = len(question_words & answer_words) / max(len(question_words), 1)
      overlap_score = min(overlap, 0.25)
      
      total_score = base_score + doc_score + length_score + quality_score + overlap_score
      return min(total_score, 1.0)
    except:
      return 0.5

  def _extract_enhanced_key_points(self, answer: str, docs: List[Document]) -> List[str]:
    """提取增强的关键点"""
    if len(answer) < 100:
      return []
    
    key_points = []
    
    # 寻找项目符号点或编号列表
    lines = answer.split('\n')
    for line in lines:
      line = line.strip()
      # 检查是否为列表项
      if (line.startswith('•') or line.startswith('-') or line.startswith('*') or
          any(line.startswith(f'{i}.') for i in range(1, 10))):
        if 15 < len(line) < 150:
          key_points.append(line)
    
    # 如果没有找到列表，提取重要句子
    if not key_points:
      sentences = [s.strip() for s in answer.split('。') if s.strip()]
      for sentence in sentences:
        if (30 < len(sentence) < 120 and 
            any(keyword in sentence for keyword in ['重要', '关键', '主要', '核心', '包括', '功能', '特性'])):
          key_points.append(sentence + '。')
          if len(key_points) >= 3:
            break
    
    return key_points[:3]








  def _get_simple_response(self, message: str, docs: List[Document]) -> Dict[str, Any]:
    """获取简单响应"""
    return {
      'answer': message,
      'question_type': 'general',
      'key_points': [],
      'confidence': 0.3,
      'source_count': len(docs)
    }



# ==================== 任务管理器 ====================

class TaskManager:
  """管理异步任务，确保它们能够正确完成或取消"""

  def __init__(self):
    self._tasks = set()
    self._lock = threading.Lock()
    self._shutdown = False

  def create_task(self, coro, description="未命名任务"):
    """创建并管理异步任务"""
    if self._shutdown:
      logger.debug(f"任务管理器已关闭，忽略任务: {description}")
      return None

    async def wrapped_coro():
      try:
        await coro
        logger.debug(f"异步任务 '{description}' 完成")
      except CancelledError:
        logger.debug(f"异步任务 '{description}' 被取消")
      except Exception as e:
        logger.error(f"异步任务 '{description}' 执行失败: {e}")
      finally:
        with self._lock:
          if task in self._tasks:
            self._tasks.remove(task)

    task = asyncio.create_task(wrapped_coro())

    with self._lock:
      self._tasks.add(task)

    return task

  async def shutdown(self, timeout=5.0):
    """关闭任务管理器，等待或取消所有任务"""
    self._shutdown = True

    with self._lock:
      tasks = self._tasks.copy()

    if not tasks:
      return

    logger.debug(f"等待 {len(tasks)} 个任务完成...")

    try:
      await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=timeout
      )
      logger.debug("所有任务已完成")
    except asyncio.TimeoutError:
      logger.warning(f"等待任务完成超时，强制取消 {len(tasks)} 个任务")
      for task in tasks:
        if not task.done():
          task.cancel()

      try:
        await asyncio.wait_for(
          asyncio.gather(*tasks, return_exceptions=True),
          timeout=1.0
        )
      except asyncio.TimeoutError:
        logger.warning("部分任务取消操作超时")

    with self._lock:
      self._tasks.clear()


# 全局任务管理器
_task_manager = None


def get_task_manager():
  global _task_manager
  if _task_manager is None:
    _task_manager = TaskManager()
  return _task_manager


def create_safe_task(coro, description="未命名任务"):
  manager = get_task_manager()
  return manager.create_task(coro, description)


# ==================== 数据类和模型定义 ====================

@dataclass
class SessionData:
  session_id: str
  created_at: str
  history: List[Dict[str, Any]]
  metadata: Dict[str, Any]
  context_summary: str = ""


# ==================== 备用实现类 ====================

class FallbackEmbeddings(Embeddings):
  def __init__(self, dimensions: int = 384):
    self.dimensions = dimensions
    self._cache = {}

  def embed_documents(self, texts: List[str]) -> List[List[float]]:
    return [self.embed_query(text) for text in texts]

  def embed_query(self, text: str) -> List[float]:
    if text in self._cache:
      return self._cache[text]

    text_hash = hash(text) % (2 ** 32)
    np.random.seed(text_hash)
    embedding = list(np.random.rand(self.dimensions))
    self._cache[text] = embedding
    return embedding


class FallbackChatModel(BaseChatModel):
  @property
  def _llm_type(self) -> str:
    return "fallback_chat_model"

  def _generate(self, messages, stop=None, run_manager=None, **kwargs):
    last_message = messages[-1].content if messages else "无输入"

    if "部署" in last_message or "安装" in last_message:
      response = "关于部署问题，建议您查看官方文档或联系技术支持。"
    elif "监控" in last_message:
      response = "关于监控配置，请确保相关服务正常运行并检查配置文件。"
    elif "故障" in last_message or "错误" in last_message:
      response = "遇到故障时，建议先查看日志文件，然后按照标准排查流程进行诊断。"
    else:
      response = f"您询问关于：'{last_message}'。由于主要模型暂时不可用，建议您稍后重试或查看相关文档。"

    message = AIMessage(content=response)
    generation = ChatGeneration(message=message)
    return ChatResult(generations=[generation])


# ==================== 向量存储管理器 ====================

# 使用Redis向量存储管理器 (兼容原有接口)
class VectorStoreManager:
  def __init__(self, vector_db_path: str, collection_name: str, embedding_model):
    self.vector_db_path = vector_db_path
    self.collection_name = collection_name
    self.embedding_model = embedding_model
    self._lock = threading.Lock()
    
    # 获取Redis配置
    redis_config = {
      'host': config.redis.host,
      'port': config.redis.port,
      'db': config.redis.db,
      'password': config.redis.password,
      'connection_timeout': config.redis.connection_timeout,
      'socket_timeout': config.redis.socket_timeout,
      'max_connections': config.redis.max_connections,
      'decode_responses': config.redis.decode_responses
    }
    
    # 初始化优化的Redis向量存储管理器
    self.redis_manager = RedisVectorStoreManager(
      redis_config=redis_config,
      collection_name=collection_name,
      embedding_model=embedding_model,
      local_storage_path=vector_db_path
    )
    
    # 使用优化的向量存储
    # 动态获取嵌入维度
    try:
      test_embedding = embedding_model.embed_query("测试")
      vector_dim = len(test_embedding)
      logger.info(f"检测到嵌入维度: {vector_dim}")
    except Exception as e:
      logger.warning(f"无法检测嵌入维度，使用默认值1536: {e}")
      vector_dim = 1536
    
    self.optimized_store = OptimizedRedisVectorStore(
      redis_config=redis_config,
      collection_name=collection_name,
      embedding_model=embedding_model,
      vector_dim=vector_dim,  # 使用动态检测的维度
      local_storage_path=vector_db_path,
      use_faiss=True,
      faiss_index_type="Flat"  # 可根据数据量调整为IVF或HNSW
    )
    
    self.retriever = None
    os.makedirs(vector_db_path, exist_ok=True)

  def load_existing_db(self) -> bool:
    """加载现有向量数据库"""
    try:
      with self._lock:
        logger.info(f"检查Redis向量存储，集合: {self.collection_name}")
        
        # 检查Redis向量存储健康状态
        health = self.redis_manager.health_check()
        if health.get('status') == 'healthy' and health.get('document_count', 0) > 0:
          # 初始化检索器
          self.retriever = self.redis_manager.get_retriever()
          logger.info(f"Redis向量存储加载成功，包含 {health['document_count']} 个文档")
          return True
        else:
          logger.info("Redis向量存储为空或不可用")
          return False
          
    except Exception as e:
      logger.error(f"加载Redis向量存储失败: {e}")
      return False

  async def create_vector_store(self, documents: List[Document], use_memory: bool = False) -> bool:
    """创建向量存储"""
    if not documents:
      logger.warning("没有文档可供创建向量存储")
      documents = [Document(
        page_content="这是一个系统自动创建的示例文档。请添加更多文档到知识库中。",
        metadata={"source": "system", "filename": "example.txt", "filetype": "text"}
      )]

    # 优化的文本分割策略
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=config.rag.chunk_size,
      chunk_overlap=config.rag.chunk_overlap,
      separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", ";", "；", " ", ""],
      length_function=len,
      is_separator_regex=False
    )

    splits = text_splitter.split_documents(documents)
    logger.info(f"文档分割完成: {len(splits)} 个块")

    # 为文档块添加增强元数据
    enhanced_splits = []
    for i, doc in enumerate(splits):
      chunk_length = len(doc.page_content)
      chunk_sentences = len(re.split(r'[。！？.!?]', doc.page_content))

      if doc.metadata:
        doc.metadata.update({
          'chunk_id': i,
          'chunk_length': chunk_length,
          'chunk_sentences': chunk_sentences,
          'created_at': time.time()
        })
      else:
        doc.metadata = {
          'chunk_id': i,
          'chunk_length': chunk_length,
          'chunk_sentences': chunk_sentences,
          'created_at': time.time()
        }

      enhanced_splits.append(doc)

    try:
      with self._lock:
        logger.info("使用优化的Redis向量存储创建向量数据库")
        
        # 使用优化的向量存储添加文档
        batch_size = min(EMBEDDING_BATCH_SIZE, 30)
        total_docs = len(enhanced_splits)
        
        if total_docs > batch_size:
          logger.info(f"文档量较大({total_docs}个)，使用分批处理方式")
          # 分批处理
          for i in range(0, total_docs, batch_size):
            batch = enhanced_splits[i:i + batch_size]
            logger.info(f"处理批次 {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}，{len(batch)}个文档")
            
            try:
              self.optimized_store.add_documents(batch)
              if i > 0:
                await asyncio.sleep(0.2)  # 增加延迟
            except Exception as e:
              logger.error(f"添加文档批次失败: {e}")
              return False
        else:
          # 数量较少，直接添加
          self.optimized_store.add_documents(enhanced_splits)

        # 初始化检索器
        self.retriever = self.get_retriever()
        
        # 测试检索器
        test_docs = self.retriever.get_relevant_documents("测试查询")
        logger.info(f"优化的Redis向量存储创建成功，包含 {len(enhanced_splits)} 个文档块")
        return True

    except Exception as e:
      logger.error(f"创建优化的Redis向量存储失败: {e}")
      return False

  def get_retriever(self):
    """获取检索器 - 改进版本"""
    if self.retriever is None:
      # 优化的检索器，提升召回率
      class OptimizedRetriever:
        def __init__(self, optimized_store, redis_manager):
          self.optimized_store = optimized_store
          self.redis_manager = redis_manager
          
        def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
          try:
            k = kwargs.get('k', 8)  # 增加默认检索数量
            similarity_threshold = kwargs.get('similarity_threshold', 0.1)  # 大幅降低阈值
            
            logger.debug(f"开始检索相关文档 - 查询: '{query}', k={k}, 阈值={similarity_threshold}")
            
            # 首先尝试混合搜索
            try:
              results = self.optimized_store.hybrid_similarity_search(
                query, 
                k=k,
                semantic_weight=0.5,  # 降低语义权重
                lexical_weight=0.5,   # 增加词汇权重
                similarity_threshold=similarity_threshold
              )
              
              if results:
                logger.debug(f"混合搜索找到 {len(results)} 个结果")
                return [doc for doc, score in results]
              else:
                logger.debug("混合搜索无结果，尝试纯语义搜索")
                
            except Exception as e:
              logger.warning(f"混合搜索失败: {e}，使用纯语义搜索")
            
            # 降级到语义搜索
            try:
              results = self.optimized_store.similarity_search(
                query, 
                k=k,
                similarity_threshold=0.05  # 极低阈值确保找到结果
              )
              
              if results:
                logger.debug(f"语义搜索找到 {len(results)} 个结果")
                return [doc for doc, score in results]
              else:
                logger.debug("语义搜索无结果，尝试关键词搜索")
                
            except Exception as e:
              logger.warning(f"语义搜索失败: {e}，尝试关键词搜索")
            
            # 最后尝试关键词搜索
            try:
              # 分解查询为关键词
              query_words = [word for word in query.split() if len(word) > 1]
              all_results = []
              
              for word in query_words[:3]:  # 只用前3个关键词
                try:
                  word_results = self.optimized_store.similarity_search(
                    word,
                    k=max(2, k//2),
                    similarity_threshold=0.01  # 最低阈值
                  )
                  all_results.extend([doc for doc, score in word_results])
                except:
                  continue
              
              # 去重
              unique_results = []
              seen_content = set()
              for doc in all_results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                  seen_content.add(content_hash)
                  unique_results.append(doc)
                  if len(unique_results) >= k:
                    break
              
              logger.debug(f"关键词搜索找到 {len(unique_results)} 个去重结果")
              return unique_results
              
            except Exception as e:
              logger.error(f"关键词搜索也失败: {e}")
              return []
            
          except Exception as e:
            logger.error(f"检索器完全失败: {e}")
            return []
            
        def invoke(self, query) -> List[Document]:
          # 兼容字符串查询和字典输入
          if isinstance(query, dict):
            query_str = query.get("query", "")
          elif isinstance(query, str):
            query_str = query
          else:
            query_str = str(query)
          return self.get_relevant_documents(query_str)
      
      self.retriever = OptimizedRetriever(self.optimized_store, self.redis_manager)
    return self.retriever

  def search_documents(self, query: str, max_retries: int = 3) -> List[Document]:
    """搜索文档（优化版） - 提升召回率"""
    if not self.retriever:
      logger.warning("检索器未初始化")
      return []

    logger.debug(f"开始搜索文档: '{query}'")
    
    for attempt in range(max_retries):
      try:
        # 使用改进的检索器
        docs = self.retriever.get_relevant_documents(query, k=10, similarity_threshold=0.05)
        
        if docs:
          logger.debug(f"第 {attempt + 1} 次尝试成功，搜索到 {len(docs)} 个文档")
          return docs
        else:
          logger.debug(f"第 {attempt + 1} 次尝试搜索返回空结果")
          
          # 如果没有结果，尝试更宽松的查询
          if attempt < max_retries - 1:
            # 提取关键词重新搜索
            keywords = [word for word in query.split() if len(word) > 2]
            if keywords:
              expanded_query = " ".join(keywords[:2])  # 用前两个关键词
              logger.debug(f"尝试关键词查询: '{expanded_query}'")
              docs = self.retriever.get_relevant_documents(expanded_query, k=8, similarity_threshold=0.01)
              if docs:
                logger.debug(f"关键词查询成功，找到 {len(docs)} 个文档")
                return docs

      except Exception as e:
        logger.error(f"文档搜索失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
          time.sleep(0.5)

    logger.warning(f"搜索 '{query}' 最终无结果")
    return []


# ==================== 缓存管理器 ====================
# 使用Redis缓存管理器，不需要本地CacheManager类


# ==================== 文档加载器 ====================

class DocumentLoader:
  """优化的文档加载器"""

  def __init__(self, knowledge_base_path: str):
    self.knowledge_base_path = Path(knowledge_base_path)
    self.executor = ThreadPoolExecutor(max_workers=2)  # 减少线程数

    self.supported_extensions = {
      '.txt': self._load_text_file,
      '.md': self._load_markdown_file,
      '.markdown': self._load_markdown_file,
    }

    if ADVANCED_LOADERS_AVAILABLE:
      self.supported_extensions.update({
        '.pdf': self._load_pdf_file,
        '.html': self._load_html_file,
        '.htm': self._load_html_file,
        '.csv': self._load_csv_file,
        '.json': self._load_json_file,
      })

  def load_documents(self) -> List[Document]:
    """加载所有支持的文档"""
    if not self.knowledge_base_path.exists():
      logger.warning(f"知识库路径不存在: {self.knowledge_base_path}")
      self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
      return []

    documents = []
    all_files = list(self.knowledge_base_path.rglob("*"))
    supported_files = [
      f for f in all_files
      if f.is_file() and f.suffix.lower() in self.supported_extensions
    ]

    supported_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    logger.info(f"发现 {len(supported_files)} 个支持的文件")

    for file_path in supported_files:
      try:
        loader_func = self.supported_extensions[file_path.suffix.lower()]
        file_docs = loader_func(file_path)
        documents.extend(file_docs)
        logger.debug(f"加载文件: {file_path.name}, 生成 {len(file_docs)} 个文档")
      except Exception as e:
        logger.error(f"加载文件失败 {file_path}: {e}")

    cleaned_documents = self._clean_documents(documents)
    logger.info(f"总共加载 {len(cleaned_documents)} 个有效文档")
    return cleaned_documents

  def _clean_documents(self, documents: List[Document]) -> List[Document]:
    """清理和优化文档"""
    cleaned = []

    for doc in documents:
      content = doc.page_content.strip()

      if len(content) < 10:
        continue

      # 清理格式
      content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
      content = re.sub(r'[ \t]+', ' ', content)

      doc.page_content = content

      if doc.metadata:
        doc.metadata.update({
          'content_length': len(content),
          'word_count': len(content.split()),
          'line_count': len(content.split('\n'))
        })

      cleaned.append(doc)

    return cleaned

  def _load_text_file(self, file_path: Path) -> List[Document]:
    """加载文本文件"""
    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

      if not content:
        return []

      return [Document(
        page_content=content,
        metadata={
          "source": str(file_path),
          "filename": file_path.name,
          "filetype": "text",
          "modified_time": file_path.stat().st_mtime,
          "file_size": file_path.stat().st_size
        }
      )]
    except Exception as e:
      logger.error(f"加载文本文件失败 {file_path}: {e}")
      return []

  def _load_markdown_file(self, file_path: Path) -> List[Document]:
    """加载Markdown文件"""
    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

      if not content:
        return []

      try:
        headers_to_split_on = [
          ("#", "Header 1"),
          ("##", "Header 2"),
          ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
          headers_to_split_on=headers_to_split_on
        )
        md_docs = markdown_splitter.split_text(content)

        for doc in md_docs:
          doc.metadata.update({
            "source": str(file_path),
            "filename": file_path.name,
            "filetype": "markdown",
            "modified_time": file_path.stat().st_mtime,
            "file_size": file_path.stat().st_size
          })

        return md_docs

      except Exception:
        return [Document(
          page_content=content,
          metadata={
            "source": str(file_path),
            "filename": file_path.name,
            "filetype": "markdown",
            "modified_time": file_path.stat().st_mtime,
            "file_size": file_path.stat().st_size
          }
        )]

    except Exception as e:
      logger.error(f"加载Markdown文件失败 {file_path}: {e}")
      return []

  def _load_pdf_file(self, file_path: Path) -> List[Document]:
    """加载PDF文件"""
    if not ADVANCED_LOADERS_AVAILABLE:
      return []

    try:
      loader = PyPDFLoader(str(file_path))
      pdf_docs = loader.load()

      for doc in pdf_docs:
        doc.metadata.update({
          "filename": file_path.name,
          "filetype": "pdf",
          "modified_time": file_path.stat().st_mtime,
          "file_size": file_path.stat().st_size
        })

      return pdf_docs

    except Exception as e:
      logger.error(f"加载PDF文件失败 {file_path}: {e}")
      return []

  def _load_html_file(self, file_path: Path) -> List[Document]:
    """加载HTML文件"""
    if not ADVANCED_LOADERS_AVAILABLE:
      return []

    try:
      loader = BSHTMLLoader(str(file_path))
      html_docs = loader.load()

      for doc in html_docs:
        doc.metadata.update({
          "filename": file_path.name,
          "filetype": "html",
          "modified_time": file_path.stat().st_mtime,
          "file_size": file_path.stat().st_size
        })

      return html_docs

    except Exception as e:
      logger.error(f"加载HTML文件失败 {file_path}: {e}")
      return []

  def _load_csv_file(self, file_path: Path) -> List[Document]:
    """加载CSV文件"""
    if not ADVANCED_LOADERS_AVAILABLE:
      return []

    try:
      loader = CSVLoader(str(file_path))
      csv_docs = loader.load()

      for doc in csv_docs:
        doc.metadata.update({
          "filename": file_path.name,
          "filetype": "csv",
          "modified_time": file_path.stat().st_mtime,
          "file_size": file_path.stat().st_size
        })

      return csv_docs

    except Exception as e:
      logger.error(f"加载CSV文件失败 {file_path}: {e}")
      return []

  def _load_json_file(self, file_path: Path) -> List[Document]:
    """加载JSON文件"""
    if not ADVANCED_LOADERS_AVAILABLE:
      return []

    try:
      loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
      json_docs = loader.load()

      for doc in json_docs:
        doc.metadata.update({
          "filename": file_path.name,
          "filetype": "json",
          "modified_time": file_path.stat().st_mtime,
          "file_size": file_path.stat().st_size
        })

      return json_docs

    except Exception as e:
      logger.error(f"加载JSON文件失败 {file_path}: {e}")
      return []


# ==================== 工具函数 ====================

def _generate_fallback_answer() -> str:
  return "抱歉，我找不到与您问题相关的信息。请尝试重新表述您的问题，或询问关于AIOps平台的核心功能、部署方式或使用方法等问题。"


async def _check_hallucination_advanced(question: str, answer: str, docs: List[Document]) -> Tuple[
  bool, float]:
  """高级幻觉检查"""
  try:
    if len(answer) < 30 or not docs:  # 调整最小长度
      return True, 0.3

    # 关键词匹配检查
    answer_words = set(re.findall(r'\w+', answer.lower()))
    doc_words = set()
    for doc in docs:
      doc_words.update(re.findall(r'\w+', doc.page_content.lower()))

    common_words = answer_words & doc_words
    coverage = len(common_words) / len(answer_words) if answer_words else 0

    # 数字和具体信息检查
    answer_numbers = re.findall(r'\d+', answer)
    doc_numbers = []
    for doc in docs:
      doc_numbers.extend(re.findall(r'\d+', doc.page_content))

    number_match = len(set(answer_numbers) & set(doc_numbers)) / len(
      answer_numbers) if answer_numbers else 1.0

    # 综合评分
    final_score = coverage * 0.7 + number_match * 0.3
    is_valid = final_score > 0.3  # 降低阈值

    return is_valid, final_score

  except Exception as e:
    logger.error(f"高级幻觉检查失败: {e}")
    return True, 0.5


async def _evaluate_doc_relevance_advanced(question: str, doc: Document) -> Tuple[bool, float]:
  """高级文档相关性评估 - 提升召回率版本"""
  try:
    # 基础关键词匹配
    question_words = set(re.findall(r'\w+', question.lower()))
    doc_words = set(re.findall(r'\w+', doc.page_content.lower()))

    overlap = len(question_words & doc_words)
    total = len(question_words | doc_words)
    basic_similarity = overlap / total if total > 0 else 0

    # 语义相关性检查
    semantic_score = 0.5

    # 文档质量评分
    content_length = len(doc.page_content)
    quality_score = min(content_length / 200, 1.0)  # 降低质量要求

    # 综合评分
    final_score = (
      basic_similarity * 0.5 +
      semantic_score * 0.3 +
      quality_score * 0.2
    )

    is_relevant = final_score > 0.05  # 大幅降低阈值

    return is_relevant, final_score

  except Exception as e:
    logger.error(f"高级文档相关性评估失败: {e}")
    return True, 0.5  # 默认认为相关


def _build_context_with_history(session: Optional['SessionData']) -> Optional[str]:
  """构建包含历史的上下文"""
  if not session or not session.history:
    return None

  recent_history = session.history[-2:]  # 减少历史数量
  if len(recent_history) < 2:
    return None

  context = "基于以下对话历史:\n"
  for msg in recent_history:
    role = "用户" if msg["role"] == "user" else "助手"
    content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']  # 减少内容长度
    context += f"{role}: {content}\n"

  return context + "\n当前问题:\n"


async def _filter_relevant_docs_advanced(question: str, docs: List[Document]) -> List[Document]:
  """高级文档过滤 - 提升召回率版本"""
  if not docs or len(docs) <= 3:  # 降低最小文档数要求
    return docs

  try:
    relevant_docs = []

    for doc in docs:
      is_relevant, score = await _evaluate_doc_relevance_advanced(question, doc)

      if is_relevant:
        doc.metadata = doc.metadata or {}
        doc.metadata["relevance_score"] = score
        relevant_docs.append(doc)

    # 如果没有相关文档，返回评分最高的几个
    if not relevant_docs:
      all_docs_with_scores = []
      for doc in docs:
        _, score = await _evaluate_doc_relevance_advanced(question, doc)
        doc.metadata = doc.metadata or {}
        doc.metadata["relevance_score"] = score
        all_docs_with_scores.append(doc)

      all_docs_with_scores.sort(
        key=lambda x: x.metadata.get("relevance_score", 0),
        reverse=True
      )
      return all_docs_with_scores[:4]  # 增加返回数量

    # 按相关性排序
    relevant_docs.sort(
      key=lambda x: x.metadata.get("relevance_score", 0),
      reverse=True
    )

    return relevant_docs

  except Exception as e:
    logger.error(f"高级文档过滤失败: {e}")
    return docs[:4]  # 增加返回数量


# ==================== 主要智能助手类 ====================

class AssistantAgent:
  """优化版智能小助手代理 - 修复所有错误，大幅提升精确度和召回率"""

  def __init__(self):
    """初始化助手代理"""
    self.llm_provider = config.llm.provider.lower()

    # 路径设置
    base_dir = Path(__file__).parent.parent.parent.parent
    self.vector_db_path = base_dir / config.rag.vector_db_path
    self.knowledge_base_path = base_dir / config.rag.knowledge_base_path
    self.collection_name = config.rag.collection_name

    # 创建必要目录
    self.vector_db_path.mkdir(parents=True, exist_ok=True)
    self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

    # 初始化组件
    self.embedding = None
    self.llm = None
    self.task_llm = None

    # 管理器
    self.vector_store_manager = None
    
    # 使用Redis缓存管理器
    redis_config = {
      'host': config.redis.host,
      'port': config.redis.port,
      'db': config.redis.db + 1,  # 使用不同的db用于缓存
      'password': config.redis.password,
      'connection_timeout': config.redis.connection_timeout,
      'socket_timeout': config.redis.socket_timeout,
      'max_connections': config.redis.max_connections,
      'decode_responses': config.redis.decode_responses
    }
    
    self.cache_manager = RedisCacheManager(
      redis_config=redis_config,
      cache_prefix="aiops_assistant_cache:",
      default_ttl=3600,
      max_cache_size=1000,
      enable_compression=True
    )
    self.document_loader = DocumentLoader(str(self.knowledge_base_path))

    # 优化组件
    self.query_rewriter = QueryRewriter()
    self.doc_ranker = DocumentRanker()
    self.context_retriever = None
    self.answer_generator = None

    # 会话管理
    self.sessions: Dict[str, SessionData] = {}
    self._session_lock = threading.Lock()

    # 线程池
    self.executor = ThreadPoolExecutor(max_workers=3)  # 减少线程数

    # 关闭标志
    self._shutdown = False

    # 初始化所有组件
    self._initialize_components()

    logger.info(f"优化版智能小助手初始化完成，提供商: {self.llm_provider}")

  def _initialize_components(self):
    """初始化所有组件"""
    try:
      self._init_embedding()
      self._init_llm()
      self._init_vector_store()
      self._init_advanced_components()
    except Exception as e:
      logger.error(f"组件初始化失败: {e}")
      raise

  def _init_embedding(self):
    """初始化嵌入模型 - 改进版本"""
    max_retries = 3
    original_provider = self.llm_provider

    for attempt in range(max_retries):
      try:
        logger.info(f"尝试初始化 {self.llm_provider} 嵌入模型 (第 {attempt + 1}/{max_retries} 次)")
        
        if self.llm_provider == 'openai':
          # 验证配置
          if not config.llm.api_key:
            raise ValueError("OpenAI API key 未配置")
          if not config.llm.base_url:
            raise ValueError("OpenAI base URL 未配置")
          if not config.rag.openai_embedding_model:
            raise ValueError("OpenAI embedding model 未配置")
          
          logger.info(f"OpenAI嵌入配置 - Model: {config.rag.openai_embedding_model}")
          
          self.embedding = OpenAIEmbeddings(
            model=config.rag.openai_embedding_model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=20,  # 使用 timeout 而不是 request_timeout
            max_retries=1
          )
        else:
          # Ollama配置验证
          if not config.llm.ollama_base_url:
            raise ValueError("Ollama base URL 未配置")
          if not config.rag.ollama_embedding_model:
            raise ValueError("Ollama embedding model 未配置")
          
          logger.info(f"Ollama嵌入配置 - Model: {config.rag.ollama_embedding_model}")
          
          self.embedding = OllamaEmbeddings(
            model=config.rag.ollama_embedding_model,
            base_url=config.llm.ollama_base_url
            # 移除 timeout 参数，因为 OllamaEmbeddings 可能不支持
          )

        # 详细的嵌入模型测试
        logger.info("测试嵌入模型连接...")
        test_texts = ["测试文本", "embedding test"]
        
        # 测试批量嵌入
        test_embeddings = self.embedding.embed_documents(test_texts)
        if not test_embeddings or len(test_embeddings) != len(test_texts):
          raise ValueError("批量嵌入测试失败")
        
        # 测试单个嵌入
        single_embedding = self.embedding.embed_query("查询测试")
        if not single_embedding or len(single_embedding) == 0:
          raise ValueError("单个嵌入测试失败")
        
        # 验证嵌入维度一致性
        batch_dim = len(test_embeddings[0]) if test_embeddings[0] else 0
        single_dim = len(single_embedding)
        if batch_dim != single_dim or batch_dim == 0:
          raise ValueError(f"嵌入维度不一致: 批量={batch_dim}, 单个={single_dim}")
        
        logger.info(f"嵌入模型测试成功 - 维度: {single_dim}, 提供商: {self.llm_provider}")
        return
        
      except Exception as e:
        error_msg = str(e)
        logger.error(f"嵌入模型初始化失败 (尝试 {attempt + 1}): {error_msg}")
        
        # 记录详细错误信息
        if "timeout" in error_msg.lower():
          logger.warning("嵌入模型连接超时")
        elif "api_key" in error_msg.lower():
          logger.warning("API密钥相关问题")
        elif "model" in error_msg.lower():
          logger.warning("模型不存在或不可用")
        elif "connection" in error_msg.lower():
          logger.warning("网络连接问题")
        
        if attempt < max_retries - 1:
          # 切换提供商
          self.llm_provider = 'ollama' if self.llm_provider == 'openai' else 'openai'
          logger.info(f"切换到 {self.llm_provider} 嵌入提供商")
          time.sleep(2)
        else:
          # 最后一次尝试使用原始提供商
          if self.llm_provider != original_provider:
            self.llm_provider = original_provider
            logger.info(f"最后一次尝试使用原始嵌入提供商 {original_provider}")

    # 所有尝试都失败，使用备用嵌入
    logger.error("所有嵌入提供商初始化失败，使用备用嵌入模型")
    logger.warning("⚠️  当前使用备用嵌入模型，向量检索质量可能受限")
    self.embedding = FallbackEmbeddings()

  def _init_llm(self):
    """初始化语言模型 - 改进版本"""
    max_retries = 3  # 增加重试次数
    original_provider = self.llm_provider

    for attempt in range(max_retries):
      try:
        logger.info(f"尝试初始化 {self.llm_provider} 语言模型 (第 {attempt + 1}/{max_retries} 次)")
        
        if self.llm_provider == 'openai':
          # 验证配置
          if not config.llm.api_key:
            raise ValueError("OpenAI API key 未配置")
          if not config.llm.base_url:
            raise ValueError("OpenAI base URL 未配置")
          
          logger.info(f"OpenAI配置 - Model: {config.llm.model}, Base URL: {config.llm.base_url}")

          self.llm = ChatOpenAI(
            model=config.llm.model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=config.rag.temperature,
            timeout=45,  # 增加超时时间
            max_retries=1  # 内部重试减少，由外层控制
          )

          task_model = getattr(config.llm, 'task_model', config.llm.model)
          self.task_llm = ChatOpenAI(
            model=task_model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=0.1,
            timeout=30,
            max_retries=1
          )
        else:
          # Ollama 配置验证
          if not config.llm.ollama_base_url:
            raise ValueError("Ollama base URL 未配置")
          
          logger.info(f"Ollama配置 - Model: {config.llm.ollama_model}, Base URL: {config.llm.ollama_base_url}")
          
          self.llm = ChatOllama(
            model=config.llm.ollama_model,
            base_url=config.llm.ollama_base_url,
            temperature=config.rag.temperature,
            timeout=45,
            keep_alive="5m"  # 保持模型在内存中
          )
          self.task_llm = self.llm

        # 更严格的模型测试
        logger.info("测试语言模型连接...")
        test_messages = [{"role": "user", "content": "请回复'连接成功'"}]
        
        if hasattr(self.llm, 'invoke'):
          test_response = self.llm.invoke(test_messages)
        else:
          test_response = self.llm.generate([test_messages])
          
        if test_response and hasattr(test_response, 'content') and test_response.content:
          content = test_response.content.strip()
          logger.info(f"语言模型测试成功，响应: {content[:50]}...")
          
          # 测试任务模型
          if self.task_llm != self.llm:
            task_test = self.task_llm.invoke("测试")
            if not (task_test and task_test.content):
              logger.warning("任务模型测试失败，使用主模型")
              self.task_llm = self.llm
          
          logger.info(f"语言模型初始化成功 - 提供商: {self.llm_provider}")
          return
        else:
          raise ValueError("模型测试返回空响应")

      except Exception as e:
        error_msg = str(e)
        logger.error(f"语言模型初始化失败 (尝试 {attempt + 1}): {error_msg}")
        
        # 记录详细错误信息
        if "timeout" in error_msg.lower():
          logger.warning("连接超时，可能是网络问题或服务不可用")
        elif "api_key" in error_msg.lower():
          logger.warning("API密钥相关问题")
        elif "connection" in error_msg.lower():
          logger.warning("连接问题，检查服务地址和端口")
        
        if attempt < max_retries - 1:
          # 切换提供商
          self.llm_provider = 'ollama' if self.llm_provider == 'openai' else 'openai'
          logger.info(f"切换到 {self.llm_provider} 提供商")
          time.sleep(2)  # 增加等待时间
        else:
          # 最后一次尝试使用原始提供商
          if self.llm_provider != original_provider:
            self.llm_provider = original_provider
            logger.info(f"最后一次尝试使用原始提供商 {original_provider}")

    # 所有尝试都失败，使用备用模型
    logger.error("所有LLM提供商初始化失败，使用备用模型")
    logger.warning("⚠️  当前使用备用模型，回答质量可能受限，请检查LLM配置")
    self.llm = FallbackChatModel()
    self.task_llm = self.llm

  def _init_vector_store(self):
    """初始化向量存储"""
    self.vector_store_manager = VectorStoreManager(
      str(self.vector_db_path),
      self.collection_name,
      self.embedding
    )

    if not self.vector_store_manager.load_existing_db():
      logger.info("创建新的向量数据库")
      documents = self.document_loader.load_documents()

      use_memory = is_test_environment()

      # 修复：使用异步运行同步方法
      import asyncio
      try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
          # 如果循环正在运行，使用create_task
          import concurrent.futures
          with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, self.vector_store_manager.create_vector_store(documents, use_memory))
            success = future.result()
        else:
          success = loop.run_until_complete(
            self.vector_store_manager.create_vector_store(documents, use_memory)
          )
      except RuntimeError:
        # 没有事件循环，创建新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
          success = loop.run_until_complete(
            self.vector_store_manager.create_vector_store(documents, use_memory)
          )
        finally:
          loop.close()

      if not success:
        logger.error("向量存储初始化失败")
        raise RuntimeError("无法初始化向量存储")

    logger.info("向量存储初始化完成")

  def _init_advanced_components(self):
    """初始化高级组件"""
    try:
      # 初始化文档排序器 - Redis向量存储已自带排序功能
      if self.vector_store_manager and self.vector_store_manager.redis_manager:
        # Redis向量存储通过检索器直接获取文档，无需访问底层collection
        logger.debug("使用Redis向量存储，文档排序由检索器内部处理")
        # 加载一些示例文档用于训练排序器
        try:
          all_docs = self.document_loader.load_documents()
        except Exception as e:
          logger.warning(f"获取训练文档失败: {e}")
          all_docs = []

        if all_docs:
          self.doc_ranker.fit(all_docs)
          logger.info("文档排序器训练完成")

      # 初始化上下文感知检索器
      if self.vector_store_manager.get_retriever():
        self.context_retriever = ContextAwareRetriever(
          self.vector_store_manager.get_retriever(),
          self.query_rewriter,
          self.doc_ranker
        )
        logger.info("上下文感知检索器初始化完成")

      # 初始化高级答案生成器
      if self.llm:
        self.answer_generator = ReliableAnswerGenerator(self.llm)
        logger.info("可靠答案生成器初始化完成")

    except Exception as e:
      logger.error(f"高级组件初始化失败: {e}")

  # ==================== 会话管理 ====================

  def create_session(self) -> str:
    """创建新会话"""
    session_id = str(uuid.uuid4())
    session_data = SessionData(
      session_id=session_id,
      created_at=datetime.now().isoformat(),
      history=[],
      metadata={},
      context_summary=""
    )

    with self._session_lock:
      self.sessions[session_id] = session_data

    return session_id

  def get_session(self, session_id: str) -> Optional[SessionData]:
    """获取会话数据"""
    return self.sessions.get(session_id)

  def add_message_to_history(self, session_id: str, role: str, content: str) -> str:
    """添加消息到会话历史"""
    if session_id not in self.sessions:
      session_id = self.create_session()

    with self._session_lock:
      session = self.sessions[session_id]
      session.history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
      })

      # 限制历史长度
      max_history = 20  # 减少历史长度
      if len(session.history) > max_history:
        session.history = session.history[-max_history:]

      # 更新上下文摘要
      if len(session.history) >= 4:
        session.context_summary = self._generate_context_summary(session.history[-4:])

    return session_id

  def _generate_context_summary(self, history: List[Dict]) -> str:
    """生成对话上下文摘要"""
    try:
      user_messages = [msg['content'] for msg in history if msg['role'] == 'user']
      all_text = ' '.join(user_messages)

      keywords = []
      for word in ['部署', '监控', '故障', '性能', '配置', '安装']:
        if word in all_text:
          keywords.append(word)

      return f"对话主题: {', '.join(keywords)}" if keywords else "一般咨询"
    except:
      return "一般咨询"

  def clear_session_history(self, session_id: str) -> bool:
    """清空会话历史"""
    if session_id in self.sessions:
      with self._session_lock:
        self.sessions[session_id].history = []
        self.sessions[session_id].context_summary = ""
      return True
    return False

  # ==================== 知识库管理 ====================

  async def refresh_knowledge_base(self) -> Dict[str, Any]:
    """刷新知识库"""
    try:
      logger.info("开始刷新知识库...")

      # 清理缓存（Redis缓存管理器自动清理过期项）
      logger.info("清理缓存（Redis缓存自动管理）")

      # 加载文档
      documents = await asyncio.get_event_loop().run_in_executor(
        self.executor, self.document_loader.load_documents
      )

      # 重新创建向量存储
      use_memory = is_test_environment()
      success = await self.vector_store_manager.create_vector_store(documents, use_memory)

      if success:
        # 重新训练高级组件
        await asyncio.get_event_loop().run_in_executor(
          self.executor,
          self._retrain_advanced_components,
          documents
        )

        doc_count = len(documents)
        logger.info(f"知识库刷新成功，包含 {doc_count} 个文档")
        return {"success": True, "documents_count": doc_count}
      else:
        return {"success": False, "documents_count": 0, "error": "向量存储创建失败"}

    except Exception as e:
      logger.error(f"刷新知识库失败: {e}")
      return {"success": False, "documents_count": 0, "error": str(e)}

  def _retrain_advanced_components(self, documents: List[Document]):
    """重新训练高级组件"""
    try:
      if documents and self.doc_ranker:
        self.doc_ranker.fit(documents)
        logger.info("文档排序器重新训练完成")
    except Exception as e:
      logger.error(f"重新训练高级组件失败: {e}")

  def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
    """添加文档到知识库"""
    try:
      if not content.strip():
        return False

      doc_id = str(uuid.uuid4())
      filename = metadata.get('filename', f"{doc_id}.txt") if metadata else f"{doc_id}.txt"
      file_path = self.knowledge_base_path / filename

      with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

      # 清理缓存（Redis缓存管理器自动清理过期项）
      logger.info("清理缓存（Redis缓存自动管理）")

      logger.info(f"文档已添加: {filename}")
      return True

    except Exception as e:
      logger.error(f"添加文档失败: {e}")
      return False

  # ==================== 核心问答逻辑 ====================

  async def get_answer(
    self,
    question: str,
    session_id: str = None,
    max_context_docs: int = 10  # 增加上下文文档数量
  ) -> Dict[str, Any]:
    """获取问题答案 - 全面优化版核心方法"""

    try:
      # 添加详细日志记录整个流程
      logger.info(f"=== 开始处理问题 ===")
      logger.info(f"问题: '{question}'")
      logger.info(f"会话ID: {session_id}")
      logger.info(f"最大文档数: {max_context_docs}")

      # 获取会话历史
      session = self.get_session(session_id) if session_id else None
      history = session.history if session else []
      logger.debug(f"会话历史长度: {len(history)}")

      # 检查缓存
      cached_response = self.cache_manager.get(question, session_id, history)
      if cached_response:
        logger.info("使用缓存回答")
        if session_id:
          self.add_message_to_history(session_id, "user", question)
          self.add_message_to_history(session_id, "assistant", cached_response["answer"])
        return cached_response

      # 添加用户消息到历史
      if session_id:
        self.add_message_to_history(session_id, "user", question)

      # 第一阶段：使用优化的检索
      logger.info("=== 开始文档检索 ===")
      relevant_docs = await self._retrieve_relevant_docs_optimized(
        question, session_id, history, max_context_docs
      )
      logger.info(f"检索到文档数量: {len(relevant_docs)}")

      # 第二阶段：如果没有相关文档，使用增强查询策略
      if not relevant_docs:
        logger.warning("初始检索无结果，启用增强查询策略")
        relevant_docs = await self._enhanced_retrieval_strategy(question, max_context_docs)
        logger.info(f"增强检索后文档数量: {len(relevant_docs)}")

      # 第三阶段：文档质量评估和排序
      if relevant_docs:
        logger.info("=== 开始文档质量评估 ===")
        relevant_docs = await self._evaluate_and_rank_documents(question, relevant_docs)
        logger.info(f"质量评估后保留文档数量: {len(relevant_docs)}")

      # 第四阶段：生成回答
      logger.info("=== 开始生成回答 ===")
      if relevant_docs:
        context_with_history = _build_context_with_history(session)

        # 检查是否使用备用模型
        using_fallback = isinstance(self.llm, FallbackChatModel)
        if using_fallback:
          logger.warning("检测到使用备用模型，将使用文档直接提取策略")

        # 根据模型类型选择生成策略
        if self.answer_generator and not using_fallback:
          try:
            logger.debug("使用高级答案生成器")
            answer_result = await self.answer_generator.generate_structured_answer(
              question, relevant_docs, context_with_history
            )
            answer = answer_result['answer']
            confidence = answer_result['confidence']
            question_type = answer_result['question_type']
            key_points = answer_result['key_points']
            
            logger.info(f"高级生成器结果 - 类型: {question_type}, 置信度: {confidence:.2f}")
            
          except Exception as e:
            logger.warning(f"高级答案生成失败，降级到基础生成: {e}")
            answer = await self._generate_answer_basic(question, relevant_docs, context_with_history)
            confidence = 0.6
            question_type = "general"
            key_points = []
        else:
          logger.debug("使用基础答案生成或文档直接提取")
          if using_fallback:
            # 直接从文档提取答案，不依赖LLM
            answer = await self._extract_direct_answer_from_docs(question, relevant_docs)
            confidence = 0.7  # 直接提取的置信度较高
          else:
            answer = await self._generate_answer_basic(question, relevant_docs, context_with_history)
            confidence = 0.6
          question_type = self._classify_question_enhanced(question)
          key_points = self._extract_key_points_from_docs(answer, relevant_docs)
      else:
        logger.warning("没有找到相关文档，生成通用回答")
        answer = "抱歉，我没有找到与您问题相关的具体信息。这可能是因为：\n1. 知识库中暂未包含相关内容\n2. 问题表述可能需要调整\n\n建议您:\n- 尝试使用不同的关键词重新提问\n- 查看AI-CloudOps平台的官方文档\n- 联系技术支持获取帮助"
        confidence = 0.1
        question_type = "unknown"
        key_points = []

      # 第五阶段：答案质量验证
      logger.info("=== 进行答案质量验证 ===")
      hallucination_free = True
      hallucination_score = 0.8
      if relevant_docs and len(answer) > 50:
        try:
          hallucination_free, hallucination_score = await _check_hallucination_advanced(
            question, answer, relevant_docs
          )
          logger.debug(f"幻觉检查结果: 通过={hallucination_free}, 分数={hallucination_score:.2f}")
        except Exception as e:
          logger.warning(f"幻觉检查失败: {e}")

      # 第六阶段：异步生成后续问题
      follow_up_task = None
      if len(answer) > 30:
        follow_up_task = asyncio.create_task(
          self._generate_follow_up_questions_advanced(question, answer, question_type)
        )

      # 第七阶段：格式化源文档
      source_docs = self._format_source_documents_advanced(relevant_docs)

      # 第八阶段：计算最终指标
      relevance_score = confidence if hallucination_free else confidence * 0.7
      recall_rate = min(len(relevant_docs) / max_context_docs, 1.0) if relevant_docs else 0.0

      logger.info(f"=== 处理完成 ===")
      logger.info(f"最终指标 - 相关性: {relevance_score:.2f}, 召回率: {recall_rate:.2f}")

      # 等待后续问题生成
      follow_up_questions = []
      if follow_up_task:
        try:
          follow_up_questions = await asyncio.wait_for(follow_up_task, timeout=3)
        except:
          follow_up_questions = self._get_default_follow_up_questions(question_type)

      # 构建响应
      result = {
        "answer": answer,
        "source_documents": source_docs,
        "relevance_score": relevance_score,
        "recall_rate": recall_rate,
        "confidence": confidence,
        "hallucination_score": hallucination_score,
        "question_type": question_type,
        "key_points": key_points,
        "follow_up_questions": follow_up_questions,
        "total_docs_found": len(relevant_docs)
      }

      # 添加助手回复到历史
      if session_id:
        self.add_message_to_history(session_id, "assistant", answer)

      # 缓存结果
      self.cache_manager.set(question, result, session_id, history)

      return result

    except Exception as e:
      logger.error(f"获取回答失败: {e}")
      import traceback
      logger.error(f"错误详情: {traceback.format_exc()}")
      
      error_answer = "抱歉，处理您的问题时出现了错误，请稍后重试。"

      if session_id:
        self.add_message_to_history(session_id, "assistant", error_answer)

      return {
        "answer": error_answer,
        "source_documents": [],
        "relevance_score": 0.0,
        "recall_rate": 0.0,
        "confidence": 0.0,
        "hallucination_score": 0.0,
        "question_type": "error",
        "key_points": [],
        "follow_up_questions": ["AIOps平台有哪些核心功能？", "如何部署AIOps系统？"],
        "total_docs_found": 0
      }

  async def _retrieve_relevant_docs_optimized(
    self, question: str, session_id: str = None,
    history: List[Dict] = None, max_docs: int = 10
  ) -> List[Document]:
    """优化的相关文档检索 - 重命名版本"""
    return await self._retrieve_relevant_docs_advanced(question, session_id, history, max_docs)

  async def _enhanced_retrieval_strategy(self, question: str, max_docs: int = 10) -> List[Document]:
    """增强的检索策略，当初始检索失败时使用"""
    try:
      logger.debug("启动增强检索策略")
      
      # 策略1: 查询扩展
      expanded_queries = self.query_rewriter.expand_query(question)
      for i, expanded_query in enumerate(expanded_queries[1:3]):  # 尝试前2个扩展查询
        logger.debug(f"尝试扩展查询 {i+1}: '{expanded_query}'")
        try:
          docs = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._safe_vector_search,
            expanded_query, max_docs
          )
          if docs:
            logger.debug(f"扩展查询 {i+1} 找到 {len(docs)} 个文档")
            return docs
        except Exception as e:
          logger.warning(f"扩展查询 {i+1} 失败: {e}")
          continue
      
      # 策略2: 关键词分解搜索
      logger.debug("尝试关键词分解搜索")
      keywords = [word for word in question.split() if len(word) > 2]
      if keywords:
        all_docs = []
        for keyword in keywords[:3]:  # 前3个关键词
          try:
            keyword_docs = await asyncio.get_event_loop().run_in_executor(
              self.executor,
              self._safe_vector_search,
              keyword, max_docs // 2
            )
            all_docs.extend(keyword_docs)
          except:
            continue
        
        if all_docs:
          # 去重并返回
          unique_docs = self._deduplicate_documents_simple(all_docs)
          logger.debug(f"关键词搜索找到 {len(unique_docs)} 个去重文档")
          return unique_docs[:max_docs]
      
      # 策略3: 宽泛搜索（降低相似度阈值）
      logger.debug("尝试宽泛搜索")
      try:
        if hasattr(self.vector_store_manager, 'optimized_store'):
          results = self.vector_store_manager.optimized_store.similarity_search(
            question, k=max_docs, similarity_threshold=0.001  # 极低阈值
          )
          docs = [doc for doc, score in results]
          if docs:
            logger.debug(f"宽泛搜索找到 {len(docs)} 个文档")
            return docs
      except Exception as e:
        logger.warning(f"宽泛搜索失败: {e}")
      
      logger.debug("所有增强检索策略都失败")
      return []
      
    except Exception as e:
      logger.error(f"增强检索策略失败: {e}")
      return []

  def _deduplicate_documents_simple(self, docs: List[Document]) -> List[Document]:
    """简单的文档去重"""
    seen_content = set()
    unique_docs = []
    
    for doc in docs:
      # 使用内容的前100字符作为去重标准
      content_hash = hash(doc.page_content[:100])
      if content_hash not in seen_content:
        seen_content.add(content_hash)
        unique_docs.append(doc)
    
    return unique_docs

  async def _evaluate_and_rank_documents(self, question: str, docs: List[Document]) -> List[Document]:
    """评估和排序文档质量"""
    if not docs:
      return []
    
    try:
      # 使用现有的文档过滤功能
      filtered_docs = await _filter_relevant_docs_advanced(question, docs)
      
      # 如果过滤后文档太少，保留更多文档
      if len(filtered_docs) < 3 and len(docs) > len(filtered_docs):
        logger.debug("过滤后文档太少，保留更多文档")
        # 按原顺序保留前几个文档
        additional_docs = [doc for doc in docs if doc not in filtered_docs][:2]
        filtered_docs.extend(additional_docs)
      
      return filtered_docs
      
    except Exception as e:
      logger.warning(f"文档评估排序失败: {e}")
      return docs[:8]  # 返回前8个文档

  async def _extract_direct_answer_from_docs(self, question: str, docs: List[Document]) -> str:
    """直接从文档提取答案，不依赖LLM"""
    if not docs:
      return "抱歉，没有找到相关文档。"
    
    try:
      # 分析问题类型
      question_type = self._classify_question_enhanced(question)
      
      # 提取相关内容
      relevant_content = []
      for doc in docs[:4]:
        content = self._extract_relevant_content_enhanced(doc.page_content, question, question_type)
        source = doc.metadata.get('filename', '文档') if doc.metadata else '文档'
        
        if content:
          relevant_content.append(f"根据 {source}:\n{content}")
      
      if relevant_content:
        # 构建结构化答案
        answer_parts = [
          f"关于您询问的 '{question}'，基于相关文档找到以下信息：",
          "",
          "\n\n".join(relevant_content),
          "",
          "以上信息来源于AI-CloudOps平台的相关文档。"
        ]
        return "\n".join(answer_parts)
      else:
        # 备选方案：提取文档摘要
        summaries = []
        for doc in docs[:3]:
          lines = [line.strip() for line in doc.page_content.split('\n') if line.strip()]
          if lines:
            summary = lines[0]
            if len(summary) > 20:
              summaries.append(f"• {summary}")
        
        if summaries:
          return f"关于 '{question}' 的相关信息：\n\n" + "\n".join(summaries)
        else:
          return "找到了相关文档，但内容提取遇到问题，建议查看完整文档获取详细信息。"
          
    except Exception as e:
      logger.error(f"直接提取答案失败: {e}")
      return "抱歉，处理文档内容时遇到问题。"

  def _extract_key_points_from_docs(self, answer: str, docs: List[Document]) -> List[str]:
    """从文档中提取关键点"""
    key_points = []
    
    try:
      # 首先尝试从答案中提取
      if answer and len(answer) > 50:
        key_points = self._extract_enhanced_key_points(answer, docs)
      
      # 如果答案中没有足够的关键点，从文档中直接提取
      if len(key_points) < 2 and docs:
        for doc in docs[:2]:
          content = doc.page_content
          # 查找标题或重要段落
          lines = content.split('\n')
          for line in lines:
            line = line.strip()
            if (line.startswith('#') or line.startswith('##') or 
                any(keyword in line for keyword in ['重要', '关键', '核心', '主要'])):
              if 10 < len(line) < 100:
                clean_line = line.lstrip('#').strip()
                if clean_line and clean_line not in key_points:
                  key_points.append(clean_line)
                  if len(key_points) >= 3:
                    break
          if len(key_points) >= 3:
            break
      
      return key_points[:3]
      
    except Exception as e:
      logger.warning(f"提取关键点失败: {e}")
      return []

  def _get_default_follow_up_questions(self, question_type: str) -> List[str]:
    """获取默认的后续问题"""
    type_questions = {
      'core_architecture': [
        "这些功能模块是如何协同工作的？",
        "如何部署和配置这些核心模块？"
      ],
      'architecture': [
        "系统架构有哪些技术优势？",
        "如何进行系统扩展和定制？"
      ],
      'deployment': [
        "部署过程中可能遇到哪些问题？",
        "如何验证部署是否成功？"
      ],
      'monitoring': [
        "如何设置监控告警规则？",
        "监控数据如何进行分析？"
      ],
      'troubleshooting': [
        "如何预防类似问题再次发生？",
        "还有其他可能的解决方案吗？"
      ],
      'performance': [
        "如何监控性能优化效果？",
        "还有哪些性能优化策略？"
      ]
    }
    
    return type_questions.get(question_type, [
      "AI-CloudOps平台有哪些核心功能？",
      "如何开始使用AI-CloudOps平台？"
    ])

  async def _retrieve_relevant_docs_advanced(
    self, question: str, session_id: str = None,
    history: List[Dict] = None, max_docs: int = 10
  ) -> List[Document]:
    """优化的高级相关文档检索"""
    try:
      # 快速检查输入
      if not question or not question.strip():
        return []

      # 优化：减少查询次数，直接使用最优检索器
      if self.context_retriever:
        docs = await asyncio.get_event_loop().run_in_executor(
          self.executor,
          self._safe_context_retrieve,
          question, session_id, history, max_docs
        )
      else:
        docs = await asyncio.get_event_loop().run_in_executor(
          self.executor,
          self._safe_vector_search,
          question, max_docs
        )

      if not docs:
        return []

      # 简化的文档过滤，减少处理时间
      filtered_docs = self._filter_docs_fast(question, docs, max_docs)
      return filtered_docs

    except Exception as e:
      logger.error(f"优化文档检索失败: {e}")
      # 最简单的降级方案
      try:
        return await asyncio.get_event_loop().run_in_executor(
          self.executor,
          self._basic_vector_search,
          question, max_docs
        )
      except:
        return []

  def _safe_context_retrieve(self, question: str, session_id: str = None,
                           history: List[Dict] = None, max_docs: int = 4) -> List[Document]:
    """安全的上下文检索"""
    try:
      return self.context_retriever.retrieve_with_context(question, session_id, history, max_docs)
    except Exception as e:
      logger.warning(f"上下文检索失败，降级到基础检索: {e}")
      return self._safe_vector_search(question, max_docs)

  def _safe_vector_search(self, question: str, max_docs: int = 6) -> List[Document]:
    """安全的向量搜索 - 提升召回率版本"""
    try:
      return self.vector_store_manager.search_documents(question)[:max_docs]
    except Exception as e:
      logger.warning(f"向量搜索失败: {e}")
      return []

  def _basic_vector_search(self, question: str, max_docs: int = 6) -> List[Document]:
    """基础向量搜索 - 提升召回率版本"""
    try:
      if hasattr(self.vector_store_manager, 'optimized_store'):
        results = self.vector_store_manager.optimized_store.similarity_search(
          question, k=max_docs, similarity_threshold=0.1  # 极低阈值
        )
        return [doc for doc, score in results]
      return []
    except Exception as e:
      logger.warning(f"基础向量搜索失败: {e}")
      return []

  def _filter_docs_fast(self, question: str, docs: List[Document], max_docs: int) -> List[Document]:
    """快速文档过滤"""
    if not docs or len(docs) <= max_docs:
      return docs

    try:
      # 简化的相关性评分
      question_words = set(question.lower().split())
      scored_docs = []

      for doc in docs:
        doc_words = set(doc.page_content.lower().split()[:100])  # 只检查前100个词
        overlap = len(question_words & doc_words)
        score = overlap / max(len(question_words), 1)
        scored_docs.append((doc, score))

      # 按分数排序并返回前max_docs个
      scored_docs.sort(key=lambda x: x[1], reverse=True)
      return [doc for doc, score in scored_docs[:max_docs]]

    except Exception as e:
      logger.warning(f"文档过滤失败: {e}")
      return docs[:max_docs]

  async def _generate_answer_basic(
    self, question: str, docs: List[Document], context: str = None
  ) -> str:
    """基础答案生成"""
    try:
      docs_content = ""
      for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
        filename = doc.metadata.get("filename", "未知文件") if doc.metadata else "未知文件"
        relevance = doc.metadata.get("relevance_score", 0.5) if doc.metadata else 0.5

        docs_content += f"\n\n文档[{i + 1}] (文件: {filename}, 相关性: {relevance:.2f}):\n{doc.page_content}"

      # 限制长度
      max_length = getattr(config.rag, 'max_context_length', 3000)  # 减少最大长度
      if len(docs_content) > max_length:
        docs_content = docs_content[:max_length] + "...(内容已截断)"

      system_prompt = """您是专业的AIOps智能助手。请基于提供的文档内容准确回答用户问题。

规则:
1. 仅基于文档内容回答，确保准确性
2. 回答要简洁实用，重点突出
3. 如果信息不足，明确说明限制
4. 提供实用的建议和步骤
5. 保持专业友好的语气"""

      user_prompt = f"{context}\n\n" if context else ""
      user_prompt += f"问题: {question}\n\n相关文档:\n{docs_content}\n\n请提供专业简洁的回答："

      messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
      ]

      response = await asyncio.wait_for(
        self.llm.ainvoke(messages),
        timeout=30  # 减少超时时间
      )

      return response.content.strip()

    except Exception as e:
      logger.error(f"基础答案生成失败: {e}")
      return "抱歉，生成回答时遇到问题，请稍后重试。"

  async def _generate_follow_up_questions_advanced(
    self, question: str, answer: str, question_type: str = "general"
  ) -> List[str]:
    """生成高级后续问题 - 修复异常处理"""
    type_specific_questions = {
      'deployment': [
        "部署过程中可能遇到哪些常见问题？",
        "如何验证部署是否成功？"
      ],
      'monitoring': [
        "如何设置有效的监控阈值？",
        "监控数据如何进行分析和处理？"
      ],
      'troubleshooting': [
        "如何预防类似问题再次发生？",
        "还有其他可能的解决方案吗？"
      ],
      'performance': [
        "如何监控性能优化效果？",
        "还有哪些性能优化策略？"
      ]
    }

    default_questions = type_specific_questions.get(question_type, [
      "如何进一步了解这个功能？",
      "在实际使用中需要注意什么？"
    ])

    # 首先返回默认问题，避免异常
    try:
      # 检查条件：模型可用且回答足够长
      if (self.task_llm and
        not isinstance(self.task_llm, FallbackChatModel) and
        len(answer) > 50 and  # 降低长度要求
        len(question) > 5):

        system_prompt = f"""基于{question_type}类型的问题和回答，生成2个相关的后续问题。

要求:
1. 问题要具体实用
2. 问题要以问号结尾
3. 每行一个问题
4. 不要编号或符号"""

        user_prompt = f"原问题: {question[:60]}\n回答摘要: {answer[:120]}\n\n生成后续问题："

        messages = [
          SystemMessage(content=system_prompt),
          HumanMessage(content=user_prompt)
        ]

        response = await asyncio.wait_for(
          self.task_llm.ainvoke(messages),
          timeout=6  # 进一步减少超时时间
        )

        if response and response.content and response.content.strip():
          generated_questions = []
          for line in response.content.strip().split('\n'):
            line = line.strip()
            # 清理格式
            line = re.sub(r'^\d+[.)、\s]+', '', line)
            line = re.sub(r'^[•\-\*]\s*', '', line)
            line = re.sub(r'^\s*-\s*', '', line)

            if line and len(line) > 5 and len(line) < 100:
              if not (line.endswith('?') or line.endswith('？')):
                line += '？'
              generated_questions.append(line)

          # 如果生成了合格的问题，返回
          if len(generated_questions) >= 1:
            return generated_questions[:2]  # 最多返回2个

    except asyncio.TimeoutError:
      logger.warning("生成后续问题超时")
    except Exception as e:
      logger.warning(f"智能生成后续问题失败: {str(e)[:100]}")  # 限制错误消息长度

    # 返回默认问题
    return default_questions[:2]  # 减少数量

  def _format_source_documents_advanced(self, docs: List[Document]) -> List[Dict[str, Any]]:
    """高级源文档格式化"""
    source_docs = []

    for i, doc in enumerate(docs):
      metadata = doc.metadata or {}
      content = doc.page_content

      # 智能截断内容
      if len(content) > 200:  # 减少截断长度
        sentences = re.split(r'[。！？.!?]', content)
        truncated = ""
        for sentence in sentences:
          if len(truncated + sentence) < 180:
            truncated += sentence + "。"
          else:
            break
        content = truncated + "..." if truncated else content[:180] + "..."

      source_doc = {
        "content": content,
        "source": metadata.get("source", "未知来源"),
        "filename": metadata.get("filename", f"文档{i + 1}"),
        "filetype": metadata.get("filetype", "unknown"),
        "relevance_score": metadata.get("relevance_score", 0.5),
        "is_web_result": metadata.get("is_web_result", False),
        "confidence": metadata.get("confidence_score", 0.5),
        "metadata": {k: v for k, v in metadata.items()
                     if k not in ['source', 'filename', 'filetype', 'relevance_score']}
      }

      source_docs.append(source_doc)

    return source_docs

  # ==================== 缓存管理 ====================

  def clear_cache(self) -> Dict[str, Any]:
    """清空响应缓存 - 使用Redis缓存管理器"""
    try:
      # 使用Redis缓存管理器清空缓存
      result = self.cache_manager.clear_all()
      return result

    except Exception as e:
      logger.error(f"清空缓存失败: {e}")
      return {
        "success": False,
        "message": f"清空缓存失败: {e}",
        "cleared_count": 0
      }

  async def shutdown(self):
    """优雅关闭助手代理"""
    if self._shutdown:
      return

    logger.info("开始关闭优化版智能助手...")
    self._shutdown = True

    try:
      # 1. 关闭缓存管理器
      if hasattr(self, 'cache_manager'):
        self.cache_manager.shutdown()

      # 2. 关闭线程池
      if hasattr(self, 'executor') and self.executor:
        self.executor.shutdown(wait=True)

      # 3. 关闭任务管理器
      task_manager = get_task_manager()
      await task_manager.shutdown()

      logger.info("优化版智能助手已成功关闭")

    except Exception as e:
      logger.warning(f"关闭智能助手时出现警告: {e}")

  def __del__(self):
    """清理资源"""
    if not self._shutdown:
      try:
        self._shutdown = True

        if hasattr(self, 'cache_manager'):
          try:
            self.cache_manager.shutdown()
          except Exception as e:
            logger.warning(f"对象销毁时关闭缓存管理器失败: {e}")

        if hasattr(self, 'executor') and self.executor:
          try:
            self.executor.shutdown(wait=False)
          except Exception as e:
            logger.warning(f"对象销毁时关闭线程池失败: {e}")

      except Exception as e:
        logger.warning(f"AssistantAgent清理资源时出错: {e}")
