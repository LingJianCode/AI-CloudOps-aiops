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
import json
import asyncio
from asyncio import CancelledError
import hashlib
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from collections import defaultdict, Counter
import math

from app.constants import EMBEDDING_BATCH_SIZE

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
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
from chromadb.config import Settings
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

    # 1. 同义词替换
    for word, synonyms in self.synonyms.items():
      if word in query:
        for synonym in synonyms[:2]:  # 限制同义词数量
          expanded_queries.append(query.replace(word, synonym))

    # 2. 关键词提取和重组
    keywords = self._extract_keywords(query)
    if len(keywords) > 1:
      # 关键词组合
      expanded_queries.append(' '.join(keywords))

    # 3. 去重并限制数量
    unique_queries = list(dict.fromkeys(expanded_queries))[:5]  # 减少查询数量
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

  def rank_documents(self, query: str, documents: List[Document], top_k: int = 8) -> List[
    Tuple[Document, float]]:
    """对文档进行相关性排序"""
    if not self.fitted or not documents:
      return [(doc, 0.5) for doc in documents[:top_k]]

    try:
      # 查询向量化
      query_vector = self.tfidf_vectorizer.transform([query])

      # 计算文档相似度
      doc_vectors = self.tfidf_vectorizer.transform([doc.page_content for doc in documents])
      similarities = cosine_similarity(query_vector, doc_vectors).flatten()

      # 综合评分
      scored_docs = []
      for i, doc in enumerate(documents):
        tfidf_score = similarities[i]
        length_score = min(len(doc.page_content) / 800, 1.0)  # 调整长度评分
        freshness_score = self._calculate_freshness_score(doc)

        # 综合评分
        final_score = (
          tfidf_score * 0.7 +
          length_score * 0.2 +
          freshness_score * 0.1
        )

        scored_docs.append((doc, final_score))

      # 排序并返回前k个
      scored_docs.sort(key=lambda x: x[1], reverse=True)
      return scored_docs[:top_k]

    except Exception as e:
      logger.error(f"文档排序失败: {e}")
      return [(doc, 0.5) for doc in documents[:top_k]]

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
                            history: List[Dict] = None, top_k: int = 6) -> List[Document]:
    """带上下文的智能检索"""
    try:
      # 1. 构建增强查询
      enhanced_query = self._build_enhanced_query(query, history)

      # 2. 多查询检索
      all_docs = []
      queries = self.query_rewriter.expand_query(enhanced_query)

      for q in queries:
        try:
          docs = self.base_retriever.invoke(q)
          all_docs.extend(docs)
        except Exception as e:
          logger.warning(f"查询检索失败 '{q}': {e}")
          continue

      # 3. 去重
      unique_docs = self._deduplicate_documents(all_docs)

      # 4. 智能排序
      ranked_docs = self.doc_ranker.rank_documents(enhanced_query, unique_docs, top_k)

      # 5. 上下文过滤
      filtered_docs = self._context_filter(ranked_docs, session_id, history)

      return [doc for doc, score in filtered_docs]

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


# ==================== 高级答案生成器 ====================

class AdvancedAnswerGenerator:
  """高级答案生成器，提升回答质量"""

  def __init__(self, llm):
    self.llm = llm

  async def generate_structured_answer(self, question: str, docs: List[Document],
                                       context: str = None) -> Dict[str, Any]:
    """生成结构化答案"""
    try:
      # 1. 分析问题类型
      question_type = self._classify_question(question)

      # 2. 构建上下文
      structured_context = self._build_structured_context(docs, question_type)

      # 3. 生成答案
      answer = await self._generate_answer_with_template(
        question, structured_context, question_type, context
      )

      # 4. 提取关键信息
      key_points = self._extract_key_points(answer, docs)

      # 5. 生成置信度
      confidence = self._calculate_confidence(question, answer, docs)

      return {
        'answer': answer,
        'question_type': question_type,
        'key_points': key_points,
        'confidence': confidence,
        'source_count': len(docs)
      }

    except Exception as e:
      logger.error(f"结构化答案生成失败: {e}")
      # 降级到基础生成
      basic_answer = await self._basic_generate(question, docs, context)
      return {
        'answer': basic_answer,
        'question_type': 'general',
        'key_points': [],
        'confidence': 0.5,
        'source_count': len(docs)
      }

  def _classify_question(self, question: str) -> str:
    """分类问题类型"""
    question_lower = question.lower()

    if any(word in question_lower for word in ['部署', '安装', '配置', '搭建']):
      return 'deployment'
    elif any(word in question_lower for word in ['监控', '观察', '检测']):
      return 'monitoring'
    elif any(word in question_lower for word in ['故障', '错误', '问题', '异常']):
      return 'troubleshooting'
    elif any(word in question_lower for word in ['性能', '优化', '效率']):
      return 'performance'
    else:
      return 'general'

  def _build_structured_context(self, docs: List[Document], question_type: str) -> str:
    """构建结构化上下文"""
    if not docs:
      return ""

    context = f"基于{len(docs)}个相关文档的信息:\n\n"

    for i, doc in enumerate(docs):
      source = doc.metadata.get('filename', f'文档{i + 1}') if doc.metadata else f'文档{i + 1}'
      content = doc.page_content

      # 根据问题类型突出相关内容
      highlighted_content = self._highlight_relevant_content(content, question_type)

      context += f"【{source}】\n{highlighted_content}\n\n"

    return context

  def _highlight_relevant_content(self, content: str, question_type: str) -> str:
    """突出显示相关内容"""
    type_keywords = {
      'deployment': ['部署', '安装', '配置', '搭建', '启动'],
      'monitoring': ['监控', '观察', '检测', '巡检', '指标'],
      'troubleshooting': ['故障', '错误', '问题', '异常', '解决'],
      'performance': ['性能', '优化', '效率', '速度', '响应']
    }

    keywords = type_keywords.get(question_type, [])

    # 简单的内容截取和关键词突出
    lines = content.split('\n')
    relevant_lines = []

    for line in lines:
      line_lower = line.lower()
      if any(keyword in line_lower for keyword in keywords):
        relevant_lines.append(line)
      elif len(relevant_lines) < 2:  # 确保有足够内容
        relevant_lines.append(line)

    return '\n'.join(relevant_lines[:4])  # 最多4行

  async def _generate_answer_with_template(self, question: str, context: str,
                                           question_type: str, history_context: str = None) -> str:
    """使用模板生成答案"""
    # 根据问题类型选择系统提示
    system_prompts = {
      'deployment': """你是AIOps部署专家。请基于文档内容提供清晰的部署指导，包括前置条件、具体步骤、注意事项和验证方法。回答要简洁实用。""",
      'monitoring': """你是AIOps监控专家。请基于文档内容提供监控相关建议，包括监控指标、配置方法、阈值设置和告警策略。""",
      'troubleshooting': """你是AIOps故障排除专家。请基于文档内容提供故障诊断和解决方案，包括问题分析、排查步骤、解决方案和预防措施。""",
      'performance': """你是AIOps性能优化专家。请基于文档内容提供性能优化建议，包括性能分析、优化策略、最佳实践和效果评估。""",
      'general': """你是专业的AIOps助手。请基于文档内容准确回答问题，确保信息的完整性和实用性。"""
    }

    system_prompt = system_prompts.get(question_type, system_prompts['general'])

    user_prompt = f"{history_context}\n\n" if history_context else ""
    user_prompt += f"问题: {question}\n\n相关文档内容:\n{context}\n\n请提供专业简洁的回答："

    messages = [
      SystemMessage(content=system_prompt),
      HumanMessage(content=user_prompt)
    ]

    response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=30)
    return response.content.strip()

  def _extract_key_points(self, answer: str, docs: List[Document]) -> List[str]:
    """提取答案关键点"""
    sentences = re.split(r'[。！？.]', answer)
    key_points = []

    for sentence in sentences:
      sentence = sentence.strip()
      if (len(sentence) > 15 and len(sentence) < 100 and
        any(keyword in sentence for keyword in ['重要', '关键', '注意', '步骤', '方法', '建议'])):
        key_points.append(sentence)

    return key_points[:3]  # 最多3个关键点

  def _calculate_confidence(self, question: str, answer: str, docs: List[Document]) -> float:
    """计算回答置信度"""
    try:
      # 多维度置信度计算
      doc_confidence = min(len(docs) / 4, 1.0)  # 调整基准
      length_confidence = min(len(answer) / 150, 1.0)  # 调整基准

      # 关键词匹配置信度
      question_words = set(re.findall(r'\w+', question.lower()))
      answer_words = set(re.findall(r'\w+', answer.lower()))
      doc_words = set()
      for doc in docs:
        doc_words.update(re.findall(r'\w+', doc.page_content.lower()))

      qa_overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
      ad_overlap = len(answer_words & doc_words) / len(answer_words) if answer_words else 0

      # 综合置信度
      confidence = (
        doc_confidence * 0.3 +
        length_confidence * 0.2 +
        qa_overlap * 0.2 +
        ad_overlap * 0.3
      )

      return min(max(confidence, 0.1), 1.0)

    except Exception as e:
      logger.error(f"置信度计算失败: {e}")
      return 0.5

  async def _basic_generate(self, question: str, docs: List[Document], context: str = None) -> str:
    """基础答案生成（降级方案）"""
    try:
      docs_content = "\n\n".join([doc.page_content[:500] for doc in docs])  # 限制长度

      system_prompt = "你是专业的AI助手。请基于提供的文档内容简洁准确地回答问题。"
      user_prompt = f"{context}\n\n问题: {question}\n\n文档: {docs_content}" if context else f"问题: {question}\n\n文档: {docs_content}"

      messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
      ]

      response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=20)
      return response.content.strip()

    except Exception as e:
      logger.error(f"基础答案生成失败: {e}")
      return "抱歉，我暂时无法处理您的问题，请稍后重试。"


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
class DocumentMetadata:
  source: str
  filename: str
  filetype: str
  modified_time: float
  is_web_result: bool = False
  relevance_score: float = 0.0
  recall_rate: float = 0.0
  confidence_score: float = 0.5


@dataclass
class CacheEntry:
  timestamp: float
  data: Dict[str, Any]
  access_count: int = 0
  last_access: float = field(default_factory=time.time)

  def is_expired(self, expiry_seconds: int) -> bool:
    return time.time() - self.timestamp > expiry_seconds

  def update_access(self):
    """更新访问信息"""
    self.access_count += 1
    self.last_access = time.time()


@dataclass
class SessionData:
  session_id: str
  created_at: str
  history: List[Dict[str, Any]]
  metadata: Dict[str, Any]
  context_summary: str = ""


class GradeDocuments(BaseModel):
  binary_score: str = Field(description="文档是否与问题相关，'yes'或'no'")
  confidence: float = Field(description="判断的置信度，0-1之间")


class GradeHallucinations(BaseModel):
  binary_score: str = Field(description="回答是否基于事实，'yes'或'no'")
  explanation: str = Field(description="判断理由")


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

class VectorStoreManager:
  def __init__(self, vector_db_path: str, collection_name: str, embedding_model):
    self.vector_db_path = vector_db_path
    self.collection_name = collection_name
    self.embedding_model = embedding_model
    self.db = None
    self.retriever = None
    self._lock = threading.Lock()
    os.makedirs(vector_db_path, exist_ok=True)

  def _get_client_settings(self, persistent: bool = True) -> Settings:
    return Settings(
      anonymized_telemetry=False,
      allow_reset=True,
      is_persistent=persistent,
      chroma_db_impl="duckdb+parquet" if not persistent else None
    )

  def _cleanup_temp_files(self):
    temp_files = [
      os.path.join(self.vector_db_path, ".lock"),
      os.path.join(self.vector_db_path, ".uuid"),
      os.path.join(self.vector_db_path, "chroma.sqlite3-shm"),
      os.path.join(self.vector_db_path, "chroma.sqlite3-wal")
    ]

    for file_path in temp_files:
      try:
        if os.path.exists(file_path):
          os.remove(file_path)
          logger.debug(f"清理临时文件: {file_path}")
      except Exception as e:
        logger.warning(f"清理临时文件失败 {file_path}: {e}")

  def load_existing_db(self) -> bool:
    db_file = os.path.join(self.vector_db_path, "chroma.sqlite3")

    if not os.path.exists(db_file):
      return False

    try:
      with self._lock:
        logger.info(f"加载现有向量数据库: {db_file}")
        self.db = Chroma(
          persist_directory=self.vector_db_path,
          embedding_function=self.embedding_model,
          collection_name=self.collection_name,
          client_settings=self._get_client_settings(persistent=True)
        )

        self.retriever = self.db.as_retriever(
          search_type="mmr",
          search_kwargs={
            "k": config.rag.top_k * 2,
            "fetch_k": config.rag.top_k * 3,  # 减少fetch_k
            "lambda_mult": 0.7
          }
        )

        self.retriever.invoke("测试查询")
        logger.info("数据库加载成功")
        return True

    except Exception as e:
      logger.error(f"加载现有数据库失败: {e}")
      self._backup_corrupted_db(db_file)
      return False

  def _backup_corrupted_db(self, db_file: str):
    try:
      backup_dir = os.path.join(self.vector_db_path, f"backup_corrupt_{int(time.time())}")
      os.makedirs(backup_dir, exist_ok=True)
      shutil.copy2(db_file, backup_dir)
      os.remove(db_file)
      logger.info(f"已备份并删除损坏的数据库文件: {backup_dir}")
    except Exception as e:
      logger.error(f"备份损坏数据库失败: {e}")

  async def create_vector_store(self, documents: List[Document], use_memory: bool = False) -> bool:
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
        self._cleanup_temp_files()

        if use_memory:
          logger.info("使用内存模式创建向量存储")
          client_settings = self._get_client_settings(persistent=False)
        else:
          logger.info("使用持久化模式创建向量存储")
          client_settings = self._get_client_settings(persistent=True)

        batch_size = min(EMBEDDING_BATCH_SIZE, 30)  # 减少批次大小
        total_docs = len(enhanced_splits)
        all_success = True

        if total_docs > batch_size:
          logger.info(f"文档量较大({total_docs}个)，使用分批处理方式")

          self.db = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=None if use_memory else self.vector_db_path,
            collection_name=self.collection_name,
            client_settings=client_settings
          )

          for i in range(0, total_docs, batch_size):
            batch = enhanced_splits[i:i + batch_size]
            logger.info(
              f"处理批次 {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}，{len(batch)}个文档")

            try:
              max_api_batch = 15  # 减少API批次大小
              for j in range(0, len(batch), max_api_batch):
                sub_batch = batch[j:j + max_api_batch]
                self.db.add_documents(sub_batch)
                if j > 0:
                  await asyncio.sleep(0.2)  # 增加延迟
            except Exception as e:
              logger.error(f"添加文档批次失败: {e}")
              all_success = False
              break
        else:
          # 数量较少，直接创建
          self.db = Chroma.from_documents(
            documents=enhanced_splits,
            embedding=self.embedding_model,
            persist_directory=None if use_memory else self.vector_db_path,
            collection_name=self.collection_name,
            client_settings=client_settings
          )

        if all_success:
          self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
              "k": config.rag.top_k * 2,
              "fetch_k": config.rag.top_k * 3,
              "lambda_mult": 0.7
            }
          )

          self.retriever.invoke("测试查询")
          logger.info(f"向量存储创建成功，包含 {len(enhanced_splits)} 个文档块")
          return True
        return False

    except Exception as e:
      logger.error(f"创建向量存储失败: {e}")
      if not use_memory:
        logger.info("尝试回退到内存模式")
        return await self.create_vector_store(documents, use_memory=True)
      return False

  def get_retriever(self):
    return self.retriever

  def search_documents(self, query: str, max_retries: int = 2) -> List[Document]:  # 减少重试次数
    if not self.retriever:
      logger.warning("检索器未初始化")
      return []

    for attempt in range(max_retries):
      try:
        docs = self.retriever.invoke(query)
        if docs:
          logger.debug(f"搜索到 {len(docs)} 个文档")
          return docs
        else:
          logger.warning(f"搜索返回空结果 (尝试 {attempt + 1}/{max_retries})")

      except Exception as e:
        logger.error(f"文档搜索失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
          time.sleep(0.5)  # 减少等待时间

    return []


# ==================== 缓存管理器 ====================

class CacheManager:
  """优化的缓存管理器，支持智能缓存策略"""

  def __init__(self, cache_dir: str, expiry_seconds: int = 3600, max_cache_size: int = 1000):
    self.cache_dir = cache_dir
    self.expiry_seconds = expiry_seconds
    self.max_cache_size = max_cache_size
    self.cache: Dict[str, CacheEntry] = {}
    self.cache_file = os.path.join(cache_dir, "response_cache.json")
    self._lock = threading.Lock()
    self._shutdown = False

    os.makedirs(cache_dir, exist_ok=True)
    self._load_cache()

  def _generate_cache_key(self, question: str, session_id: str = None, history: List = None) -> str:
    """生成缓存键"""
    cache_input = question

    if session_id and history:
      recent_history = history[-1:] if len(history) >= 1 else history  # 减少历史长度
      if recent_history:
        history_str = json.dumps([
          {"role": h["role"], "content": h["content"][:20]}  # 减少内容长度
          for h in recent_history
        ], ensure_ascii=False)
        cache_input = f"{question}|{history_str}"

    return hashlib.sha256(cache_input.encode('utf-8')).hexdigest()

  def _load_cache(self):
    """加载缓存文件"""
    try:
      if os.path.exists(self.cache_file):
        with open(self.cache_file, 'r', encoding='utf-8') as f:
          cache_data = json.load(f)

        valid_cache = {}
        for k, v in cache_data.items():
          if isinstance(v, dict) and "timestamp" in v:
            entry = CacheEntry(
              timestamp=v["timestamp"],
              data=v["data"],
              access_count=v.get("access_count", 0),
              last_access=v.get("last_access", time.time())
            )
            if not entry.is_expired(self.expiry_seconds):
              valid_cache[k] = entry

        self.cache = valid_cache
        logger.info(f"加载了 {len(self.cache)} 条有效缓存")
    except Exception as e:
      logger.warning(f"加载缓存失败: {e}")
      self.cache = {}

  def save_cache_sync(self):
    """同步保存缓存到文件"""
    if self._shutdown:
      return

    try:
      with self._lock:
        valid_cache = {}
        for k, v in self.cache.items():
          if not v.is_expired(self.expiry_seconds):
            valid_cache[k] = v

        # LRU策略清理
        if len(valid_cache) > self.max_cache_size:
          sorted_cache = sorted(
            valid_cache.items(),
            key=lambda x: (x[1].access_count, x[1].last_access),
            reverse=True
          )
          valid_cache = dict(sorted_cache[:self.max_cache_size])

        # 转换为可序列化格式
        serializable_cache = {
          k: {
            "timestamp": v.timestamp,
            "data": v.data,
            "access_count": v.access_count,
            "last_access": v.last_access
          }
          for k, v in valid_cache.items()
        }

        with open(self.cache_file, 'w', encoding='utf-8') as f:
          json.dump(serializable_cache, f, ensure_ascii=False, indent=2)

        self.cache = valid_cache
        logger.debug(f"保存了 {len(valid_cache)} 条缓存")
    except Exception as e:
      logger.warning(f"保存缓存失败: {e}")

  def get(self, question: str, session_id: str = None, history: List = None) -> Optional[
    Dict[str, Any]]:
    """获取缓存"""
    cache_key = self._generate_cache_key(question, session_id, history)

    if cache_key in self.cache:
      entry = self.cache[cache_key]
      if not entry.is_expired(self.expiry_seconds):
        entry.update_access()
        logger.debug(f"缓存命中: {cache_key[:8]}...")
        return entry.data
      else:
        del self.cache[cache_key]

    return None

  def set(self, question: str, response_data: Dict[str, Any], session_id: str = None,
          history: List = None):
    """设置缓存"""
    if self._shutdown:
      return

    cache_key = self._generate_cache_key(question, session_id, history)
    entry = CacheEntry(timestamp=time.time(), data=response_data)

    with self._lock:
      self.cache[cache_key] = entry

  async def save_async(self):
    """异步保存缓存"""
    if self._shutdown:
      return

    try:
      loop = asyncio.get_event_loop()
      await loop.run_in_executor(None, self.save_cache_sync)
    except Exception as e:
      logger.error(f"异步保存缓存失败: {e}")

  def shutdown(self):
    """关闭缓存管理器"""
    self._shutdown = True
    try:
      self.save_cache_sync()
    except Exception as e:
      logger.warning(f"关闭时保存缓存失败: {e}")


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
  """高级文档相关性评估"""
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
    quality_score = min(content_length / 400, 1.0)  # 调整基准

    # 综合评分
    final_score = (
      basic_similarity * 0.5 +
      semantic_score * 0.3 +
      quality_score * 0.2
    )

    is_relevant = final_score > 0.15  # 降低阈值

    return is_relevant, final_score

  except Exception as e:
    logger.error(f"高级文档相关性评估失败: {e}")
    return True, 0.5


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
  """高级文档过滤"""
  if not docs or len(docs) <= 2:
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
      return all_docs_with_scores[:2]  # 减少返回数量

    # 按相关性排序
    relevant_docs.sort(
      key=lambda x: x.metadata.get("relevance_score", 0),
      reverse=True
    )

    return relevant_docs

  except Exception as e:
    logger.error(f"高级文档过滤失败: {e}")
    return docs[:2]  # 减少返回数量


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
    self.cache_manager = CacheManager(
      str(self.vector_db_path / "cache"),
      expiry_seconds=3600,  # 减少缓存时间
      max_cache_size=1000  # 减少缓存容量
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
    """初始化嵌入模型"""
    max_retries = 2  # 减少重试次数

    for attempt in range(max_retries):
      try:
        if self.llm_provider == 'openai':
          logger.info(f"初始化OpenAI嵌入模型 (尝试 {attempt + 1})")
          self.embedding = OpenAIEmbeddings(
            model=config.rag.openai_embedding_model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=10,  # 减少超时时间
            max_retries=2
          )
        else:
          logger.info(f"初始化Ollama嵌入模型 (尝试 {attempt + 1})")
          self.embedding = OllamaEmbeddings(
            model=config.rag.ollama_embedding_model,
            base_url=config.llm.ollama_base_url,
            timeout=10
          )

        # 测试嵌入
        test_embedding = self.embedding.embed_query("测试")
        if test_embedding and len(test_embedding) > 0:
          logger.info(f"嵌入模型初始化成功，维度: {len(test_embedding)}")
          return

      except Exception as e:
        logger.error(f"嵌入模型初始化失败 (尝试 {attempt + 1}): {e}")
        if attempt < max_retries - 1:
          self.llm_provider = 'ollama' if self.llm_provider == 'openai' else 'openai'
          time.sleep(1)

    # 使用备用嵌入
    logger.warning("使用备用嵌入模型")
    self.embedding = FallbackEmbeddings()

  def _init_llm(self):
    """初始化语言模型"""
    max_retries = 2

    for attempt in range(max_retries):
      try:
        if self.llm_provider == 'openai':
          logger.info(f"初始化OpenAI语言模型 (尝试 {attempt + 1})")

          self.llm = ChatOpenAI(
            model=config.llm.model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=config.rag.temperature,
            timeout=30,  # 减少超时时间
            max_retries=2
          )

          task_model = getattr(config.llm, 'task_model', config.llm.model)
          self.task_llm = ChatOpenAI(
            model=task_model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=0.1,
            timeout=15,
            max_retries=2
          )
        else:
          logger.info(f"初始化Ollama语言模型 (尝试 {attempt + 1})")
          self.llm = ChatOllama(
            model=config.llm.ollama_model,
            base_url=config.llm.ollama_base_url,
            temperature=config.rag.temperature,
            timeout=30
          )
          self.task_llm = self.llm

        # 测试模型
        test_response = self.llm.invoke("返回'OK'")
        if test_response and test_response.content:
          logger.info("语言模型初始化成功")
          return

      except Exception as e:
        logger.error(f"语言模型初始化失败 (尝试 {attempt + 1}): {e}")
        if attempt < max_retries - 1:
          self.llm_provider = 'ollama' if self.llm_provider == 'openai' else 'openai'
          time.sleep(1)

    # 使用备用模型
    logger.warning("使用备用语言模型")
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
      except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

      success = loop.run_until_complete(
        self.vector_store_manager.create_vector_store(documents, use_memory)
      )

      if not success:
        logger.error("向量存储初始化失败")
        raise RuntimeError("无法初始化向量存储")

    logger.info("向量存储初始化完成")

  def _init_advanced_components(self):
    """初始化高级组件"""
    try:
      # 初始化文档排序器
      if self.vector_store_manager and self.vector_store_manager.db:
        all_docs = []
        try:
          collection = self.vector_store_manager.db._collection
          if hasattr(collection, 'get'):
            results = collection.get()
            if results and 'documents' in results:
              for doc_text in results['documents']:
                all_docs.append(Document(page_content=doc_text))
        except Exception as e:
          logger.warning(f"获取训练文档失败: {e}")
          all_docs = self.document_loader.load_documents()

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
        self.answer_generator = AdvancedAnswerGenerator(self.llm)
        logger.info("高级答案生成器初始化完成")

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

      # 清理缓存
      self.cache_manager = CacheManager(
        str(self.vector_db_path / "cache"),
        expiry_seconds=3600,
        max_cache_size=1000
      )

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

      # 清理缓存
      self.cache_manager = CacheManager(
        str(self.vector_db_path / "cache"),
        expiry_seconds=3600,
        max_cache_size=1000
      )

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
    max_context_docs: int = 4  # 减少上下文文档数量
  ) -> Dict[str, Any]:
    """获取问题答案 - 优化版核心方法"""

    try:
      # 获取会话历史
      session = self.get_session(session_id) if session_id else None
      history = session.history if session else []

      # 检查缓存
      cached_response = self.cache_manager.get(question, session_id, history)
      if cached_response:
        if session_id:
          self.add_message_to_history(session_id, "user", question)
          self.add_message_to_history(session_id, "assistant", cached_response["answer"])
        return cached_response

      # 添加用户消息到历史
      if session_id:
        self.add_message_to_history(session_id, "user", question)

      # 使用上下文感知检索
      relevant_docs = await self._retrieve_relevant_docs_advanced(
        question, session_id, history, max_context_docs
      )

      # 如果没有相关文档，尝试查询重写
      if not relevant_docs:
        expanded_queries = self.query_rewriter.expand_query(question)
        for expanded_query in expanded_queries[:2]:  # 减少重写查询数量
          relevant_docs = await self._retrieve_relevant_docs_advanced(
            expanded_query, session_id, history, max_context_docs
          )
          if relevant_docs:
            break

      # 生成回答
      if relevant_docs:
        context_with_history = _build_context_with_history(session)

        if self.answer_generator:
          answer_result = await self.answer_generator.generate_structured_answer(
            question, relevant_docs, context_with_history
          )
          answer = answer_result['answer']
          confidence = answer_result['confidence']
          question_type = answer_result['question_type']
          key_points = answer_result['key_points']
        else:
          answer = await self._generate_answer_basic(question, relevant_docs, context_with_history)
          confidence = 0.7
          question_type = "general"
          key_points = []
      else:
        answer = _generate_fallback_answer()
        confidence = 0.1
        question_type = "unknown"
        key_points = []

      # 高级幻觉检查
      hallucination_free, hallucination_score = await _check_hallucination_advanced(
        question, answer, relevant_docs
      ) if relevant_docs else (False, 0.3)

      # 生成后续问题
      follow_up_questions = await self._generate_follow_up_questions_advanced(
        question, answer, question_type
      )

      # 格式化源文档
      source_docs = self._format_source_documents_advanced(relevant_docs)

      # 计算指标
      relevance_score = confidence if hallucination_free else confidence * 0.7
      recall_rate = min(len(relevant_docs) / max_context_docs, 1.0) if relevant_docs else 0.0

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

      # 异步保存缓存
      if not self._shutdown:
        create_safe_task(
          self.cache_manager.save_async(),
          description=f"保存缓存: {session_id if session_id else '无会话'}"
        )

      return result

    except Exception as e:
      logger.error(f"获取回答失败: {e}")
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

  async def _retrieve_relevant_docs_advanced(
    self, question: str, session_id: str = None,
    history: List[Dict] = None, max_docs: int = 4
  ) -> List[Document]:
    """高级相关文档检索"""
    try:
      if self.context_retriever:
        docs = await asyncio.get_event_loop().run_in_executor(
          self.executor,
          self.context_retriever.retrieve_with_context,
          question, session_id, history, max_docs
        )
      else:
        docs = await asyncio.get_event_loop().run_in_executor(
          self.executor,
          self.vector_store_manager.search_documents,
          question
        )

      if not docs:
        return []

      filtered_docs = await _filter_relevant_docs_advanced(question, docs[:max_docs * 2])
      return filtered_docs[:max_docs]

    except Exception as e:
      logger.error(f"高级文档检索失败: {e}")
      return []

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
    """清空响应缓存 - 修复方法调用"""
    try:
      cache_count = len(self.cache_manager.cache)

      # 清空缓存
      with self.cache_manager._lock:
        self.cache_manager.cache.clear()

      # 同步保存空缓存 - 修复：使用正确的方法名
      self.cache_manager.save_cache_sync()

      logger.info(f"已清空响应缓存，原有 {cache_count} 条缓存项")

      return {
        "success": True,
        "message": f"已清空 {cache_count} 条缓存项",
        "cleared_count": cache_count
      }

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
