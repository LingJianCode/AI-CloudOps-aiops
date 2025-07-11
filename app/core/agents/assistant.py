"""
RAG智能助手模块 - AI-CloudOps 系统的知识库驱动智能助手

本模块实现了基于检索增强生成（RAG）技术的智能助手代理，能够结合本地知识库和
网络搜索来回答用户问题。助手集成了多种文档加载器、向量数据库、缓存机制和
对话管理功能，提供准确、相关且及时的技术支持。

主要特性：
1. 多模态文档处理 - 支持文本、PDF、Markdown、HTML、CSV、JSON等格式
2. 智能检索系统 - 基于向量相似度的高效文档检索
3. 上下文管理 - 维护对话历史和会话状态
4. 智能缓存机制 - 减少重复计算，提高响应速度
5. 网络搜索集成 - 结合实时信息获取能力
6. 多语言LLM支持 - 兼容OpenAI和Ollama等多种模型
7. 异步任务管理 - 高效的并发处理能力
8. 质量控制 - 包含幻觉检测和相关性评估

技术架构：
- 向量数据库：ChromaDB用于文档向量存储和检索
- 文档处理：LangChain集成的多种文档加载器
- 嵌入模型：支持OpenAI Embeddings和Ollama Embeddings
- 语言模型：支持多种聊天模型进行答案生成
- 缓存层：基于哈希的智能响应缓存
- 任务管理：异步任务生命周期管理

Author: AI-CloudOps Team
Date: 2024
"""

import os
# ==================== 核心依赖导入 ====================
# 时间和异步处理相关
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
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

from app.constants import EMBEDDING_BATCH_SIZE

# ==================== LangChain核心组件 ====================
# 向量数据库和嵌入模型
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
# 文档处理和分割
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

# ==================== 高级文档加载器（可选） ====================
# 尝试导入高级文档处理能力，如果不可用则回退到基础功能
try:
    from langchain_community.document_loaders import (
        TextLoader, PyPDFLoader, DirectoryLoader,
        UnstructuredMarkdownLoader, CSVLoader, JSONLoader, BSHTMLLoader
    )
    from langchain_community.utilities import TavilySearchAPIWrapper
    ADVANCED_LOADERS_AVAILABLE = True
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    # 如果高级加载器不可用，记录状态但不影响基础功能
    ADVANCED_LOADERS_AVAILABLE = False
    WEB_SEARCH_AVAILABLE = False

# ==================== 数值计算和系统配置 ====================
import numpy as np
from chromadb.config import Settings

from app.config.settings import config

logger = logging.getLogger("aiops.assistant")

# ==================== 工具函数 ====================

def is_test_environment() -> bool:
    """
    检查当前是否在测试环境中运行

    通过检查pytest模块是否已加载来判断是否在测试环境中。
    测试环境下会使用内存数据库而不是持久化存储，以避免测试数据污染。

    Returns:
        bool: True表示在测试环境，False表示在生产环境
    """
    import sys
    return 'pytest' in sys.modules

# ==================== 任务管理器 ====================

class TaskManager:
    """
    异步任务管理器 - 管理和协调异步任务的生命周期

    该类负责创建、跟踪和清理异步任务，确保系统资源得到正确管理。
    它提供了任务创建、监控、超时处理和优雅关闭等功能，防止任务泄漏
    和资源浪费。

    主要功能：
    1. 任务创建和注册
    2. 任务生命周期跟踪
    3. 异常处理和错误恢复
    4. 优雅关闭和资源清理
    5. 防止任务泄漏

    Attributes:
        _tasks (set): 当前活跃的任务集合
        _lock (threading.Lock): 线程安全锁
        _shutdown (bool): 关闭标志
    """

    def __init__(self):
        """
        初始化任务管理器

        设置任务集合、线程锁和关闭标志，准备任务管理环境。
        """
        self._tasks = set()  # 存储所有活跃任务的引用
        self._lock = threading.Lock()  # 确保线程安全
        self._shutdown = False  # 关闭状态标志

    def create_task(self, coro, description="未命名任务"):
        """
        创建并管理异步任务

        包装协程为异步任务，添加错误处理和生命周期管理。
        任务完成后会自动从管理器中移除，避免内存泄漏。

        Args:
            coro: 要执行的协程对象
            description (str): 任务描述，用于日志和调试

        Returns:
            asyncio.Task: 创建的任务对象，如果系统已关闭则返回None

        Features:
        1. 自动异常捕获和日志记录
        2. 任务完成后自动清理
        3. 支持取消操作
        4. 线程安全的任务注册
        """
        if self._shutdown:
            logger.debug(f"任务管理器已关闭，忽略任务: {description}")
            return None

        async def wrapped_coro():
            """
            包装的协程，添加异常处理和清理逻辑
            """
            try:
                await coro
                logger.debug(f"异步任务 '{description}' 完成")
            except CancelledError:
                logger.debug(f"异步任务 '{description}' 被取消")
            except Exception as e:
                logger.error(f"异步任务 '{description}' 执行失败: {e}")
            finally:
                # 任务完成或异常时从管理器中移除
                with self._lock:
                    if task in self._tasks:
                        self._tasks.remove(task)

        # 创建并注册任务
        task = asyncio.create_task(wrapped_coro())

        with self._lock:
            self._tasks.add(task)

        return task

    async def shutdown(self, timeout=5.0):
        """
        优雅关闭任务管理器，等待或取消所有任务

        首先设置关闭标志，然后尝试等待所有任务完成。如果超时，
        会强制取消未完成的任务。这确保了系统能够优雅地关闭而
        不会留下悬挂的任务。

        Args:
            timeout (float): 等待任务完成的超时时间（秒）

        Process:
        1. 设置关闭标志，防止新任务创建
        2. 获取当前所有活跃任务
        3. 等待任务完成或超时
        4. 超时后强制取消未完成的任务
        5. 清理任务集合
        """
        self._shutdown = True

        # 获取当前任务快照
        with self._lock:
            tasks = self._tasks.copy()

        if not tasks:
            return

        logger.debug(f"等待 {len(tasks)} 个任务完成...")

        try:
            # 等待所有任务完成，设置超时保护
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            logger.debug("所有任务已完成")
        except asyncio.TimeoutError:
            logger.warning(f"等待任务完成超时，强制取消 {len(tasks)} 个任务")
            # 取消所有未完成的任务
            for task in tasks:
                if not task.done():
                    task.cancel()

            # 再等待一小段时间让取消操作完成
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning("部分任务取消操作超时")

        # 清理任务集合
        with self._lock:
            self._tasks.clear()

# ==================== 全局任务管理器实例 ====================
# 单例模式的全局任务管理器，确保整个应用程序中任务的统一管理
_task_manager = None

def get_task_manager():
    """
    获取全局任务管理器实例

    使用单例模式确保整个应用程序共享同一个任务管理器，
    这样可以统一管理所有异步任务的生命周期。

    Returns:
        TaskManager: 全局任务管理器实例
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager

def create_safe_task(coro, description="未命名任务"):
    """
    创建一个安全的异步任务

    使用全局任务管理器创建任务，确保任务被正确跟踪和管理。
    这是创建异步任务的推荐方式，提供了统一的错误处理和资源管理。

    Args:
        coro: 要执行的协程
        description (str): 任务描述

    Returns:
        asyncio.Task: 创建的任务对象
    """
    manager = get_task_manager()
    return manager.create_task(coro, description)

# ==================== 数据类和模型定义 ====================

# ==================== 数据类和模型定义 ====================

@dataclass
class DocumentMetadata:
    """
    文档元数据数据类

    存储文档的元信息，包括来源、类型、修改时间等，用于文档管理和排序。
    这些元数据在文档检索和结果展示时非常有用，可以帮助用户理解信息来源。

    Attributes:
        source (str): 文档来源路径或URL
        filename (str): 文件名称
        filetype (str): 文件类型（如text, pdf, markdown等）
        modified_time (float): 最后修改时间戳
        is_web_result (bool): 是否为网络搜索结果，默认False
        relevance_score (float): 相关性评分，默认0.0
        recall_rate (float): 召回率，默认0.0
    """
    source: str
    filename: str
    filetype: str
    modified_time: float
    is_web_result: bool = False
    relevance_score: float = 0.0
    recall_rate: float = 0.0

@dataclass
class CacheEntry:
    """
    缓存条目数据类

    表示一个缓存条目，包含时间戳和存储的数据。用于管理响应缓存，
    提高系统响应速度和减少重复计算。缓存条目包含过期时间管理，
    确保数据的时效性。

    Attributes:
        timestamp (float): 缓存创建时间戳
        data (Dict[str, Any]): 缓存的数据内容

    Methods:
        is_expired(expiry_seconds): 检查缓存是否过期
    """
    timestamp: float
    data: Dict[str, Any]

    def is_expired(self, expiry_seconds: int) -> bool:
        """
        检查缓存条目是否已过期

        Args:
            expiry_seconds (int): 过期时间（秒）

        Returns:
            bool: True表示已过期，False表示仍有效
        """
        return time.time() - self.timestamp > expiry_seconds

@dataclass
class SessionData:
    """
    会话数据数据类

    存储用户会话的完整信息，包括会话 ID、创建时间、对话历史和元数据。
    支持多轮对话的上下文保持，使得助手能够理解对话上下文和历史信息。

    Attributes:
        session_id (str): 唯一的会话标识符
        created_at (str): 会话创建时间（ISO格式）
        history (List[Dict[str, Any]]): 对话历史消息列表
        metadata (Dict[str, Any]): 会话元数据和额外信息
    """
    session_id: str
    created_at: str
    history: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class GradeDocuments(BaseModel):
    """
    文档相关性评估模型

    用于LLM评估文档与用户问题的相关性。这个模型使用Pydantic验证，
    确保输出格式的一致性和可靠性。相关性评估用于过滤无关文档，
    提高答案的准确性和相关性。

    Attributes:
        binary_score (str): 二元评分，'yes'表示相关，'no'表示不相关
    """
    binary_score: str = Field(description="文档是否与问题相关，'yes'或'no'")

class GradeHallucinations(BaseModel):
    """
    幻觉检测模型

    用于检测LLM生成的回答是否存在幻觉（即编造或不准确的信息）。
    通过检查回答是否基于提供的文档事实，确保答案的可靠性和真实性。
    这是保证RAG系统输出质量的重要组件。

    Attributes:
        binary_score (str): 二元评分，'yes'表示基于事实，'no'表示可能存在幻觉
    """
    binary_score: str = Field(description="回答是否基于事实，'yes'或'no'")

# ==================== 备用实现类 ====================

class FallbackEmbeddings(Embeddings):
    """
    备用嵌入实现 - 在主要嵌入模型不可用时的备选方案

    使用简单的哈希和随机数生成来模拟嵌入向量。虽然这不是一个真正的
    语义嵌入，但能确保系统在没有外部嵌入服务的情况下仍然可以运行。
    这个实现使用确定性的哈希算法，确保相同的文本始终产生相同的向量。

    Attributes:
        dimensions (int): 嵌入向量的维度，默认384

    Methods:
        embed_documents: 为文档列表生成嵌入向量
        embed_query: 为单个查询生成嵌入向量
    """

    def __init__(self, dimensions: int = 384):
        """
        初始化备用嵌入实现

        Args:
            dimensions (int): 嵌入向量的维度大小
        """
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成嵌入向量

        Args:
            texts (List[str]): 要嵌入的文本列表

        Returns:
            List[List[float]]: 每个文本对应的嵌入向量列表
        """
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        为单个文本生成嵌入向量

        使用文本的哈希值作为随机数种子，生成确定性的向量。
        这确保相同的文本始终产生相同的向量。

        Args:
            text (str): 要嵌入的文本

        Returns:
            List[float]: 生成的嵌入向量
        """
        # 使用文本哈希生成确定性向量，避免负数哈希值
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        return list(np.random.rand(self.dimensions))

class FallbackChatModel(BaseChatModel):
    """
    备用聊天模型 - 在主要LLM不可用时的备选方案

    提供基本的应急响应能力，当所有主要的LLM服务都不可用时使用。
    虽然这不是一个真正的智能模型，但能确保系统不会完全失败，
    并向用户提供明确的错误信息和指导。

    Methods:
        _llm_type: 返回模型类型标识
        _generate: 生成备用响应
    """

    @property
    def _llm_type(self) -> str:
        """
        返回模型类型标识

        Returns:
            str: 模型类型名称
        """
        return "fallback_chat_model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """
        生成备用响应

        创建一个通用的错误响应，告知用户主要模型不可用的情况。

        Args:
            messages: 输入消息列表
            stop: 停止条件（未使用）
            run_manager: 运行管理器（未使用）
            **kwargs: 其他参数

        Returns:
            ChatResult: 包含备用响应的结果对象
        """
        last_message = messages[-1].content if messages else "无输入"
        response = f"我是备用助手。您的问题是：'{last_message}'。由于主要模型暂时不可用，功能受限。请稍后重试。"
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

# ==================== 向量存储管理器 ====================

class VectorStoreManager:
    """
    向量存储管理器 - 负责向量数据库的创建、维护和查询

    这个类封装了所有与向量数据库相关的操作，包括数据库创建、文档存储、
    相似性搜索等。它支持内存和持久化两种模式，能够处理大量文档的
    分批处理，并提供完善的错误处理和恢复机制。

    主要功能：
    1. 数据库创建和初始化
    2. 文档分割和向量化
    3. 大批量文档的分批处理
    4. 相似性搜索和检索
    5. 数据库维护和优化
    6. 错误处理和数据恢复

    Attributes:
        vector_db_path (str): 向量数据库存储路径
        collection_name (str): 集合名称
        embedding_model: 嵌入模型实例
        db: ChromaDB实例
        retriever: 文档检索器
        _lock: 线程安全锁
    """

    def __init__(self, vector_db_path: str, collection_name: str, embedding_model):
        """
        初始化向量存储管理器

        Args:
            vector_db_path (str): 向量数据库的文件系统路径
            collection_name (str): ChromaDB集合名称
            embedding_model: 用于生成文档嵌入的模型实例
        """
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.db = None  # ChromaDB实例，延迟初始化
        self.retriever = None  # 文档检索器
        self._lock = threading.Lock()  # 线程安全锁

        # 确保目录存在，为数据库创建做准备
        os.makedirs(vector_db_path, exist_ok=True)

    def _get_client_settings(self, persistent: bool = True) -> Settings:
        """
        获取ChromaDB客户端配置设置

        根据使用场景（持久化或内存）配置ChromaDB的参数，优化性能和存储。

        Args:
            persistent (bool): 是否使用持久化存储，默认True

        Returns:
            Settings: ChromaDB配置对象

        配置选项：
        - anonymized_telemetry: 禁用遥测数据收集
        - allow_reset: 允许数据库重置
        - is_persistent: 是否持久化存储
        - chroma_db_impl: 数据库实现类型
        """
        return Settings(
            anonymized_telemetry=False,  # 禁用遥测，保护隐私
            allow_reset=True,  # 允许数据库重置，方便测试和调试
            is_persistent=persistent,  # 设置持久化模式
            chroma_db_impl="duckdb+parquet" if not persistent else None  # 内存模式使用DuckDB
        )

    def _cleanup_temp_files(self):
        """
        清理临时文件，避免数据库锁定问题

        删除可能导致数据库访问问题的临时文件，包括锁文件、
        WAL文件和其他临时文件。这些文件可能在系统异常关闭时残留。

        清理的文件类型：
        - .lock: 数据库锁文件
        - .uuid: UUID文件
        - chroma.sqlite3-shm: SQLite共享内存文件
        - chroma.sqlite3-wal: SQLite写前日志文件
        """
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
        """
        加载现有的向量数据库

        尝试从磁盘加载已存在的向量数据库，如果数据库不存在或损坏，
        会返回False。成功加载后会创建检索器并进行基本测试。

        Returns:
            bool: True表示成功加载，False表示需要创建新数据库

        加载流程：
        1. 检查数据库文件是否存在
        2. 初始化ChromaDB实例
        3. 创建文档检索器
        4. 执行测试查询验证功能
        5. 处理加载失败和数据损坏
        """
        db_file = os.path.join(self.vector_db_path, "chroma.sqlite3")

        if not os.path.exists(db_file):
            return False

        try:
            with self._lock:
                logger.info(f"加载现有向量数据库: {db_file}")
                # 初始化ChromaDB实例
                self.db = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name,
                    client_settings=self._get_client_settings(persistent=True)
                )

                # 创建检索器并配置参数
                self.retriever = self.db.as_retriever(
                    search_kwargs={"k": config.rag.top_k}
                )

                # 测试数据库功能性，确保正常工作
                test_results = self.retriever.invoke("测试查询")
                logger.info(f"数据库加载成功，测试查询返回 {len(test_results)} 个结果")
                return True

        except Exception as e:
            logger.error(f"加载现有数据库失败: {e}")
            # 数据库损坏时备份并删除
            self._backup_corrupted_db(db_file)
            return False

    def _backup_corrupted_db(self, db_file: str):
        """
        备份损坏的数据库文件

        当数据库文件损坏无法加载时，将其备份到特殊目录并删除原文件。
        这样可以保留损坏数据供后续分析，同时允许系统创建新的数据库。

        Args:
            db_file (str): 损坏的数据库文件路径

        备份流程：
        1. 创建带时间戳的备份目录
        2. 复制损坏的数据库文件
        3. 删除原数据库文件
        4. 记录备份操作结果
        """
        try:
            # 创建带时间戳的备份目录
            backup_dir = os.path.join(self.vector_db_path, f"backup_corrupt_{int(time.time())}")
            os.makedirs(backup_dir, exist_ok=True)

            # 复制损坏的数据库文件到备份目录
            shutil.copy2(db_file, backup_dir)

            # 删除损坏的原文件，为新数据库让路
            os.remove(db_file)

            logger.info(f"已备份并删除损坏的数据库文件: {backup_dir}")
        except Exception as e:
            logger.error(f"备份损坏数据库失败: {e}")

    def create_vector_store(self, documents: List[Document], use_memory: bool = False) -> bool:
        """创建向量存储"""
        if not documents:
            logger.warning("没有文档可供创建向量存储")
            documents = [Document(
                page_content="这是一个系统自动创建的示例文档。请添加更多文档到知识库中。",
                metadata={"source": "system", "filename": "example.txt", "filetype": "text"}
            )]

        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        splits = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成: {len(splits)} 个块")

        try:
            with self._lock:
                self._cleanup_temp_files()

                if use_memory:
                    logger.info("使用内存模式创建向量存储")
                    client_settings = self._get_client_settings(persistent=False)
                else:
                    logger.info("使用持久化模式创建向量存储")
                    client_settings = self._get_client_settings(persistent=True)

                # 处理批量大小限制，将文档分批处理
                batch_size = EMBEDDING_BATCH_SIZE
                total_docs = len(splits)
                all_success = True

                if total_docs > batch_size:
                    logger.info(f"文档量较大({total_docs}个)，使用分批处理方式")

                    # 创建初始空集合
                    self.db = Chroma(
                        embedding_function=self.embedding_model,
                        persist_directory=None if use_memory else self.vector_db_path,
                        collection_name=self.collection_name,
                        client_settings=client_settings
                    )

                    # 分批添加文档，每批使用更小的批量来防止API限制
                    for i in range(0, total_docs, batch_size):
                        batch = splits[i:i+batch_size]
                        logger.info(f"处理批次 {i//batch_size + 1}/{(total_docs+batch_size-1)//batch_size}，{len(batch)}个文档块")

                        try:
                            # 对于大批量，进一步拆分为更小的子批次
                            max_api_batch = 50  # 安全设置，低于API的最大限制64
                            for j in range(0, len(batch), max_api_batch):
                                sub_batch = batch[j:j+max_api_batch]
                                if j > 0:
                                    logger.debug(f"  - 子批次 {j//max_api_batch + 1}/{(len(batch)+max_api_batch-1)//max_api_batch}, {len(sub_batch)}个文档")
                                self.db.add_documents(sub_batch)
                        except Exception as e:
                            logger.error(f"添加文档批次失败: {e}")
                            all_success = False
                            break
                else:
                    # 数量较少，直接创建
                    self.db = Chroma.from_documents(
                        documents=splits,
                        embedding=self.embedding_model,
                        persist_directory=None if use_memory else self.vector_db_path,
                        collection_name=self.collection_name,
                        client_settings=client_settings
                    )

                if all_success:
                    self.retriever = self.db.as_retriever(
                        search_kwargs={"k": config.rag.top_k}
                    )

                    # 测试创建的数据库
                    test_results = self.retriever.invoke("测试查询")
                    logger.info(f"向量存储创建成功，包含 {len(splits)} 个文档块")
                    return True
                return False

        except Exception as e:
            logger.error(f"创建向量存储失败: {e}")
            if not use_memory:
                logger.info("尝试回退到内存模式")
                return self.create_vector_store(documents, use_memory=True)
            return False

    def get_retriever(self):
        """
        获取文档检索器实例

        返回配置好的文档检索器，用于执行相似性搜索。
        检索器已经配置了适当的参数，包括返回的文档数量等。

        Returns:
            文档检索器实例，如果数据库未初始化则返回None
        """
        return self.retriever

    def search_documents(self, query: str, max_retries: int = 3) -> List[Document]:
        """
        搜索相关文档

        使用向量相似性搜索找到与查询最相关的文档。支持重试机制，
        在遇到临时问题时自动重试。返回的文档按相似度排序。

        Args:
            query (str): 搜索查询字符串
            max_retries (int): 最大重试次数，默认3次

        Returns:
            List[Document]: 相关文档列表，按相似度降序排列

        搜索流程：
        1. 检查检索器是否已初始化
        2. 执行向量相似性搜索
        3. 处理搜索异常和重试
        4. 记录搜索结果和性能指标
        """
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
                    logger.warning(f"搜索返回空结果 (尝试 {attempt+1}/{max_retries})")

            except Exception as e:
                logger.error(f"文档搜索失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待一秒后重试

        return []

# ==================== 缓存管理器 ====================

class CacheManager:
    """缓存管理器，负责响应缓存的管理"""

    def __init__(self, cache_dir: str, expiry_seconds: int = 3600):
        self.cache_dir = cache_dir
        self.expiry_seconds = expiry_seconds
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
            # 使用最近的对话历史生成缓存键
            recent_history = history[-3:] if len(history) >= 3 else history
            if recent_history:
                history_str = json.dumps([
                    {"role": h["role"], "content": h["content"][:50]}
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

                # 过滤过期缓存
                valid_cache = {}
                for k, v in cache_data.items():
                    entry = CacheEntry(timestamp=v["timestamp"], data=v["data"])
                    if not entry.is_expired(self.expiry_seconds):
                        valid_cache[k] = entry

                self.cache = valid_cache
                logger.info(f"加载了 {len(self.cache)} 条有效缓存")
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            self.cache = {}

    def _save_cache_sync(self):
        """同步保存缓存到文件"""
        if self._shutdown:
            return

        try:
            with self._lock:
                # 清理过期缓存
                valid_cache = {
                    k: {"timestamp": v.timestamp, "data": v.data}
                    for k, v in self.cache.items()
                    if not v.is_expired(self.expiry_seconds)
                }

                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(valid_cache, f, ensure_ascii=False, indent=2)

                self.cache = {
                    k: CacheEntry(timestamp=v["timestamp"], data=v["data"])
                    for k, v in valid_cache.items()
                }

                logger.debug(f"保存了 {len(valid_cache)} 条缓存")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def get(self, question: str, session_id: str = None, history: List = None) -> Optional[Dict[str, Any]]:
        """获取缓存"""
        cache_key = self._generate_cache_key(question, session_id, history)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired(self.expiry_seconds):
                logger.debug(f"缓存命中: {cache_key[:8]}...")
                return entry.data
            else:
                # 删除过期缓存
                del self.cache[cache_key]

        return None

    def set(self, question: str, response_data: Dict[str, Any], session_id: str = None, history: List = None):
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
            # 在线程池中执行同步保存操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_cache_sync)
        except Exception as e:
            logger.error(f"异步保存缓存失败: {e}")

    def shutdown(self):
        """关闭缓存管理器"""
        self._shutdown = True
        # 执行最后一次同步保存
        try:
            self._save_cache_sync()
        except Exception as e:
            logger.warning(f"关闭时保存缓存失败: {e}")

# ==================== 文档加载器 ====================

class DocumentLoader:
    """文档加载器，负责加载各种格式的文档"""

    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 支持的文件扩展名
        self.supported_extensions = {
            '.txt': self._load_text_file,
            '.md': self._load_markdown_file,
            '.markdown': self._load_markdown_file,
        }

        # 如果高级加载器可用，添加更多支持
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

        # 按修改时间排序
        supported_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        logger.info(f"发现 {len(supported_files)} 个支持的文件")

        # 并行加载文件
        for file_path in supported_files:
            try:
                loader_func = self.supported_extensions[file_path.suffix.lower()]
                file_docs = loader_func(file_path)
                documents.extend(file_docs)
                logger.debug(f"加载文件: {file_path.name}, 生成 {len(file_docs)} 个文档")
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")

        logger.info(f"总共加载 {len(documents)} 个文档")
        return documents

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
                    "modified_time": file_path.stat().st_mtime
                }
            )]
        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {e}")
            return []

    def _load_markdown_file(self, file_path: Path) -> List[Document]:
        """加载Markdown文件，使用标题分割"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return []

            # 尝试使用Markdown标题分割
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

                # 添加元数据
                for doc in md_docs:
                    doc.metadata.update({
                        "source": str(file_path),
                        "filename": file_path.name,
                        "filetype": "markdown",
                        "modified_time": file_path.stat().st_mtime
                    })

                return md_docs

            except Exception:
                # 如果分割失败，作为单个文档处理
                return [Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "filetype": "markdown",
                        "modified_time": file_path.stat().st_mtime
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
                    "modified_time": file_path.stat().st_mtime
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
                    "modified_time": file_path.stat().st_mtime
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
                    "modified_time": file_path.stat().st_mtime
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
                    "modified_time": file_path.stat().st_mtime
                })

            return json_docs

        except Exception as e:
            logger.error(f"加载JSON文件失败 {file_path}: {e}")
            return []

# ==================== 主要的智能助手类 ====================

class AssistantAgent:
    """智能小助手代理 - 优化版"""

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
        self.web_search = None

        # 管理器
        self.vector_store_manager = None
        self.cache_manager = CacheManager(str(self.vector_db_path / "cache"))
        self.document_loader = DocumentLoader(str(self.knowledge_base_path))

        # 缓存存储
        self.response_cache = {}

        # 会话管理
        self.sessions: Dict[str, SessionData] = {}
        self._session_lock = threading.Lock()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 关闭标志
        self._shutdown = False

        # 初始化所有组件
        self._initialize_components()

        logger.info(f"智能小助手初始化完成，提供商: {self.llm_provider}")

    def _save_cache(self):
        """保存响应缓存到文件"""
        try:
            if hasattr(self, 'cache_manager') and not self._shutdown:
                self.cache_manager._save_cache_sync()
                logger.debug("响应缓存已保存")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 1. 初始化嵌入模型
            self._init_embedding()

            # 2. 初始化语言模型
            self._init_llm()

            # 3. 初始化向量存储
            self._init_vector_store()

            # 4. 初始化网络搜索
            self._init_web_search()

        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise

    def _init_embedding(self):
        """初始化嵌入模型，带有重试和回退机制"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self.llm_provider == 'openai':
                    logger.info(f"初始化OpenAI嵌入模型 (尝试 {attempt+1})")
                    self.embedding = OpenAIEmbeddings(
                        model=config.rag.openai_embedding_model,
                        api_key=config.llm.api_key,
                        base_url=config.llm.base_url,
                        timeout=10,
                        max_retries=2
                    )
                else:
                    logger.info(f"初始化Ollama嵌入模型 (尝试 {attempt+1})")
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
                logger.error(f"嵌入模型初始化失败 (尝试 {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    # 切换提供商重试
                    self.llm_provider = 'ollama' if self.llm_provider == 'openai' else 'openai'
                    time.sleep(1)

        # 使用备用嵌入
        logger.warning("使用备用嵌入模型")
        self.embedding = FallbackEmbeddings()

    def _init_llm(self):
        """初始化语言模型"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self.llm_provider == 'openai':
                    logger.info(f"初始化OpenAI语言模型 (尝试 {attempt+1})")

                    # 主聊天模型
                    self.llm = ChatOpenAI(
                        model=config.llm.model,
                        api_key=config.llm.api_key,
                        base_url=config.llm.base_url,
                        temperature=config.rag.temperature,
                        timeout=30,
                        max_retries=2
                    )

                    # 任务模型
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
                    logger.info(f"初始化Ollama语言模型 (尝试 {attempt+1})")
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
                logger.error(f"语言模型初始化失败 (尝试 {attempt+1}): {e}")
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

        # 尝试加载现有数据库
        if not self.vector_store_manager.load_existing_db():
            # 如果没有现有数据库，创建新的
            logger.info("创建新的向量数据库")
            documents = self.document_loader.load_documents()

            # 检查是否在测试环境
            use_memory = is_test_environment()
            success = self.vector_store_manager.create_vector_store(documents, use_memory)

            if not success:
                logger.error("向量存储初始化失败")
                raise RuntimeError("无法初始化向量存储")

        logger.info("向量存储初始化完成")

    def _init_web_search(self):
        """初始化网络搜索"""
        if WEB_SEARCH_AVAILABLE and config.tavily.api_key and config.tavily.api_key.strip() != "":
            try:
                self.web_search = TavilySearchAPIWrapper(
                    api_key=config.tavily.api_key,
                    max_results=config.tavily.max_results
                )
                # 验证API密钥是否有效
                test_result = self.web_search.results("test", max_results=1)
                if test_result:
                    logger.info("网络搜索工具初始化成功")
                else:
                    logger.warning("网络搜索初始化测试未返回结果，可能API密钥无效")
                    self.web_search = None
            except Exception as e:
                logger.warning(f"网络搜索工具初始化失败: {e}")
                self.web_search = None
        else:
            self.web_search = None
            if not WEB_SEARCH_AVAILABLE:
                logger.info("网络搜索功能不可用：缺少必要的库")
            elif not config.tavily.api_key or config.tavily.api_key.strip() == "":
                logger.info("网络搜索功能未启用：未配置Tavily API密钥")
            else:
                logger.info("网络搜索功能未启用：未知原因")

    # ==================== 会话管理 ====================

    def create_session(self) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        session_data = SessionData(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            history=[],
            metadata={}
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
            max_history = 20
            if len(session.history) > max_history:
                session.history = session.history[-max_history:]

        return session_id

    def clear_session_history(self, session_id: str) -> bool:
        """清空会话历史"""
        if session_id in self.sessions:
            with self._session_lock:
                self.sessions[session_id].history = []
            return True
        return False

    # ==================== 知识库管理 ====================

    def _load_documents(self) -> List[Document]:
        """加载文档（测试兼容方法）"""
        return self.document_loader.load_documents()

    def _get_relevant_docs(self, question: str) -> List[Document]:
        """获取相关文档（测试兼容方法）"""
        return self.vector_store_manager.search_documents(question)

    async def refresh_knowledge_base(self) -> Dict[str, Any]:
        """刷新知识库"""
        try:
            logger.info("开始刷新知识库...")

            # 清理缓存
            self.cache_manager = CacheManager(str(self.vector_db_path / "cache"))

            # 加载文档
            documents = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.document_loader.load_documents
            )

            # 重新创建向量存储
            use_memory = is_test_environment()
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.vector_store_manager.create_vector_store,
                documents,
                use_memory
            )

            if success:
                doc_count = len(documents)
                logger.info(f"知识库刷新成功，包含 {doc_count} 个文档")
                return {"success": True, "documents_count": doc_count}
            else:
                return {"success": False, "documents_count": 0, "error": "向量存储创建失败"}

        except Exception as e:
            logger.error(f"刷新知识库失败: {e}")
            return {"success": False, "documents_count": 0, "error": str(e)}

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """添加文档到知识库"""
        try:
            if not content.strip():
                return False

            # 生成文件名
            doc_id = str(uuid.uuid4())
            filename = metadata.get('filename', f"{doc_id}.txt") if metadata else f"{doc_id}.txt"
            file_path = self.knowledge_base_path / filename

            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 清理缓存
            self.cache_manager = CacheManager(str(self.vector_db_path / "cache"))

            logger.info(f"文档已添加: {filename}")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    # ==================== 网络搜索 ====================

    async def search_web(self, query: str, max_results: int = None) -> List[Dict]:
        """网络搜索"""
        if not self.web_search:
            return []

        try:
            max_results = max_results or config.tavily.max_results
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.web_search.results,
                query,
                max_results
            )
            return results
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return []

    async def _safe_web_search(self, query: str, max_results: int = None) -> Tuple[List[Dict], str]:
        """安全的网络搜索，返回结果和可能的错误信息"""
        if not self.web_search:
            return [], "网络搜索功能未启用"

        try:
            max_results = max_results or config.tavily.max_results
            results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.web_search.results,
                    query,
                    max_results
                ),
                timeout=10.0  # 设置合理的超时时间
            )

            if not results:
                return [], "未找到相关的网络搜索结果"

            logger.info(f"网络搜索返回 {len(results)} 个结果")
            return results, None

        except asyncio.TimeoutError:
            logger.error("网络搜索操作超时")
            return [], "网络搜索操作超时，请稍后重试"
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return [], f"网络搜索过程中出现错误: {str(e)[:100]}"

    # ==================== 核心问答逻辑 ====================

    async def get_answer(
        self,
        question: str,
        session_id: str = None,
        use_web_search: bool = False,
        max_context_docs: int = 4
    ) -> Dict[str, Any]:
        """获取问题答案 - 核心方法"""

        try:
            # 获取会话历史
            session = self.get_session(session_id) if session_id else None
            history = session.history if session else []

            # 检查缓存
            cached_response = self.cache_manager.get(question, session_id, history)
            if cached_response:
                # 添加到会话历史
                if session_id:
                    self.add_message_to_history(session_id, "user", question)
                    self.add_message_to_history(session_id, "assistant", cached_response["answer"])
                return cached_response

            # 添加用户消息到历史
            if session_id:
                self.add_message_to_history(session_id, "user", question)

            # 网络搜索（如果启用）
            web_results = []
            web_search_error = None
            if use_web_search:
                web_results, web_search_error = await self._safe_web_search(question)
                if web_search_error:
                    logger.warning(f"网络搜索失败: {web_search_error}")
                elif web_results:
                    logger.info(f"网络搜索返回 {len(web_results)} 个结果")
                else:
                    logger.warning("网络搜索未返回任何结果")

            # 检索相关文档
            relevant_docs = await self._retrieve_relevant_docs(question, max_context_docs)

            # 合并网络搜索结果
            if web_results:
                web_docs = self._convert_web_results_to_docs(web_results)
                relevant_docs.extend(web_docs)

            # 如果没有相关文档，尝试重写问题
            if not relevant_docs:
                rewritten_question = await self._rewrite_question(question)
                if rewritten_question != question:
                    relevant_docs = await self._retrieve_relevant_docs(rewritten_question, max_context_docs)

            # 生成回答
            if relevant_docs:
                context_with_history = self._build_context_with_history(session)
                answer = await self._generate_answer(question, relevant_docs, context_with_history)
            else:
                answer = self._generate_fallback_answer(use_web_search, web_search_error)

            # 检查幻觉
            hallucination_free = await self._check_hallucination(question, answer, relevant_docs) if relevant_docs else False

            # 生成后续问题
            follow_up_questions = await self._generate_follow_up_questions(question, answer)

            # 格式化源文档
            source_docs = self._format_source_documents(relevant_docs)

            # 构建响应
            result = {
                "answer": answer,
                "source_documents": source_docs,
                "relevance_score": 1.0 if hallucination_free else 0.5,
                "recall_rate": len(relevant_docs) / max_context_docs if relevant_docs else 0.0,
                "follow_up_questions": follow_up_questions
            }

            # 添加助手回复到历史
            if session_id:
                self.add_message_to_history(session_id, "assistant", answer)

            # 缓存结果
            self.cache_manager.set(question, result, session_id, history)

            # 创建异步保存任务，但不等待它完成
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
                "follow_up_questions": ["AIOps平台有哪些核心功能？", "如何部署AIOps系统？"]
            }

    async def _retrieve_relevant_docs(self, question: str, max_docs: int) -> List[Document]:
        """检索相关文档"""
        try:
            # 检索文档
            docs = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.vector_store_manager.search_documents,
                question
            )

            if not docs:
                return []

            # 过滤相关文档
            relevant_docs = await self._filter_relevant_docs(question, docs[:max_docs])

            return relevant_docs

        except Exception as e:
            logger.error(f"检索文档失败: {e}")
            return []

    async def _filter_relevant_docs(self, question: str, docs: List[Document]) -> List[Document]:
        """过滤相关文档"""
        if not docs or len(docs) <= 2:
            return docs

        try:
            relevant_docs = []

            for doc in docs:
                is_relevant, score = await self._evaluate_doc_relevance(question, doc)

                if is_relevant:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["relevance_score"] = score
                    relevant_docs.append(doc)

            # 如果没有相关文档，返回前几个
            if not relevant_docs:
                return docs[:3]

            # 按相关性排序
            relevant_docs.sort(
                key=lambda x: x.metadata.get("relevance_score", 0),
                reverse=True
            )

            return relevant_docs

        except Exception as e:
            logger.error(f"过滤文档失败: {e}")
            return docs[:3]

    async def _evaluate_doc_relevance(self, question: str, doc: Document) -> Tuple[bool, float]:
        """评估文档相关性"""
        try:
            # 简单的关键词匹配
            question_words = set(question.lower().split())
            doc_words = set(doc.page_content.lower().split())

            # 计算重叠度
            overlap = len(question_words & doc_words)
            total = len(question_words | doc_words)
            similarity = overlap / total if total > 0 else 0

            # 基于相似度判断相关性
            is_relevant = similarity > 0.1
            score = min(similarity * 2, 1.0)  # 归一化到[0,1]

            return is_relevant, score

        except Exception as e:
            logger.error(f"评估文档相关性失败: {e}")
            return True, 0.5  # 默认相关

    def _convert_web_results_to_docs(self, web_results: List[Dict]) -> List[Document]:
        """将网络搜索结果转换为文档"""
        if not web_results:
            return []

        docs = []
        try:
            for result in web_results:
                if not isinstance(result, dict):
                    logger.warning(f"跳过无效的网络搜索结果格式: {type(result)}")
                    continue

                title = result.get('title', '未知标题')
                url = result.get('url', '未知来源')
                content = result.get('content', '无内容')

                # 限制内容长度，避免过长文档
                max_content_length = 1000
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "...(内容已截断)"

                formatted_content = f"标题: {title}\n"
                formatted_content += f"来源: {url}\n"
                formatted_content += f"内容: {content}"

                doc = Document(
                    page_content=formatted_content,
                    metadata={
                        "source": url,
                        "title": title,
                        "is_web_result": True,
                        "filetype": "web",
                        "modified_time": time.time()
                    }
                )
                docs.append(doc)

            logger.debug(f"成功转换 {len(docs)} 个网络搜索结果为文档")
            return docs

        except Exception as e:
            logger.error(f"转换网络搜索结果失败: {e}")
            return docs  # 返回已处理的文档

    def _build_context_with_history(self, session: Optional[SessionData]) -> Optional[str]:
        """构建包含历史的上下文"""
        if not session or not session.history:
            return None

        # 获取最近的对话
        recent_history = session.history[-6:]  # 最近3轮对话
        if len(recent_history) < 2:
            return None

        context = "以下是之前的对话历史:\n"
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            context += f"{role}: {msg['content']}\n"

        return context + "\n"

    async def _rewrite_question(self, question: str) -> str:
        """重写问题以提高检索效果"""
        try:
            if len(question) < 10:
                return question

            system_prompt = """重写用户问题，使其更适合搜索。保持问题本意，只返回重写后的问题。"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"问题: {question}")
            ]

            response = await asyncio.wait_for(
                self.task_llm.ainvoke(messages),
                timeout=5
            )

            rewritten = response.content.strip()
            return rewritten if rewritten != question else question

        except Exception as e:
            logger.warning(f"重写问题失败: {e}")
            return question

    async def _generate_answer(
        self,
        question: str,
        docs: List[Document],
        context_with_history: Optional[str] = None
    ) -> str:
        """生成回答"""
        try:
            # 构建文档内容
            docs_content = ""
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "未知") if doc.metadata else "未知"
                filename = doc.metadata.get("filename", "未知文件") if doc.metadata else "未知文件"

                # 更详细的文档标识
                docs_content += f"\n\n文档[{i+1}] (文件: {filename}, 来源: {source}):\n{doc.page_content}"

            # 限制长度
            max_length = getattr(config.rag, 'max_context_length', 4000)
            if len(docs_content) > max_length:
                docs_content = docs_content[:max_length] + "...(内容已截断)"

            # 构建提示
            system_prompt = """您是专业的AI助手。请基于提供的文档内容回答用户问题。

规则:
1. 仅基于文档内容回答，不要编造信息
2. 回答要简洁清晰，直接解决问题
3. 如果文档信息不足，明确说明
4. 使用专业友好的语气
5. 语言与用户问题保持一致"""

            user_prompt = f"{context_with_history}\n\n" if context_with_history else ""
            user_prompt += f"问题: {question}\n\n文档内容:\n{docs_content}\n\n请回答问题："

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=30
            )

            return response.content.strip()

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return "抱歉，生成回答时遇到问题，请稍后重试。"

    def _generate_fallback_answer(self, web_search_attempted: bool = False, web_search_error: str = None) -> str:
        """生成备用回答"""
        if web_search_attempted and web_search_error:
            return f"抱歉，我无法回答这个问题。我尝试了网络搜索，但出现了问题：{web_search_error}。请尝试重新表述您的问题，或询问关于AIOps平台的其他问题。"
        elif web_search_attempted:
            return "抱歉，我无法回答这个问题。我尝试了网络搜索，但没有找到相关信息。请尝试重新表述您的问题，或询问关于AIOps平台的问题，我会更好地帮助您。"
        else:
            return "抱歉，我找不到与您问题相关的信息。请尝试重新表述您的问题，或询问关于AIOps平台的核心功能、部署方式或使用方法等问题。"

    async def _check_hallucination(self, question: str, answer: str, docs: List[Document]) -> bool:
        """检查回答是否存在幻觉"""
        try:
            if len(answer) < 80 or not docs:
                return True

            # 简单检查 - 基于关键词匹配
            answer_words = set(answer.lower().split())
            doc_words = set()

            for doc in docs:
                doc_words.update(doc.page_content.lower().split())

            # 计算回答中有多少词汇来自文档
            common_words = answer_words & doc_words
            coverage = len(common_words) / len(answer_words) if answer_words else 0

            # 如果覆盖率较高，认为没有幻觉
            return coverage > 0.3

        except Exception as e:
            logger.error(f"幻觉检查失败: {e}")
            return True  # 默认通过

    async def _generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        """生成后续问题"""
        default_questions = [
            "AIOps平台有哪些核心功能？",
            "如何部署和配置AIOps系统？",
            "AIOps如何帮助解决运维问题？"
        ]

        try:
            if len(answer) < 100:
                return default_questions[:3]

            system_prompt = """生成3个与当前话题相关的后续问题，每行一个，以问号结尾。"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"原问题: {question}\n回答: {answer[:300]}")
            ]

            response = await asyncio.wait_for(
                self.task_llm.ainvoke(messages),
                timeout=5
            )

            # 解析问题
            questions = []
            for line in response.content.strip().split("\n"):
                line = re.sub(r"^\d+[\.\)、]\s*", "", line.strip())
                if line and (line.endswith("?") or line.endswith("？")):
                    questions.append(line)
                elif len(line) > 10:
                    questions.append(line + "?")

            return questions[:3] if len(questions) >= 2 else default_questions[:3]

        except Exception as e:
            logger.error(f"生成后续问题失败: {e}")
            return default_questions[:3]

    def _format_source_documents(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """格式化源文档"""
        source_docs = []

        for doc in docs:
            metadata = doc.metadata or {}
            content = doc.page_content

            # 截断长内容
            if len(content) > 200:
                content = content[:200] + "..."

            source_docs.append({
                "content": content,
                "source": metadata.get("source", "未知来源"),
                "is_web_result": metadata.get("is_web_result", False),
                "metadata": metadata
            })

        return source_docs

    async def shutdown(self):
        """优雅关闭助手代理"""
        if self._shutdown:
            return

        logger.info("开始关闭智能助手...")
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

            logger.info("智能助手已成功关闭")

        except Exception as e:
            logger.warning(f"关闭智能助手时出现警告: {e}")

    def __del__(self):
        """清理资源"""
        if not self._shutdown:
            try:
                # 标记为关闭状态
                self._shutdown = True

                # 关闭缓存管理器
                if hasattr(self, 'cache_manager'):
                    try:
                        self.cache_manager.shutdown()
                    except Exception as e:
                        logger.warning(f"对象销毁时关闭缓存管理器失败: {e}")

                # 关闭线程池
                if hasattr(self, 'executor') and self.executor:
                    try:
                        self.executor.shutdown(wait=False)
                    except Exception as e:
                        logger.warning(f"对象销毁时关闭线程池失败: {e}")

            except Exception as e:
                logger.warning(f"AssistantAgent清理资源时出错: {e}")
