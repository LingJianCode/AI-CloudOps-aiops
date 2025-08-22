#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: Redis向量存储
"""

import asyncio
import hashlib
import logging
import pickle
import re
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger("aiops.vector_store")


@dataclass
class SearchConfig:
    """检索配置"""

    semantic_weight: float = 0.6
    lexical_weight: float = 0.4
    similarity_threshold: float = 0.3
    use_mmr: bool = True  # 最大边际相关性
    mmr_lambda: float = 0.5
    use_cache: bool = True
    cache_ttl: int = 3600


class EnhancedRedisVectorStore:
    """Redis向量存储"""

    def __init__(
        self,
        redis_config: Dict[str, Any],
        collection_name: str,
        embedding_model: Embeddings,
        vector_dim: int = 1536,
        index_type: str = "HNSW",  # FLAT, IVF, HNSW
    ):
        self.redis_config = redis_config
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_dim = vector_dim
        self.index_type = index_type

        # Redis连接池
        self.pool = redis.ConnectionPool(**redis_config)
        self.client = redis.Redis(connection_pool=self.pool)

        # 索引管理
        self.index_manager = IndexManager(self.client, collection_name)

        # 查询缓存
        self.query_cache = QueryCache(self.client, ttl=3600)

        # 批处理队列
        self.batch_queue = BatchQueue(max_size=50)

        self._initialize()

    def _initialize(self):
        try:
            self.client.ping()
            logger.info(f"Redis向量存储初始化成功: {self.collection_name}")

            # 创建索引
            self.index_manager.create_index(self.vector_dim, self.index_type)

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            raise

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档 - 优化版本"""
        if not documents:
            return []

        # 批处理嵌入生成
        embeddings = await self._batch_embed(documents)

        # 并行存储
        doc_ids = await self._parallel_store(documents, embeddings)

        # 更新索引
        await self.index_manager.update_index(doc_ids, embeddings)

        # 清理相关缓存
        self.query_cache.invalidate_pattern("*")

        logger.info(f"成功添加 {len(doc_ids)} 个文档")
        return doc_ids

    async def _batch_embed(self, documents: List[Document]) -> List[np.ndarray]:
        """批量生成嵌入"""
        texts = [doc.page_content for doc in documents]

        # 智能批处理
        batch_size = 25  # 优化的批大小
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # 异步生成嵌入
            batch_embeddings = await asyncio.to_thread(
                self.embedding_model.embed_documents, batch
            )
            embeddings.extend(batch_embeddings)

            # 避免API限流
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        return [np.array(e, dtype=np.float32) for e in embeddings]

    async def _parallel_store(
        self, documents: List[Document], embeddings: List[np.ndarray]
    ) -> List[str]:
        """并行存储文档"""
        doc_ids = []

        # 使用管道批量操作
        pipe = self.client.pipeline()

        for doc, embedding in zip(documents, embeddings):
            doc_id = self._generate_doc_id(doc.page_content)
            doc_ids.append(doc_id)

            # 存储文档
            doc_key = f"{self.collection_name}:doc:{doc_id}"
            pipe.hset(
                doc_key,
                mapping={
                    "content": doc.page_content,
                    "metadata": pickle.dumps(doc.metadata),
                },
            )

            # 存储向量
            vec_key = f"{self.collection_name}:vec:{doc_id}"
            pipe.set(vec_key, embedding.tobytes())

            # 更新倒排索引（限制词数，降低写放大）
            terms = list(self._tokenize_text(doc.page_content))[:100]
            for term in terms:
                term_key = f"{self.collection_name}:term:{term}"
                pipe.sadd(term_key, doc_id)

        # 执行批量操作
        await asyncio.to_thread(pipe.execute)

        return doc_ids

    async def similarity_search(
        self, query: str, k: int = 5, config: Optional[SearchConfig] = None
    ) -> List[Tuple[Document, float]]:
        """优化的相似度搜索"""
        config = config or SearchConfig()

        # 检查缓存
        if config.use_cache:
            cache_key = self._get_search_cache_key(query, k, config)
            cached = self.query_cache.get(cache_key)
            if cached:
                logger.debug(f"搜索缓存命中: {query[:50]}...")
                return cached

        # 生成查询向量
        query_embedding = await asyncio.to_thread(
            self.embedding_model.embed_query, query
        )
        query_vector = np.array(query_embedding, dtype=np.float32)

        # 执行混合搜索
        results = await self._hybrid_search(
            query, query_vector, k * 3, config  # 获取更多候选
        )

        # MMR去重和重排序
        if config.use_mmr and len(results) > k:
            results = self._mmr_rerank(results, query_vector, k, config.mmr_lambda)
        else:
            results = results[:k]

        # 缓存结果
        if config.use_cache and results:
            self.query_cache.set(cache_key, results, config.cache_ttl)

        return results

    async def _hybrid_search(
        self, query: str, query_vector: np.ndarray, k: int, config: SearchConfig
    ) -> List[Tuple[Document, float]]:
        """混合搜索"""
        # 并行执行语义和词汇搜索
        semantic_task = asyncio.create_task(self._semantic_search(query_vector, k))
        lexical_task = asyncio.create_task(self._lexical_search(query, k))

        semantic_results, lexical_results = await asyncio.gather(
            semantic_task, lexical_task
        )

        # 融合结果
        return self._fuse_results(
            semantic_results,
            lexical_results,
            config.semantic_weight,
            config.lexical_weight,
            config.similarity_threshold,
        )

    async def _semantic_search(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[Document, float]]:
        """语义搜索 - 优先使用 fallback 搜索"""
        try:
            # 尝试使用原始实现
            doc_ids_scores = await self.index_manager.search_vectors(query_vector, k)

            # 获取文档
            results = []
            for doc_id, score in doc_ids_scores:
                doc = await self._get_document(doc_id)
                if doc:
                    results.append((doc, score))

            # 如果原始实现没有结果，使用 fallback
            if not results:
                logger.info("索引搜索无结果，使用fallback搜索")
                return await self._fallback_semantic_search(query_vector, k)

            return results
        except Exception as e:
            logger.warning(f"索引搜索失败，使用fallback搜索: {e}")
            # 使用简单的余弦相似度搜索作为fallback
            return await self._fallback_semantic_search(query_vector, k)

    async def _lexical_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """词汇搜索 - 基于倒排索引"""
        # 分词（与索引一致）
        query_terms = set(self._tokenize_text(query))

        # 搜索倒排索引
        doc_scores = defaultdict(float)

        for term in query_terms:
            term_key = f"{self.collection_name}:term:{term}"
            doc_ids = await asyncio.to_thread(self.client.smembers, term_key)

            for doc_id in doc_ids:
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode()
                doc_scores[doc_id] += 1.0 / len(query_terms)

        # 排序并批量获取文档
        top_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        if not top_items:
            return []

        doc_ids = [doc_id for doc_id, _ in top_items]
        docs_map = await self._get_documents_bulk(doc_ids)

        results = []
        for doc_id, score in top_items:
            doc = docs_map.get(doc_id)
            if doc:
                results.append((doc, score))
        return results

    def _fuse_results(
        self,
        semantic_results: List[Tuple[Document, float]],
        lexical_results: List[Tuple[Document, float]],
        semantic_weight: float,
        lexical_weight: float,
        threshold: float,
    ) -> List[Tuple[Document, float]]:
        """融合搜索结果"""
        # 创建文档到分数的映射
        doc_scores = defaultdict(float)
        doc_map = {}

        # 处理语义结果
        for doc, score in semantic_results:
            doc_id = self._generate_doc_id(doc.page_content)
            doc_scores[doc_id] += score * semantic_weight
            doc_map[doc_id] = doc

        # 处理词汇结果
        for doc, score in lexical_results:
            doc_id = self._generate_doc_id(doc.page_content)
            doc_scores[doc_id] += score * lexical_weight
            doc_map[doc_id] = doc

        # 排序和过滤
        results = []
        for doc_id, score in doc_scores.items():
            if score >= threshold:
                results.append((doc_map[doc_id], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _mmr_rerank(
        self,
        results: List[Tuple[Document, float]],
        query_vector: np.ndarray,
        k: int,
        lambda_param: float,
    ) -> List[Tuple[Document, float]]:
        """最大边际相关性重排序"""
        if not results:
            return []

        # 批量提取文档向量（减少往返）
        doc_ids = [self._generate_doc_id(doc.page_content) for doc, _ in results]
        vec_keys = [f"{self.collection_name}:vec:{doc_id}" for doc_id in doc_ids]
        pipe = self.client.pipeline()
        for key in vec_keys:
            pipe.get(key)
        vec_bytes_list = pipe.execute()

        doc_vectors = []
        for vec_bytes in vec_bytes_list:
            if vec_bytes:
                doc_vectors.append(np.frombuffer(vec_bytes, dtype=np.float32))
            else:
                doc_vectors.append(np.zeros(self.vector_dim, dtype=np.float32))

        # MMR选择
        selected = []
        selected_indices = set()

        while len(selected) < k and len(selected_indices) < len(results):
            best_score = -1
            best_idx = -1

            for i, (doc, orig_score) in enumerate(results):
                if i in selected_indices:
                    continue

                # 计算与查询的相似度
                query_sim = self._cosine_similarity(query_vector, doc_vectors[i])

                # 计算与已选文档的最大相似度
                max_selected_sim = 0
                for j in selected_indices:
                    sim = self._cosine_similarity(doc_vectors[i], doc_vectors[j])
                    max_selected_sim = max(max_selected_sim, sim)

                # MMR分数
                mmr_score = (
                    lambda_param * query_sim - (1 - lambda_param) * max_selected_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(results[best_idx])
                selected_indices.add(best_idx)

        return selected

    async def _get_document(self, doc_id: str) -> Optional[Document]:
        """获取文档"""
        try:
            doc_key = f"{self.collection_name}:doc:{doc_id}"
            doc_data = await asyncio.to_thread(self.client.hgetall, doc_key)

            if not doc_data:
                return None

            content = doc_data.get(b"content", b"").decode()
            metadata = pickle.loads(doc_data.get(b"metadata", pickle.dumps({})))

            return Document(page_content=content, metadata=metadata)

        except Exception as e:
            logger.error(f"获取文档失败 {doc_id}: {e}")
            return None

    async def _get_documents_bulk(self, doc_ids: List[str]) -> Dict[str, Document]:
        """批量获取文档，减少往返"""
        if not doc_ids:
            return {}

        try:
            pipe = self.client.pipeline()
            for doc_id in doc_ids:
                doc_key = f"{self.collection_name}:doc:{doc_id}"
                pipe.hgetall(doc_key)
            results = await asyncio.to_thread(pipe.execute)

            docs: Dict[str, Document] = {}
            for doc_id, data in zip(doc_ids, results):
                if not data:
                    continue
                content = data.get(b"content", b"").decode()
                metadata = pickle.loads(data.get(b"metadata", pickle.dumps({})))
                docs[doc_id] = Document(page_content=content, metadata=metadata)
            return docs
        except Exception as e:
            logger.error(f"批量获取文档失败: {e}")
            return {}

    def _generate_doc_id(self, content: str) -> str:
        """生成文档ID"""
        return hashlib.md5(content.encode()).hexdigest()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    async def _fallback_semantic_search(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[Document, float]]:
        """Fallback语义搜索，使用余弦相似度，提高召回率"""
        try:
            logger.info("使用fallback语义搜索")

            # 使用SCAN+pipeline，维护Top-K的小根堆
            pattern = f"{self.collection_name}:vec:*"
            cursor = 0
            top_k = []  # (sim, doc_id)
            threshold = 0.1

            while True:
                cursor, keys = await asyncio.to_thread(
                    self.client.scan, cursor, match=pattern, count=500
                )
                if not keys:
                    if cursor == 0:
                        break
                if keys:
                    pipe = self.client.pipeline()
                    for key in keys:
                        pipe.get(key)
                    values = await asyncio.to_thread(pipe.execute)

                    for key, vec_bytes in zip(keys, values):
                        if not vec_bytes:
                            continue
                        stored_vector = np.frombuffer(vec_bytes, dtype=np.float32)
                        sim = self._cosine_similarity_fallback(
                            query_vector, stored_vector
                        )
                        if sim <= threshold:
                            continue
                        # 解析doc_id
                        kstr = (
                            key.decode()
                            if isinstance(key, (bytes, bytearray))
                            else str(key)
                        )
                        doc_id = kstr.split(":")[-1]

                        if len(top_k) < k * 5:
                            heapq.heappush(top_k, (sim, doc_id))
                        else:
                            if sim > top_k[0][0]:
                                heapq.heapreplace(top_k, (sim, doc_id))

                if cursor == 0:
                    break

            if not top_k:
                return []

            # 取相似度最高的前k个并批量取文档
            top_k.sort(key=lambda x: x[0], reverse=True)
            selected = top_k[:k]
            sel_ids = [doc_id for _, doc_id in selected]
            docs_map = await self._get_documents_bulk(sel_ids)

            results: List[Tuple[Document, float]] = []
            for sim, doc_id in selected:
                doc = docs_map.get(doc_id)
                if doc:
                    results.append((doc, float(sim)))

            logger.info(f"Fallback搜索找到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"Fallback搜索失败: {e}")
            return []

    def _cosine_similarity_fallback(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        try:
            # 归一化向量
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # 计算余弦相似度
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0

    def _get_search_cache_key(self, query: str, k: int, config: SearchConfig) -> str:
        """生成搜索缓存键"""
        params = f"{query}:{k}:{config.semantic_weight}:{config.lexical_weight}"
        return hashlib.md5(params.encode()).hexdigest()

    def _tokenize_text(self, text: str) -> List[str]:
        """简单分词：小写、去掉非字母数字下划线、按空白切分"""
        if not text:
            return []
        lowered = text.lower()
        cleaned = re.sub(r"[^\w\s]", " ", lowered)
        return [t for t in cleaned.split() if t]


class IndexManager:
    """索引管理器"""

    def __init__(self, redis_client, collection_name: str):
        self.client = redis_client
        self.collection_name = collection_name
        self.index_key = f"{collection_name}:index"

        # 尝试加载FAISS
        try:
            import faiss

            self.faiss_available = True
            self.faiss_index = None
        except ImportError:
            self.faiss_available = False
            logger.warning("FAISS不可用，使用基础索引")

    def create_index(self, dim: int, index_type: str = "FLAT"):
        """创建索引"""
        if not self.faiss_available:
            return

        import faiss

        if index_type == "FLAT":
            self.faiss_index = faiss.IndexFlatIP(dim)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, 100)
        elif index_type == "HNSW":
            self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
        else:
            self.faiss_index = faiss.IndexFlatIP(dim)

        logger.info(f"创建{index_type}索引，维度: {dim}")

    async def update_index(self, doc_ids: List[str], embeddings: List[np.ndarray]):
        """更新索引"""
        if not self.faiss_available or not self.faiss_index:
            return

        # 添加到FAISS索引
        vectors = np.array(embeddings, dtype=np.float32)

        # 归一化向量（用于内积相似度）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)

        # 训练索引（如果需要）
        if hasattr(self.faiss_index, "is_trained") and not self.faiss_index.is_trained:
            self.faiss_index.train(vectors)

        # 添加向量
        start_idx = self.faiss_index.ntotal
        self.faiss_index.add(vectors)

        # 更新ID映射
        for i, doc_id in enumerate(doc_ids):
            idx_key = f"{self.collection_name}:idx:{start_idx + i}"
            await asyncio.to_thread(self.client.set, idx_key, doc_id)

    async def search_vectors(
        self, query_vector: np.ndarray, k: int
    ) -> List[Tuple[str, float]]:
        """搜索向量"""
        if (
            not self.faiss_available
            or not self.faiss_index
            or self.faiss_index.ntotal == 0
        ):
            return []

        # 归一化查询向量
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)

        # FAISS搜索
        scores, indices = self.faiss_index.search(
            query_vector, min(k, self.faiss_index.ntotal)
        )

        # 获取文档ID
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            idx_key = f"{self.collection_name}:idx:{idx}"
            doc_id = await asyncio.to_thread(self.client.get, idx_key)

            if doc_id:
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode()
                results.append((doc_id, float(score)))

        return results


class QueryCache:
    """查询缓存"""

    def __init__(self, redis_client, ttl: int = 3600):
        self.client = redis_client
        self.ttl = ttl
        self.prefix = "query_cache:"

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            data = self.client.get(self.prefix + key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.warning(f"缓存读取失败: {e}")
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        try:
            self.client.setex(self.prefix + key, ttl or self.ttl, pickle.dumps(value))
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")

    def invalidate_pattern(self, pattern: str):
        """清理匹配的缓存"""
        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(
                    cursor, match=self.prefix + pattern, count=100
                )
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")


class BatchQueue:
    """批处理队列"""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.queue = []
        self.lock = asyncio.Lock()

    async def add(self, item: Any) -> Optional[List[Any]]:
        """添加项目，如果达到批量大小则返回批次"""
        async with self.lock:
            self.queue.append(item)

            if len(self.queue) >= self.max_size:
                batch = self.queue[: self.max_size]
                self.queue = self.queue[self.max_size :]
                return batch

        return None

    async def flush(self) -> List[Any]:
        """清空队列"""
        async with self.lock:
            batch = self.queue[:]
            self.queue = []
            return batch
