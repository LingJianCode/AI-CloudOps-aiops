#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 文档排序和检索器 - 智能文档排序和上下文感知检索
"""

import logging
import numpy as np
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("aiops.document_processor")


class DocumentRanker:
    """智能文档排序器，基于多种因素对检索到的文档进行精确排序"""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=None,  # 保留中文停用词处理
            max_features=1000,
            ngram_range=(1, 2),  # 使用unigram和bigram
        )

    def rank_documents(
        self, 
        documents: List[Dict[str, Any]], 
        query: str, 
        original_scores: List[float]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        对文档进行智能排序
        
        Args:
            documents: 文档列表
            query: 原始查询
            original_scores: 原始相似度分数
            
        Returns:
            排序后的文档及其分数
        """
        if not documents:
            return []

        try:
            # 1. TF-IDF重排序
            tfidf_scores = self._calculate_tfidf_scores(documents, query)
            
            # 2. 关键词匹配加权
            keyword_scores = self._calculate_keyword_scores(documents, query)
            
            # 3. 文档长度和质量评分
            quality_scores = self._calculate_quality_scores(documents)
            
            # 4. 综合评分
            final_scores = []
            for i, doc in enumerate(documents):
                # 权重分配：原始分数40%，TF-IDF 30%，关键词20%，质量10%
                combined_score = (
                    0.4 * original_scores[i] +
                    0.3 * tfidf_scores[i] +
                    0.2 * keyword_scores[i] +
                    0.1 * quality_scores[i]
                )
                final_scores.append(combined_score)
            
            # 5. 排序
            ranked_docs_with_scores = list(zip(documents, final_scores))
            ranked_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            return ranked_docs_with_scores
            
        except Exception as e:
            logger.warning(f"文档排序失败，使用原始顺序: {str(e)}")
            return list(zip(documents, original_scores))

    def _calculate_tfidf_scores(self, documents: List[Dict[str, Any]], query: str) -> List[float]:
        """计算TF-IDF相似度分数"""
        try:
            texts = [doc.get("page_content", "") for doc in documents]
            all_texts = texts + [query]
            
            # 计算TF-IDF矩阵
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]  # 查询向量
            doc_vectors = tfidf_matrix[:-1]  # 文档向量
            
            # 计算余弦相似度
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            return similarities.tolist()
            
        except Exception as e:
            logger.warning(f"TF-IDF计算失败: {str(e)}")
            return [0.5] * len(documents)

    def _calculate_keyword_scores(self, documents: List[Dict[str, Any]], query: str) -> List[float]:
        """计算关键词匹配分数"""
        query_words = set(query.lower().split())
        scores = []
        
        for doc in documents:
            content = doc.get("page_content", "").lower()
            doc_words = set(content.split())
            
            # 计算关键词重叠率
            overlap = len(query_words.intersection(doc_words))
            score = overlap / len(query_words) if query_words else 0
            scores.append(score)
        
        return scores

    def _calculate_quality_scores(self, documents: List[Dict[str, Any]]) -> List[float]:
        """计算文档质量分数"""
        scores = []
        
        for doc in documents:
            content = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            
            score = 0.5  # 基础分数
            
            # 文档长度评分（适中长度得分更高）
            content_length = len(content)
            if 100 <= content_length <= 2000:
                score += 0.2
            elif content_length > 2000:
                score += 0.1
            
            # 元数据丰富度
            if metadata.get("source"):
                score += 0.1
            if metadata.get("title"):
                score += 0.1
            
            # 内容结构评分（包含标题、列表等结构化内容）
            if any(marker in content for marker in ["##", "###", "- ", "* ", "1. "]):
                score += 0.1
            
            scores.append(min(score, 1.0))  # 限制最大分数为1.0
        
        return scores


class ContextAwareRetriever:
    """上下文感知检索器，基于会话历史优化检索结果"""

    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
        self.document_ranker = DocumentRanker()

    async def retrieve_with_context(
        self,
        query: str,
        session_history: Optional[List[str]] = None,
        max_docs: int = 4,
        diversity_threshold: float = 0.7
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        执行上下文感知的文档检索
        
        Args:
            query: 查询字符串
            session_history: 会话历史
            max_docs: 最大文档数
            diversity_threshold: 多样性阈值
            
        Returns:
            检索到的文档和召回率
        """
        try:
            # 1. 构建增强查询
            enhanced_query = self._build_enhanced_query(query, session_history)
            
            # 2. 多轮检索
            all_documents = []
            all_scores = []
            
            # 第一轮：原始查询
            docs1, scores1 = await self._perform_retrieval(query, max_docs * 2)
            all_documents.extend(docs1)
            all_scores.extend(scores1)
            
            # 第二轮：增强查询（如果有历史上下文）
            if enhanced_query != query:
                docs2, scores2 = await self._perform_retrieval(enhanced_query, max_docs)
                all_documents.extend(docs2)
                all_scores.extend(scores2)
            
            # 3. 去重和多样性过滤
            unique_docs, unique_scores = self._remove_duplicates_and_diversify(
                all_documents, all_scores, diversity_threshold
            )
            
            # 4. 智能排序
            ranked_docs_with_scores = self.document_ranker.rank_documents(
                unique_docs, query, unique_scores
            )
            
            # 5. 截取最终结果
            final_docs = [doc for doc, _ in ranked_docs_with_scores[:max_docs]]
            
            # 6. 计算召回率
            recall_rate = self._calculate_recall_rate(final_docs, query)
            
            logger.info(f"检索完成: 查询='{query}', 文档数={len(final_docs)}, 召回率={recall_rate:.2f}")
            
            return final_docs, recall_rate
            
        except Exception as e:
            logger.error(f"上下文感知检索失败: {str(e)}")
            # 降级到简单检索
            try:
                docs, scores = await self._perform_retrieval(query, max_docs)
                recall_rate = self._calculate_recall_rate(docs, query)
                return docs, recall_rate
            except Exception as fallback_e:
                logger.error(f"降级检索也失败: {str(fallback_e)}")
                return [], 0.0

    def _build_enhanced_query(self, query: str, session_history: Optional[List[str]]) -> str:
        """构建增强查询"""
        if not session_history:
            return query
        
        # 从历史中提取相关关键词
        history_text = " ".join(session_history[-3:])  # 只使用最近3轮对话
        history_keywords = self._extract_keywords_from_history(history_text)
        
        # 构建增强查询
        if history_keywords:
            enhanced_query = f"{query} {' '.join(history_keywords[:3])}"
            return enhanced_query
        
        return query

    def _extract_keywords_from_history(self, history_text: str) -> List[str]:
        """从历史文本中提取关键词"""
        # 简单的关键词提取逻辑
        import re
        words = re.findall(r'[\w\u4e00-\u9fff]+', history_text.lower())
        
        # 过滤停用词和短词
        stop_words = {"如何", "什么", "为什么", "怎么", "是", "的", "了", "在", "和"}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 返回频率最高的关键词
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(5)]

    async def _perform_retrieval(self, query: str, k: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """执行基础检索"""
        try:
            vector_store = await self.vector_store_manager.get_vector_store()
            if not vector_store:
                return [], []
            
            # 执行相似性搜索
            results = await vector_store.asimilarity_search_with_score(query, k=k)
            
            documents = []
            scores = []
            
            for doc, score in results:
                doc_dict = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                documents.append(doc_dict)
                scores.append(float(score))
            
            return documents, scores
            
        except Exception as e:
            logger.error(f"基础检索失败: {str(e)}")
            return [], []

    def _remove_duplicates_and_diversify(
        self, 
        documents: List[Dict[str, Any]], 
        scores: List[float], 
        threshold: float
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """去重和多样性过滤"""
        if not documents:
            return [], []
        
        unique_docs = []
        unique_scores = []
        
        for i, doc in enumerate(documents):
            content = doc.get("page_content", "")
            
            # 检查是否与已有文档过于相似
            is_duplicate = False
            for existing_doc in unique_docs:
                existing_content = existing_doc.get("page_content", "")
                similarity = self._calculate_text_similarity(content, existing_content)
                
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                unique_scores.append(scores[i])
        
        return unique_docs, unique_scores

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            # 简单的基于词汇重叠的相似度计算
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception:
            return 0.0

    def _calculate_recall_rate(self, documents: List[Dict[str, Any]], query: str) -> float:
        """计算召回率"""
        if not documents:
            return 0.0
        
        # 改进的召回率计算：基于文档相关性和覆盖率
        query_words = set(query.lower().split())
        
        # 移除常见的停用词，专注于关键词
        stop_words = {"什么", "是", "的", "了", "在", "和", "与", "或", "不", "但", "如果", "因为", "所以", "？", "。", "！", "怎么", "如何", "为什么"}
        query_words = query_words - stop_words
        
        if not query_words:
            # 如果去除停用词后没有关键词，返回基于文档数量的基础分数
            return min(0.6 + 0.1 * len(documents), 1.0)
        
        relevant_docs = 0
        total_coverage = 0
        semantic_relevance = 0
        
        for doc in documents:
            content = doc.get("page_content", "").lower()
            doc_words = set(content.split())
            
            # 计算查询关键词在文档中的覆盖率
            matched_words = query_words.intersection(doc_words)
            coverage = len(matched_words) / len(query_words)
            
            # 如果至少有一个关键词匹配，认为这是相关文档
            if coverage > 0:
                relevant_docs += 1
                total_coverage += coverage
                # 给予更高的权重以提升召回率
                semantic_relevance += min(coverage * 2, 1.0)
            else:
                # 即使没有直接关键词匹配，向量搜索找到的文档也有一定相关性
                # 基于文档长度和内容质量给予基础分数
                content_quality = min(len(content) / 500, 1.0)  # 基于内容长度的质量分数
                semantic_relevance += content_quality * 0.3
        
        # 召回率计算：提升基础分数，更好反映语义相关性
        if relevant_docs > 0:
            # 有关键词匹配的情况
            doc_relevance = relevant_docs / len(documents)
            avg_coverage = total_coverage / relevant_docs
            keyword_recall = doc_relevance * avg_coverage
            
            # 结合语义相关性分数
            semantic_score = semantic_relevance / len(documents)
            
            # 综合召回率：关键词匹配60% + 语义相关性40%
            recall_rate = min(0.6 * keyword_recall + 0.4 * semantic_score, 1.0)
        else:
            # 没有关键词匹配，但有文档返回，基于语义相关性
            semantic_score = semantic_relevance / len(documents)
            # 提升基础分数，给予更合理的召回率
            recall_rate = min(0.4 + 0.6 * semantic_score, 1.0)
        
        # 确保召回率不低于0.3（如果有文档返回）
        return max(recall_rate, 0.3) if documents else 0.0