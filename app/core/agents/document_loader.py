#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI-CloudOps-aiops
Author: Bamboo
Email: bamboocloudops@gmail.com
License: Apache 2.0
Description: 文档加载器 - 智能文档加载和处理
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import (
        TextLoader,
        UnstructuredMarkdownLoader,
    )
    ADVANCED_LOADERS_AVAILABLE = True
except ImportError:
    ADVANCED_LOADERS_AVAILABLE = False

logger = logging.getLogger("aiops.document_loader")


class DocumentLoader:
    """智能文档加载器，支持多种文档格式的加载和处理"""

    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.supported_extensions = {
            '.md': self._load_markdown,
            '.txt': self._load_text,
        }

    async def load_all_documents(self) -> List[Document]:
        documents = []
        
        if not self.knowledge_base_path.exists():
            logger.warning(f"知识库路径不存在: {self.knowledge_base_path}")
            return documents

        try:
            for file_path in self.knowledge_base_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    try:
                        file_docs = await self._load_single_file(file_path)
                        documents.extend(file_docs)
                    except Exception as e:
                        logger.error(f"加载文档失败 {file_path}: {str(e)}")
                        continue

            logger.info(f"成功加载 {len(documents)} 个文档片段")
            return documents

        except Exception as e:
            logger.error(f"批量加载文档失败: {str(e)}")
            return documents

    async def _load_single_file(self, file_path: Path) -> List[Document]:
        extension = file_path.suffix.lower()
        loader_func = self.supported_extensions.get(extension)
        
        if not loader_func:
            return []

        try:
            return loader_func(file_path)
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {str(e)}")
            return []

    def _load_markdown(self, file_path: Path) -> List[Document]:
        try:
            if ADVANCED_LOADERS_AVAILABLE:
                loader = UnstructuredMarkdownLoader(str(file_path))
                documents = loader.load()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(
                    page_content=content,
                    metadata={"source": str(file_path), "type": "markdown"}
                )]

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            
            split_docs = []
            for doc in documents:
                splits = markdown_splitter.split_text(doc.page_content)
                for split in splits:
                    metadata = doc.metadata.copy()
                    metadata.update(split.metadata)
                    metadata["source"] = str(file_path)
                    metadata["file_type"] = "markdown"
                    
                    split_docs.append(Document(
                        page_content=split.page_content,
                        metadata=metadata
                    ))
            
            return split_docs if split_docs else documents

        except Exception as e:
            logger.error(f"加载Markdown文件失败 {file_path}: {str(e)}")
            return []

    def _load_text(self, file_path: Path) -> List[Document]:
        try:
            if ADVANCED_LOADERS_AVAILABLE:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(
                    page_content=content,
                    metadata={"source": str(file_path), "type": "text"}
                )]

            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "file_type": "text",
                    "title": file_path.stem
                })

            return documents

        except Exception as e:
            logger.error(f"加载文本文件失败 {file_path}: {str(e)}")
            return []


    def add_document_from_content(self, content: str, metadata: dict = None) -> Document:
        if metadata is None:
            metadata = {}
        
        metadata.setdefault("source", "user_input")
        metadata.setdefault("type", "text")
        metadata.setdefault("created_at", str(Path(__file__).stat().st_mtime))
        
        return Document(page_content=content, metadata=metadata)

    def validate_document(self, document: Document) -> bool:
        if not document:
            return False
        
        if not document.page_content or len(document.page_content.strip()) < 10:
            return False
        
        if not document.metadata or not document.metadata.get("source"):
            return False
        
        return True

    def get_document_stats(self, documents: List[Document]) -> dict:
        if not documents:
            return {"total": 0, "total_chars": 0, "avg_length": 0, "file_types": {}}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            "total": len(documents),
            "total_chars": total_chars,
            "avg_length": total_chars // len(documents) if documents else 0,
            "file_types": file_types
        }