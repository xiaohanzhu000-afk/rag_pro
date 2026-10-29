"""
================================================================================
向量数据库模块
================================================================================

负责 Chroma 向量数据库的创建、加载与文档索引。

核心功能：
  - 创建新的向量数据库（从文档块构建）
  - 加载已有向量数据库（从磁盘恢复）
  - 检查向量数据库是否存在
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import CHROMA_CONFIG
from core.embedding import get_embedding_model

logger = logging.getLogger(__name__)


def create_vectorstore(
    chunks: list[Document],
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """
    创建向量数据库并索引文档块。

    将分割后的文档块通过 Embedding 模型转为向量，存入 Chroma 并持久化。

    Args:
        chunks:     分割后的文档块列表
        persist_dir: Chroma 持久化目录路径，默认使用配置文件中的路径
        collection_name: 集合名称，默认使用配置文件中的名称

    Returns:
        Chroma: 已索引文档的向量数据库实例
    """
    if persist_dir is None:
        persist_dir = CHROMA_CONFIG["persist_directory"]
    if collection_name is None:
        collection_name = CHROMA_CONFIG["collection_name"]

    # 确保目录存在
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    embedding_model = get_embedding_model()

    logger.info(
        "正在创建向量数据库: 集合=%s, 文档块数=%d, 持久化路径=%s",
        collection_name, len(chunks), persist_dir,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    logger.info("向量数据库创建完成: 已索引 %d 个向量", count)
    return vectorstore


def load_vectorstore(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """
    从磁盘加载已有的向量数据库。

    Args:
        persist_dir:    Chroma 持久化目录路径
        collection_name: 集合名称

    Returns:
        Chroma: 已加载的向量数据库实例

    Raises:
        FileNotFoundError: 向量数据库目录不存在
    """
    if persist_dir is None:
        persist_dir = CHROMA_CONFIG["persist_directory"]
    if collection_name is None:
        collection_name = CHROMA_CONFIG["collection_name"]

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"向量数据库目录不存在: {persist_dir}\n"
            f"请先运行 build_index.py 构建索引"
        )

    embedding_model = get_embedding_model()

    logger.info("正在从磁盘加载向量数据库: %s", persist_dir)

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_name=collection_name,
    )

    count = vectorstore._collection.count()
    logger.info("向量数据库加载成功: 共 %d 个向量", count)
    return vectorstore


def vectorstore_exists(
    persist_dir: str | None = None,
    collection_name: str | None = None,
) -> bool:
    """
    检查向量数据库是否已经存在。

    Args:
        persist_dir:    Chroma 持久化目录路径
        collection_name: 集合名称

    Returns:
        bool: 存在返回 True，否则 False
    """
    if persist_dir is None:
        persist_dir = CHROMA_CONFIG["persist_directory"]

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        return False

    # 进一步检查是否有数据
    try:
        vectorstore = load_vectorstore(persist_dir, collection_name)
        count = vectorstore._collection.count()
        return count > 0
    except Exception:
        return False
