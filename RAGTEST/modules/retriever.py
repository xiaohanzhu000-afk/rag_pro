"""
================================================================================
检索模块（Retriever）
================================================================================

负责从向量数据库中检索与查询最相关的文档块。

支持的检索策略:
  - similarity_search:       基础相似度检索
  - similarity_search_with_score: 带分数的检索
  - mmr_search:              最大边际相关性检索（兼顾多样性和相关性）
  - as_retriever:            转为 LangChain Retriever 接口
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import RETRIEVER_CONFIG

logger = logging.getLogger(__name__)


def create_retriever(
    vectorstore: Chroma,
    search_type: str | None = None,
    k: int | None = None,
):
    """
    从向量数据库创建 LangChain Retriever 接口。

    Args:
        vectorstore:  Chroma 向量数据库实例
        search_type:  检索类型 (similarity / mmr / similarity_score_threshold)
        k:            返回结果数量

    Returns:
        BaseRetriever: LangChain Retriever 实例
    """
    search_kwargs = {"k": k or RETRIEVER_CONFIG["search_kwargs"]["k"]}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = k * 4 if k else 20

    retriever = vectorstore.as_retriever(
        search_type=search_type or RETRIEVER_CONFIG["search_type"],
        search_kwargs=search_kwargs,
    )
    logger.info("Retriever 已创建: type=%s, k=%d", search_type or RETRIEVER_CONFIG["search_type"], search_kwargs["k"])
    return retriever


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
) -> list[Document]:
    """
    基础相似度检索。

    Args:
        vectorstore: Chroma 向量数据库实例
        query:       查询文本
        k:           返回结果数量

    Returns:
        List[Document]: 最相关的文档块列表
    """
    return vectorstore.similarity_search(query, k=k)


def similarity_search_with_score(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
) -> list[tuple[Document, float]]:
    """
    带分数的相似度检索。

    Args:
        vectorstore: Chroma 向量数据库实例
        query:       查询文本
        k:           返回结果数量

    Returns:
        List[tuple[Document, float]]: 文档块和 L2 距离的元组列表（距离越小越相关）
    """
    return vectorstore.similarity_search_with_score(query, k=k)


def mmr_search(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """
    MMR（最大边际相关性）检索。

    在保证相关性的前提下，尽可能选择内容多样化的结果，
    避免返回多个高度重复的内容块。

    Args:
        vectorstore: Chroma 向量数据库实例
        query:       查询文本
        k:           返回结果数量
        fetch_k:     候选池大小
        lambda_mult: 相关性权重（0~1，越大越注重相关性）

    Returns:
        List[Document]: 多样化且相关的文档块列表
    """
    return vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def format_retrieved_docs(docs: list[Document]) -> str:
    """
    将检索到的文档列表格式化为上下文字符串，供 LLM 使用。

    Args:
        docs: 检索到的文档列表

    Returns:
        str: 格式化后的上下文字符串
    """
    if not docs:
        return "未检索到相关文档。"

    formatted = []
    for i, doc in enumerate(docs):
        source = Path(doc.metadata.get("source", "unknown")).name
        formatted.append(
            f"[文档{i + 1} 来源: {source}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)
