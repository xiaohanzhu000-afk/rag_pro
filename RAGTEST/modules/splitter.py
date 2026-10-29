"""
================================================================================
文档分割模块（Text Splitter）
================================================================================

负责将长文档分割为适合向量检索的文本块（Chunks）。

核心策略：RecursiveCharacterTextSplitter
  - 按 separators 优先级递归分割
  - 针对中文文档优化了分割符顺序
  - 保留块间重叠以防止语义断裂
"""

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import SPLITTER_CONFIG

logger = logging.getLogger(__name__)


def create_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    创建文本分割器实例。

    参数均可选，不传则使用配置文件中的默认值。

    Args:
        chunk_size:    每块最大字符数
        chunk_overlap: 块间重叠字符数
        separators:    分割符优先级列表

    Returns:
        RecursiveCharacterTextSplitter: 配置好的分割器
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or SPLITTER_CONFIG["chunk_size"],
        chunk_overlap=chunk_overlap or SPLITTER_CONFIG["chunk_overlap"],
        separators=separators or SPLITTER_CONFIG["separators"],
        length_function=SPLITTER_CONFIG["length_function"],
        keep_separator=SPLITTER_CONFIG["keep_separator"],
        add_start_index=SPLITTER_CONFIG["add_start_index"],
    )


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    将文档列表分割为文本块。

    Args:
        documents:     待分割的 Document 对象列表
        chunk_size:    每块最大字符数（可选，默认 500）
        chunk_overlap: 块间重叠字符数（可选，默认 100）

    Returns:
        List[Document]: 分割后的 Document 对象列表
    """
    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    logger.info(
        "开始分割文档: 原始 %d 个文档, chunk_size=%d, chunk_overlap=%d",
        len(documents),
        splitter._chunk_size,
        splitter._chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    # 统计信息
    if chunks:
        lengths = [len(c.page_content) for c in chunks]
        logger.info(
            "文档分割完成: %d 个原始文档 → %d 个文本块, "
            "平均长度=%.0f 字符, 范围=[%d, %d]",
            len(documents), len(chunks),
            sum(lengths) / len(lengths),
            min(lengths), max(lengths),
        )

    return chunks
