"""
================================================================================
RAG Chain 模块
================================================================================

使用 LangChain LCEL（LangChain Expression Language）组装 RAG 流水线。

Chain 数据流:
  question → retriever → format_docs → prompt → llm → output_parser → answer

组件说明:
  - retriever:     从向量数据库检索相关文档
  - format_docs:   将 Document 列表拼接为上下文字符串
  - prompt:        将检索到的上下文 + 问题填入 Prompt 模板
  - llm:           大模型基于上下文生成回答
  - output_parser: 将 LLM 输出转为纯文本字符串
"""

import logging
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.settings import RAG_SYSTEM_PROMPT
from core.llm import get_llm_model
from modules.retriever import create_retriever, format_retrieved_docs

logger = logging.getLogger(__name__)


def build_rag_chain(vectorstore: Chroma):
    """
    构建完整的 RAG Chain（LCEL 方式）。

    Args:
        vectorstore: Chroma 向量数据库实例

    Returns:
        Runnable: 可调用的 RAG Chain，调用 invoke(question) 即可获取回答

    Usage:
        chain = build_rag_chain(vectorstore)
        answer = chain.invoke("员工迟到有什么处罚？")
    """
    # 1. 创建检索器
    retriever = create_retriever(vectorstore)

    # 2. 创建 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # 3. 获取 LLM
    llm = get_llm_model()

    # 4. 使用 LCEL 管道符组装 Chain
    rag_chain = (
        {
            "context": retriever | format_retrieved_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG Chain 构建完成")
    return rag_chain


def build_rag_chain_with_source(vectorstore: Chroma):
    """
    构建带来源追溯的 RAG Chain。

    与 build_rag_chain 的区别：返回值包含 context 和 answer 两部分，
    可以用于 UI 展示检索来源。

    Args:
        vectorstore: Chroma 向量数据库实例

    Returns:
        Runnable: 可调用的 Chain，返回 {"context": str, "answer": str}
    """
    import json
    from langchain_core.runnables import RunnableLambda

    retriever = create_retriever(vectorstore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm_model()

    def extract_retrieved_docs(input_dict: dict) -> str:
        """从输入中提取并格式化检索到的文档"""
        docs = input_dict.get("context", [])
        return format_retrieved_docs(docs)

    rag_chain_with_source = (
        {
            "context": retriever | RunnableLambda(extract_retrieved_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 同时返回检索到的文档
    def combine_result(question: str):
        docs = retriever.invoke(question)
        context_str = format_retrieved_docs(docs)
        answer = rag_chain_with_source.invoke(question)
        return {
            "answer": answer,
            "context": context_str,
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
        }

    logger.info("RAG Chain (含来源追溯) 构建完成")
    return combine_result
