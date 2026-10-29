"""
================================================================================
RAG Agent 智能体模块
================================================================================

使用 LangChain Agent + Tool 实现智能检索增强问答。

Agent 工作流程（ReAct 模式）：
  Thought → Action（调用 search_knowledge_base 工具）→ Observation → Final Answer

与 Chain 方式的区别：
  - Chain: 固定流水线，按部就班执行
  - Agent: 智能决策，可以多次检索、评估结果、决定是否需要更多信息
"""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.tools import tool

from config.settings import AGENT_SYSTEM_PROMPT
from core.llm import get_llm_model
from modules.retriever import format_retrieved_docs

logger = logging.getLogger(__name__)


def create_rag_agent(vectorstore: Chroma, system_prompt: str | None = None):
    """
    创建 RAG Agent 智能体。

    Agent 会根据用户问题自动决定是否检索知识库，
    以及如何利用检索结果生成准确回答。

    Args:
        vectorstore:   Chroma 向量数据库实例
        system_prompt: 自定义系统提示词（可选）

    Returns:
        Agent: LangChain Agent 实例

    Usage:
        agent = create_rag_agent(vectorstore)
        response = agent.invoke({
            "messages": [{"role": "user", "content": "保密制度有哪些要求？"}]
        })
        answer = response["messages"][-1].content
    """

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        在企业知识库中搜索相关文档信息。

        当需要查阅企业制度、规范、流程等内部文档时，必须使用此工具。
        文档包括：企业行为规范、财务支付流程、招聘简章、食堂管理规范等。

        Args:
            query: 用户想要查询的问题或关键词

        Returns:
            str: 检索到的最相关文档片段（格式化后的文本）
        """
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "未在知识库中找到相关文档信息。"
        return format_retrieved_docs(docs)

    # 创建 Agent
    agent = create_agent(
        model=get_llm_model(),
        tools=[search_knowledge_base],
        system_prompt=system_prompt or AGENT_SYSTEM_PROMPT,
    )

    logger.info("RAG Agent 创建完成，工具: search_knowledge_base")
    return agent


def run_agent_query(agent, question: str) -> str:
    """
    使用 Agent 处理单个问题并返回回答。

    Args:
        agent:    RAG Agent 实例
        question: 用户问题

    Returns:
        str: Agent 的回答文本
    """
    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        # Agent 最后一条消息即为最终回答
        answer = response["messages"][-1].content
        return answer
    except Exception as e:
        logger.error("Agent 查询失败: %s", e)
        return f"查询出错: {e}"
