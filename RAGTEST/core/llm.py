"""
================================================================================
LLM 模型模块
================================================================================

负责大语言模型的初始化与管理。

采用单例模式确保全局只有一个 LLM 实例。
"""
import logging
from langchain_openai import ChatOpenAI
from config.settings import LLM_CONFIG

logger = logging.getLogger(__name__)

# 全局单例
_llm_instance: ChatOpenAI | None = None


def get_llm_model() -> ChatOpenAI:
    """
    获取 LLM 模型实例（单例模式）。

    Returns:
        ChatOpenAI: 通义千问模型实例（通过 DashScope 兼容接口）
    """
    global _llm_instance

    if _llm_instance is None:
        logger.info(
            "初始化 LLM 模型: %s (temperature=%.2f, max_tokens=%d)",
            LLM_CONFIG["model"],
            LLM_CONFIG["temperature"],
            LLM_CONFIG["max_tokens"],
        )
        _llm_instance = ChatOpenAI(
            model=LLM_CONFIG["model"],
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            streaming=LLM_CONFIG.get("streaming", False),
        )
        logger.info("LLM 模型初始化完成")

    return _llm_instance


def reset_llm_model() -> None:
    """重置 LLM 模型实例。"""
    global _llm_instance
    _llm_instance = None
    logger.info("LLM 模型实例已重置")
