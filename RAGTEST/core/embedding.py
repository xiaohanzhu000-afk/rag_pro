"""
================================================================================
Embedding 模型模块
================================================================================

负责向量嵌入模型的初始化与管理。

采用单例模式确保全局只有一个 Embedding 实例，避免重复初始化。
"""
import logging
from langchain_openai import OpenAIEmbeddings
from config.settings import EMBEDDING_CONFIG

logger = logging.getLogger(__name__)

# 全局单例
_embedding_instance: OpenAIEmbeddings | None = None


def get_embedding_model() -> OpenAIEmbeddings:
    """
    获取 Embedding 模型实例（单例模式）。

    Returns:
        OpenAIEmbeddings: text-embedding-3-small 模型实例
    """
    global _embedding_instance

    if _embedding_instance is None:
        logger.info(
            "初始化 Embedding 模型: %s (维度: %s)",
            EMBEDDING_CONFIG["model"],
            EMBEDDING_CONFIG.get("dimensions", "默认"),
        )
        _embedding_instance = OpenAIEmbeddings(
            model=EMBEDDING_CONFIG["model"],
            api_key=EMBEDDING_CONFIG["api_key"],
            base_url=EMBEDDING_CONFIG["base_url"],
            dimensions=EMBEDDING_CONFIG.get("dimensions", 1536),
        )
        logger.info("Embedding 模型初始化完成")

    return _embedding_instance


def reset_embedding_model() -> None:
    """重置 Embedding 模型实例（用于测试或配置变更后重新初始化）。"""
    global _embedding_instance
    _embedding_instance = None
    logger.info("Embedding 模型实例已重置")
