"""
================================================================================
文档加载模块（Document Loader）
================================================================================

支持多种文档格式的统一加载，将不同格式的文件统一转为 LangChain Document 对象。

支持的格式:
  - .txt  纯文本文件
  - .md   Markdown 文件
  - .docx Word 文档
  - .pdf  PDF 文档
  - .png / .jpg / .jpeg  图片文件（OCR 文字识别）

设计原则:
  - 策略模式：每种文件格式对应一个加载策略
  - 统一接口：所有加载器返回 List[Document]
  - 可扩展：新增文件格式只需添加新的加载策略
"""

import logging
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    PyPDFLoader,
)

from config.settings import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# ============================================================================
# 各格式加载策略
# ============================================================================

def _load_text(file_path: Path) -> list[Document]:
    """加载 .txt / .md 纯文本文件"""
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    # 为每个 Document 补充文件名元数据
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = file_path.suffix
    logger.info("  [TEXT] 加载完成: %s (%d 字符)", file_path.name, len(docs[0].page_content) if docs else 0)
    return docs


def _load_docx(file_path: Path) -> list[Document]:
    """加载 .docx Word 文档"""
    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = file_path.suffix
    logger.info("  [DOCX] 加载完成: %s (%d 字符)", file_path.name, len(docs[0].page_content) if docs else 0)
    return docs


def _load_pdf(file_path: Path) -> list[Document]:
    """加载 .pdf PDF 文档"""
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = file_path.suffix
    total_chars = sum(len(d.page_content) for d in docs)
    logger.info("  [PDF] 加载完成: %s (%d 页, %d 字符)", file_path.name, len(docs), total_chars)
    return docs


def _load_image(file_path: Path) -> list[Document]:
    """
    加载图片文件（尝试 OCR 提取文字）。

    优先使用 pytesseract 进行 OCR 文字识别，
    如果未安装则尝试使用 PIL 读取图片基本信息并记录元数据。

    注意：OCR 需要系统安装 tesseract-ocr。
    """
    try:
        import pytesseract
        from PIL import Image

        logger.info("  [IMAGE] 正在对图片进行 OCR 识别: %s", file_path.name)
        img = Image.open(str(file_path))
        text = pytesseract.image_to_string(img, lang="chi_sim+eng")

        if not text or not text.strip():
            logger.warning("  [IMAGE] 未从图片中识别到文字: %s", file_path.name)
            return []

        doc = Document(
            page_content=text.strip(),
            metadata={
                "source": str(file_path),
                "file_type": file_path.suffix,
                "extraction_method": "ocr",
            },
        )
        logger.info("  [IMAGE] OCR 完成: %s (%d 字符)", file_path.name, len(doc.page_content))
        return [doc]

    except ImportError as e:
        logger.warning(
            "  [IMAGE] 缺少 OCR 依赖 (%s)，跳过图片: %s。"
            "安装方式: pip install pytesseract Pillow，并安装 tesseract-ocr",
            e, file_path.name,
        )
        # 返回空列表，跳过无法处理的图片
        return []
    except Exception as e:
        logger.error("  [IMAGE] 处理图片失败: %s, 错误: %s", file_path.name, e)
        return []


# ============================================================================
# Loader 注册表：扩展名 → 加载函数
# ============================================================================

_LOADER_REGISTRY: dict[str, Callable[[Path], list[Document]]] = {
    "text":  _load_text,
    "docx":  _load_docx,
    "pdf":   _load_pdf,
    "image": _load_image,
}


# ============================================================================
# 公共 API
# ============================================================================

def load_document(file_path: str | Path) -> list[Document]:
    """
    加载单个文档文件。

    根据文件扩展名自动选择合适的加载策略。

    Args:
        file_path: 文件路径

    Returns:
        List[Document]: 加载后的 Document 对象列表

    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = file_path.suffix.lower()
    loader_type = SUPPORTED_EXTENSIONS.get(ext)

    if loader_type is None:
        raise ValueError(
            f"不支持的文件格式: {ext}\n"
            f"支持的格式: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    loader_func = _LOADER_REGISTRY.get(loader_type)
    if loader_func is None:
        raise ValueError(f"未注册的加载器类型: {loader_type}")

    return loader_func(file_path)


def load_documents(data_dir: str | Path) -> list[Document]:
    """
    批量加载目录下的所有支持格式的文档文件。

    Args:
        data_dir: 知识库数据目录路径

    Returns:
        List[Document]: 所有加载后的 Document 对象列表
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_documents: list[Document] = []
    skipped_files: list[str] = []

    # 扫描目录下的所有文件
    files = sorted(
        [f for f in data_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    logger.info("开始加载文档，数据目录: %s，共发现 %d 个文件", data_dir, len(files))

    for file_path in files:
        try:
            docs = load_document(file_path)
            if docs:
                all_documents.extend(docs)
            else:
                skipped_files.append(file_path.name)
        except Exception as e:
            logger.error("加载文件失败: %s, 错误: %s", file_path.name, e)
            skipped_files.append(file_path.name)

    # 汇总
    logger.info("=" * 50)
    logger.info("文档加载完成:")
    logger.info("  成功加载: %d 个文件 → %d 个 Document", len(files) - len(skipped_files), len(all_documents))
    logger.info("  总字符数: %d", sum(len(d.page_content) for d in all_documents))
    if skipped_files:
        logger.warning("  跳过的文件: %s", ", ".join(skipped_files))
    logger.info("=" * 50)

    return all_documents


def get_loader_stats(data_dir: str | Path) -> dict:
    """
    获取数据目录的统计信息（不实际加载文档内容）。

    Args:
        data_dir: 数据目录路径

    Returns:
        dict: 包含文件数量、类型分布等统计信息
    """
    data_dir = Path(data_dir)
    stats = {
        "total_files": 0,
        "by_type": {},
        "supported": [],
        "unsupported": [],
    }

    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            stats["total_files"] += 1
            loader_type = SUPPORTED_EXTENSIONS[ext]
            stats["by_type"][loader_type] = stats["by_type"].get(loader_type, 0) + 1
            stats["supported"].append(file_path.name)
        else:
            stats["unsupported"].append(file_path.name)

    return stats
