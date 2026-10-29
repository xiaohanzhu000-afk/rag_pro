"""
================================================================================
企业 RAG 系统 - 知识库索引构建脚本
================================================================================

离线执行，将 data/ 目录下的所有文档构建为向量索引。

执行流程:
  1. 扫描 data/ 目录，加载所有支持格式的文档
  2. 将长文档分割为文本块（Chunks）
  3. 使用 Embedding 模型将文本块转为向量
  4. 向量存储到 Chroma 向量数据库（持久化到磁盘）

使用方式:
  # 构建索引（使用默认配置）
  python build_index.py

  # 清空旧索引并重新构建
  python build_index.py --rebuild

  # 查看数据目录统计（不构建索引）
  python build_index.py --stats

  # 自定义分割参数
  python build_index.py --chunk-size 800 --chunk-overlap 150

依赖:
  pip install -r requirements.txt
================================================================================
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# 确保可以从当前目录导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATA_DIR, CHROMA_PERSIST_DIR, CHROMA_CONFIG
from modules.loader import load_documents, get_loader_stats
from modules.splitter import split_documents
from core.vector_store import create_vectorstore, vectorstore_exists
from utils.helpers import setup_logging, print_banner, Timer

logger = logging.getLogger(__name__)


def build_index(
    rebuild: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> bool:
    """
    构建知识库向量索引。

    Args:
        rebuild:       是否清空已有索引并重建
        chunk_size:    文本块最大字符数
        chunk_overlap: 块间重叠字符数

    Returns:
        bool: 构建成功返回 True
    """
    print_banner("企业 RAG 系统 - 知识库索引构建")

    # ---- 检查是否需要重建 ----
    if not rebuild and vectorstore_exists():
        logger.info("向量数据库已存在，跳过索引构建。使用 --rebuild 强制重建。")
        return True

    if rebuild and CHROMA_PERSIST_DIR.exists():
        logger.info("正在清空旧的向量数据库...")
        shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 步骤 1：加载文档 ----
    with Timer("文档加载"):
        try:
            documents = load_documents(DATA_DIR)
        except FileNotFoundError as e:
            logger.error("文档加载失败: %s", e)
            return False

    if not documents:
        logger.error("未加载到任何文档！请确保 data/ 目录下有支持格式的文件。")
        return False

    print(f"\n  [OK] 步骤 1/3：共加载 {len(documents)} 个文档")

    # ---- 步骤 2：分割文档 ----
    with Timer("文档分割"):
        chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    print(f"  [OK] 步骤 2/3：共分割为 {len(chunks)} 个文本块")

    # ---- 步骤 3：向量化并存储 ----
    with Timer("向量化与存储"):
        try:
            vectorstore = create_vectorstore(chunks)
        except Exception as e:
            logger.error("向量数据库创建失败: %s", e)
            return False

    count = vectorstore._collection.count()
    print(f"  [OK] 步骤 3/3：已索引 {count} 个向量到 Chroma")

    # ---- 完成 ----
    print("\n" + "=" * 60)
    print("  索引构建完成！")
    print(f"  存储位置: {CHROMA_PERSIST_DIR}")
    print(f"  向量数量: {count}")
    print(f"  集合名称: {CHROMA_CONFIG['collection_name']}")
    print("=" * 60)
    return True


def show_stats() -> None:
    """显示数据目录的统计信息。"""
    print_banner("数据目录统计")

    stats = get_loader_stats(DATA_DIR)

    print(f"\n  数据目录: {DATA_DIR}")
    print(f"  支持格式的文件: {stats['total_files']} 个")

    if stats["by_type"]:
        print(f"\n  文件类型分布:")
        for file_type, count in stats["by_type"].items():
            print(f"    - {file_type}: {count} 个")

    if stats["supported"]:
        print(f"\n  可处理文件列表:")
        for name in stats["supported"]:
            print(f"    ✓ {name}")

    if stats["unsupported"]:
        print(f"\n  不支持的文件:")
        for name in stats["unsupported"]:
            print(f"    ✗ {name}")

    print()


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="企业 RAG 系统 - 知识库索引构建工具",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="清空已有索引并重新构建",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="仅显示数据目录统计信息，不构建索引",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="文本块最大字符数 (默认: 500)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="文本块间重叠字符数 (默认: 100)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(level=args.log_level)

    if args.stats:
        show_stats()
        return

    success = build_index(
        rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
