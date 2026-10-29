"""
================================================================================
企业 RAG 系统 - 交互式问答主程序
================================================================================

提供两种运行模式:
  模式 1：Chain 模式 — 使用 LCEL Chain 流水线（快速、简单）
  模式 2：Agent 模式 — 使用 AI Agent 智能体（更强推理能力）

使用方式:
  # 默认启动（Chain 模式）
  python main.py

  # 使用 Agent 模式
  python main.py --mode agent

  # 查看帮助
  python main.py --help

前置条件:
  必须先运行 build_index.py 构建知识库索引
================================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

# 确保可以从当前目录导入项目模块
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DATA_DIR
from core.vector_store import load_vectorstore, vectorstore_exists
from chains.rag_chain import build_rag_chain
from agents.rag_agent import create_rag_agent, run_agent_query
from modules.retriever import format_retrieved_docs
from utils.helpers import setup_logging, print_banner

logger = logging.getLogger(__name__)

# 最近一次提问（用于查看检索来源）
_last_question: str | None = None
_last_retrieved_source: str | None = None


def show_help_text() -> None:
    """显示帮助信息。"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                 企业 RAG 问答系统 - 帮助                       ║
╠══════════════════════════════════════════════════════════════╣
║  命令：                                                      ║
║    /help     - 显示此帮助信息                                 ║
║    /source   - 查看上次回答引用的原始文档片段                  ║
║    /exit     - 退出系统                                       ║
║                                                              ║
║  提示：                                                      ║
║    - 直接输入问题即可获得基于企业制度的回答                     ║
║    - 使用 /source 可验证回答是否基于真实文档                   ║
║    - 示例问题：                                               ║
║      · 员工迟到有什么处罚？                                    ║
║      · 采购金额超过5000元需要什么审批流程？                    ║
║      · 保密制度有哪些要求？                                    ║
║      · 食堂管理有什么规定？                                    ║
╚══════════════════════════════════════════════════════════════╝
""")


def show_source(vectorstore) -> None:
    """显示上次检索到的文档来源。"""
    global _last_question

    if _last_question is None:
        print("\n  还没有提问过，请先提问。")
        return

    from modules.retriever import similarity_search

    docs = similarity_search(vectorstore, _last_question, k=4)

    print(f"\n{'=' * 60}")
    print(f"  查询: \"{_last_question}\"")
    print(f"  检索到 {len(docs)} 个相关文档块：")
    print(f"{'=' * 60}")

    for i, doc in enumerate(docs):
        source = Path(doc.metadata.get("source", "unknown")).name
        print(f"\n  [{i + 1}] 来源: {source}")
        print(f"       字符数: {len(doc.page_content)}")
        print(f"       内容: {doc.page_content[:200]}...")
    print()


def run_chain_mode(vectorstore) -> None:
    """
    Chain 模式：使用 LCEL 链式调用进行问答。

    特点：简单直接，固定流程，适合大多数问答场景。
    """
    global _last_question

    print_banner("企业 RAG 问答系统 - Chain 模式")
    print("  技术栈: LangChain 1.3 | DashScope(qwen-turbo) | text-embedding-3-small | Chroma")
    print("  输入 /help 查看帮助, /exit 退出\n")

    # 构建 RAG Chain
    logger.info("正在构建 RAG Chain...")
    rag_chain = build_rag_chain(vectorstore)
    print("  [OK] RAG Chain 已就绪\n")

    # 示例问题
    print("  示例问题：")
    examples = [
        "员工迟到有什么处罚？",
        "保密制度有哪些要求？",
        "采购金额超过5000元需要什么审批流程？",
        "食堂管理有什么规定？",
    ]
    for ex in examples:
        print(f"    · {ex}")
    print()

    # 主循环
    while True:
        try:
            question = input(">>> ").strip()

            if not question:
                continue

            if question == "/exit":
                print("\n  再见！\n")
                break

            if question == "/help":
                show_help_text()
                continue

            if question == "/source":
                show_source(vectorstore)
                continue

            # 执行 RAG 问答
            _last_question = question
            print("  正在检索知识库并生成回答...", end="", flush=True)

            answer = rag_chain.invoke(question)

            print("\r" + " " * 40 + "\r", end="")  # 清除等待提示
            print(f"\n{'─' * 60}")
            print(f"  {answer}")
            print(f"{'─' * 60}")
            print("  (输入 /source 查看引用来源)\n")

        except KeyboardInterrupt:
            print("\n\n  再见！\n")
            break
        except Exception as e:
            logger.error("查询出错: %s", e)
            print(f"\n  出错了: {e}\n")


def run_agent_mode(vectorstore) -> None:
    """
    Agent 模式：使用 AI Agent 智能体进行问答。

    特点：Agent 可以自主决策、多次检索、智能推理。
    """
    global _last_question

    print_banner("企业 RAG 问答系统 - Agent 模式")
    print("  技术栈: LangChain 1.3 | DashScope(qwen-turbo) | text-embedding-3-small | Chroma")
    print("  Agent 可自主决策，多次检索知识库以获得最佳答案")
    print("  输入 /help 查看帮助, /exit 退出\n")

    # 创建 Agent
    logger.info("正在创建 RAG Agent...")
    agent = create_rag_agent(vectorstore)
    print("  [OK] RAG Agent 已就绪\n")

    # 示例问题
    print("  示例问题：")
    examples = [
        "员工离职后还需要保守公司机密吗？保密多久？",
        "采购金额超过一万元需要谁审批？",
        "违反企业行为规范会受到哪些处罚？",
        "食堂的卫生管理有哪些要求？",
    ]
    for ex in examples:
        print(f"    · {ex}")
    print()

    # 主循环
    while True:
        try:
            question = input(">>> ").strip()

            if not question:
                continue

            if question == "/exit":
                print("\n  再见！\n")
                break

            if question == "/help":
                show_help_text()
                continue

            if question == "/source":
                show_source(vectorstore)
                continue

            # 执行 Agent 问答
            _last_question = question
            print("  Agent 正在思考并检索知识库...", end="", flush=True)

            answer = run_agent_query(agent, question)

            print("\r" + " " * 40 + "\r", end="")
            print(f"\n{'─' * 60}")
            print(f"  {answer}")
            print(f"{'─' * 60}")
            print("  (输入 /source 查看引用来源)\n")

        except KeyboardInterrupt:
            print("\n\n  再见！\n")
            break
        except Exception as e:
            logger.error("Agent 查询出错: %s", e)
            print(f"\n  出错了: {e}\n")


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(
        description="企业 RAG 系统 - 交互式问答",
    )
    parser.add_argument(
        "--mode", type=str, default="chain",
        choices=["chain", "agent"],
        help="运行模式: chain (默认) / agent",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(level=args.log_level)

    # 检查向量数据库是否存在
    if not vectorstore_exists():
        print()
        print("=" * 60)
        print("  [错误] 向量数据库不存在！")
        print("=" * 60)
        print(f"\n  请先运行 build_index.py 构建知识库索引：")
        print(f"    python build_index.py")
        print(f"\n  数据目录: {DATA_DIR}")
        print(f"\n  如果数据目录中有文件但构建失败，请运行以下命令查看详情：")
        print(f"    python build_index.py --stats")
        print()
        sys.exit(1)

    # 加载向量数据库
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        logger.error("加载向量数据库失败: %s", e)
        sys.exit(1)

    # 根据模式运行
    if args.mode == "agent":
        run_agent_mode(vectorstore)
    else:
        run_chain_mode(vectorstore)


if __name__ == "__main__":
    main()
