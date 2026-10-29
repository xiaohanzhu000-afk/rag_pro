"""
============================================================================
RAG 学习案例 - 步骤 6：交互式 RAG 问答应用
============================================================================
本步骤创建一个命令行交互式的 RAG 问答应用，你可以：

  1. 自由输入问题
  2. 查看 LLM 基于文档生成的回答
  3. 同时查看检索到的原始文档片段（方便验证）
  4. 支持特殊命令：/help /source /exit

使用方式：
  python 6_rag_qa.py

============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from env_utils import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    OPENAI_BASE_URL,
)

# ============================================================================
# 初始化 RAG 组件
# ============================================================================

base_dir = os.path.dirname(__file__)
chroma_persist_dir = os.path.join(base_dir, "chroma_db")

def initialize_rag():
    """初始化 RAG 所需的全部组件"""
    print("\n" + "=" * 70)
    print("  🚀 正在初始化 RAG 系统...")
    print("=" * 70)

    # 1. 加载向量数据库
    if not os.path.exists(chroma_persist_dir):
        print("\n❌ 向量数据库不存在！")
        print("   请先运行以下命令创建向量数据库：")
        print("   python 3_embedding_chroma.py")
        sys.exit(1)

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="sk-eEnWubQSh3rNm06a1hRj8DIW4zZx7Tak0TTCQYwq2QjkVgV2",
        base_url=OPENAI_BASE_URL,
    )

    vector_store = Chroma(
        persist_directory=chroma_persist_dir,
        embedding_function=embeddings_model,
        collection_name="enterprise_policies",
    )

    doc_count = vector_store._collection.count()
    print(f"  ✅ 向量数据库已加载 ({doc_count} 个文档块)")

    # 2. 初始化 LLM
    llm = ChatOpenAI(
        model="qwen-turbo",
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.3,
        max_tokens=1024,
    )
    print(f"  ✅ LLM 已就绪 (qwen-turbo)")

    # 3. 创建检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    print(f"  ✅ 检索器已就绪 (k=4)")

    # 4. 创建 Prompt 模板
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个专业的企业制度咨询助手。请严格基于以下【参考文档】中的内容回答问题。

【回答规则】
1. 只使用参考文档中明确提到的信息，不要编造或推测
2. 如果参考文档中没有相关信息，请明确说明：\"根据现有制度文档，未找到相关信息\"
3. 回答要条理清晰，使用序号或分点列出关键信息
4. 如果涉及处罚、流程等内容，请引用制度中的具体规定

【参考文档】
{context}""",
        ),
        ("human", "{question}"),
    ])

    def format_docs(docs: list[Document]) -> str:
        """将检索到的文档列表格式化为上下文字符串"""
        formatted = []
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            formatted.append(f"[文档{i+1} 来源:{source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # 5. 组装 Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print(f"  ✅ RAG Chain 已就绪")
    return rag_chain, retriever


def show_help():
    """显示帮助信息"""
    help_text = """
╔══════════════════════════════════════════════════════════════╗
║                  RAG 问答系统 - 帮助                          ║
╠══════════════════════════════════════════════════════════════╣
║  命令：                                                      ║
║    /help    - 显示此帮助信息                                  ║
║    /source  - 显示上次回答引用的文档来源                       ║
║    /exit    - 退出系统                                        ║
║                                                              ║
║  提示：                                                      ║
║    - 直接输入问题即可获得基于企业制度的回答                     ║
║    - 可提问的内容包括：                                        ║
║      · 员工行为规范（考勤、礼仪、廉洁等）                      ║
║      · 员工购物及财务支付流程                                  ║
║    - 使用 /source 可查看回答引用的原始文档                     ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def show_source(retriever, last_question: str | None):
    """显示检索到的文档来源"""
    if last_question is None:
        print("\n⚠️  还没有提问过，请先提问。")
        return

    docs: list[Document] = retriever.invoke(last_question)

    print(f"\n{'=' * 60}")
    print(f"📋 问题: \"{last_question}\"")
    print(f"📄 检索到 {len(docs)} 个相关文档块：")
    print(f"{'=' * 60}")

    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"\n  ┌─ 文档块 #{i+1} ")
        print(f"  ├─ 来源文件: {source}")
        print(f"  ├─ 字符数: {len(doc.page_content)}")
        print(f"  └─ 内容:")
        print(f"     {doc.page_content[:300]}")
        if len(doc.page_content) > 300:
            print(f"     ... (共 {len(doc.page_content)} 字符)")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """交互式 RAG 问答主循环"""

    # 显示启动横幅
    print("\n" + "=" * 70)
    print("""
   ██████╗   █████╗   ██████╗     问答系统
   ██╔══██╗ ██╔══██╗ ██╔════╝
   ██████╔╝ ███████║ ██║  ███╗  企业制度智能问答
   ██╔══██╗ ██╔══██║ ██║   ██║
   ██║  ██║ ██║  ██║ ╚██████╔╝  基于 RAG 检索增强生成
   ╚═╝  ╚═╝ ╚═╝  ╚═╝  ╚═════╝
    """)
    print("=" * 70)
    print("  知识库: 企业行为规范 + 企业员工购物及财务支付全流程手册")
    print("  技术栈: LangChain V1.3 + Chroma + Qwen-turbo")
    print("  输入 /help 查看帮助, /exit 退出")
    print("=" * 70)

    # 初始化
    rag_chain, retriever = initialize_rag()

    print("\n" + "=" * 70)
    print("  ✨ 系统就绪，开始提问吧！")
    print("=" * 70)

    # 示例问题
    print("\n💡 示例问题：")
    examples = [
        "员工迟到有什么处罚？",
        "保密制度有哪些要求？",
        "员工购物流程是怎样的？",
        "员工是否可以收受客户礼品？",
    ]
    for ex in examples:
        print(f"   · {ex}")
    print()

    # 交互循环
    last_question: str | None = None

    while True:
        try:
            question = input("🔍 请输入问题: ").strip()

            if not question:
                continue

            # 处理特殊命令
            if question == "/exit":
                print("\n👋 感谢使用 RAG 问答系统，再见！\n")
                break

            if question == "/help":
                show_help()
                continue

            if question == "/source":
                show_source(retriever, last_question)
                continue

            # 正常问答
            last_question = question
            print("🤔 正在检索知识库并生成回答...", end="", flush=True)

            answer = rag_chain.invoke(question)

            print("\r" + " " * 40 + "\r", end="")  # 清除等待提示
            print(f"\n{'─' * 60}")
            print(f"🤖 回答:")
            print(f"{'─' * 60}")
            print(answer)
            print(f"{'─' * 60}")
            print("💡 提示: 输入 /source 查看回答引用的原始文档")
            print()

        except KeyboardInterrupt:
            print("\n\n👋 感谢使用 RAG 问答系统，再见！\n")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            print("   请重试或输入 /exit 退出。")


if __name__ == "__main__":
    main()
