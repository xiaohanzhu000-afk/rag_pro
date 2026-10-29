"""
============================================================================
RAG 学习案例 - 步骤 5：构建完整的 RAG Chain
============================================================================
本步骤演示如何组装一个端到端的 RAG（检索增强生成）链：

  RAG Chain 完整流程：
  ┌──────────────────────────────────────────────────────────────┐
  │  用户提问                                                     │
  │     ↓                                                        │
  │  ① 向量检索（Retriever）—— 从 Chroma 中检索相关文档           │
  │     ↓                                                        │
  │  ② 上下文组装（Prompt） —— 将检索结果填入 Prompt 模板          │
  │     ↓                                                        │
  │  ③ LLM 生成（Qwen）    —— 基于上下文生成回答                   │
  │     ↓                                                        │
  │  ④ 返回答案                                                   │
  └──────────────────────────────────────────────────────────────┘

本案例使用：
  - LLM:    阿里千问（qwen-turbo）通过 DashScope API
  - Embedding: text-embedding-3-small
  - 向量库:  Chroma（本地持久化）
  - 框架:   LangChain V1.3
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
# 1. 加载向量数据库
# ============================================================================
print("=" * 70)
print("【步骤1】加载 Chroma 向量数据库")
print("=" * 70)

base_dir = os.path.dirname(__file__)
chroma_persist_dir = os.path.join(base_dir, "chroma_db")

if not os.path.exists(chroma_persist_dir):
    print("❌ 向量数据库不存在！请先运行 3_embedding_chroma.py")
    exit(1)

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

print(f"✅ 向量数据库加载成功 ({vector_store._collection.count()} 个文档块)")


# ============================================================================
# 2. 初始化 LLM（阿里千问 DashScope）
# ============================================================================
print("\n" + "=" * 70)
print("【步骤2】初始化 LLM —— 阿里千问（via DashScope）")
print("=" * 70)

# DashScope 提供了 OpenAI 兼容接口，因此可以直接用 ChatOpenAI
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    temperature=0.3,   # 较低温度，让回答更严谨
    max_tokens=1024,
)

print(f"🤖 LLM 模型: qwen-turbo")
print(f"🌐 API 地址: {DASHSCOPE_BASE_URL}")
print(f"🌡️  温度: 0.3 (较低，回答更严谨)")


# ============================================================================
# 3. 设计 RAG Prompt 模板
# ============================================================================
print("\n" + "=" * 70)
print("【步骤3】设计 RAG Prompt 模板")
print("=" * 70)

# Prompt 是 RAG 的关键组件 —— 它告诉 LLM：
#   1. 如何利用检索到的上下文
#   2. 以什么角色/风格回答
#   3. 遇到不确定的情况如何处理
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个专业的企业制度咨询助手。请严格基于以下【参考文档】中的内容回答问题。

【回答规则】
1. 只使用参考文档中明确提到的信息，不要编造或推测
2. 如果参考文档中没有相关信息，请明确回答："根据现有制度文档，未找到相关信息"
3. 回答要条理清晰，使用序号或分点列出关键信息
4. 如果涉及处罚、流程等内容，请引用制度中的具体规定

【参考文档】
{context}""",
    ),
    ("human", "{question}"),
])

print("Prompt 模板预览：")
print(rag_prompt.pretty_print())


# ============================================================================
# 4. 构建 RAG Chain
# ============================================================================
print("\n" + "=" * 70)
print("【步骤4】构建 RAG Chain（LCEL 方式）")
print("=" * 70)
print("""
Chain 结构：
  question → retriever → format_docs → prompt → llm → output_parser

各组件说明：
  - retriever:     从向量数据库检索相关文档
  - format_docs:   将 Document 列表拼接为文本字符串
  - prompt:        将检索到的上下文 + 问题填入 Prompt 模板
  - llm:           大模型生成回答
  - output_parser: 将 LLM 输出转为纯文本字符串
""")

# 创建检索器
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},  # 每次检索 4 个最相关的文档块
)

# 格式化检索到的文档：将多个 Document 拼接为一个上下文字符串
def format_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为上下文字符串"""
    formatted = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        formatted.append(f"[文档{i+1} 来源:{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# 使用 LCEL（LangChain Expression Language）组装 Chain
rag_chain = (
    {
        "context": retriever | format_docs,   # 检索 + 格式化 → 上下文
        "question": RunnablePassthrough(),     # 用户问题直接传递
    }
    | rag_prompt                               # 填入 Prompt 模板
    | llm                                      # LLM 生成回答
    | StrOutputParser()                        # 解析为纯文本
)

print("✅ RAG Chain 构建完成！")
print(f"   检索器配置: similarity, k=4")


# ============================================================================
# 5. 测试 RAG Chain
# ============================================================================
print("\n" + "=" * 70)
print("【步骤5】测试 RAG Chain —— 问答演示")
print("=" * 70)

# 准备几个测试问题
test_questions = [
    "员工迟到会有什么处罚？",
    "员工是否可以收受客户送的礼品？",
    "企业的保密制度有哪些要求？",
    "员工请假的流程是什么？",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─' * 60}")
    print(f"❓ 问题 {i}: {question}")
    print(f"{'─' * 60}")

    # 调用 RAG Chain
    answer = rag_chain.invoke(question)
    print(f"🤖 回答:\n{answer}")


# ============================================================================
# 6. 展示检索到的上下文（调试用）
# ============================================================================
print("\n" + "=" * 70)
print("【步骤6】调试信息：查看最后一次检索到的上下文")
print("=" * 70)

last_question = test_questions[-1]
retrieved_docs: list[Document] = retriever.invoke(last_question)

print(f"问题: \"{last_question}\"")
print(f"\n检索到 {len(retrieved_docs)} 个相关文档块：")
for i, doc in enumerate(retrieved_docs):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"\n  📄 文档块 #{i} (来源: {source})")
    print(f"  {'─' * 50}")
    print(f"  {doc.page_content[:200]}...")

print(f"\n📝 组装后的 Prompt 上下文大小: {len(format_docs(retrieved_docs))} 字符")


# ============================================================================
# 知识总结
# ============================================================================
print("\n" + "=" * 70)
print("📚 知识卡片")
print("=" * 70)
print("""
RAG (Retrieval-Augmented Generation) 核心架构：
────────────────────────────────────────────
    用户问题
       │
       ▼
  ┌──────────┐    ┌───────────┐    ┌────────┐
  │ Retriever│ → │  Prompt   │ → │  LLM   │ → 回答
  │ (检索)   │    │ (上下文)   │    │ (生成)  │
  └──────────┘    └───────────┘    └────────┘
       ↑
  ┌──────────┐
  │  Chroma  │ (向量数据库)
  └──────────┘

LCEL (LangChain Expression Language) 管道操作符：
  - |  (管道符): 将前一个组件的输出传给后一个组件的输入
  - RunnablePassthrough(): 将数据原样传递
  - 字典组装 {}:  同时运行多个并行分支，合并为字典输出

RAG 相比直接使用 LLM 的优势：
  ✅ 基于真实文档，减少"幻觉"
  ✅ 可溯源到原始文档
  ✅ 知识可动态更新（更新向量库即可，无需微调）
  ✅ 适合企业内部知识库场景
""")
