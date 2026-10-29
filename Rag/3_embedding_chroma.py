"""
============================================================================
RAG 学习案例 - 步骤 3：文本 Embedding 与 Chroma 向量存储
============================================================================
本步骤演示：
  1. 使用 OpenAI text-embedding-3-small 模型将文本转换为向量
  2. 将向量存储到 Chroma 向量数据库
  3. 了解 Embedding 和向量数据库的工作原理

什么是 Embedding（嵌入）？
─────────────────────────
  Embedding 是将文本转换为固定维度向量的过程。
  语义相似的文本 → 向量空间中距离更近
  语义不同的文本 → 向量空间中距离更远

  例如：
  "员工不能迟到"  → [0.12, -0.34, 0.56, ...]（1536维向量）
  "考勤管理严格"  → [0.15, -0.31, 0.52, ...]（相近的向量）
  "财务报表编制"  → [-0.78, 0.45, -0.23, ...]（较远的向量）

text-embedding-3-small 模型特点：
  - 维度：1536（small）或 3072（large）
  - 支持多语言（中英文效果优秀）
  - 性价比高，适合 RAG 场景

什么是 Chroma？
───────────────
  Chroma 是一个开源的轻量级向量数据库，专为 AI 应用设计：
  - 本地运行，无需额外服务
  - 支持持久化存储（不会丢失数据）
  - 内置多种相似度检索算法（余弦相似度、欧氏距离等）
  - 与 LangChain 深度集成
============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from env_utils import OPENAI_BASE_URL

# ============================================================================
# 1. 加载并分割文档（复用步骤 1 和 2 的逻辑）
# ============================================================================
print("=" * 70)
print("【步骤1】加载并分割文档")
print("=" * 70)

base_dir = os.path.dirname(__file__)

# 加载文档
loader_txt = TextLoader(
    file_path=os.path.join(base_dir, "企业行为规范.txt"),
    encoding="utf-8",
)
loader_docx = Docx2txtLoader(
    file_path=os.path.join(base_dir, "企业员工购物及财务支付全流程手册.docx"),
)
all_docs = loader_txt.load() + loader_docx.load()
print(f"✅ 加载了 {len(all_docs)} 个原始文档")

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=[
        "\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ".", "",
    ],
    length_function=len,
)
split_docs = text_splitter.split_documents(all_docs)
print(f"✅ 分割为 {len(split_docs)} 个文本块")


# ============================================================================
# 2. 初始化 Embedding 模型
# ============================================================================
print("\n" + "=" * 70)
print("【步骤2】初始化 text-embedding-3-small Embedding 模型")
print("=" * 70)

# 使用 OpenAI 的 text-embedding-3-small 模型
# 注意：虽然调用的是 OpenAI 的 Embedding API，但这里使用的是代理地址
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="sk-eEnWubQSh3rNm06a1hRj8DIW4zZx7Tak0TTCQYwq2QjkVgV2",
    base_url=OPENAI_BASE_URL,
    # dimensions=1536,  # 可选：指定输出维度（small 默认 1536，可缩减到 512 节省存储）
)

print(f"📐 Embedding 模型: text-embedding-3-small")
print(f"📐 向量维度: 1536")

# 演示：对单条文本进行 Embedding
print("\n演示单条 Embedding：")
sample_text = "员工必须遵守企业考勤制度"
sample_vector = embeddings_model.embed_query(sample_text)
print(f"  输入文本: \"{sample_text}\"")
print(f"  输出向量: [{sample_vector[0]:.4f}, {sample_vector[1]:.4f}, ..., {sample_vector[-1]:.4f}]")
print(f"  向量长度: {len(sample_vector)} 维")


# ============================================================================
# 3. 存入 Chroma 向量数据库
# ============================================================================
print("\n" + "=" * 70)
print("【步骤3】将文档向量存入 Chroma 向量数据库")
print("=" * 70)

# 设置 Chroma 持久化存储路径
chroma_persist_dir = os.path.join(base_dir, "chroma_db")
print(f"📁 Chroma 存储路径: {chroma_persist_dir}")

# from_documents() 会：
#   1. 调用 Embedding 模型将每个 Document 转为向量
#   2. 将向量 + 元数据存入 Chroma
#   3. 自动创建索引以支持快速检索
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings_model,
    persist_directory=chroma_persist_dir,  # 持久化到磁盘
    collection_name="enterprise_policies",  # 集合名称（类似数据库中的表）
)

print(f"✅ 向量数据库创建成功！")
print(f"   存储文档块数: {vector_store._collection.count()} 块")
print(f"   集合名称: enterprise_policies")
print(f"   持久化路径: {chroma_persist_dir}")


# ============================================================================
# 4. 验证：测试相似度检索
# ============================================================================
print("\n" + "=" * 70)
print("【步骤4】验证：相似度检索测试")
print("=" * 70)

# 测试查询：通过语义相似度找到最相关的文档块
test_query = "员工迟到会有什么处罚？"
print(f"🔍 查询: \"{test_query}\"")

# similarity_search 返回最相关的 k 个文档
results: list[Document] = vector_store.similarity_search(
    query=test_query,
    k=3,  # 返回最相似的前 3 个
)

print(f"\n检索结果 (Top {len(results)})：")
for i, doc in enumerate(results):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"\n  [{i}] 来源: {source}")
    print(f"      相关性排名: #{i+1}")
    print(f"      内容预览: {doc.page_content[:120]}...")

print(f"\n✅ 向量数据库工作正常！检索结果与查询语义高度相关。")


# ============================================================================
# 知识总结
# ============================================================================
print("\n" + "=" * 70)
print("📚 知识卡片")
print("=" * 70)
print("""
Embedding + 向量数据库 核心概念：
────────────────────────────────────────────
1. Embedding 将文本 → 固定维度向量（text-embedding-3-small: 1536维）
2. 语义相似的文本在向量空间中距离更近
3. Chroma 负责存储向量 + 提供高效检索
4. 检索过程：查询文本 → Embedding → 向量 → 相似度计算 → 返回 Top-K

架构示意：
  文档 → Loader → Splitter → Embedding → Chroma
                                              ↑
  用户查询 → Embedding → 向量检索 ─────────────┘
                     ↓
              返回 Top-K 相关文档块

Chroma 特性：
  - persist_directory: 数据持久化，重启不丢失
  - collection_name:   逻辑隔离，支持多项目共用
  - from_documents():   一步完成 Embedding + 存储
""")
