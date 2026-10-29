"""
============================================================================
RAG 学习案例 - 步骤 4：向量检索（Retrieval）
============================================================================
本步骤演示如何使用 Chroma 向量数据库进行多种检索方式：

  1. 相似度检索（Similarity Search）—— 最基础的检索方式
  2. 带分数的相似度检索 —— 返回相似度分数
  3. MMR（最大边际相关性）检索 —— 兼顾相关性和多样性
  4. 元数据过滤检索 —— 按文档来源等条件筛选

检索是 RAG 的核心环节，质量直接影响最终回答的准确性。
============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from env_utils import OPENAI_BASE_URL

# ============================================================================
# 1. 加载已有的向量数据库
# ============================================================================
print("=" * 70)
print("【步骤1】加载已有的 Chroma 向量数据库")
print("=" * 70)

base_dir = os.path.dirname(__file__)
chroma_persist_dir = os.path.join(base_dir, "chroma_db")

# 检查向量数据库是否已存在
if not os.path.exists(chroma_persist_dir):
    print("❌ 向量数据库不存在！请先运行 3_embedding_chroma.py")
    print(f"   期望路径: {chroma_persist_dir}")
    exit(1)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="sk-eEnWubQSh3rNm06a1hRj8DIW4zZx7Tak0TTCQYwq2QjkVgV2",
    base_url=OPENAI_BASE_URL,
)

# 从持久化目录加载已有的向量数据库
vector_store = Chroma(
    persist_directory=chroma_persist_dir,
    embedding_function=embeddings_model,
    collection_name="enterprise_policies",
)

doc_count = vector_store._collection.count()
print(f"✅ 向量数据库加载成功！")
print(f"   存储文档块数: {doc_count} 块")
print(f"   存储路径: {chroma_persist_dir}")


# ============================================================================
# 2. 基础相似度检索
# ============================================================================
print("\n" + "=" * 70)
print("【检索方式一】基础相似度检索（similarity_search）")
print("=" * 70)

query_1 = "员工请假的流程是什么？"

results_1: list[Document] = vector_store.similarity_search(
    query=query_1,
    k=3,  # 返回最相似的 3 个结果
)

print(f"🔍 查询: \"{query_1}\"")
print(f"\n检索结果 (Top {len(results_1)})：")
for i, doc in enumerate(results_1):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"\n  [{i}] 来源: {source}")
    print(f"      内容: {doc.page_content[:150]}...")


# ============================================================================
# 3. 带分数的相似度检索
# ============================================================================
print("\n" + "=" * 70)
print("【检索方式二】带分数的相似度检索（similarity_search_with_score）")
print("=" * 70)

query_2 = "员工收受礼品会有什么后果？"

results_2 = vector_store.similarity_search_with_score(
    query=query_2,
    k=3,
)

print(f"🔍 查询: \"{query_2}\"")
print(f"\n检索结果 (Top {len(results_2)})：")
for i, (doc, score) in enumerate(results_2):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    # Chroma 返回的是距离值（越小越相似），转换为相似度
    similarity = 1 / (1 + score)
    print(f"\n  [{i}] 来源: {source} | 距离: {score:.4f} | 相似度: {similarity:.4f}")
    print(f"      内容: {doc.page_content[:120]}...")


# ============================================================================
# 4. MMR 检索 —— 兼顾相关性和多样性
# ============================================================================
print("\n" + "=" * 70)
print("【检索方式三】MMR 检索（最大边际相关性）")
print("=" * 70)
print("""
MMR (Maximal Marginal Relevance) 算法：
  - 在保证相关性的前提下，尽可能选择内容多样化的结果
  - 避免返回多个高度相似的内容块
  - 参数 lambda_mult:
    - 接近 1：更注重相关性（类似普通 similarity_search）
    - 接近 0：更注重多样性（返回内容差异更大的结果）
""")

query_3 = "企业的考勤制度是怎样的？"

# 先用普通检索查看
results_normal = vector_store.similarity_search(query_3, k=3)

# 再用 MMR 检索
results_mmr: list[Document] = vector_store.max_marginal_relevance_search(
    query=query_3,
    k=3,
    fetch_k=10,       # 先从候选池中取 10 个，再从中选 3 个
    lambda_mult=0.5,  # 相关性权重 50%，多样性权重 50%
)

print(f"🔍 查询: \"{query_3}\"")
print(f"\n普通检索 (similarity_search) vs MMR检索 对比：")
print(f"{'─' * 50}")
print(f"| {'排名':^4} | {'普通检索(前80字)':^40} |")
print(f"{'─' * 50}")
for i, doc in enumerate(results_normal):
    print(f"| {i:^4} | {doc.page_content[:40]:<40} |")
print(f"{'─' * 50}")
print(f"| {'排名':^4} | {'MMR检索(前80字)':^40} |")
print(f"{'─' * 50}")
for i, doc in enumerate(results_mmr):
    print(f"| {i:^4} | {doc.page_content[:40]:<40} |")
print(f"{'─' * 50}")


# ============================================================================
# 5. 元数据过滤检索
# ============================================================================
print("\n" + "=" * 70)
print("【检索方式四】元数据过滤检索")
print("=" * 70)

query_4 = "财务支付流程"

# 分别从两个来源检索
for source_name, source_file in [
    ("企业行为规范", "企业行为规范.txt"),
    ("财务支付手册", "企业员工购物及财务支付全流程手册.docx"),
]:
    results_filtered = vector_store.similarity_search(
        query=query_4,
        k=2,
        filter={"source": os.path.join(base_dir, source_file)},
    )
    print(f"\n📂 来源过滤: {source_name}")
    for i, doc in enumerate(results_filtered):
        print(f"  [{i}] {doc.page_content[:100]}...")


# ============================================================================
# 6. 将检索器转换为 LangChain Retriever 接口
# ============================================================================
print("\n" + "=" * 70)
print("【额外】转换为 Retriever 接口")
print("=" * 70)
print("""
Retriever 是 LangChain 的检索器抽象接口：
  - 统一了各种检索方式的调用接口
  - 可作为 RAG Chain 的组件直接使用
  - 支持配置检索参数（如返回数量 k）
""")

retriever = vector_store.as_retriever(
    search_type="similarity",  # similarity / mmr / similarity_score_threshold
    search_kwargs={"k": 4},    # 返回 4 个结果
)

# 测试 Retriever
test_docs = retriever.invoke("保密制度有哪些要求？")
print(f"Retriever 返回 {len(test_docs)} 个文档：")
for i, doc in enumerate(test_docs):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"  [{i}] 来源: {source} | 内容: {doc.page_content[:80]}...")


# ============================================================================
# 知识总结
# ============================================================================
print("\n" + "=" * 70)
print("📚 知识卡片")
print("=" * 70)
print("""
四种检索方式对比：
────────────────────────────────────────────
检索方式                        适用场景
────────────────────────────────────────────
similarity_search              通用场景，快速获取最相关内容
similarity_search_with_score   需要评估检索质量、设置阈值过滤
MMR (最大边际相关性)             需要答案覆盖多个不同角度时
元数据过滤                      需按来源/日期/类别等条件限定检索

LangChain Retriever 接口：
  - 统一的 invoke(query) 调用方式
  - 返回 List[Document]
  - 可直接嵌入 RAG Chain

检索质量对 RAG 的影响：
  ✅ 高质量检索 → LLM 获得准确上下文 → 生成准确回答
  ❌ 低质量检索 → LLM 获得无关上下文 → 可能产生幻觉
""")
