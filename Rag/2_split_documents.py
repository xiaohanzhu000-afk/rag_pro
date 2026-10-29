"""
============================================================================
RAG 学习案例 - 步骤 2：文档分割（Document Splitting）
============================================================================
本步骤演示如何使用最优的分割策略将长文档切分为小块（Chunks）。

为什么需要文档分割？
  1. LLM 的上下文窗口有限，长文档无法一次性传入
  2. 小块文档能实现更精准的语义检索
  3. 合适的重叠（overlap）能避免语义断裂

分割策略选择：
  ┌─────────────────────────────────────────────────────────────┐
  │ RecursiveCharacterTextSplitter —— 本案例选择此方法            │
  │                                                             │
  │ 原理：按照优先级从高到低尝试分割符：                           │
  │   "\\n\\n"  →  "\\n"  →  "。"  →  "，"  →  " "  →  ""       │
  │                                                             │
  │ 优势：                                                       │
  │  ✅ 对中文文档最友好（特别加入了中文标点分割符）               │
  │  ✅ 尽量保持段落/句子的完整性                                 │
  │  ✅ 递归降级策略确保所有文本都能被合理切分                     │
  │  ✅ 适合结构化和半结构化文档                                   │
  └─────────────────────────────────────────────────────────────┘

  对比其他分割器：
  - CharacterTextSplitter： 仅按单一字符分割，容易切断句子
  - SemanticChunker：       按语义相似度分割，计算成本高
  - MarkdownTextSplitter：  仅适用于 Markdown 格式

参数说明：
  - chunk_size:      每个文本块的最大字符数
  - chunk_overlap:   相邻块之间的重叠字符数（避免语义在边界丢失）
  - separators:      分割符优先级列表
  - length_function: 计算文本长度的函数（默认 len）
============================================================================
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ============================================================================
# 1. 重新加载所有文档（与步骤 1 一致）
# ============================================================================
print("=" * 70)
print("【步骤1】加载文档")
print("=" * 70)

base_dir = os.path.dirname(__file__)

loader_txt = TextLoader(
    file_path=os.path.join(base_dir, "企业行为规范.txt"),
    encoding="utf-8",
)
loader_docx = Docx2txtLoader(
    file_path=os.path.join(base_dir, "企业员工购物及财务支付全流程手册.docx"),
)

all_docs: list[Document] = loader_txt.load() + loader_docx.load()
print(f"✅ 加载了 {len(all_docs)} 个原始文档")


# ============================================================================
# 2. 配置 RecursiveCharacterTextSplitter
# ============================================================================
print("\n" + "=" * 70)
print("【步骤2】配置 RecursiveCharacterTextSplitter（递归字符分割器）")
print("=" * 70)

# 为中文文档定制分割符优先级
# 从语义完整性高的分割符开始，逐步降级
chinese_separators = [
    "\n\n",     # 第1优先级：段落间空行（保持段落完整性）
    "\n",       # 第2优先级：换行符（保持行完整性）
    "。",       # 第3优先级：中文句号（保持句子完整性）
    "；",       # 第4优先级：中文分号
    "！",       # 第5优先级：感叹号
    "？",       # 第6优先级：问号
    "，",       # 第7优先级：中文逗号
    "、",       # 第8优先级：顿号
    " ",        # 第9优先级：空格
    ".",        # 第10优先级：英文句号（兼容特殊情况）
    "",         # 最后保底：逐字符分割
]

text_splitter = RecursiveCharacterTextSplitter(
    # 每个 chunk 最大 500 字符 —— 适合中文语义单元
    chunk_size=500,
    # 重叠 80 字符 —— 约 1-2 句话，防止语义在边界断裂
    chunk_overlap=80,
    # 使用定制的中文分割符
    separators=chinese_separators,
    # 按字符数计算长度（中文更适合按字符而非 token）
    length_function=len,
    # 如果 chunk 在分割后略超 chunk_size，允许 10 字符的弹性
    # 避免产生过小的碎片 chunk
    # 注意：此参数在较新版本中可能名为 add_start_index
)

print("分割器配置：")
print(f"  chunk_size      = {text_splitter._chunk_size}")
print(f"  chunk_overlap   = {text_splitter._chunk_overlap}")
print(f"  separators      = {text_splitter._separators}")


# ============================================================================
# 3. 执行文档分割
# ============================================================================
print("\n" + "=" * 70)
print("【步骤3】执行文档分割")
print("=" * 70)

# split_documents() 接受 List[Document]，返回分割后的 List[Document]
split_docs: list[Document] = text_splitter.split_documents(all_docs)

print(f"📊 分割结果：")
print(f"   原始文档数：{len(all_docs)} 个")
print(f"   分割后块数：{len(split_docs)} 个")
print(f"   平均分割比：{len(split_docs) / len(all_docs):.1f}x")

# 统计各 chunk 的长度分布
chunk_lengths = [len(doc.page_content) for doc in split_docs]
print(f"   Chunk 长度范围：{min(chunk_lengths)} ~ {max(chunk_lengths)} 字符")
print(f"   Chunk 平均长度：{sum(chunk_lengths) / len(chunk_lengths):.0f} 字符")


# ============================================================================
# 4. 展示分割效果
# ============================================================================
print("\n" + "=" * 70)
print("【步骤4】分割效果展示（前 3 个 chunk）")
print("=" * 70)

for i, doc in enumerate(split_docs[:3]):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"\n{'─' * 60}")
    print(f"Chunk #{i} | 来源: {source} | 长度: {len(doc.page_content)} 字符")
    print(f"元数据: {doc.metadata}")
    print(f"{'─' * 60}")
    # 显示前 200 字符
    content_preview = doc.page_content[:200].replace("\n", "\\n")
    print(f"内容: {content_preview}...")

# 展示重叠效果（相邻两个 chunk）
print(f"\n{'=' * 60}")
print("【重叠效果验证】以 Chunk#0 和 Chunk#1 为例：")
print(f"{'=' * 60}")
if len(split_docs) >= 2:
    chunk0_end = split_docs[0].page_content[-100:]
    chunk1_start = split_docs[1].page_content[:100]
    print(f"Chunk#0 结尾: ...{chunk0_end}")
    print(f"Chunk#1 开头: {chunk1_start}...")
    print("\n重叠分析：")
    # 找共同子串
    for i in range(min(50, len(chunk0_end)), 0, -1):
        suffix = chunk0_end[-i:]
        if suffix in chunk1_start:
            print(f"  发现 {i} 字符的重叠文本：\"{suffix}\"")
            break


# ============================================================================
# 知识总结
# ============================================================================
print("\n" + "=" * 70)
print("📚 知识卡片")
print("=" * 70)
print("""
RecursiveCharacterTextSplitter 工作原理：
────────────────────────────────────────────
1. 按 separators[0] (\\n\\n) 尝试分割
2. 如果分割后仍有 chunk 超过 chunk_size，降级到 separators[1] (\\n)
3. 递归降级，直到 separators[-1] ("") 逐字符分割
4. 相邻 chunk 之间保留 chunk_overlap 字符的重叠

参数调优建议：
  - chunk_size:  中文建议 300~800，英文建议 500~1500
  - chunk_overlap: 通常为 chunk_size 的 10%~20%
  - separators:  根据文档语言和格式定制分割符顺序

为什么选择此方案？
  - 企业行为规范：结构化条款文档 → 递归分割能保持条款完整性
  - 财务手册：    流程化操作文档 → 递归分割能保持步骤连续性
""")
