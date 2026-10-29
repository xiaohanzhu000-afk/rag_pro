"""
============================================================================
RAG 学习案例 - 步骤 1：文档加载
============================================================================
本步骤演示如何使用 LangChain 的 Document Loader 加载不同类型的文档。

文档加载是 RAG（检索增强生成）的第一步：
- 将原始文档（txt、docx、pdf 等）加载为 LangChain 的 Document 对象
- Document 对象包含两个核心属性：
  - page_content: 文档的文本内容
  - metadata:   文档的元数据（如来源、页码等）

支持的 Loader：
  - TextLoader: 加载 .txt 纯文本文件
  - Docx2txtLoader: 加载 .docx Word 文档（基于 python-docx）
============================================================================
"""

import os
import sys

# 将项目根目录加入 path，以便导入 env_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_core.documents import Document

# ============================================================================
# 1. 加载 .txt 文件
# ============================================================================
print("=" * 70)
print("【步骤1】加载 .txt 文件：企业行为规范")
print("=" * 70)

txt_path = os.path.join(os.path.dirname(__file__), "企业行为规范.txt")
loader_txt = TextLoader(
    file_path=txt_path,
    encoding="utf-8",   # 指定中文编码，避免乱码
)
docs_txt: list[Document] = loader_txt.load()

print(f"✅ 成功加载：共 {len(docs_txt)} 个文档块")
print(f"   文档字符数：{len(docs_txt[0].page_content)} 字符")
print(f"   文档元数据：{docs_txt[0].metadata}")
print()
print("--- 文档内容预览（前 500 字符）---")
print(docs_txt[0].page_content[:500])
print("...")


# ============================================================================
# 2. 加载 .docx 文件
# ============================================================================
print("\n" + "=" * 70)
print("【步骤2】加载 .docx 文件：企业员工购物及财务支付全流程手册")
print("=" * 70)

docx_path = os.path.join(
    os.path.dirname(__file__),
    "企业员工购物及财务支付全流程手册.docx"
)
loader_docx = Docx2txtLoader(file_path=docx_path)
docs_docx: list[Document] = loader_docx.load()

print(f"✅ 成功加载：共 {len(docs_docx)} 个文档块")
print(f"   文档字符数：{len(docs_docx[0].page_content)} 字符")
print(f"   文档元数据：{docs_docx[0].metadata}")
print()
print("--- 文档内容预览（前 500 字符）---")
print(docs_docx[0].page_content[:500])
print("...")


# ============================================================================
# 3. 汇总所有文档
# ============================================================================
print("\n" + "=" * 70)
print("【步骤3】文档汇总")
print("=" * 70)

all_docs: list[Document] = docs_txt + docs_docx

total_chars = sum(len(doc.page_content) for doc in all_docs)

print(f"📄 文档总数：{len(all_docs)} 个")
print(f"📊 总字符数：{total_chars} 字符")
print()
print("各文档统计：")
for i, doc in enumerate(all_docs):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"  [{i}] 来源：{source} | 字符数：{len(doc.page_content)}")


# ============================================================================
# 知识总结
# ============================================================================
print("\n" + "=" * 70)
print("📚 知识卡片")
print("=" * 70)
print("""
LangChain Document Loader 核心概念：
────────────────────────────────────────────
1. Document 对象 = page_content（文本） + metadata（元数据）
2. 不同 Loader 处理不同格式：TextLoader(文本) / Docx2txtLoader(Word) / PyPDFLoader(PDF)
3. load() 方法返回 List[Document]，一个文件可能被拆为多个 Document
4. metadata 的 "source" 字段记录了文件来源路径
""")
