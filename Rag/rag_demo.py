"""
================================================================================
LangChain 1.3 - RAG 学习案例（企业文档智能问答系统）
================================================================================

本模块是一个完整的 RAG (Retrieval-Augmented Generation) 教学案例，
每一步都有详细的中文注释和说明，帮助你从零理解 RAG 的完整工作流程。

核心步骤：
  1. 文档加载 —— 加载 .txt 和 .docx 两种格式的企业文档
  2. 文档分割 —— 探索最优的中文文档分割策略
  3. 向量嵌入 —— 使用 OpenAI text-embedding-3-small 模型
  4. 向量存储 —— Chroma 本地向量数据库
  5. 文档索引 —— 将分割后的文档存入向量数据库
  6. 检索测试 —— 语义相似度搜索验证
  7. RAG 问答 —— Agent 工具调用实现检索增强生成

技术栈：
  - 框架: LangChain 1.3
  - LLM:  通义千问 (DashScope)
  - Embedding: OpenAI text-embedding-3-small
  - 向量数据库: Chroma (本地持久化)
  - 文档格式: .txt + .docx

依赖安装:
  pip install langchain langchain-openai langchain-community langchain-text-splitters chromadb python-docx python-dotenv
"""

import os
import sys
from pathlib import Path

# ============================================================================
# 【环境准备】导入必要的模块
# ============================================================================

# 将项目根目录加入 sys.path，以便导入 env_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, OPENAI_BASE_URL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.tools import tool

# ============================================================================
# 【路径配置】文档位于 Rag/ 目录下
# ============================================================================
SCRIPT_DIR = Path(__file__).parent          # Rag/ 目录
CHROMA_DIR = SCRIPT_DIR / "chroma_db"       # Chroma 持久化目录

# 文档路径（直接放在 Rag/ 目录下）
TXT_FILE = SCRIPT_DIR / "企业行为规范.txt"
DOCX_FILE = SCRIPT_DIR / "企业员工购物及财务支付全流程手册.docx"

# 确保目录存在
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 【Embedding 配置】OpenAI text-embedding-3-small
# ============================================================================
# text-embedding-3-small 特点：
#   - 维度: 1536（也可指定为 512，维度越小越快但精度略降）
#   - 支持中英文，对中文语义理解好
#   - 性价比高，适合生产环境
# ============================================================================
EMBEDDING_API_KEY = "sk-eEnWubQSh3rNm06a1hRj8DIW4zZx7Tak0TTCQYwq2QjkVgV2"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=EMBEDDING_API_KEY,
    base_url=OPENAI_BASE_URL,      # 通过 env_utils 配置的代理地址
    dimensions=1536,               # 使用完整 1536 维，获得最佳精度
)

# ============================================================================
# 【LLM 配置】DashScope 通义千问
# ============================================================================
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    temperature=0.3,               # RAG 场景建议用较低温度，保证回答准确
    max_tokens=1024,
)


# ============================================================================
# 辅助函数：打印分步标题
# ============================================================================
def print_step(step_num: int, title: str, description: str = ""):
    """统一的分步标题打印格式"""
    print("\n" + "=" * 75)
    print(f" 步骤 {step_num}：{title}")
    print("=" * 75)
    if description:
        print(f"\n💡 {description}\n")


# ============================================================================
# 步骤 1：文档加载（Document Loading）
# ============================================================================
def step1_load_documents():
    """
    【第一步：文档加载】

    目的：
      将不同格式的文档统一加载为 LangChain 的 Document 对象。
      Document 对象包含两个核心属性：
        - page_content: 文档的文本内容
        - metadata:    文档的元数据（如来源、页码等）

    用到的 Loader:
      - TextLoader:     加载 .txt 纯文本文件
      - Docx2txtLoader: 加载 .docx Word 文档（自动提取文本）

    为什么选择这两种 Loader？
      - TextLoader 最基础，适合所有纯文本场景
      - Docx2txtLoader 基于 python-docx 解析 .docx，保留段落结构
    """
    print_step(1, "文档加载（Document Loading）",
               "将 .txt 和 .docx 文档统一加载为 LangChain Document 对象")

    all_documents = []

    # ---- 加载 .txt 文件 ----
    if TXT_FILE.exists():
        print(f"[加载] {TXT_FILE.name}")
        txt_loader = TextLoader(str(TXT_FILE), encoding="utf-8")
        txt_docs = txt_loader.load()
        all_documents.extend(txt_docs)
        print(f"  ├─ 文档数量: {len(txt_docs)}")
        print(f"  ├─ 内容长度: {len(txt_docs[0].page_content)} 字符")
        print(f"  ├─ 元数据: {txt_docs[0].metadata}")
        print(f"  └─ 前80字: {txt_docs[0].page_content[:80]}...")
    else:
        print(f"[警告] 找不到文件: {TXT_FILE}")
        print(f"       请确保文档存在于 {DATA_DIR} 目录下")

    # ---- 加载 .docx 文件 ----
    if DOCX_FILE.exists():
        print(f"\n[加载] {DOCX_FILE.name}")
        docx_loader = Docx2txtLoader(str(DOCX_FILE))
        docx_docs = docx_loader.load()
        all_documents.extend(docx_docs)
        print(f"  ├─ 文档数量: {len(docx_docs)}")
        print(f"  ├─ 内容长度: {len(docx_docs[0].page_content)} 字符")
        print(f"  ├─ 元数据: {docx_docs[0].metadata}")
        print(f"  └─ 前80字: {docx_docs[0].page_content[:80]}...")
    else:
        print(f"\n[警告] 找不到文件: {DOCX_FILE}")
        print(f"       请先运行 generate_docs.py 生成 .docx 文档：")
        print(f"       python Rag/generate_docs.py")

    print(f"\n{'─' * 50}")
    print(f"[汇总] 共加载 {len(all_documents)} 个文档")
    print(f"[汇总] 总字符数: {sum(len(d.page_content) for d in all_documents)}")

    return all_documents


# ============================================================================
# 步骤 2：文档分割（Text Splitting）—— 探索最优策略
# ============================================================================
def step2_split_documents(documents):
    """
    【第二步：文档分割 —— 探索最优分割策略】

    ========== 为什么需要分割？ ==========
    1. LLM 有 Token（上下文窗口）限制，无法一次性处理整本文档
    2. 小块检索更精准：检索器返回的是"块"而非整篇文档
    3. 降低成本：每次只把最相关的块传给 LLM

    ========== 分割策略对比 ==========

    策略 A：「粗粒度分割」chunk_size=1000, overlap=200
      适合：长篇文章、制度手册
      优点：每块信息量大，回答更完整
      缺点：可能包含无关内容，检索精度略低

    策略 B：「细粒度分割」chunk_size=300, overlap=50
      适合：FAQ、对话记录
      优点：检索精准，命中率高
      缺点：可能丢失上下文，回答碎片化

    策略 C：「平衡分割」chunk_size=500, overlap=100  ← 推荐
      适合：大多数企业文档场景
      优点：兼顾信息完整性和检索精度
      缺点：无明显缺点

    ========== 针对中文文档的最优分割器选择 ==========
    使用 RecursiveCharacterTextSplitter：
      - 递归分割：按 separators 优先级依次尝试分割
      - 中文优化：separators 中包含中文标点符号

    separators 优先级说明：
      1. "\\n\\n"  → 优先按段落分割（保留语义完整性）
      2. "\\n"     → 其次按行分割
      3. "。"      → 按句号分割（中文句子边界）
      4. "！"      → 按感叹号分割
      5. "？"      → 按问号分割
      6. "；"      → 按分号分割
      7. "，"      → 按逗号分割
      8. " "       → 按空格分割
      9. ""        → 最后才按字符分割（兜底）

    这种设计确保在中文文档上，分割行为优先尊重句子边界，
    不会在句子中间截断，从而保护语义完整性。
    """
    print_step(2, "文档分割（Text Splitting）—— 探索最优策略",
               "对比不同分割参数，选择最适合企业文档的分割方案")

    # ---------------------------------------------------------------
    # 2.1 配置最优分割器
    # ---------------------------------------------------------------
    print("【最优分割器配置】")
    print(f"  算法:    RecursiveCharacterTextSplitter（递归字符分割器）")
    print(f"  块大小:  500 字符")
    print(f"  重叠量:  100 字符（20% 重叠，防止关键信息被截断）")
    print(f"  分割优先级: 段落 > 行 > 句子边界 > 标点 > 字符")

    optimal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # 每块最大 500 字符
        chunk_overlap=100,    # 块之间重叠 100 字符
        length_function=len,  # 使用 Python 内置 len() 计算长度
        separators=[
            "\n\n",           # 最高优先级：段落分隔
            "\n",             # 行分隔
            "。",             # 中文句号
            "！",             # 中文感叹号
            "？",             # 中文问号
            "；",             # 中文分号
            "，",             # 中文逗号
            " ",              # 空格
            ""                # 最低优先级：逐字符
        ],
        keep_separator=True,  # 保留分隔符在块中
        add_start_index=True, # 添加起始位置索引（方便追溯原始位置）
    )

    # ---------------------------------------------------------------
    # 2.2 执行分割
    # ---------------------------------------------------------------
    print("\n【分割执行】")
    optimal_chunks = optimal_splitter.split_documents(documents)
    print(f"  原始文档: {len(documents)} 个")
    print(f"  分割后块: {len(optimal_chunks)} 个")
    print(f"  平均块大小: {sum(len(c.page_content) for c in optimal_chunks) / len(optimal_chunks):.0f} 字符")

    # ---------------------------------------------------------------
    # 2.3 展示分割效果
    # ---------------------------------------------------------------
    print(f"\n【前 5 块预览】")
    for i, chunk in enumerate(optimal_chunks[:5], 1):
        preview = chunk.page_content[:100].replace("\n", "\\n")
        print(f"\n  块 {i} ({len(chunk.page_content)} 字符):")
        print(f"    {preview}...")
        if hasattr(chunk, 'metadata') and 'start_index' in chunk.metadata:
            print(f"    原始起始位置: {chunk.metadata['start_index']}")

    # ---------------------------------------------------------------
    # 2.4 对比：如果不用重叠会怎样？
    # ---------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("【对比实验】无重叠分割 vs 有重叠分割")
    no_overlap_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    no_overlap_chunks = no_overlap_splitter.split_documents(documents)

    # 检查是否有信息被截断（简单检测：查看相邻块的尾部/头部）
    if len(no_overlap_chunks) >= 2:
        print(f"\n  无重叠分割结果: {len(no_overlap_chunks)} 块")
        print(f"  第 1 块结尾: ...{no_overlap_chunks[0].page_content[-50:]}")
        print(f"  第 2 块开头: {no_overlap_chunks[1].page_content[:50]}...")
        print(f"  → 注意：块之间没有衔接，关键信息（如\"应符合以下条件：\"）可能被截断")

    print(f"\n  有重叠分割结果: {len(optimal_chunks)} 块")
    if len(optimal_chunks) >= 2:
        print(f"  第 1 块结尾: ...{optimal_chunks[0].page_content[-50:]}")
        print(f"  第 2 块开头: {optimal_chunks[1].page_content[:50]}...")
        print(f"  → 注意：有 100 字符重叠，语义承接更连贯")

    # ---------------------------------------------------------------
    # 2.5 总结：为什么这是最优分割策略？
    # ---------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("【总结】为什么选择这个分割策略？")
    print("  1. chunk_size=500:")
    print("     - 对中文文档而言，约等于 300-350 个中文字符的实质性内容")
    print("     - 足够容纳一个完整的段落或几个要点")
    print("     - 不会因太大而包含无关信息，也不会因太小而碎片化")
    print("  2. chunk_overlap=100 (20%):")
    print("     - 确保分割边界的关键信息不会丢失")
    print("     - 对于问答场景，问题可能恰好落在分割边界上")
    print("  3. separators 优先级: 段落 > 句子 > 字:")
    print("     - 遵循中文自然语言的语义边界")
    print("     - 避免在句子中间生硬截断")
    print("  4. keep_separator=True:")
    print("     - 保留原文标点，LLM 更容易理解上下文")

    return optimal_chunks


# ============================================================================
# 步骤 3：向量嵌入（Embeddings）
# ============================================================================
def step3_create_embeddings():
    """
    【第三步：向量嵌入】

    目的：
      将文本转换为高维向量（数字数组），
      语义相近的文本在向量空间中距离更近。

    模型选择：OpenAI text-embedding-3-small
      - 维度: 1536（高维度 = 更强的语义表达能力）
      - 支持: 中文 + 英文（多语言模型）
      - 性能: 速度快，成本低（$0.02/1M tokens）
      - 对比: 比 text-embedding-ada-002 精度高 20%，价格低 50%

    常用 Embedding 模型对比：
      ┌──────────────────────────────┬────────┬──────────┬────────┐
      │ 模型                          │ 维度    │ 语言支持  │ 成本   │
      ├──────────────────────────────┼────────┼──────────┼────────┤
      │ text-embedding-3-small       │ 1536   │ 多语言    │ 低     │
      │ text-embedding-3-large       │ 3072   │ 多语言    │ 中     │
      │ text-embedding-ada-002       │ 1536   │ 多语言    │ 中     │
      │ bge-large-zh-v1.5 (本地)     │ 1024   │ 中文      │ 免费   │
      │ all-MiniLM-L6-v2 (本地)      │ 384    │ 英文为主  │ 免费   │
      └──────────────────────────────┴────────┴──────────┴────────┘

    我们选择 text-embedding-3-small 的理由：
      1. 对中文语义理解出色
      2. 1536 维向量提供足够的精度
      3. API 调用稳定，不占用本地资源
    """
    print_step(3, "向量嵌入（Embeddings）",
               "使用 OpenAI text-embedding-3-small 将文本转换为向量")

    # ---------------------------------------------------------------
    # 3.1 嵌入单个查询文本
    # ---------------------------------------------------------------
    print("[嵌入测试 1] 单个查询")
    query = "企业员工的行为规范有哪些要求？"
    print(f"  输入文本: {query}")

    query_vector = embeddings.embed_query(query)
    print(f"  输出向量维度: {len(query_vector)}")
    print(f"  向量前5个值: {[round(v, 6) for v in query_vector[:5]]}")

    # ---------------------------------------------------------------
    # 3.2 批量嵌入文档
    # ---------------------------------------------------------------
    print(f"\n[嵌入测试 2] 批量文档")
    sample_texts = [
        "员工应当诚实守信，不得虚报工作信息。",
        "采购金额超过5000元需附供应商报价对比。",
        "公司倡导诚信创新的核心价值观。",
    ]
    sample_vectors = embeddings.embed_documents(sample_texts)

    # ---------------------------------------------------------------
    # 3.3 计算语义相似度（验证嵌入效果）
    # ---------------------------------------------------------------
    print(f"\n[嵌入测试 3] 语义相似度验证")
    import numpy as np

    def cosine_similarity(v1, v2):
        """余弦相似度：值越接近 1 表示语义越相近"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # 比较三对文本的相似度
    text_pairs = [
        # 文本A, 文本B, 关系说明
        ("员工应当诚实守信", "员工必须诚信正直", "同义表达，相似度应高"),
        ("员工应当诚实守信", "采购金额超5000元需审批", "不同主题，相似度应低"),
        ("公司倡导诚信创新", "员工必须诚信正直", "同主题不同表达，相似度中等"),
    ]

    for text_a, text_b, desc in text_pairs:
        v_a = embeddings.embed_query(text_a)
        v_b = embeddings.embed_query(text_b)
        sim = cosine_similarity(v_a, v_b)
        print(f"  [{desc}]")
        print(f"    A: {text_a}")
        print(f"    B: {text_b}")
        print(f"    相似度: {sim:.4f}")

    print(f"\n  → 验证结论：语义相近的文本确实拥有更高的向量相似度！")
    print(f"  → 这就是 RAG 检索的数学基础。")

    return embeddings


# ============================================================================
# 步骤 4+5：Chroma 向量数据库 + 文档索引
# ============================================================================
def step4_create_vectorstore(chunks):
    """
    【第四+五步：Chroma 向量数据库 + 文档索引】

    Chroma 简介：
      - 开源的向量数据库，专为 AI 应用设计
      - 支持本地持久化存储（数据不会丢失）
      - 无需注册账号或配置服务器
      - 内置多种相似度搜索算法

    为什么选择 Chroma 而不是 Pinecone？
      - Chroma：本地运行，零配置，适合学习和开发
      - Pinecone：云服务，适合生产环境的分布式场景
      - 对于我们的学习案例，Chroma 是最佳选择

    Chroma 的存储原理：
      文档 → Embedding模型 → 1536维向量 → Chroma存储
      查询 → Embedding模型 → 1536维向量 → Chroma搜索 → 返回最相似的Top-K文档
    """
    print_step(4, "Chroma 向量数据库 + 文档索引",
               "将分割后的 文档块 嵌入向量 并存入 Chroma 数据库")

    print("【Chroma 配置】")
    print(f"  数据库类型: Chroma (本地)")
    print(f"  存储路径:   {CHROMA_DIR}")
    print(f"  Embedding:  text-embedding-3-small (1536维)")
    print(f"  相似度算法: 余弦相似度 (cosine)")

    # ---------------------------------------------------------------
    # 4.1 创建向量数据库并索引文档
    # ---------------------------------------------------------------
    # Chroma.from_documents() 自动完成：
    #   1. 对每个文档块调用 embeddings.embed_documents() 生成向量
    #   2. 将文档内容和向量一起存入 Chroma
    #   3. 持久化到磁盘
    print(f"\n[索引中...] 正在将 {len(chunks)} 个文档块写入 Chroma")
    print(f"  这可能需要几秒钟，请耐心等待...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),   # 持久化到磁盘
        collection_name="enterprise_docs",    # 集合名称（类似数据库的"表"）
    )

    print(f"\n[索引完成]")
    print(f"  已索引文档块: {len(chunks)} 个")
    print(f"  向量维度:     1536")
    print(f"  存储位置:     {CHROMA_DIR}")

    # ---------------------------------------------------------------
    # 4.2 验证持久化：重新加载
    # ---------------------------------------------------------------
    print(f"\n[验证] 测试从磁盘重新加载数据库...")
    loaded_vs = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="enterprise_docs",
    )
    count = loaded_vs._collection.count()
    print(f"  重新加载成功！数据库中有 {count} 个向量")
    print(f"  → 持久化生效，重启程序无需重新索引")

    return vectorstore


# ============================================================================
# 步骤 6：检索测试（验证检索质量）
# ============================================================================
def step5_test_retrieval(vectorstore):
    """
    【第六步：检索测试】

    目的：
      验证向量数据库的检索效果，确保相似度搜索能返回正确的文档块。

    检索原理：
      1. 用户问题 → Embedding模型 → 查询向量
      2. 查询向量 vs 数据库中所有向量 → 余弦相似度计算
      3. 返回相似度最高的 Top-K 个文档块

    similarity_search() 参数说明：
      - query:  用户查询文本
      - k:      返回最相似的前 K 个文档块（建议 3-5）
    """
    print_step(5, "检索测试（验证检索质量）",
               "用多个问题测试向量数据库的搜索引擎效果")

    # ---------------------------------------------------------------
    # 5.1 设计测试问题
    # ---------------------------------------------------------------
    test_queries = [
        # (类别,                      查询问题,                           预期命中文档)
        ("行为规范-诚信", "员工应该遵守哪些诚信要求？",                  "企业行为规范"),
        ("行为规范-保密", "员工离职后还需要保密吗？保密多久？",          "企业行为规范"),
        ("行为规范-处罚", "违反公司行为规范会受到什么处罚？",            "企业行为规范"),
        ("采购-额度审批", "采购金额超过一万元需要谁审批？",              "购物及财务手册"),
        ("采购-供应商",   "如何选择合格供应商？",                       "购物及财务手册"),
        ("财务-报销",     "差旅费报销有什么规定？",                     "购物及财务手册"),
    ]

    # ---------------------------------------------------------------
    # 5.2 执行检索
    # ---------------------------------------------------------------
    for category, query, expected_source in test_queries:
        print(f"\n{'─' * 60}")
        print(f"【{category}】查询: {query}")
        print(f"  预期来源: {expected_source}")

        results = vectorstore.similarity_search(query, k=3)

        for j, doc in enumerate(results, 1):
            # 尝试判断文档来源
            source = doc.metadata.get("source", "未知")
            filename = Path(source).name if source != "未知" else "未知"
            preview = doc.page_content[:80].replace("\n", " ")

            # 计算相关性分数（如果有）
            score_info = ""
            print(f"  结果 #{j} | 来源: {filename}")
            print(f"         | 内容: {preview}...{score_info}")

    # ---------------------------------------------------------------
    # 5.3 使用 similarity_search_with_score 查看相关性分数
    # ---------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("【带分数的检索】查看 L2 距离（越小越相关）")
    query = "财务支付的审批流程是怎样的？"
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"\n  结果 #{i} | L2距离: {score:.4f}")
        print(f"  内容: {preview}...")

    print(f"\n  → L2 距离越小，表示向量空间中越接近，语义越相关")
    print(f"  → 注意：Chroma 默认使用 L2 距离，越小越好。")

    return vectorstore


# ============================================================================
# 步骤 7：RAG Agent 问答
# ============================================================================
def step6_rag_agent(vectorstore):
    """
    【第七步：RAG Agent 问答 —— 检索增强生成】

    这是 RAG 系统的核心：
      1. 用户提问
      2. Agent 识别到需要查询知识库
      3. 调用 search_enterprise_docs 检索相关文档
      4. 将检索结果注入 LLM 上下文
      5. LLM 基于检索结果生成准确答案

    与普通 LLM 对话的区别：
      ❌ 普通对话: LLM 只能用训练数据回答（可能过时/不准确/产生幻觉）
      ✅ RAG 对话: LLM 结合企业文档回答（基于真实数据，准确可靠）

    Agent 工作原理（ReAct 模式）：
      Thought → Action（调用工具）→ Observation（工具返回结果）→ Final Answer
    """
    print_step(6, "RAG Agent 问答 —— 检索增强生成",
               "将向量检索封装为工具，创建能「查阅企业内部文档」的智能问答 Agent")

    # ---------------------------------------------------------------
    # 6.1 创建检索工具
    # ---------------------------------------------------------------
    @tool
    def search_enterprise_docs(query: str) -> str:
        """
        在企业文档知识库中搜索相关信息。
        文档包括：《企业行为规范》、《企业员工购物及财务支付全流程手册》

        Args:
            query: 用户想要查询的问题或关键词

        Returns:
            检索到的最相关文档片段
        """
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "未找到相关文档信息。"
        results = []
        for i, doc in enumerate(docs, 1):
            source = Path(doc.metadata.get("source", "")).name
            results.append(f"[文档{i} - {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(results)

    # ---------------------------------------------------------------
    # 6.2 创建 Agent
    # ---------------------------------------------------------------
    system_prompt = """你是一个企业知识库助手，拥有访问企业内部文档的能力。

你的知识库包含以下文档：
1. 《企业行为规范》—— 员工职业道德、工作纪律、商务行为等规范
2. 《企业员工购物及财务支付全流程手册》—— 采购流程、付款审批、报销管理等

工作规则：
1. 当用户提问时，必须先使用 search_enterprise_docs 工具检索相关文档
2. 基于检索到的真实文档内容回答，不要凭空捏造
3. 如果文档中没有相关信息，诚实告知用户"根据现有文档，未找到相关信息"
4. 回答时引用具体的规范条款（如"根据《企业行为规范》第X条..."）"""

    agent = create_agent(
        model=llm,
        tools=[search_enterprise_docs],
        system_prompt=system_prompt,
    )

    print("  [OK] Agent 已创建，工具: search_enterprise_docs\n")

    # ---------------------------------------------------------------
    # 6.3 测试问答
    # ---------------------------------------------------------------
    test_questions = [
        "员工离职后还需要保守公司机密吗？",
        "采购金额超过5000元需要什么审批流程？",
        "出差住宿费有没有报销标准？超出的部分怎么办？",
        "违反《企业行为规范》会受到哪些处罚？",
        "如何向财务部申请付款？需要准备哪些材料？",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"{'─' * 60}")
        print(f"[问题 {i}] {question}")
        print(f"{'─' * 60}")

        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            # 取最后一条消息（Agent 的最终回答）
            answer = response["messages"][-1].content
            print(f"\n[回答]\n{answer}\n")
        except Exception as e:
            print(f"[错误] {e}\n")

    return agent


# ============================================================================
# 步骤 8：交互式问答（可选）
# ============================================================================
def step7_interactive_qa(vectorstore):
    """
    【第八步：交互式问答】

    进入交互模式，可以自由提问，直到输入 'quit' 退出。
    """
    print_step(7, "交互式问答（可选）",
               "自由提问模式，输入 quit 退出")

    @tool
    def search_enterprise_docs(query: str) -> str:
        """在企业文档知识库中搜索相关信息"""
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "未找到相关文档信息。"
        results = []
        for i, doc in enumerate(docs, 1):
            source = Path(doc.metadata.get("source", "")).name
            results.append(f"[文档{i} - {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(results)

    agent = create_agent(
        model=llm,
        tools=[search_enterprise_docs],
        system_prompt="你是企业知识库助手。先检索再回答，基于文档内容作答。",
    )

    print("输入你的问题（输入 'quit' 退出）：\n")

    while True:
        try:
            question = input(">>> ").strip()
            if question.lower() == "quit":
                print("再见！")
                break
            if not question:
                continue

            response = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            answer = response["messages"][-1].content
            print(f"\n{answer}\n")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"[错误] {e}\n")


# ============================================================================
# 主程序
# ============================================================================
def main():
    """
    RAG 学习案例 —— 完整流程

    你可以选择运行模式：
      1. 完整演示（默认）   —— 逐步执行所有步骤，每步有详细讲解
      2. 交互式问答          —— 直接进入问答模式
    """
    print("\n" + "=" * 75)
    print("  LangChain 1.3 — RAG 学习案例")
    print("  企业文档智能问答系统")
    print("=" * 75)
    print("\n  技术栈: LangChain 1.3 | DashScope(qwen) | text-embedding-3-small | Chroma")
    print("  文档: 《企业行为规范》+《企业员工购物及财务支付全流程手册》")
    print("=" * 75)

    # -------- 模式选择 --------
    print("\n请选择运行模式：")
    print("  [1] 完整演示 —— 逐步执行 RAG 所有步骤（推荐学习）")
    print("  [2] 交互问答 —— 直接进入自由提问模式")
    choice = input("\n请输入选项 (1/2，默认为 1): ").strip() or "1"

    if choice == "2":
        # ---- 快速模式：跳过讲解，直接构建 RAG 并进入问答 ----
        print("\n[快速模式] 正在构建 RAG 系统...")
        # 加载文档
        all_docs = []
        if TXT_FILE.exists():
            all_docs.extend(TextLoader(str(TXT_FILE), encoding="utf-8").load())
        if DOCX_FILE.exists():
            all_docs.extend(Docx2txtLoader(str(DOCX_FILE)).load())

        if not all_docs:
            print("[错误] 没有找到任何文档！")
            return

        # 分割 + 索引 + 问答
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)
        print(f"  已加载 {len(all_docs)} 个文档 → {len(chunks)} 个文档块")

        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
            persist_directory=str(CHROMA_DIR), collection_name="enterprise_docs"
        )
        print(f"  已索引到 Chroma: {vectorstore._collection.count()} 个向量")

        step7_interactive_qa(vectorstore)

    else:
        # ---- 完整演示模式 ----
        try:
            # 步骤 1: 加载文档
            documents = step1_load_documents()
            if not documents:
                print("\n[提示] 请确保文档文件存在于 Rag/data/ 目录下")
                return
            input("\n按 Enter 继续到下一步...")

            # 步骤 2: 分割文档
            chunks = step2_split_documents(documents)
            input("\n按 Enter 继续到下一步...")

            # 步骤 3: 向量嵌入
            step3_create_embeddings()
            input("\n按 Enter 继续到下一步...")

            # 步骤 4+5: 向量数据库 + 索引
            vectorstore = step4_create_vectorstore(chunks)
            input("\n按 Enter 继续到下一步...")

            # 步骤 6: 检索测试
            step5_test_retrieval(vectorstore)
            input("\n按 Enter 继续到下一步...")

            # 步骤 7: RAG Agent
            step6_rag_agent(vectorstore)

            # 询问是否进入交互模式
            print("\n" + "=" * 75)
            print("  核心演示完成！")
            print("=" * 75)
            if input("\n是否进入交互问答模式？(y/n): ").strip().lower() == "y":
                step7_interactive_qa(vectorstore)

        except KeyboardInterrupt:
            print("\n\n程序已中断。")
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 75)
    print("  学习完成！")
    print("=" * 75)
    print("""
    RAG 工作流总结：

      离线阶段（一次）:
        文档 → 加载 → 分割 → 嵌入 → Chroma 向量数据库

      在线阶段（每次查询）:
        问题 → Agent → 检索工具 → 相似度搜索 → LLM + 上下文 → 答案

    核心要点:
      1. Document Loaders  —— 加载不同格式的文档
      2. Text Splitters     —— 最优分割策略：chunk_size=500, overlap=100
      3. Embeddings         —— 文本转向量，语义相近则向量相近
      4. Chroma             —— 本地向量数据库，持久化存储
      5. Agent + Tool       —— 检索工具 + LLM = 智能问答
    """)


if __name__ == "__main__":
    main()
