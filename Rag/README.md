# RAG 学习案例 —— 企业文档智能问答系统

基于 **LangChain 1.3** 构建的完整 RAG (Retrieval-Augmented Generation) 学习案例，实现对《企业行为规范》和《企业员工购物及财务支付全流程手册》的智能问答。

---

## 目录

- [什么是 RAG？](#什么是-rag)
- [项目文件结构](#项目文件结构)
- [技术栈](#技术栈)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [学习路径](#学习路径)
  - [路径 A：分步学习（推荐新手）](#路径-a分步学习)
  - [路径 B：一体化演示（快速体验）](#路径-b一体化演示)
- [RAG 工作流程详解](#rag-工作流程详解)
  - [步骤 1：文档加载](#步骤-1文档加载)
  - [步骤 2：文档分割（最优策略）](#步骤-2文档分割最优策略)
  - [步骤 3：向量嵌入 + Chroma 存储](#步骤-3向量嵌入--chroma-存储)
  - [步骤 4：向量检索](#步骤-4向量检索)
  - [步骤 5：RAG Chain 组装](#步骤-5rag-chain-组装)
  - [步骤 6：交互式问答](#步骤-6交互式问答)
- [关键设计决策](#关键设计决策)
- [常见问题](#常见问题)
- [扩展方向](#扩展方向)

---

## 什么是 RAG？

**RAG (Retrieval-Augmented Generation) = 检索增强生成**

RAG 让 LLM 能够访问外部知识库，解决三大痛点：

| 痛点 | 无 RAG | 有 RAG |
|------|--------|--------|
| 训练数据过时 | LLM 只知道训练截止日前的知识 | 可以访问最新的企业文档 |
| 私有数据盲区 | LLM 不知道企业内部规定 | 检索企业内部文档后回答 |
| 幻觉问题 | LLM 可能凭空编造答案 | 基于真实文档内容回答 |

**RAG 工作流：**

```
┌──────────────────────────────────────────────┐
│             离线阶段（一次性）                  │
│  文档 → 加载 → 分割 → 嵌入 → Chroma向量数据库   │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│             在线阶段（每次查询）                 │
│  问题 → Embedding → 相似度搜索 → LLM+上下文 → 答案│
└──────────────────────────────────────────────┘
```

---

## 项目文件结构

```
Rag/
├── README.md                           # 本文档（学习与使用指南）
│
├── 企业行为规范.txt                      # 文档1：员工职业道德和行为规范
├── 企业员工购物及财务支付全流程手册.docx   # 文档2：采购和财务支付流程
│
├── generate_docs.py                    # Word 文档生成脚本（已有 .docx 无需运行）
│
├── 1_load_documents.py                 # 步骤1：Document Loader 加载文档
├── 2_split_documents.py                # 步骤2：最优分割策略（RecursiveCharacterTextSplitter）
├── 3_embedding_chroma.py               # 步骤3：Embedding + Chroma 向量数据库
├── 4_retrieve.py                       # 步骤4：多种检索方式演示
├── 5_rag_chain.py                      # 步骤5：LCEL 组装完整 RAG 链
├── 6_rag_qa.py                         # 步骤6：交互式 RAG 问答应用
│
├── rag_demo.py                         # 一体化演示：全部步骤 + 交互问答
│
└── chroma_db/                          # Chroma 向量数据库持久化目录（自动生成）
    └── ...                             # 数据库文件
```

---

## 技术栈

| 组件 | 技术选择 | 说明 |
|------|----------|------|
| 框架 | LangChain 1.3 | LCEL 管道 + Agent + Tool |
| LLM | 通义千问 qwen-turbo | 通过 DashScope API，兼容 OpenAI 接口 |
| Embedding | OpenAI text-embedding-3-small | 1536维，中英文多语言，$0.02/1M tokens |
| 向量数据库 | Chroma | 本地运行，零配置，支持持久化 |
| 文档加载 | TextLoader + Docx2txtLoader | .txt 和 .docx 双格式支持 |
| 文本分割 | RecursiveCharacterTextSplitter | 中文优化分隔符，递归语义分割 |

---

## 环境准备

### 1. 安装依赖

```bash
pip install langchain langchain-openai langchain-community \
            langchain-chroma langchain-text-splitters \
            chromadb python-docx python-dotenv
```

### 2. 配置环境变量

在项目根目录的 `.env` 文件中：

```bash
# DashScope（LLM 模型 —— 通义千问）
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# OpenAI 兼容代理（Embedding 模型使用）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

> **Embedding API Key 说明：** 代码中已内置 Embedding 的 API Key。Embedding 通过 `OPENAI_BASE_URL` 指定的代理地址调用，支持任何兼容 OpenAI 接口的服务。

### 3. 确认文档

确保以下两个文档存在于 `Rag/` 目录下：
- `Rag/企业行为规范.txt` 
- `Rag/企业员工购物及财务支付全流程手册.docx` 

> 如需重新生成 .docx：`python Rag/generate_docs.py`

---

## 快速开始

### 路径 A：分步学习（推荐新手）

逐个运行步骤文件，每步独立执行并有详细讲解：

```bash
cd Rag

# 步骤1：加载文档（理解 Document 对象）
python 1_load_documents.py

# 步骤2：分割文档（理解为什么 chunk_size=500 是最优选择）
python 2_split_documents.py

# 步骤3：Embedding + Chroma（理解文本→向量→存储全流程）
python 3_embedding_chroma.py

# 步骤4：检索（对比 4 种检索方式的效果）
python 4_retrieve.py

# 步骤5：RAG Chain（理解 LCEL 管道组装）
python 5_rag_chain.py

# 步骤6：交互问答（自由提问，体验完整 RAG 效果）
python 6_rag_qa.py
```

### 路径 B：一体化演示（快速体验）

一个脚本包含所有步骤 + 交互问答：

```bash
cd Rag
python rag_demo.py
```

按提示选择：
- **[1] 完整演示** —— 逐步执行 RAG 所有步骤，每步有详细讲解（推荐学习）
- **[2] 交互问答** —— 跳过讲解，直接进入自由提问模式

---

## RAG 工作流程详解

### 步骤 1：文档加载

**对应文件：** `1_load_documents.py`

```python
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

# .txt 文件
loader_txt = TextLoader("企业行为规范.txt", encoding="utf-8")
docs_txt = loader_txt.load()

# .docx 文件
loader_docx = Docx2txtLoader("企业员工购物及财务支付全流程手册.docx")
docs_docx = loader_docx.load()

# 汇总
all_docs = docs_txt + docs_docx
# all_docs 是 List[Document]，每个 Document 有 page_content 和 metadata
```

**核心概念：**
- `Document.page_content`：文档的纯文本内容
- `Document.metadata`：来源文件路径等元信息

---

### 步骤 2：文档分割（最优策略）

**对应文件：** `2_split_documents.py`

**为什么选择 RecursiveCharacterTextSplitter？**

| 对比项 | RecursiveCharacterTextSplitter | CharacterTextSplitter | SemanticChunker |
|--------|-------------------------------|----------------------|-----------------|
| 分割方式 | 按优先级递归尝试分割符 | 固定分隔符一刀切 | 按语义相似度 |
| 中文支持 | ✅ 支持中文标点分隔符 | ❌ 容易切断句子 | ⚠️ 计算成本高 |
| 语义保护 | ✅ 优先保持段落/句子完整 | ❌ 不保护 | ✅ 保护最好 |
| 性能 | ✅ 快 | ✅ 最快 | ❌ 慢 |
| 适合场景 | 结构化/半结构化文档 | 简单文本 | 高质量要求 |

**最优参数配置：**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块 500 字符，约等于 300-350 个中文实质字符
    chunk_overlap=80,     # 20% 重叠，防止关键信息在边界被截断
    separators=[
        "\n\n",           # 第1优先级：段落分隔
        "\n",             # 第2优先级：行分隔
        "。",             # 第3优先级：中文句号
        "；", "！", "？", # 第4-6优先级：其他中文标点
        "，", "、",       # 第7-8优先级：逗号、顿号
        " ", ".",         # 第9-10优先级：空格、英文句号
        "",               # 最后保底：逐字符
    ],
    length_function=len,
)
```

**为什么 chunk_size=500？**

| 文档类型 | 推荐值 | 原因 |
|----------|--------|------|
| 企业制度规范 | 400-600 | 每个条款约 200-400 字，500 可容纳完整条款 |
| FAQ 问答 | 200-300 | 问答对通常较短 |
| 技术长文 | 800-1200 | 技术概念需要更多上下文 |

我们的文档是规章制度，500 字恰好容纳一条完整规定。

**为什么 chunk_overlap=80？**
- 过小(0)：可能导致"应符合以下条件："这类过渡句被截断
- 过大(200+)：重复内容太多，浪费 token
- 80(约16%)：在信息完整性和成本之间取得平衡

---

### 步骤 3：向量嵌入 + Chroma 存储

**对应文件：** `3_embedding_chroma.py`

**Embedding 原理：** 将文本映射为高维向量，语义相近的文本在向量空间中距离更近。

```
"员工应当诚实守信" → [0.12, -0.34, 0.56, ..., 0.78]  (1536维)
"员工必须诚信正直" → [0.15, -0.31, 0.52, ..., 0.74]  ← 距离近！
"财务报表编制流程" → [-0.78, 0.45, -0.23, ..., -0.11] ← 距离远！
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 初始化 Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key="your-api-key",
    base_url=OPENAI_BASE_URL,
)

# 一步完成：Embedding + 存入 Chroma + 持久化
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="enterprise_policies",
)
```

**Chroma vs 其他向量数据库：**

| 数据库 | 部署 | 持久化 | 学习曲线 | 适合场景 |
|--------|------|--------|----------|----------|
| Chroma | 本地 | ✅ | 低 | 学习、开发、小项目 |
| FAISS | 本地(内存) | ❌ | 低 | 离线批量处理 |
| Pinecone | 云服务 | ✅ | 中 | 生产环境 |

---

### 步骤 4：向量检索

**对应文件：** `4_retrieve.py`

四种检索方式及适用场景：

| 方式 | API | 适用场景 |
|------|-----|----------|
| **相似度检索** | `similarity_search(query, k)` | 日常问答，快速获取最相关内容 |
| **带分数检索** | `similarity_search_with_score(query, k)` | 需要评估置信度，设置阈值过滤 |
| **MMR 检索** | `max_marginal_relevance_search(query, k)` | 需要答案覆盖多个角度 |
| **元数据过滤** | `similarity_search(query, k, filter={})` | 按来源/分类限定检索范围 |

```python
# 基础检索
docs = vector_store.similarity_search("员工请假的流程？", k=3)

# 带分数检索（L2距离，越小越相关）
results = vector_store.similarity_search_with_score("员工请假的流程？", k=3)
for doc, score in results:
    similarity = 1 / (1 + score)  # 转换为0-1的相似度
```

---

### 步骤 5：RAG Chain 组装

**对应文件：** `5_rag_chain.py`

使用 **LCEL (LangChain Expression Language)** 管道组装完整 RAG 链：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt 模板：指导 LLM 如何使用检索到的上下文
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是企业制度助手。基于以下【参考文档】回答问题：
    {context}
    规则：只用文档中的信息，不要编造。"""),
    ("human", "{question}"),
])

# 格式化检索到的文档
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# LCEL 管道：| 操作符连接各组件
rag_chain = (
    {
        "context": retriever | format_docs,   # 检索 + 格式化
        "question": RunnablePassthrough(),     # 问题直接传递
    }
    | prompt                                    # 填入模板
    | llm                                       # LLM 生成
    | StrOutputParser()                         # 解析输出
)

# 单次调用
answer = rag_chain.invoke("员工迟到有什么处罚？")
```

**LCEL 管道数据流：**

```
question ──→ {                                      } ──→ prompt ──→ llm ──→ answer
              context: retriever → format_docs   ↑
              question: RunnablePassthrough()    ↑
```

---

### 步骤 6：交互式问答

**对应文件：** `6_rag_qa.py` 或 `rag_demo.py` 模式2

```
🔍 请输入问题: 采购金额超过一万元需要谁审批？
🤔 正在检索知识库并生成回答...

根据《企业员工购物及财务支付全流程手册》，采购金额超过10,000元属于三级审批，
需要经过以下流程：
1. 部门主管审核
2. 财务经理复核
3. 分管副总审批
4. 总经理终审
然后才能进行付款。

💡 提示: 输入 /source 查看回答引用的原始文档
```

---

## 关键设计决策

### 1. 为什么选择 RecursiveCharacterTextSplitter？

- **递归降级策略**：优先在段落/句子边界处分割，保护语义完整性
- **中文友好**：支持中文标点（。！？；，、）作为分割标记
- **广泛适用**：企业制度、流程手册等结构化文档的最佳选择

### 2. 为什么选择 text-embedding-3-small？

- **1536维**：精度和速度的最佳平衡点
- **多语言**：对中文语义理解优秀
- **成本友好**：$0.02/百万 tokens，比 text-embedding-ada-002 便宜 50%
- **维度可调**：可降到 512 维以节省存储（牺牲少量精度）

### 3. 为什么选择 Chroma？

- **零配置**：本地运行，无需注册云服务
- **持久化**：重启程序数据不丢失
- **轻量级**：适合学习和开发阶段
- **与 LangChain 深度集成**：一行代码完成 Embedding+存储

### 4. 为什么选择 LCEL 而非 Chain 类？

LangChain 1.3 推荐使用 LCEL（| 管道操作符）代替旧的 Chain 类：
- ✅ 类型安全，IDE 能提供更好的补全
- ✅ 流式支持更好（支持 `stream()` 方法）
- ✅ 组件可复用，易于调试
- ✅ 是 LangChain 1.0+ 的官方推荐方式

---

## 常见问题

### Q1：chroma_db 已存在，需要重新索引吗？

不需要。Chroma 会自动加载已有数据。如果文档有更新，删除 `Rag/chroma_db/` 目录后重新运行即可。

```bash
# 清空数据库，重新索引
rmdir /s Rag\chroma_db
python Rag/3_embedding_chroma.py
```

### Q2：Embedding API 报 401 错误？

确认代码中的 `EMBEDDING_API_KEY` 和 `.env` 中的 `OPENAI_BASE_URL` 配置正确。Embedding 通过 `OPENAI_BASE_URL` 指定的代理地址调用。

### Q3：如何用自己的文档替换？

将你的 `.txt` 或 `.docx` 文件放到 `Rag/` 目录下，修改对应步骤文件中的文件路径即可。

### Q4：为什么有时检索结果不准确？

可能原因及解决方法：
1. **chunk_size 不合适**：制度类文档建议 400-600
2. **查询表述多义**：尝试用更精确的关键词
3. **文档分词差**：确保原文档文本清晰、格式规范
4. **相似度阈值**：用 `similarity_search_with_score` 查看分数，设置阈值过滤

### Q5：检索器返回的文件路径为空？

这是正常现象。`Docx2txtLoader` 加载的文档 metadata 中的 path 可能包含绝对路径，使用 `os.path.basename()` 提取文件名即可。

---

## 扩展方向

掌握基础 RAG 后，可以深入学习：

| 方向 | 说明 | 关键技术 |
|------|------|----------|
| 混合检索 | 关键词 + 语义搜索 | BM25Retriever + EnsembleRetriever |
| 重排序 | 二次排序提高精度 | Cohere Rerank / bge-reranker |
| 查询改写 | 优化用户提问 | LLM 改写 + 多轮查询 |
| 多模态 RAG | 支持图片/表格 | 多模态 Embedding 模型 |
| 流式输出 | 逐字输出提升体验 | `stream()` 方法 |
| RAG 评估 | 量化检索质量 | RAGAS 评估框架 |
| 生产部署 | 规模化运维 | FastAPI + Docker + 监控 |

---

## 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/introduction/)
- [Chroma 向量数据库文档](https://docs.trychroma.com/)
- [OpenAI Embeddings 指南](https://platform.openai.com/docs/guides/embeddings)
- [LangChain LCEL 教程](https://python.langchain.com/docs/concepts/lcel/)
- [DashScope 通义千问 API](https://help.aliyun.com/zh/dashscope/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629) — Agent 推理模式理论基础
