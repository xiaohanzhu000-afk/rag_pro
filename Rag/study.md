# RAG（检索增强生成）面试学习材料

> 基于 LangChain V1.3 + Chroma 企业制度问答系统的实战总结

---

## 一、RAG 核心概念

### 1.1 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将**信息检索**与**大语言模型生成**相结合的技术架构。

```
传统 LLM 问答：  用户问题 → LLM（依赖训练数据）→ 回答  ❌ 可能幻觉/知识过时

RAG 问答：       用户问题 → 检索相关文档 → LLM（基于文档）→ 回答  ✅ 准确/可溯源
```

**核心思想**：给 LLM 配上"外脑"——在生成回答之前，先从外部知识库中检索相关信息作为上下文。

### 1.2 为什么需要 RAG？

| 问题 | 纯 LLM | RAG 方案 |
|------|--------|----------|
| 知识时效性 | 训练数据截止日期后无法知晓 | 更新文档即可 |
| 幻觉问题 | 可能编造不存在的事实 | 严格基于检索到的真实文档 |
| 可溯源性 | 无法追溯信息来源 | 可定位到具体文档段落 |
| 私有知识 | 无法访问企业内部文档 | 基于企业自己的文档回答 |
| 更新成本 | 需要重新微调模型 | 更新向量库即可（秒级） |

---

## 二、RAG 完整架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      离线阶段（建库）                             │
│                                                                 │
│  原始文档                                                     │
│    │                                                          │
│    ▼                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ ① Load  │ → │ ② Split     │ → │ ③ Embed     │        │
│  │ 文档加载  │    │ 文档分割      │    │ 文本 → 向量   │        │
│  └──────────┘    └──────────────┘    └──────┬───────┘        │
│                                              │                 │
│                                              ▼                 │
│                                    ┌──────────────────┐       │
│                                    │ ④ Vector Store  │       │
│                                    │ 向量数据库(Chroma) │       │
│                                    └──────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      在线阶段（问答）                             │
│                                                                 │
│  用户问题                                                     │
│    │                                                          │
│    ▼                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ ⑤ Embed │ → │ ⑥ Retrieve  │ → │ ⑦ Generate  │        │
│  │ 问题向量化 │    │ 相似度检索    │    │ LLM 生成回答  │        │
│  └──────────┘    └──────────────┘    └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、七大核心环节详解

### 环节 ①：文档加载（Document Loading）

**作用**：将不同格式的原始文档转换为统一的 `Document` 对象。

**LangChain Document 对象结构**：
```python
class Document:
    page_content: str    # 文档文本内容
    metadata: dict       # 元数据（来源、页码、作者等）
```

**常见 Loader 及适用场景**：

| Loader | 包路径 | 适用格式 | 注意事项 |
|--------|--------|----------|----------|
| TextLoader | langchain_community | .txt | 中文需指定 `encoding="utf-8"` |
| Docx2txtLoader | langchain_community | .docx | 依赖 `python-docx` |
| PyPDFLoader | langchain_community | .pdf | 扫描版 PDF 需 OCR |
| CSVLoader | langchain_community | .csv | 每行转为一个 Document |
| UnstructuredLoader | langchain_community | 通用 | 支持多种格式，较重 |

**面试重点**：不同文档格式的加载策略、元数据的重要性（后续可用于过滤检索）。

---

### 环节 ②：文档分割（Document Splitting）★★★ 高频考点

**为什么需要分割？**
1. LLM 上下文窗口有限（如 4K/8K/128K tokens）
2. 小块检索更精准（大块包含多种语义，检索准确率下降）
3. 避免 Embedding 模型输入超限（text-embedding-3-small 最大 8191 tokens）

**核心矛盾**：
```
chunk 太小 → 语义不完整，检索碎片化
chunk 太大 → 语义混杂，检索不准；超 LLM 上下文
```

**分割器对比（面试必问）**：

| 分割器 | 原理 | 优点 | 缺点 | 适用场景 |
|--------|------|------|------|----------|
| **RecursiveCharacterTextSplitter** ★ | 按分隔符优先级递归切分 | 保持段落/句子完整 | 需定制分隔符 | **通用推荐** |
| CharacterTextSplitter | 按单一字符切分 | 简单 | 容易切断句子 | 极简场景 |
| SemanticChunker | 按 Embedding 相似度断点切分 | 语义边界准确 | 计算成本高 | 高质量需求 |
| MarkdownTextSplitter | 按 Markdown 标题层级切分 | 保留文档结构 | 仅限 Markdown | 技术文档 |
| TokenTextSplitter | 按 Token 数切分 | 精确控制 Token | 不考虑语义 | 对 Token 数敏感的场景 |

**RecursiveCharacterTextSplitter 工作原理（深入理解）**：
```python
separators = ["\n\n", "\n", "。", "；", "，", " ", "", ]

# 递归流程：
# 1. 先用 "\n\n"（段落边界）尝试切分
# 2. 切分后如果某块仍超过 chunk_size
#    → 对该块用下一个分隔符 "\n" 继续切分
# 3. 递归降级直到最后一个分隔符 ""（逐字符切分）
#   → 保证所有文本最终都能被切分
```

**参数调优指南**：

| 参数 | 中文建议值 | 英文建议值 | 说明 |
|------|-----------|-----------|------|
| chunk_size | 300~800 | 500~1500 | 中文单字信息密度高 |
| chunk_overlap | chunk_size × 10%~20% | 同左 | 防止语义在边界丢失 |
| separators | 加入中文标点 | 英文标点即可 | 语言相关 |

**面试延伸**：
- **父子文档检索（Parent Document Retriever）**：检索用小 chunk，返回大 chunk 给 LLM 生成
- **多级索引策略**：先检索摘要级 chunk，再定位到详细 chunk
- **chunk_size 与 chunk_overlap 的关系**：overlap 太少 → 信息丢失，太多 → 冗余增加

---

### 环节 ③：Embedding（文本向量化）

**Embedding 是什么？**
将文本映射到高维向量空间，使得语义相近的文本向量距离也相近。

```
"员工不能迟到"  → [0.12, -0.34, 0.56, ..., -0.21]  (1536维向量)
"考勤管理严格"  → [0.15, -0.31, 0.52, ..., -0.18]  ← 向量距离近
"编写财务报表"  → [-0.78, 0.45, -0.23, ..., 0.67]  ← 向量距离远
```

**主流 Embedding 模型对比**：

| 模型 | 维度 | 最大输入 | 中文支持 | 费用 | 适用场景 |
|------|------|----------|----------|------|----------|
| text-embedding-3-small | 1536(可降512) | 8191 tokens | ★★★★ | 低 | 通用 RAG |
| text-embedding-3-large | 3072(可降256) | 8191 tokens | ★★★★ | 中 | 高精度需求 |
| text-embedding-ada-002 | 1536 | 8191 tokens | ★★★ | 中 | 旧版兼容 |
| bge-large-zh-v1.5 | 1024 | 512 tokens | ★★★★★ | 免费(本地) | 纯中文场景 |
| m3e-base | 768 | 512 tokens | ★★★★ | 免费(本地) | 中文轻量 |

**两个核心 API**：
```python
# 文档向量化（批量）
embeddings.embed_documents(["文本1", "文本2", ...])
# → List[List[float]]  每个文本对应一个向量

# 查询向量化（单条）
embeddings.embed_query("查询文本")
# → List[float]  单条向量
```

**面试重点**：
- **对称 vs 非对称 Embedding**：文档和查询是否用同一个模型（大多数场景用对称）
- **向量维度的选择**：维度越高→表达能力越强→存储和计算成本越高
- **归一化**：余弦相似度要求向量已归一化（L2 归一化）
- **Fine-tune Embedding**：在领域数据上微调 Embedding 模型可显著提升检索质量

---

### 环节 ④：向量数据库（Vector Store）

**向量数据库的核心能力**：
1. 存储高维向量及对应元数据
2. 高效 ANN（近似最近邻）检索
3. 支持 CRUD 操作

**主流向量数据库对比**：

| 数据库 | 类型 | 部署 | 规模 | 适用场景 |
|--------|------|------|------|----------|
| **Chroma** | 开源嵌入式 | 本地 | 小~中 | **学习/原型** |
| FAISS | 开源库 | 本地/内存 | 中~大 | 高性能、无持久化 |
| Milvus | 开源分布式 | 服务器 | 大~超大 | 生产环境 |
| Pinecone | 云服务 | SaaS | 不限 | 免运维需求 |
| Qdrant | 开源 | 本地/服务器 | 中~大 | 高性能 Rust 实现 |
| Weaviate | 开源 | 本地/服务器 | 中~大 | 混合搜索 |

**Chroma 核心操作**：
```python
# 创建并存储
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db",    # 持久化路径
    collection_name="my_collection",    # 集合（逻辑隔离）
)

# 加载已有库
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection",
)
```

**相似度度量方式**：
| 度量 | 公式 | 特点 |
|------|------|------|
| 余弦相似度 | cos(θ) = A·B/(|A|×|B|) | 最常用，关注方向 |
| 欧氏距离 | √Σ(Ai-Bi)² | 关注绝对距离 |
| 点积 | A·B | 简单但受向量模影响 |

---

### 环节 ⑤⑥：检索策略（Retrieval）★★★ 高频考点

**四种核心检索方式**：

#### 1. 基础相似度检索
```python
docs = vector_store.similarity_search(query, k=4)
```
最简单直接，返回与查询最相关的 k 个文档。

#### 2. 带分数的相似度检索
```python
docs_with_scores = vector_store.similarity_search_with_score(query, k=4)
# 返回 List[(Document, float)]，score 是距离值
```
用于设置相似度阈值过滤低质量结果。

#### 3. MMR（最大边际相关性）
```python
docs = vector_store.max_marginal_relevance_search(
    query, k=4, fetch_k=20, lambda_mult=0.5
)
```
- **原理**：在保证相关性的同时，惩罚与已选文档过于相似的候选文档
- **lambda_mult**：0~1，越小越注重多样性
- **适用场景**：需要覆盖不同角度回答的问题

#### 4. 元数据过滤
```python
docs = vector_store.similarity_search(
    query, k=4,
    filter={"source": "/path/to/doc.txt"}
)
```
按来源、日期、类别等限定检索范围。

**高级检索策略**：

| 策略 | 说明 | 优势 |
|------|------|------|
| 混合检索 | 向量检索 + BM25 关键词检索 | 补充语义检索的词汇匹配缺陷 |
| 多路召回 | 多 query 并行检索 + 结果合并 | 覆盖不同表述的用户意图 |
| Self-Query | LLM 自动生成查询 + 元数据过滤条件 | 智能筛选 |
| Rerank | 初检索 → 精排模型打分 → 取 Top-K | 大幅提升检索精度 |
| HyDE | 先生成假设文档 → 用假设文档检索 | 缩小 query-document 语义差距 |

**面试重点：检索质量评估**
| 指标 | 含义 |
|------|------|
| Hit Rate | Top-K 结果中包含正确答案的比例 |
| MRR | 第一个正确答案排名的倒数均值 |
| NDCG | 考虑排名位置的归一化折损累计增益 |

---

### 环节 ⑦：Prompt 设计与 LLM 生成

**RAG Prompt 设计原则**：

```
1. 角色定义：明确 LLM 的身份和职责
2. 上下文注入：清晰标记参考文档区域
3. 行为约束：
   - 仅基于文档回答（防幻觉）
   - 不知道就说不知道（防编造）
   - 引用来源（可追溯）
4. 格式要求：结构化输出
```

**标准 RAG Prompt 模板**：
```python
system_prompt = """你是一个专业的{角色}。请严格基于以下【参考文档】回答问题。

【回答规则】
1. 只使用参考文档中的信息，不要编造
2. 如果文档中没有相关信息，请回答："未找到相关信息"
3. 回答时引用文档中的具体条款

【参考文档】
{context}"""

user_prompt = "{question}"
```

---

## 四、LangChain RAG 实现方式

### 4.1 LCEL 管道式组装

```python
rag_chain = (
    {
        "context": retriever | format_docs,   # 检索 → 格式化
        "question": RunnablePassthrough(),     # 透传问题
    }
    | prompt     # 填入模板
    | llm        # 生成
    | StrOutputParser()  # 解析输出
)

answer = rag_chain.invoke("员工迟到有什么处罚？")
```

**LCEL 核心符号**：
| 符号 | 含义 | 示例 |
|------|------|------|
| `|` | 管道：左输出 → 右输入 | `a | b` |
| `{}` | 并行字典：同时运行多个分支 | `{"a": f1, "b": f2}` |
| `RunnablePassthrough()` | 透传数据不做处理 | |

---

## 五、实战项目中的关键设计决策

### 5.1 为什么用 RecursiveCharacterTextSplitter？

本案例文档为**结构化中文制度文档**，具有以下特点：
- 条文化格式（第一条第X款）
- 中文标点分隔（句号、分号、逗号）
- 需要保持条款完整性

因此：
- ✅ 选择 `RecursiveCharacterTextSplitter` + 中文定制分隔符
- ❌ 不用 CharacterTextSplitter（会切断句子）
- ❌ 不用 SemanticChunker（额外计算成本，且结构化文档语义边界已由格式体现）

### 5.2 为什么 chunk_size=500，chunk_overlap=80？

- 中文单字信息密度高，500 字符 ≈ 约 300 tokens，适合大多数 LLM
- overlap 80 字符 ≈ 1~2 句中文，足够防止语义在边界断裂
- 两条文档共约 4000 字符 → 分割后约 12-15 个 chunk，规模适中

### 5.3 为什么用 text-embedding-3-small 而不是本地模型？

| 维度 | text-embedding-3-small | bge-large-zh (本地) |
|------|----------------------|---------------------|
| 中文效果 | ★★★★ | ★★★★★ |
| 部署方式 | API 调用 | 需 GPU/CPU |
| 费用 | 低成本 | 免费 |
| 维度 | 1536 | 1024 |
| 速度 | 快 | 取决于硬件 |

对于学习场景，API 模型更方便；生产环境中文场景可考虑 bge-large-zh。

### 5.4 为什么用 Chroma 而不是 FAISS？

- Chroma 内置持久化，重启不丢数据
- FAISS 是纯内存索引库，需手动序列化
- 学习场景 Chroma 开箱即用，更友好

---

## 六、常见面试问题 & 参考答案

### Q1：RAG 和微调（Fine-tuning）有什么区别？什么时候用哪个？

**参考答案**：
| 维度 | RAG | 微调 |
|------|-----|------|
| 原理 | 检索外部知识 | 修改模型参数 |
| 知识更新 | 更新文档即可（秒级） | 需重新训练（小时级） |
| 适用场景 | 知识密集型、频繁更新 | 风格/格式/推理模式 |
| 幻觉 | 低（基于文档） | 仍可能幻觉 |
| 成本 | 低 | 高 |
| 可解释性 | 高（可溯源） | 低 |

**选择建议**：
- 知识问答、企业文档 → RAG
- 特定风格写作、指令遵循 → 微调
- 最佳实践：RAG + 微调结合使用

### Q2：文档分割中 chunk_size 太大/太小分别有什么问题？

**参考答案**：
- **太大**：
  - 单一 chunk 包含多个主题，检索精度下降
  - 多个 chunk 拼接容易超出 LLM 上下文
  - Embedding 表示不够聚焦
- **太小**：
  - 语义不完整，关键信息被切断
  - 检索结果碎片化，LLM 难以拼凑完整答案
  - 存储和检索开销增大（chunk 数量增多）

**解法**：根据文档类型和 LLM 上下文窗口选择合适大小，配合 overlap 保留边界语义。

### Q3：如何评估 RAG 系统的检索质量？

**参考答案**：
三个核心指标：
1. **Hit Rate（命中率）**：Top-K 中是否包含正确答案
2. **MRR（平均倒数排名）**：第一个正确答案排名的倒数均值
3. **NDCG**：考虑排名位置的准确率

评估方法：
- 构建 QA 测试集（问题 + 预期答案 + 预期文档 ID）
- 使用 Ragas 框架自动评估
- 人工评估答案的忠实度和相关性

### Q4：什么是 RAG 中的"幻觉"问题？如何缓解？

**参考答案**：
RAG 中的幻觉指 LLM 生成的回答包含检索文档中不存在的信息。

缓解方案：
1. **Prompt 约束**：明确要求"仅基于文档回答，不知道就说不知道"
2. **Rerank 优化**：提升检索精度，减少无关文档干扰
3. **Faithfulness 评估**：逐句验证回答是否能在文档中找到依据
4. **引用机制**：要求 LLM 在回答中引用具体文档段落

### Q5：语义检索和关键词检索各有什么优缺点？如何结合？

**参考答案**：

| 方式 | 优点 | 缺点 |
|------|------|------|
| 语义检索（向量） | 理解语义，同义词泛化 | 可能忽略精确关键词匹配 |
| 关键词检索（BM25） | 精确匹配，专有名词友好 | 不理解语义 |

**混合检索（Hybrid Search）方案**：
1. 同时执行向量检索和 BM25 检索
2. 使用 RRF（倒数排名融合）合并结果
3. 去重后返回给 LLM

### Q6：如何处理文档更新（文档被修改/新增/删除）？

**参考答案**：
1. **全量重建**：重新 Embedding 全部文档（小规模适用）
2. **增量更新**：
   - Chroma 支持 `add_documents()` 追加
   - 支持 `update_document()` / `delete()` 按 ID 更新/删除
3. **版本管理**：记录文档版本与向量库版本的映射关系
4. **定时任务**：定期检查文档变更，自动触发重建

### Q7：RAG 系统的延迟优化有哪些手段？

**参考答案**：
1. **Embedding 加速**：使用本地部署的 Embedding 模型替代 API 调用
2. **向量索引优化**：使用 IVF/HNSW 等近似检索算法
3. **缓存策略**：对常见查询缓存检索结果和最终回答
4. **流式输出**：LLM 使用 streaming 模式，逐 token 返回
5. **异步处理**：检索和生成使用异步调用
6. **并发检索**：多 query 并行检索

---

## 七、RAG 进阶话题

### 7.1 高级 RAG 架构

```
基础 RAG：  Query → Retrieve → Generate

进阶 RAG：
  ┌──────────────────────────────────────────┐
  │ Query → Query Rewrite（查询改写）          │
  │   → Multi-Query（多角度查询）              │
  │   → Hybrid Search（混合检索）              │
  │   → Rerank（精排）                         │
  │   → Context Compression（上下文压缩）       │
  │   → Generate                               │
  └──────────────────────────────────────────┘
```

### 7.2 RAG 评估框架

| 框架 | 特点 |
|------|------|
| **Ragas** | 最流行，支持 Faithfulness/Relevance/Context Precision 等指标 |
| **TruLens** | 可视化评估，支持反馈函数 |
| **LangSmith** | LangChain 官方，全链路追踪 |
| **DeepEval** | 轻量级，支持多种指标 |

---

## 八、核心代码速记

```python
# === 完整 RAG 系统核心代码 ===

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# 1. 加载
docs = TextLoader("doc.txt", encoding="utf-8").load()

# 2. 分割
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""]
)
chunks = splitter.split_documents(docs)

# 3. Embedding + 存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings,
    persist_directory="./db"
)

# 4. 检索
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# 5. Prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下文档回答：\n{context}"),
    ("human", "{question}"),
])

# 6. LLM
llm = ChatOpenAI(model="gpt-4o")

# 7. Chain（LCEL）
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 8. 调用
answer = chain.invoke("问题？")
```

---

## 九、学习检查清单

```
□ 能画出 RAG 的完整架构图（离线建库 + 在线问答）
□ 能说出 Document Loader 至少 3 种，并知道中文编码注意事项
□ 能解释 RecursiveCharacterTextSplitter 的递归降级原理
□ 能对比至少 2 种分割器的优缺点
□ 能说清楚 chunk_size 和 chunk_overlap 如何选择
□ 理解 Embedding 的本质：文本 → 高维向量的语义映射
□ 知道至少 3 种 Embedding 模型及其中文支持情况
□ 能说出 Chroma、FAISS、Milvus 的区别
□ 掌握 similarity_search / MMR / 元数据过滤 / Retriever 接口
□ 理解 MMR 的 lambda_mult 参数含义
□ 能设计出一个合格的 RAG Prompt
□ 能写出 LCEL 管道式组装的代码
□ 能回答 RAG vs 微调的选择问题
□ 知道混合检索（Hybrid Search）是什么
□ 了解 RAG 评估指标：Hit Rate、MRR
□ 能说出 3 种 RAG 幻觉缓解方案
```

---

*祝你面试顺利！🎯*
