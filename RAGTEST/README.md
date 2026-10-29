# 企业 RAG 智能问答系统

基于 **LangChain 1.3** 构建的企业级检索增强生成（RAG）系统，
实现对企业内部多格式文档的智能问答。

---

## 目录

- [项目简介](#项目简介)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细使用指南](#详细使用指南)
- [配置说明](#配置说明)
- [知识库管理](#知识库管理)
- [API 参考](#api-参考)
- [常见问题](#常见问题)

---

## 项目简介

本系统是一个企业级的 RAG（Retrieval-Augmented Generation，检索增强生成）解决方案，
能够将企业内部的文档（制度、手册、规范等）构建为可检索的知识库，
并提供基于大模型的智能问答服务。

### 核心特性

| 特性 | 说明 |
|------|------|
| 多格式支持 | 支持 TXT、Markdown、Word(.docx)、PDF、图片(PNG/JPG) 等多种格式 |
| 中文优化 | 分割策略针对中文文档深度优化，保留语义完整性 |
| 双模式运行 | Chain 模式（快速）和 Agent 模式（智能推理） |
| 可追溯性 | 回答可追溯到原始文档，避免 AI "幻觉" |
| 模块化设计 | 各功能模块独立，便于扩展和维护 |
| 持久化存储 | 向量数据库持久化到磁盘，重启无需重建 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        企业 RAG 系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ 知识库文档 │ → │ 文档加载 │ → │ 文档分割 │ → │ 向量嵌入  │  │
│  │ .txt .md  │    │ Loader   │    │ Splitter │    │ Embedding │  │
│  │ .docx .pdf│    └──────────┘    └──────────┘    └─────┬─────┘  │
│  │ .png .jpg │                                          │        │
│  └──────────┘                                     ┌─────▼─────┐  │
│       ▲                                           │  Chroma   │  │
│       │  离线索引阶段                              │ 向量数据库 │  │
│       │                                           └─────┬─────┘  │
│  ┌────┴──────┐                                         │        │
│  │ 更新文档   │                                    ┌────▼────┐   │
│  │ build_idx │                                    │ 检索器  │   │
│  └───────────┘                                    │Retriever│   │
│                                                   └────┬────┘   │
│  ┌──────────────────────────────────────────────────────┼──────┐ │
│  │                   在线问答阶段                        │      │ │
│  │                                                      │      │ │
│  │   用户提问 ──→ ┌─────────┐ ──→ ┌──────┐ ──→ 回答     │      │ │
│  │               │ Prompt  │     │ LLM  │               │      │ │
│  │               │ 上下文组装│     │ qwen │               │      │ │
│  │               └─────────┘     └──────┘               │      │ │
│  └──────────────────────────────────────────────────────┘      │ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 框架 | **LangChain 1.3** | 使用最新 LCEL 表达式语言 |
| LLM | **通义千问 (qwen-turbo)** | 通过 DashScope API |
| Embedding | **text-embedding-3-small** | 1536 维向量，中英文支持 |
| 向量数据库 | **Chroma** | 本地持久化，零配置 |
| 文档加载 | **LangChain Loaders** | TextLoader / Docx2txtLoader / PyPDFLoader |
| 文本分割 | **RecursiveCharacterTextSplitter** | 中文语义边界优化 |

---

## 项目结构

```
RAGTEST/
│
├── data/                           # 知识库文档目录（放置原始文档）
│   ├── 企业行为规范.txt             # 纯文本文档
│   ├── 企业食堂管理规范.md          # Markdown 文档
│   ├── 企业员工购物及财务支付全流程手册.docx  # Word 文档
│   ├── 企业通用招聘简章.pdf         # PDF 文档
│   └── 企业守则图片生成.png         # 图片文件
│
├── config/                         # 配置模块
│   ├── __init__.py
│   └── settings.py                 # 集中配置文件（模型、分割、检索等参数）
│
├── core/                           # 核心组件模块
│   ├── __init__.py
│   ├── embedding.py                # Embedding 模型管理（单例模式）
│   ├── llm.py                      # LLM 模型管理（单例模式）
│   └── vector_store.py             # Chroma 向量数据库管理
│
├── modules/                        # 流水线功能模块（可插拔）
│   ├── __init__.py
│   ├── loader.py                   # 多格式文档加载器
│   ├── splitter.py                 # 文本分割器
│   └── retriever.py                # 多策略检索器
│
├── chains/                         # Chain 组装模块
│   ├── __init__.py
│   └── rag_chain.py                # RAG Chain（LCEL 方式）
│
├── agents/                         # Agent 智能体模块
│   ├── __init__.py
│   └── rag_agent.py                # RAG Agent（ReAct 模式）
│
├── utils/                          # 工具模块
│   ├── __init__.py
│   └── helpers.py                  # 日志、计时器等辅助函数
│
├── storage/                        # 持久化存储目录
│   └── chroma_db/                  # Chroma 向量数据库文件（自动生成）
│
├── logs/                           # 日志文件目录（自动生成）
│   └── rag_YYYYMMDD.log
│
├── build_index.py                  # 知识库索引构建脚本
├── main.py                         # 交互式问答主程序入口
├── requirements.txt                # 依赖清单
└── README.md                       # 本文件
```

---

## 环境要求

### Python 版本

- Python >= 3.10

### 系统依赖（可选，用于图片 OCR）

图片 OCR 功能需要安装 **Tesseract OCR**：

**Windows:**
1. 下载安装 [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
2. 安装路径保持默认（`C:\Program Files\Tesseract-OCR\`）
3. 将安装路径加入系统环境变量 PATH

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
```

> 注意：不安装 Tesseract 也不影响系统正常使用，仅图片文件会被跳过。

### 环境变量配置

在项目根目录（`langchain/`）创建 `.env` 文件：

```env
# DashScope（通义千问 LLM）
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# OpenAI 兼容接口（用于 Embedding）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

---

## 快速开始

### 第一步：安装依赖

```bash
cd RAGTEST
pip install -r requirements.txt
```

### 第二步：放置知识库文档

将企业文档放入 `data/` 目录，系统支持以下格式：

| 格式 | 扩展名 | 加载方式 |
|------|--------|---------|
| 纯文本 | `.txt` | TextLoader |
| Markdown | `.md` | TextLoader |
| Word 文档 | `.docx` | Docx2txtLoader |
| PDF 文档 | `.pdf` | PyPDFLoader |
| 图片 | `.png` `.jpg` `.jpeg` | OCR（pytesseract） |

### 第三步：构建知识库索引

```bash
# 首次构建（使用默认配置）
python build_index.py

# 查看当前数据目录统计
python build_index.py --stats

# 清空旧索引并重建
python build_index.py --rebuild

# 自定义分割参数
python build_index.py --chunk-size 800 --chunk-overlap 150 --rebuild
```

构建过程中会依次执行：
1. 扫描 `data/` 目录，加载所有文档
2. 将文档分割为适合检索的文本块
3. 使用 Embedding 模型将文本块转为向量
4. 写入 Chroma 向量数据库（持久化到 `storage/chroma_db/`）

### 第四步：启动问答系统

```bash
# 默认 Chain 模式（推荐，速度快）
python main.py

# Agent 模式（更强推理能力）
python main.py --mode agent
```

---

## 详细使用指南

### 运行模式对比

| 维度 | Chain 模式 | Agent 模式 |
|------|-----------|-----------|
| 调用方式 | `python main.py` | `python main.py --mode agent` |
| 实现方式 | LCEL 管道链 | LangChain Agent + Tool |
| 检索次数 | 固定 1 次 | Agent 自主决定（可多次） |
| 推理能力 | 较弱 | 较强 |
| 响应速度 | 快 | 较慢 |
| 适用场景 | 简单问答 | 复杂、多步推理 |

### 交互命令

启动系统后，支持以下命令：

| 命令 | 说明 |
|------|------|
| `问题文本` | 向系统提问，获取基于知识库的回答 |
| `/source` | 查看上次回答引用的原始文档片段 |
| `/help` | 显示帮助信息 |
| `/exit` | 退出系统 |

### 示例问答

```
>>> 员工迟到有什么处罚？

  根据《企业行为规范》，员工迟到处罚规定如下：
  1. 月度累计迟到3次以内（含），每次扣减绩效分5分
  2. 月度累计迟到4-6次，除扣减绩效分外，给予书面警告
  3. 月度累计迟到7次及以上，视为严重违纪，可给予降级或解除劳动合同
  ...

  (输入 /source 查看引用来源)

>>> 保密制度有哪些要求？

  根据《企业行为规范》保密制度章节，主要要求包括：
  1. 在职期间需签订的保密协议...
  2. 离职后保密义务持续24个月...
  ...

  (输入 /source 查看引用来源)
```

### 日志查看

系统日志同时输出到控制台和文件：

```bash
# 查看当天日志
cat logs/rag_20260730.log

# 调试模式
python main.py --log-level DEBUG
```

---

## 配置说明

所有配置集中在 `config/settings.py`，支持按需修改：

### LLM 模型配置

```python
LLM_CONFIG = {
    "model": "qwen-turbo",          # 模型名称，可换成 qwen-plus / qwen-max
    "temperature": 0.3,             # 温度参数（0=确定, 1=随机）
    "max_tokens": 2048,             # 最大生成 token 数
}
```

### Embedding 模型配置

```python
EMBEDDING_CONFIG = {
    "model": "text-embedding-3-small",  # 嵌入模型
    "dimensions": 1536,                  # 向量维度
}
```

### 文档分割配置

```python
SPLITTER_CONFIG = {
    "chunk_size": 500,              # 文本块最大字符数
    "chunk_overlap": 100,           # 块间重叠字符数
}
```

| chunk_size | 适用场景 |
|-----------|---------|
| 300-400 | FAQ、短问答 |
| 500 | **推荐默认值**，适合大多数企业文档 |
| 800-1000 | 长篇文章、制度手册 |

### 检索配置

```python
RETRIEVER_CONFIG = {
    "search_type": "similarity",    # similarity / mmr / similarity_score_threshold
    "search_kwargs": {"k": 4},      # 返回 Top-K 个结果
}
```

### Prompt 模板

可在 `RAG_SYSTEM_PROMPT` 中自定义系统提示词，调整回答风格和规则。

---

## 知识库管理

### 添加新文档

直接放入 `data/` 目录，然后重建索引：

```bash
python build_index.py --rebuild
```

### 更新已有文档

1. 替换 `data/` 目录下的文件
2. 重建索引：

```bash
python build_index.py --rebuild
```

### 支持的文档格式扩展

如需添加新的文档格式支持，在 `modules/loader.py` 中：

1. 编写加载函数（遵循 `_load_xxx(file_path) -> list[Document]` 签名）
2. 在 `_LOADER_REGISTRY` 中注册
3. 在 `SUPPORTED_EXTENSIONS` 中添加扩展名映射

---

## API 参考

### 在代码中调用 RAG 系统

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.vector_store import load_vectorstore
from chains.rag_chain import build_rag_chain

# 加载向量数据库
vectorstore = load_vectorstore()

# 构建 RAG Chain
rag_chain = build_rag_chain(vectorstore)

# 提问
answer = rag_chain.invoke("员工离职后保密多久？")
print(answer)
```

### 使用 Agent 模式

```python
from core.vector_store import load_vectorstore
from agents.rag_agent import create_rag_agent, run_agent_query

vectorstore = load_vectorstore()
agent = create_rag_agent(vectorstore)

answer = run_agent_query(agent, "员工的请假流程是怎样的？")
print(answer)
```

### 模块独立使用

```python
# 单独使用文档加载器
from modules.loader import load_documents, load_document
docs = load_documents("data/")

# 单独使用分割器
from modules.splitter import split_documents
chunks = split_documents(docs, chunk_size=600, chunk_overlap=120)

# 单独使用检索器
from modules.retriever import similarity_search, mmr_search
results = similarity_search(vectorstore, "考勤制度", k=5)
```

---

## 常见问题

### Q1: 构建索引时提示 `FileNotFoundError`？

确保 `data/` 目录存在且包含文档文件。运行 `python build_index.py --stats` 查看文件统计。

### Q2: 图片文件无法加载？

图片需要 OCR 支持。请确保：
1. 已安装 `pytesseract` 和 `Pillow`：`pip install pytesseract Pillow`
2. 已安装 Tesseract OCR 并加入系统 PATH

### Q3: Embedding 调用报错？

检查 `config/settings.py` 中的 `EMBEDDING_API_KEY` 和 `OPENAI_BASE_URL` 是否正确。

### Q4: LLM 调用报错？

检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL` 是否正确设置。

### Q5: 如何切换 LLM 模型？

修改 `config/settings.py` 中 `LLM_CONFIG["model"]`，支持：
- `qwen-turbo`：速度最快
- `qwen-plus`：效果更好
- `qwen-max`：最强效果

### Q6: 检索结果不准确怎么办？

尝试以下调整：
1. 减小 `chunk_size`（如 300-400）以获得更精细的检索
2. 增大 `k` 值以返回更多候选文档
3. 尝试 MMR 检索模式：修改 `RETRIEVER_CONFIG["search_type"]` 为 `"mmr"`
4. 优化 Prompt 模板，增加更详细的指令

### Q7: Chroma 数据库文件很大？

向量数据会占用磁盘空间，如需清理：
```bash
# 删除向量数据库
rm -rf storage/chroma_db/
# 重新构建
python build_index.py
```

---

## 开发计划

- [ ] Gradio Web UI 界面
- [ ] 多轮对话支持
- [ ] 流式输出（Streaming）
- [ ] 对话历史记忆
- [ ] 多语言问答支持
- [ ] 检索质量评估

---

## 许可证

Internal Use Only - 企业内部使用

---

**技术栈**: LangChain 1.3 | DashScope | Chroma | text-embedding-3-small  
**最后更新**: 2026-07-30
