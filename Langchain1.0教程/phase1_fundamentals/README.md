# Phase 1: Fundamentals (基础知识)

LangChain 1.0 基础教程 - 第一阶段

## 学习目标

掌握 LangChain 1.0 的核心概念和基础用法：
- 模型调用和消息系统
- 提示词模板
- 自定义工具
- Agent 创建和执行

## 模块列表

### 01 - Hello LangChain
**学习内容：**
- `init_chat_model` - 统一的模型初始化
- `invoke` 方法 - 三种输入格式
- 环境配置和 API 密钥管理

**关键文件：**
- `main.py` - 7 个基础示例
- `invoke_practice.py` - 实践练习
- `README.md` - 详细教程

### 02 - Prompt Templates
**学习内容：**
- `PromptTemplate` - 文本模板
- `ChatPromptTemplate` - 对话模板
- 变量替换和部分变量
- LCEL 链式调用

**关键文件：**
- `main.py` - 9 个模板示例
- `examples/template_library.py` - 15 个可复用模板
- `README.md` - 模板使用指南

### 03 - Messages
**学习内容：**
- 消息类型：HumanMessage、AIMessage、SystemMessage
- 对话历史管理
- 多轮对话的关键规则

**关键文件：**
- `main.py` - 5 个核心示例
- `test.py` - 对话测试
- `README.md` - 重点讲解

**核心难点：**
每次调用必须传入完整历史！

```python
conversation = []
conversation.append({"role": "user", "content": "我叫张三"})
r1 = model.invoke(conversation)
conversation.append({"role": "assistant", "content": r1.content})
conversation.append({"role": "user", "content": "我叫什么？"})
r2 = model.invoke(conversation)  # AI 能记住
```

### 04 - Custom Tools
**学习内容：**
- `@tool` 装饰器 - LangChain 1.0 推荐方式
- docstring 的重要性（AI 依赖它理解工具）
- 参数类型注解
- 可选参数使用 `Optional[type]`

**关键文件：**
- `main.py` - 6 个工具示例
- `tools/weather.py` - 天气工具
- `tools/calculator.py` - 计算器（多参数）
- `tools/web_search.py` - 搜索（可选参数）
- `README.md` - 工具开发指南

**最佳实践：**
```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """
    清晰的工具描述（AI 读这个！）

    参数:
        param: 参数说明

    返回:
        返回值说明
    """
    # 实现
    return "结果字符串"
```

### 05 - Simple Agent
**学习内容：**
- `create_agent` - LangChain 1.0 统一 API
- Agent = 模型 + 工具 + 自动决策
- Agent 如何选择工具
- 多轮对话处理

**关键文件：**
- `main.py` - 6 个 Agent 示例
- `test_simple.py` - 简单测试
- `README.md` - Agent 使用指南

**关键语法：**
```python
from langchain.agents import create_agent

agent = create_agent(
    model=init_chat_model("groq:llama-3.3-70b-versatile"),
    tools=[tool1, tool2],
    system_prompt="Agent 的行为指令"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "问题"}]
})

final_answer = response['messages'][-1].content
```

### 06 - Agent Loop
**学习内容：**
- Agent 执行循环详解
- 消息历史分析
- 流式输出 `.stream()`
- 调试和监控技巧

**关键文件：**
- `main.py` - 6 个执行循环示例
- `test.py` - 测试脚本
- `README.md` - 循环详解

**执行流程：**
```
用户问题 (HumanMessage)
    ↓
AI 决定 (AIMessage with tool_calls)
    ↓
执行工具 (ToolMessage)
    ↓
最终答案 (AIMessage)
```

## 快速开始

### 1. 环境搭建

```bash
# 创建虚拟环境
python -m venv venv

# 激活（Windows）
venv\Scripts\activate

# 安装依赖
pip install langchain langchain-groq python-dotenv
```

### 2. 配置 API 密钥

创建 `.env` 文件：
```
GROQ_API_KEY=your_key_here
```

### 3. 运行示例

```bash
# 运行特定模块
cd phase1_fundamentals/01_hello_langchain
python main.py

# 或者
python phase1_fundamentals/02_prompt_templates/main.py
```

## 核心知识点总结

### 1. LangChain 1.0 架构
- 构建在 LangGraph 运行时之上
- 统一的 `init_chat_model` 和 `create_agent` API
- 中间件架构（后续学习）

### 2. 模型调用
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 三种输入格式
model.invoke("简单文本")
model.invoke([{"role": "user", "content": "字典格式"}])
model.invoke([HumanMessage("消息对象")])
```

### 3. 提示词模板
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("user", "{input}")
])

chain = template | model
result = chain.invoke({"role": "助手", "input": "问题"})
```

### 4. 对话历史
```python
# 关键：每次调用传完整历史
conversation = []
conversation.append(user_msg)
response = model.invoke(conversation)
conversation.append({"role": "assistant", "content": response.content})
```

### 5. 创建工具
```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具描述 - AI 读这个！"""
    return "result"
```

### 6. 创建 Agent
```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[tool1, tool2],
    system_prompt="指令"
)

response = agent.invoke({"messages": [...]})
```

### 7. Agent 执行循环
```python
# 查看完整历史
for msg in response['messages']:
    print(msg)

# 获取最终答案
final = response['messages'][-1].content

# 流式输出
for chunk in agent.stream(input):
    # 实时处理
```

## 重要概念

### LCEL (LangChain Expression Language)
使用 `|` 操作符链接组件：
```python
chain = prompt | model | output_parser
result = chain.invoke(input)
```

### 消息类型
- **HumanMessage** - 用户输入
- **AIMessage** - AI 输出
- **SystemMessage** - 系统指令
- **ToolMessage** - 工具结果

### Agent 工作原理
1. 接收用户问题
2. 分析是否需要工具
3. 如果需要，调用工具
4. 基于工具结果生成答案
5. 返回最终答案

## 常见问题

### 1. API 密钥问题
确保 `.env` 文件中的 API 密钥正确：
```bash
GROQ_API_KEY=gsk_...
```

### 2. 导入错误
LangChain 1.0 导入路径：
```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
```

### 3. Agent 不调用工具
- 检查工具的 docstring 是否清晰
- 确保问题明确需要该工具
- 工具参数类型注解完整

### 4. 对话不记忆
必须传入完整历史：
```python
# ❌ 错误
model.invoke("你记得我的名字吗？")  # AI 不记得

# ✅ 正确
conversation = [previous_messages...] + [new_message]
model.invoke(conversation)
```

## 学习建议

1. **按顺序学习**
   - 01 → 02 → 03 → 04 → 05 → 06
   - 每个模块都有实践练习

2. **动手实践**
   - 运行每个示例
   - 修改参数观察结果
   - 完成练习题

3. **理解核心**
   - invoke 方法的三种输入
   - 对话历史管理
   - 工具的 docstring
   - Agent 执行循环

4. **查看源码**
   - 理解每个示例的实现
   - 对比不同方法的差异

## 下一步

### Phase 2: Intermediate (中级特性)

**即将学习：**
- **Module 07-09**: 内存和状态管理
  - InMemorySaver
  - 上下文管理
  - Checkpointing 持久化

- **Module 10-12**: 中间件架构
  - 自定义中间件
  - 可观测性
  - 防护栏（Guardrails）

- **Module 13-15**: 结构化输出
  - Pydantic 模型
  - 验证和重试
  - 工具与结构化输出结合

## 资源链接

- **官方文档**: https://docs.langchain.com/oss/python/langchain/
- **迁移指南**: https://docs.langchain.com/oss/python/migrate/langchain-v1

## 贡献

如有问题或建议，请提 Issue 或 PR。

---

**恭喜完成阶段一！** 🎉

你已经掌握了 LangChain 1.0 的核心基础，可以开始构建实际的 AI 应用了！
