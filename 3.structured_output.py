"""
结构化输出（Structured Output）示例
=====================================
演示三种让 LLM 返回结构化数据的方式：
  1. Pydantic 模型（推荐）：类型安全，支持嵌套和验证
  2. TypedDict：轻量级，适合简单结构
  3. JSON Schema：动态定义，适合外部配置或运行时生成

核心 API：openai_llm.with_structured_output(schema)
  将 schema 注入模型调用，模型会按照 schema 结构返回解析好的数据。
"""

from langchain.agents import create_agent
from langchain.tools import tool
from my_llm import openai_llm  # 自定义封装的 LLM 实例
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio
from pydantic import BaseModel, Field  # 方式一：Pydantic 模型定义
from typing import TypedDict, Annotated  # 方式二：TypedDict 类型定义

# ============================================================================
# 方式一：Pydantic 模型（推荐）
#   - 使用 BaseModel 定义结构，Field 提供字段描述（description 会传给 LLM 作为提示）
#   - 支持类型校验、嵌套对象、默认值等
#   - 输出是 Pydantic 实例，可以用 .dict() / .model_dump() 序列化
# ============================================================================
# class Actor(BaseModel):
#     """演员信息"""
#     name: str = Field(..., description="演员名称")
#     role: str = Field(..., description="角色")
#
# class Movie(BaseModel):
#     """电影信息"""
#     title: str = Field(..., description="电影名称")
#     director: str = Field(..., description="导演")
#     year: int = Field(..., description="上映年份")
#     rating: float = Field(..., description="评分")
#     casts: list[Actor] = Field(..., description="主演")  # 嵌套的 Pydantic 模型
#
# # 绑定 schema 到 LLM，返回的 resp 是 Movie 实例
# model_with_structured_output = openai_llm.with_structured_output(Movie)
# resp = model_with_structured_output.invoke("介绍下泰坦尼克号这部电影")
# print(resp)

# ============================================================================
# 方式二：TypedDict
#   - 更轻量，不需要 Pydantic 依赖，但缺乏运行时校验
#   - 通过 Annotated[str, "描述"] 方式给每个字段添加说明
#   - 输出是普通 dict，而非对象实例
# ============================================================================
# class Actor(TypedDict):
#     """演员信息 TypedDict"""
#     name: Annotated[str, "演员名称"]
#     role: Annotated[str, "角色"]
#
# class Movie(TypedDict):
#     """电影信息 TypedDict"""
#     title: Annotated[str, "电影名称"]
#     director: Annotated[str, "导演"]
#     year: Annotated[int, "上映年份"]
#     rating: Annotated[float, "评分"]
#     casts: Annotated[list[Actor], "主演"]
#
# model_with_structured_output = openai_llm.with_structured_output(Movie)
# resp = model_with_structured_output.invoke("介绍下泰坦尼克号这部电影")
# print(resp)

# ============================================================================
# 方式三：JSON Schema（适用于动态 / 外部配置场景）
#   - 直接传入原生 JSON Schema 字典，无需定义 Python 类
#   - 适合从配置文件、数据库或 API 动态加载 schema 的场景
#   - include_raw=True：返回完整响应（含 parsing_error 等元信息）
#   - 注意：schema 的 "title" 只能是 ASCII 字符（如 "Movie"），不能用中文
# ============================================================================
json_schema = {
    "title": "Movie",          # 模式名称，只能是 ASCII 字符，不能用中文
    "type": "object",          # 根类型为对象
    "properties": {            # 定义字段
        "title": {
            "type": "string",
            "description": "电影名称"
        },
        "director": {
            "type": "string",
            "description": "导演"
        },
        "year": {
            "type": "integer",  # 整数类型
            "description": "上映年份"
        },
        "rating": {
            "type": "number",   # 浮点数类型
            "description": "评分"
        }
    },
    "required": ["title", "director", "year", "rating"]  # 必填字段
}

# 绑定 JSON Schema 到 LLM
# include_raw=True：返回包含 raw/tool_use/parsed/parsing_error 的完整字典
model_with_structured_output = openai_llm.with_structured_output(json_schema, include_raw=True)
resp = model_with_structured_output.invoke("介绍下泰坦尼克号这部电影")
print(resp)

