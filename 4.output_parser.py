from langchain.tools import tool
from my_llm import openai_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import asyncio
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated

# 1. 定义Pydantic输出模型
class Movie(BaseModel):
    title: str = Field(..., description="电影名称")
    director: str = Field(..., description="导演")
    year: int = Field(..., description="上映年份")

# 2. 生成Pydantic解析器
parser = PydanticOutputParser(pydantic_object=Movie)

# 3. 修正拼写错误 from_temaplte → from_template
# 把格式要求注入Prompt，让大模型严格按格式输出
prompt = ChatPromptTemplate.from_template("""
回答用户问题，提取对应电影信息：
问题：{question}

{format_instructions}
""")

# 4. 拼接链：提示词 → LLM → 结构化解析
chain = prompt.partial(format_instructions=parser.get_format_instructions()) | openai_llm | parser

# 5. 调用执行
if __name__ == "__main__":
    resp = chain.invoke({"question": "介绍《泰坦尼克号》"})
    print(resp)
    # 可以直接取字段
    print(f"电影名：{resp.title}，导演：{resp.director}，上映年份：{resp.year}")