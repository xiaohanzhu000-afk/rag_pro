
import os
from langchain_openai import ChatOpenAI
from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL,DASHSCOPE_API_KEY,DASHSCOPE_BASE_URL,ZHIPU_API_KEY,ZHIPU_BASE_URL
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model

# 创建各类大模型

# openai 模型初始化 方法1.model class
openai_llm = ChatOpenAI(
    model="gpt-5.6",  # 可选 gpt-4o / gpt-4-turbo
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.7,  # 随机性 0~1
    max_tokens=1024
)


# openai 模型初始化 方法2.init_chat_model langchain v1.0.0版本新增的初始化方法 
# 建议使用此方法，因为统一格式
qwen_llm:BaseChatModel = init_chat_model(
    model="qwen-turbo",
    model_provider="openai", # 可选 openai / dashscope ，没有则填写 openai
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

zhipu_llm:BaseChatModel = init_chat_model(
    model="chatglm2-6b",    
    model_provider="zhipu",
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_BASE_URL,
)

res = zhipu_llm.invoke("简单介绍LangChain v1版本和旧版区别")
print(res.content)

