from langchain.agents import create_agent
from langchain.tools import tool
from my_llm import openai_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio

# 1.invoke
# 单条调用
resp = openai_llm.invoke("自我介绍一下。")

# 字典格式消息列表
conversations= [
    {"role": "system", "content": "你是一个翻译助手，能把汉语翻译成英语。"},
    {"role": "user", "content": "翻译：我喜欢编程"},
    {"role": "assistant", "content": "I like programming."},
    {"role": "user", "content": "翻译：我是一个中国人。"},
]
resp = openai_llm.invoke(conversations) # 返回 I am Chinese.

# 消息对象格式的消息列表
conversations= [
    SystemMessage(content="你是一个翻译助手，能把汉语翻译成英语。"),
    HumanMessage(content="翻译：我喜欢编程"),
    AIMessage(content="I like programming."),
    HumanMessage(content="翻译：我喜欢你")
]
resp = openai_llm.invoke(conversations) # I like you.
print(resp.content)

# 2.stream invoke 流式调用 返回是迭代器
resp = openai_llm.stream("自我介绍。")
for chunk in resp:
    print(chunk.content, end="", flush=True) # end="" 避免换行 flush=True 刷新输出

# 3.批量调用模型
# 全部搞定，一次返回列表
resp = openai_llm.batch(["自我介绍。", "自我安慰。", "自我赞扬。"]) 
for item in resp:
    print(item.content)
# 优先返回先回答的模型结果 deepseek 有效
resp = openai_llm.batch_as_complated(["自我介绍。", "自我安慰。", "自我赞扬。"]) 
for item in resp:
    print(item)

# 4.异步调用模型 等待返回结果时，可以做其他事情
async def invoke_async():
    resp = await openai_llm.ainvoke("自我介绍。")
    print(resp.content)
async def invoke_async1():
    resp = await openai_llm.ainvoke("自我陶醉")
    print(resp.content)
async def main():
    await asyncio.gather(invoke_async(), invoke_async1())


asyncio.run(main())