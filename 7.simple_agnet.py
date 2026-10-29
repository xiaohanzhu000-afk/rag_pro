from my_llm import openai_llm
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent# 创建工具
@tool
def get_weather(location: str) -> str:
    """根据地点查询天气
    Args:
        location: 需要查询天气的城市地点
    """
    return f"{location} 今天天气晴朗，气温25℃。"

agent = create_agent(
    model=openai_llm,
    tools=[get_weather],
    system_prompt="你是一个天气查询助手，用户会向你询问天气情况，你可以调用工具获取天气信息。",
)

resp = agent.invoke({
    "messages": [
        {"role": "user", "content": "今天北京的天气如何？"},
    ]
})

print(resp)