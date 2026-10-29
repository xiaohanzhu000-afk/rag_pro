
from langchain.agents import create_agent
from langchain.tools import tool
from my_llm import openai_llm

@tool
def get_weather(location: str) -> str:
    """Get the weather in a given location."""
    
    return f"It's sunny in {location}."

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