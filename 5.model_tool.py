from my_llm import openai_llm
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# 创建工具
@tool
def get_weather(location: str) -> str:
    """根据地点查询天气
    Args:
        location: 需要查询天气的城市地点
    """
    return f"{location} 今天天气晴朗，气温25℃。"

# 工具告诉模型
model_width_tool = openai_llm.bind_tools([get_weather])


# 准备messages
messages = []
humanMessage = HumanMessage(content="北京天气怎么样?")
messages.append(humanMessage)

# 模型调用工具返回接口，只知道去调用工具，但不会真正调用
resp = model_width_tool.invoke(messages)
messages.append(resp)

if resp.tool_calls:
    for tool_call in resp.tool_calls:
        print(f"调用工具：{tool_call}")
        if tool_call['name'] == 'get_weather':
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

final_resp = model_width_tool.invoke(messages)
print(final_resp)