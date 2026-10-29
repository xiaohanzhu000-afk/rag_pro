# 导入必要的模块
from my_llm import openai_llm
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 定义工具装饰器：将普通函数转换为 LangChain 可识别的工具
@tool
def get_weather(location: str) -> str:
    """根据地点查询天气信息

    Args:
        location: 需要查询天气的城市地点
    """
    # 返回模拟的天气信息
    return f"{location} 今天天气晴朗，气温25℃。"

@tool
def get_movie(title: str) -> str:
    """根据电影名称查询电影信息

    Args:
        title: 需要查询的电影名称
    """
    # 返回模拟的电影信息
    return f"{title} 是一部评分9.5的科幻电影，由张艺谋、王宝强主演。"


# 将工具绑定到 LLM 模型上，让模型知道可以使用这些工具
model_width_tool = openai_llm.bind_tools([get_weather, get_movie])


# 准备消息列表
messages = []
# 创建用户消息
humanMessage = HumanMessage(content="十面埋伏这部电影怎么样?")
messages.append(humanMessage)

# 调用模型，模型会决定是否需要调用工具
# 注意：这里只是返回工具调用的信息，不会真正执行工具
resp = model_width_tool.invoke(messages)

# 将模型的响应添加到消息列表中
messages.append(resp)

# 如果模型决定调用工具，则执行工具调用
if resp.tool_calls:
    print(f"模型决定调用工具：{resp.tool_calls}")
    for tool_call in resp.tool_calls:
        # 根据工具名称调用对应的工具函数
        if tool_call['name'] == 'get_weather':
            # 执行天气查询工具
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)
        elif tool_call['name'] == 'get_movie':
            # 执行电影查询工具
            tool_result = get_movie.invoke(tool_call)
            messages.append(tool_result)

# 再次调用模型，传入工具执行结果，让模型生成最终回答
final_resp = model_width_tool.invoke(messages)
print(final_resp)