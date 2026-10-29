"""
LangChain 1.3 - 文件夹操作 Agent（使用 langchain 内置文件工具）
===============================================================
内置工具（来自 langchain-community，由 langchain-classic 转发）：
- ListDirectoryTool : 查看文件夹中有哪些文件
- WriteFileTool     : 新建 / 覆盖写入文件（如 a.txt）
- ReadFileTool      : 读取文件内容
- DeleteFileTool    : 删除文件

所有工具通过 root_dir 锁定在 managed_files 文件夹内，自动阻止路径穿越。

运行：
    python 9.folder_agent.py
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools import (
    DeleteFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)

from env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL

# ============================================================
# 1. 指定受管文件夹（root_dir 锁定，Agent 只能操作这里）
# ============================================================
MANAGED_DIR = Path(__file__).parent / "managed_files"
MANAGED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. 初始化 langchain 内置文件工具（root_dir 强制安全目录）
# ============================================================
ROOT_DIR = str(MANAGED_DIR.resolve())  # 内置工具要求字符串路径
tools = [
    ListDirectoryTool(root_dir=ROOT_DIR),  # 查看文件列表
    WriteFileTool(root_dir=ROOT_DIR),      # 新建 / 覆盖写入
    ReadFileTool(root_dir=ROOT_DIR),       # 读取内容
    DeleteFileTool(root_dir=ROOT_DIR),     # 删除文件
]

# ============================================================
# 3. 创建 Agent
# ============================================================
# 默认使用千问，可切换为 openai（改下面两行即可）
model = init_chat_model(
    model="qwen-turbo",
    model_provider="openai",  # dashscope 兼容 OpenAI 协议
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    temperature=0,
)
# model = init_chat_model(
#     model="gpt-5.6",
#     model_provider="openai",
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
#     temperature=0,
# )

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "你是一个文件夹管理助手，只能操作受管文件夹 managed_files 中的文件。\n"
        "可用工具及用法：\n"
        "1. 用户要求查看文件夹里有哪些文件 / 列出文件时，调用 list_directory，参数 dir_path 传 '.'。\n"
        "2. 用户要求新建或写入文件（如 a.txt）时，调用 write_file，参数 file_path 传文件名，text 传内容。\n"
        "3. 用户要求读取文件内容时，调用 read_file，参数 file_path 传文件名。\n"
        "4. 用户要求删除文件时，调用 file_delete，参数 file_path 传文件名。\n"
        "所有文件操作必须通过工具完成，不要编造结果。\n"
        "如果工具返回错误，请如实告知用户。"
    ),
)


# ============================================================
# 4. 演示 & 交互
# ============================================================
def run(question: str, history: list | None = None) -> tuple[str, list]:
    """执行一轮对话，返回 (回答, 更新后的消息历史)"""
    messages = (history or []) + [{"role": "user", "content": question}]
    response = agent.invoke({"messages": messages})
    return response["messages"][-1].content, response["messages"]


def demo():
    print("=" * 60)
    print("演示 1：查看文件夹中有哪些文件")
    print("=" * 60)
    answer, history = run("请查看这个文件夹里面有哪些文件")
    print(f"\nAgent：{answer}\n")

    print("=" * 60)
    print("演示 2：新增一个文件 a.txt")
    print("=" * 60)
    answer, history = run("请新增一个文件 a.txt，内容写上 hello world", history)
    print(f"\nAgent：{answer}\n")

    print("=" * 60)
    print("演示 3：再次查看文件夹（确认 a.txt 已创建）")
    print("=" * 60)
    answer, history = run("现在文件夹里有哪些文件？", history)
    print(f"\nAgent：{answer}\n")


def interactive():
    print("\n" + "=" * 60)
    print("进入交互模式，可直接对话（输入 exit 退出）")
    print(f"受管文件夹：{MANAGED_DIR}")
    print("=" * 60)
    history = None
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "退出"):
            break
        if not question:
            continue
        try:
            answer, history = run(question, history)
            print(f"\nAgent：{answer}")
        except Exception as e:
            print(f"\n出错：{e}")


if __name__ == "__main__":
    # demo()
    interactive()
