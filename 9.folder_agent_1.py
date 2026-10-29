"""
LangChain 1.3 - 文件夹操作 Agent
=================================
一个可以操作指定文件夹内容的 Agent，支持：
- 查看文件夹中有哪些文件
- 新建文件（如 a.txt）
- 读取文件内容
- 覆盖写入文件
- 删除文件

运行：
    python 9.folder_agent.py
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, OPENAI_API_KEY, OPENAI_BASE_URL

# ============================================================
# 1. 指定受管文件夹（Agent 只能操作这个文件夹）
# ============================================================
MANAGED_DIR = Path(__file__).parent / "managed_files" # 受管文件夹
MANAGED_DIR.mkdir(parents=True, exist_ok=True) # 确保文件夹存在


def _resolve(filename: str) -> Path:
    """把文件名安全地解析到受管文件夹内，防止路径穿越"""
    path = (MANAGED_DIR / filename).resolve() # 解析为绝对路径
    if not path.is_relative_to(MANAGED_DIR.resolve()):
        raise ValueError(f"非法路径，只能操作文件夹 {MANAGED_DIR} 内的文件: {filename}")
    return path


# ============================================================
# 2. 定义文件夹操作工具
# ============================================================
@tool
def list_files() -> str:
    """查看受管文件夹中有哪些文件，返回文件名列表（含大小，单位为字节）。"""
    files = sorted(MANAGED_DIR.iterdir(), key=lambda p: p.name.lower())
    if not files:
        return "文件夹为空，没有任何文件。"
    lines = [f"- {p.name}（{p.stat().st_size} 字节）" for p in files if p.is_file()]
    return "文件夹中的文件：\n" + "\n".join(lines) if lines else "文件夹为空，没有任何文件。"


@tool
def create_file(filename: str, content: str = "") -> str:
    """在受管文件夹中新建一个文件。

    Args:
        filename: 文件名，例如 "a.txt"、"notes/hello.txt"
        content: 可选，文件内容（默认为空）
    """
    path = _resolve(filename)
    if path.exists():
        return f"文件 {filename} 已存在，请使用 write_file 覆盖写入。"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已创建文件 {filename}（{len(content)} 字符）。"


@tool
def read_file(filename: str) -> str:
    """读取受管文件夹中指定文件的内容。

    Args:
        filename: 文件名
    """
    path = _resolve(filename)
    if not path.is_file():
        return f"文件 {filename} 不存在，请先用 create_file 创建或 list_files 查看。"
    return path.read_text(encoding="utf-8")


@tool
def write_file(filename: str, content: str) -> str:
    """把内容写入受管文件夹中的文件（覆盖已有内容，不存在则创建）。

    Args:
        filename: 文件名
        content: 要写入的内容
    """
    path = _resolve(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"已写入文件 {filename}（{len(content)} 字符）。"


@tool
def delete_file(filename: str) -> str:
    """删除受管文件夹中的文件。

    Args:
        filename: 文件名
    """
    path = _resolve(filename)
    if not path.is_file():
        return f"文件 {filename} 不存在，无需删除。"
    path.unlink()
    return f"已删除文件 {filename}。"


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
    tools=[list_files, create_file, read_file, write_file, delete_file],
    system_prompt=(
        "你是一个文件夹管理助手，只能操作受管文件夹 managed_files 中的文件。\n"
        "规则：\n"
        "1. 用户要求查看文件时，使用 list_files 工具。\n"
        "2. 用户要求新建文件时，使用 create_file 工具。\n"
        "3. 用户要求读取内容时，使用 read_file 工具。\n"
        "4. 用户要求修改/覆盖文件时，使用 write_file 工具。\n"
        "5. 用户要求删除文件时，使用 delete_file 工具。\n"
        "所有文件操作必须通过工具完成，不要编造结果。"
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
