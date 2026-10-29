import gradio as gr

# 业务函数
def hello(name):
    return f"你好，{name}！"

# 搭建界面
demo = gr.Interface(
    fn=hello,          # 绑定执行函数
    inputs="text",     # 输入组件：文本框
    outputs="text",    # 输出组件：文本框
    title="Gradio示例",
    description="输入名字，返回问候语"
)

# 启动服务
if __name__ == "__main__":
    demo.launch()