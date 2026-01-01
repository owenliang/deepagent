import os
from datetime import datetime
from deepagents import create_deep_agent
from openai import OpenAI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver  # 导入检查点工具
from deepagents.backends import StoreBackend

def internet_search_tool(query: str):
    """Run a web search"""
    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus", 
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ],
        extra_body={
            "enable_search": True
        }
    )
    return completion.choices[0].message.content

# System prompt to steer the agent to be an expert researcher
today = datetime.now().strftime("%Y年%m月%d日")
research_instructions = f"""你是一个智能助手。你的任务是帮助用户完成各种任务。

你可以使用互联网搜索工具来获取信息。
## `internet_search`
使用此工具对给定查询进行互联网搜索。你可以指定返回结果的最大数量、主题以及是否包含原始内容。

今天的日期是：{today}
"""

# Create the deep agent with memory
model = init_chat_model(model="qwen-max",model_provider='openai',api_key=os.getenv('DASHSCOPE_API_KEY'),base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
checkpointer = MemorySaver()  # 创建内存检查点，自动保存历史

agent = create_deep_agent( # state：thread会话级的状态
    tools=[internet_search_tool],
    system_prompt=research_instructions,
    model=model,
    checkpointer=checkpointer,  # 添加检查点，启用自动记忆
    interrupt_on={'internet_search_tool':False}
)

# 多轮对话循环（使用 Checkpointer 自动记忆）
printed_msg_ids = set()  # 跟踪已打印的消息ID
thread_id = "user_session_001"  # 会话 ID，区分不同用户/会话
config = {"configurable": {"thread_id": thread_id}, "metastore": {'assistant_id': 'owenliang'}}  # 配置会话

print("开始对话（输入 'exit' 退出）：")
while True:
    user_input = input("\nHUMAN: ").strip()
    if user_input.lower() == 'exit':
        break
    
    # 使用 values 模式多次返回完整状态，这里按 message.id 去重，并按类型分类打印
    pending_resume = None
    while True:
        if pending_resume is None:
            request = {"messages": [{"role": "user", "content": user_input}]}
        else:
            from langgraph.types import Command as _Command

            request = _Command(resume=pending_resume)
            pending_resume = None

        for item in agent.stream(
            request,
            config=config,
            stream_mode="values",
        ):
            state = item[0] if isinstance(item, tuple) and len(item) == 2 else item

            # 先检查是否触发了 Human-In-The-Loop 中断
            if isinstance(state, dict) and "__interrupt__" in state:
                interrupts = state["__interrupt__"] or []
                if interrupts:
                    hitl_payload = interrupts[0].value
                    action_requests = hitl_payload.get("action_requests", [])

                    print("\n=== 需要人工审批的工具调用 ===")
                    decisions: list[dict[str, str]] = []
                    for idx, ar in enumerate(action_requests):
                        name = ar.get("name")
                        args = ar.get("args")
                        print(f"[{idx}] 工具 {name} 参数: {args}")
                        while True:
                            choice = input("  决策 (a=approve, r=reject): ").strip().lower()
                            if choice in ("a", "r"):
                                break
                        decisions.append({"type": "approve" if choice == "a" else "reject"})

                    # 下一轮调用改为 resume，同一轮用户回合继续往下跑
                    pending_resume = {"decisions": decisions}
                    break

            # 兼容 dict state 和 AgentState dataclass
            messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
            for msg in messages:
                msg_id = getattr(msg, "id", None)
                if msg_id is not None and msg_id in printed_msg_ids:
                    continue
                if msg_id is not None:
                    printed_msg_ids.add(msg_id)

                msg_type = getattr(msg, "type", None)

                if msg_type == "human":
                    # 用户输入已经在命令行里，不再重复打印
                    continue

                if msg_type == "ai":
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        # 这是发起工具调用的 AI 消息（TOOL CALL）
                        for tc in tool_calls:
                            tool_name = tc.get("name")
                            args = tc.get("args")
                            print(f"TOOL CALL [{tool_name}]: {args}")
                    # 如果 AI 同时带有自然语言内容，也一起打印
                    if getattr(msg, "content", None):
                        print(f"AI: {msg.content}")
                    continue

                if msg_type == "tool":
                    # 工具执行结果（TOOL RESPONSE）
                    tool_name = getattr(msg, "name", None) or "tool"
                    print(f"TOOL RESPONSE [{tool_name}]: {msg.content}")
                    continue

                # 兜底：其它类型直接打印出来便于调试
                print(f"[{msg_type}]: {getattr(msg, 'content', None)}")

        # 如果没有新的中断需要 resume，则整轮结束，等待下一轮用户输入
        if pending_resume is None:
            break
