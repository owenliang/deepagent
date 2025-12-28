"""修复消息历史中悬空工具调用的中间件

核心作用：防止 AI 消息中的 tool_call 没有对应的 ToolMessage 响应，
导致后续模型调用出错。自动为悬空的工具调用补充取消消息。

Middleware to patch dangling tool calls in the messages history.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """修复消息历史中悬空工具调用的中间件
    
    问题场景：
    - AI 消息包含 tool_calls，但在执行前被新消息打断
    - 导致没有对应的 ToolMessage 响应
    - 后续模型调用会因为悬空的 tool_call 而出错
    
    解决方案：
    - 检测所有 AI 消息中的 tool_calls
    - 为没有对应 ToolMessage 的 tool_call 自动补充取消消息
    
    Middleware to patch dangling tool calls in the messages history.
    """

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Agent 运行前修复悬空的工具调用
        
        返回：包含修复后消息列表的字典，使用 Overwrite 完全替换原消息列表
        """
        messages = state["messages"]
        if not messages or len(messages) == 0:
            return None  # 空消息列表，无需处理

        patched_messages = []
        # 遍历所有消息，检测并修复悬空的工具调用
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            # 检查是否为包含工具调用的 AI 消息
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 在当前消息之后查找对应的 ToolMessage
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # 发现悬空的工具调用（没有对应的 ToolMessage），补充取消消息
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        # 使用 Overwrite 完全替换消息列表（而不是 add_messages 合并）
        return {"messages": Overwrite(patched_messages)}
