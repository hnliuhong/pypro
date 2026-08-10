from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage


# --- 1. 定义工人 (Worker) 节点 ---
def researcher(state: MessagesState):
    print("--> 运行：Researcher")
    return {"messages": [AIMessage(content="[Researcher]: 已搜集到最新的 AI 行业数据。")]}


def coder(state: MessagesState):
    print("--> 运行：Coder")
    return {"messages": [AIMessage(content="[Coder]: 已根据数据编写好 Python 绘图代码。")]}


# --- 2. 定义 Supervisor 节点逻辑（主管） ---
# 教学中我们用 Python 代码模拟 LLM 的决策逻辑：
# 检查消息历史，如果没有搜集数据就找 Researcher，有数据无代码就找 Coder，都有了就 END
def supervisor_decision(state: MessagesState) -> Literal["researcher", "coder", "__end__"]:
    messages = state["messages"]
    contents = [m.content for m in messages]

    has_research = any("[Researcher]" in c for c in contents)
    has_code = any("[Coder]" in c for c in contents)

    if not has_research:
        return "researcher"
    elif not has_code:
        return "coder"
    else:
        return END  # 任务全部完成


# --- 3. 构建图 ---
builder = StateGraph(MessagesState)

# 添加工人节点
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

# 条件边：从 START 开始，由 supervisor_decision 决定第一步去哪
builder.add_conditional_edges(START, supervisor_decision)

# 工人处理完后，不直接 END，而是通过 supervisor_decision 决定下一步去哪（形成协同环路）
builder.add_conditional_edges("researcher", supervisor_decision)
builder.add_conditional_edges("coder", supervisor_decision)

graph = builder.compile()

# --- 运行测试 ---
result = graph.invoke({"messages": [{"role": "user", "content": "请分析 AI 行业数据并画图"}]})

print("\n--- 最终消息列表 ---")
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")