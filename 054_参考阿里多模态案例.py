import os
from textwrap import dedent
from langchain.agents import create_agent
from langgraph.graph import END, START, MessagesState, StateGraph

# ==========================================
# 1. 设置想要讨论的轮数 N
# ==========================================
TOTAL_ROUNDS = 3  # 👈 在这里自由修改想要讨论的轮数 N (例如: 3, 5, 10)

# ==========================================
# 2. 定义专家 Agent 角色 Prompt
# ==========================================
DESIGNER_LI_PROMPT = """你是李老师，经验丰富的教学设计师。在讨论中，专注于：
1. 定义每个模块清晰的学习目标。
2. 确保知识点由浅入深，循序渐进。
3. 提出互动性的练习和项目来巩固学习效果。"""

SCIENTIST_WANG_PROMPT = """你是王工，资深数据科学家。在讨论中，专注于：
1. 提供最核心、最常用的 Pandas 知识点。
2. 设计源于真实工作场景的案例和数据集。
3. 编写简洁、规范、易于理解的代码示例。"""

WRITER_ZHANG_PROMPT = """你是小张，内容编写者。在讨论中，专注于：
1. 用通俗易懂的语言和比喻来解释复杂概念。
2. 设计真实性高的案例场景和模块标题。
3. 确保课程的整体基调是鼓励性和启发性的。"""

SECRETARY_PROMPT = dedent("""
你是专业的会议秘书。请总结之前的全部团队讨论记录，以清晰的 Markdown 格式输出“Pandas 入门课程”大纲初稿。
需包含：模块标题、学习目标、核心概念、核心案例、代码示例、课后练习。
""")

# ==========================================
# 3. 实例化 4 个独立的 Agent
# ==========================================
designer_agent = create_agent(
    model="deepseek-v4-flash", system_prompt=DESIGNER_LI_PROMPT
)
scientist_agent = create_agent(
    model="deepseek-v4-flash", system_prompt=SCIENTIST_WANG_PROMPT
)
writer_agent = create_agent(
    model="deepseek-v4-flash", system_prompt=WRITER_ZHANG_PROMPT
)
secretary_agent = create_agent(
    model="deepseek-v4-flash", system_prompt=SECRETARY_PROMPT
)


# ==========================================
# 4. 定义图节点（关键：给返回的消息打上 name 标签）
# ==========================================
def designer_node(state: MessagesState):
    print("\n💬 [李老师 (教学设计师)]: 正在发言...")
    res = designer_agent.invoke(state)
    last_msg = res["messages"][-1]
    last_msg.name = "designer_node"  # 标记来源
    return {"messages": [last_msg]}


def scientist_node(state: MessagesState):
    print("\n💬 [王工 (数据科学家)]: 正在发言...")
    res = scientist_agent.invoke(state)
    last_msg = res["messages"][-1]
    last_msg.name = "scientist_node"  # 标记来源
    return {"messages": [last_msg]}


def writer_node(state: MessagesState):
    print("\n💬 [小张 (内容编写者)]: 正在发言...")
    res = writer_agent.invoke(state)
    last_msg = res["messages"][-1]
    last_msg.name = "writer_node"  # 一轮研讨结束的关键标记！
    return {"messages": [last_msg]}


def secretary_node(state: MessagesState):
    print("\n📝 [会议秘书]: 正在总结整理最终大纲...")
    res = secretary_agent.invoke(state)
    last_msg = res["messages"][-1]
    last_msg.name = "secretary_node"
    return {"messages": [last_msg]}


# ==========================================
# 5. 控制多轮循环的路由逻辑
# ==========================================
def check_discussion_rounds(state: MessagesState):
    # 统计小张（writer_node）发言的次数，即跑完的完整轮数
    completed_rounds = sum(
        1
        for msg in state["messages"]
        if getattr(msg, "name", None) == "writer_node"
    )

    print(f"   [进度反馈]: 已完成第 {completed_rounds} / {TOTAL_ROUNDS} 轮讨论")

    # 如果完成的轮数小于设置的 N 轮，继续回到李老师开始下一轮
    if completed_rounds < TOTAL_ROUNDS:
        print(f"🔄 --- 进入第 {completed_rounds + 1} 轮讨论 ---")
        return "designer_node"
    else:
        # 达到 N 轮，跳出循环，交给会议秘书
        print(
            f"\n✅ --- 已完成满 {TOTAL_ROUNDS} 轮讨论，提交给会议秘书整理大纲 ---"
        )
        return "secretary_node"


# ==========================================
# 6. 构建并编译 LangGraph 流程图
# ==========================================
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node("designer_node", designer_node)
builder.add_node("scientist_node", scientist_node)
builder.add_node("writer_node", writer_node)
builder.add_node("secretary_node", secretary_node)

# 设置基础流转方向
builder.add_edge(START, "designer_node")
builder.add_edge("designer_node", "scientist_node")
builder.add_edge("scientist_node", "writer_node")

# 研讨闭环条件边：依据 N 轮计数选择【循环】还是【退出总结】
builder.add_conditional_edges(
    "writer_node",
    check_discussion_rounds,
    {"designer_node": "designer_node", "secretary_node": "secretary_node"},
)

builder.add_edge("secretary_node", END)

# 编译生成 Graph
graph = builder.compile()

# 打印 ASCII 结构拓扑图
print("=== 多 Agent 共创研讨模式拓扑图 ===")
graph.get_graph().print_ascii()

# ==========================================
# 7. 执行测试
# ==========================================
announcement = (
    "团队好，我们今天的目标是共同协作，为“Pandas 数据分析入门”课程制定一个完整的、吸引人的**课程大纲和核心案例**。"
    "请大家集思广益，从教学设计师李老师开始，提出你的第一轮建议。"
)

print(f"\n会议开场：\n{announcement}\n" + "=" * 50)

final_result = graph.invoke(
    {"messages": [{"role": "user", "content": announcement}]}
)

print("\n" + "=" * 50)
print(f" 🎉 最终团队成果（经历 {TOTAL_ROUNDS} 轮研讨后生成的课程大纲）：")
print("=" * 50 + "\n")
print(final_result["messages"][-1].content)