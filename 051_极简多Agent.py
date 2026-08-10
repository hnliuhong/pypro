from typing import Literal, TypedDict
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph

# ==========================================
# 1. 定义全局 State (状态驱动)
# ==========================================
class PoemState(TypedDict):
    topic: str         # 诗歌主题
    poem: str          # 生成的诗歌内容
    status: str        # 状态："approved" 或 "rejected"
    feedback: str      # 评估反馈
    iterations: int    # 记录迭代循环次数


# 初始化模型
llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0.7)


# ==========================================
# 2. 定义节点 (Nodes)
# ==========================================

def generator_node(state: PoemState):
    """【生成节点】：负责生成或根据反馈改进诗歌"""
    topic = state["topic"]
    feedback = state.get("feedback", "")
    iterations = state.get("iterations", 0)

    print(f"\n✍️ [Generator 节点] 第 {iterations + 1} 次尝试生成...")

    if feedback:
        prompt = f"请重新写一首关于【{topic}】的五言绝句或短诗。注意上次被拒绝的原因：{feedback}"
    else:
        prompt = f"请写一首关于【{topic}】的极简短诗。"

    response = llm.invoke(prompt)

    # 只需要返回要更新的字段（增量更新）
    return {
        "poem": response.content.strip(),
        "iterations": iterations + 1,
        "feedback": ""  # 准备进入新一轮评估，清空旧反馈
    }


def evaluator_node(state: PoemState):
    """【评估节点】：评估诗歌是否符合要求（此处做硬性规则：字数不能超过 15 字）"""
    poem = state["poem"]
    poem_length = len(poem.replace(" ", "").replace("\n", "").replace("，", "").replace("。", ""))

    print(f"🔍 [Evaluator 节点] 评估诗歌... 当前字数(不含标点): {poem_length} 字")
    print(f"   诗歌内容: \"{poem}\"")

    # 规则判断：字数小于等于 15 字才算通过
    if poem_length <= 15:
        print("   结论: ✅ 足够精简，通过！")
        return {"status": "approved"}
    else:
        print("   结论: ❌ 字数太长，退回重写！")
        return {
            "status": "rejected",
            "feedback": f"你的诗有 {poem_length} 个字，太长了！请严格控制在 15 个字以内，越短越好。"
        }


# ==========================================
# 3. 定义条件路由 (Cycle & Control Flow)
# ==========================================

def should_continue(state: PoemState) -> Literal["generator", END]:
    """【控制流】：根据评估结果决定是【形成闭环循环】还是【退出图】"""
    if state["status"] == "approved":
        return END  # 满足条件，走向结束
    else:
        return "generator"  # 不满足条件，回到 generator 形成 Cycle（循环）


# ==========================================
# 4. 构建与编译图 (Build Graph)
# ==========================================

builder = StateGraph(PoemState)

# 添加节点
builder.add_node("generator", generator_node)
builder.add_node("evaluator", evaluator_node)

# 添加边：START -> generator -> evaluator
builder.add_edge(START, "generator")
builder.add_edge("generator", "evaluator")

# 添加条件边：evaluator -> (generator 或 END)
builder.add_conditional_edges("evaluator", should_continue)
graph = builder.compile()
# 打印 ASCII 图，观察 generator <-> evaluator 之间的闭环结构
# graph.get_graph().print_ascii()
# 传统的方式无法打印循环,建议生成mermai代码,然后交给豆包显示即可
mermaid_code = graph.get_graph().draw_mermaid()
print(mermaid_code)

# ==========================================
# 5. 运行测试
# ==========================================

initial_input = {"topic": "夏天的知了"}

print("\n" + "=" * 50)
print("🚀 开始运行 Evaluator-Optimizer 闭环图")
print("=" * 50)

final_state = graph.invoke(initial_input)

print("\n" + "=" * 50)
print("🎉 图运行成功退出！")
print(f"总共迭代次数: {final_state['iterations']}")
print(f"最终输出符合要求的诗:\n{final_state['poem']}")
print("=" * 50)