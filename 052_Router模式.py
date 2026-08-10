from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph

# --- 1. 定义路由专用的 Pydantic Schema ---
class RouterDecision(BaseModel):
    """分析用户意图，选择最合适的处理专家节点"""

    next_node: Literal["tech_support", "billing"] = Field(
        description="如果是设备报错、卡顿、网络等技术问题选 'tech_support'；如果是费用、账单、扣款等资金问题选 'billing'。"
    )


# 初始化大模型
llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)

# 创建 JSON 解析器并构造 Prompt 模板
parser = JsonOutputParser(pydantic_object=RouterDecision)

prompt_template = PromptTemplate(
    template="""请分析以下用户的最新请求意图，并将其分类分发给对应的专家处理。

用户请求："{user_query}"

{format_instructions}""",
    input_variables=["user_query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 使用 LCEL 管道符组合：Prompt -> LLM -> Parser
router_chain = prompt_template | llm | parser


# --- 2. 定义专家节点 ---
def tech_support_agent(state: MessagesState):
    return {
        "messages": [
            AIMessage(
                content="[技术专家]: 请尝试重启您的设备并检查网络配置。"
            )
        ]
    }


def billing_agent(state: MessagesState):
    return {
        "messages": [
            AIMessage(content="[账单专家]: 您的本月账单为 $99，已扣款成功。")
        ]
    }


# --- 3. 修改路由条件判断函数 (LLM 意图识别) ---
def route_decision(state: MessagesState) -> Literal["tech_support", "billing"]:
    last_user_msg = state["messages"][-1].content

    # 执行包含 JsonOutputParser 的链
    res = router_chain.invoke({"user_query": last_user_msg})

    next_node = res.get("next_node")
    print(f"👉 [路由决策 LLM 判定]: 用户意图属于 -> {next_node}")

    return next_node


# --- 4. 构建图 ---
builder = StateGraph(MessagesState)

# 添加节点
builder.add_node("tech_support", tech_support_agent)
builder.add_node("billing", billing_agent)

# 使用条件边：从 START 开始，通过 LLM 意图识别函数决定跳转
builder.add_conditional_edges(START, route_decision)

# 专家处理完后都指向 END
builder.add_edge("tech_support", END)
builder.add_edge("billing", END)

graph = builder.compile()
# 打印拓扑图
graph.get_graph().print_ascii()

# --- 5. 运行测试 ---
test_input = {"messages": [HumanMessage(content="我的网络很卡,应该如何解决呢？")]}
result = graph.invoke(test_input)

print("\n=== 运行结果 ===")
for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")