from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END

@tool(description="查询天气时可以调用此方法")
def get_city_weather(city:str) -> str:
    return city + " 未来几天是晴天"

agent = create_agent(model="deepseek-v4-flash",tools=[get_city_weather])
builder = StateGraph(MessagesState)
builder.add_node("llm_node", agent)

builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)
graph = builder.compile()
graph.get_graph().print_ascii()

result = graph.invoke({"messages": [{"role": "user", "content": "你好,能简单告诉我北京未来几天的天气吗?"}]})
for mes in result['messages']:
    print(mes.content)