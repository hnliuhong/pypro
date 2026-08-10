from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": 1001}}
# 如果不配置config则会抛出异常: 'configurable' keys: thread_id
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},config= thread_config)["messages"][-1].content
print(response)  # "Hi Bob! Nice to see you here. How are you doing?"
# 经过参数, 10~15 次还是会有记忆！
response = agent.invoke({"messages": [{"role": "user", "content": "你还记得我是谁吗"}]},thread_config,)["messages"][-1].content
print(response)
print('-'*100)
thread_config = {"configurable": {"thread_id": 1002}}
response = agent.invoke({"messages": [{"role": "user", "content": "你还记得我是谁吗"}]},thread_config,)["messages"][-1].content
print(response)