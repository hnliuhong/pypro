import os
from langchain.agents import create_agent
from langsmith import Client, evaluate

# ==========================================
# 步骤 1: 设置环境变量
# ==========================================
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your_langsmith_api_key_here")

# 初始化 LangSmith 客户端
client = Client()

# ==========================================
# 步骤 2: 准备 Agent 工具与创建 Agent 实例
# ==========================================
def get_refund_policy() -> str:
    """查询退货政策。"""
    return "官方政策：支持自签收之日起 7 天无理由退款。"


agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_refund_policy],
)

# ==========================================
# 步骤 3: 在 LangSmith 平台自动创建/准备测试数据集
# ==========================================
dataset_name = "Agent客服能力评估集"

if not client.has_dataset(dataset_name=dataset_name):

    client.create_examples(
        inputs=[
            {"messages": [{"role": "user", "content": "你们支持多少天退货？"}]},
            {"messages": [{"role": "user", "content": "你好，今天天气怎么样？"}]},
        ],
        outputs=[
            {"reference": "7天无理由退款"},
            {"reference": "无法查询实时天气"},
        ],
        # dataset_id=dataset.id,
    )


# ==========================================
# 步骤 4: 定义评估标准 (无需额外 import，原生函数即可)
# ==========================================
def correctness_evaluator(run, example) -> dict:
    """
    原生评估函数：对比 Agent 的最终回答与测试集的标准参考答案
    """
    print("run:",run,"example:",example)
    # 获取 Agent 的实际输出 (从最后一条消息中提取内容)
    messages = run.outputs.get("messages", [])
    agent_answer = messages[-1].content if messages else ""

    # 获取测试集里的预期答案
    reference_answer = example.outputs.get("reference", "")

    # 判断：如果预期的关键词出现在回复中，就算得分 1.0，否则 0.0
    is_correct = reference_answer in agent_answer

    return {
        "key": "correctness",  # 展示在 LangSmith 看板上的指标名称
        "score": 1.0 if is_correct else 0.0  # 得分
    }


# ==========================================
# 步骤 5: 启动批量评估并提交实验结果
# ==========================================
print("开始运行评估任务...")

results = evaluate(
    agent.invoke,   # / 前面的参数不能写名称
    data=dataset_name,
    evaluators=[correctness_evaluator],  # 传入自定义评测函数
    experiment_prefix="lesson-demo-v1",
)

print("评估完成！可登录 https://smith.langchain.com 查看结果。")