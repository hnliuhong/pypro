import os
from langchain.agents import create_agent
from langsmith import Client, evaluate

# ==========================================
# 步骤 1: 设置环境变量
# ==========================================
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your_langsmith_api_key_here")

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
# 步骤 3: 使用 examples 数组方式创建测试数据集
# ==========================================

client = Client()
dataset_name = "Agent客服能力评估集"
# ValueError: Exactly one argument in each of the following groups must be defined: dataset_name, dataset_id
if not client.has_dataset(dataset_name=dataset_name):
    # 改用 examples 参数传入结构化的 List[dict]，高内聚且易于阅读
    client.create_examples(
        dataset_name=dataset_name,
        examples=[
            {
                "inputs": {"messages": [{"role": "user", "content": "你们支持多少天退货？"}]},
                "outputs": {"reference": "7天无理由退款"},
            },
            {
                "inputs": {"messages": [{"role": "user", "content": "你好，今天天气怎么样？"}]},
                "outputs": {"reference": "无法查询实时天气"},
            },
        ],
    )


# ==========================================
# 步骤 4: 定义评估标准
# ==========================================
def correctness_evaluator(run, example) -> dict:
    """
    原生评估函数：对比 Agent 的最终回答与测试集的标准参考答案
    """
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
    agent.invoke,
    data=dataset_name,
    evaluators=[correctness_evaluator],  # 传入自定义评测函数
    experiment_prefix="lesson-demo-v1",
)

print("评估完成！可登录Smith官网 Datasets & Experiments 看结果。")