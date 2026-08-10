import os
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langsmith import Client, evaluate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# ==========================================
# 步骤 1: 设置环境变量与客户端初始化
# ==========================================
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "your_langsmith_api_key_here")

client = Client()

# ==========================================
# 步骤 2: 准备 Agent 与裁判模型
# ==========================================
def get_refund_policy() -> str:
    """查询退货政策。"""
    return "官方政策：支持自签收之日起 7 天无理由退款。"

agent = create_agent(
    model="deepseek-v4-flash",
    tools=[get_refund_policy],
)

# 1. 定义打分的 Pydantic 数据结构
class JudgeResult(BaseModel):
    score: float = Field(description="语义相似度与质量评分，范围从 0.0 到 1.0")
    reason: str = Field(description="打分的具体简要理由")

# 2. 初始化 JsonOutputParser
parser = JsonOutputParser(pydantic_object=JudgeResult)

# 3. 构建包含 format_instructions 的 PromptTemplate
judge_prompt_template = PromptTemplate(
    template="""你是一名专业的 AI 回答质量评估员。请对比 Agent 的实际回答与标准参考答案，给出评分和理由。

【预期标准答案】: {reference_answer}
【Agent 实际回答】: {agent_answer}

【打分标准】:
- 1.0: 语义完全正确，且表述清晰、完整无多余废话。
- 0.8~0.9: 语义基本一致，核心事实正确，但可能存在冗余词汇、语气轻微偏差或细节略有出入。
- 0.4~0.7: 回答了一部分内容，但缺少关键细节，或回答过于模糊。
- 0.0~0.3: 完全答非所问、拒绝回答（非预期）、事实错误或严重偏离主题。

{format_instructions}""",
    input_variables=["reference_answer", "agent_answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. 初始化裁判模型并组装 Chain
judge_llm = ChatDeepSeek(model="deepseek-v4-flash", temperature=0)
# 使用 LCEL 链：Prompt -> LLM -> JsonOutputParser
judge_chain = judge_prompt_template | judge_llm | parser

# ==========================================
# 步骤 3: 优化后的数据集创建逻辑
# ==========================================
dataset_name = "Agent客服能力评估集"

if not client.has_dataset(dataset_name=dataset_name):
    # 直接使用 dataset_name 关联，无需 dataset.id
    client.create_examples(
        inputs=[
            {"messages": [{"role": "user", "content": "你们支持多少天退货？"}]},
            {"messages": [{"role": "user", "content": "你好，今天天气怎么样？"}]},
        ],
        outputs=[
            {"reference": "7天无理由退款"},
            {"reference": "无法查询实时天气"},
        ],
        dataset_name=dataset_name,
    )

# ==========================================
# 步骤 4: LLM-as-a-Judge 评估器 (使用 JsonOutputParser)
# ==========================================
def llm_semantic_evaluator(run, example) -> dict:
    """
    使用 JsonOutputParser 的大模型裁判评估器
    """
    messages = run.outputs.get("messages", [])
    agent_answer = messages[-1].content if (messages and hasattr(messages[-1], 'content')) else str(messages[-1]) if messages else ""
    reference_answer = example.outputs.get("reference", "")

    # 执行 Chain，内部自动注入 JSON 输出格式说明，并自动将 LLM 返回解析为 Python dict
    eval_result = judge_chain.invoke({
        "reference_answer": reference_answer,
        "agent_answer": agent_answer
    })

    score = float(eval_result.get("score", 0.0))
    reason = eval_result.get("reason", "")

    print(f"裁判评价 - 得分: {score}, 原因: {reason}")

    return {
        "key": "semantic_correctness",
        "score": score,
        "comment": reason
    }

# ==========================================
# 步骤 5: 启动评估与实验提交
# ==========================================
print("开始运行 LLM 语义打分评估...")

results = evaluate(
    agent.invoke,
    data=dataset_name,
    evaluators=[llm_semantic_evaluator],
    experiment_prefix="lesson-demo-llm-judge",
)

print("评估完成！请刷新 LangSmith 控制台页面。")