import json
from typing import List, Dict, Any, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger


# 假设你有办法加载 Schema，这里模拟一个加载函数，实际请根据你的 kt_gen.py 修改
# from models.constructor.kt_gen import load_schema

class RAgent:
    """
    Reasoning Agent: 核心大脑。
    融合 GLM 的 Notebook 机制和 Youtu 的 Schema 约束。
    负责规划下一步行动 (Next Step) 或 生成最终答案 (Final Answer)。
    """

    def __init__(self, schema_text: str = ""):
        self.llm = LLMCompletionCall().langchain_model
        self.schema = schema_text if schema_text else "Entities: Paper, Author, Concept. Relations: CITES, AUTHORED_BY."

    def plan(self, question: str, notebook: List[str]) -> Dict[str, Any]:
        """
        根据当前问题、已知信息(Notebook)和Schema，规划下一步。
        """
        logger.info(f"🧠 [R-Agent] Planning next step. Notebook items: {len(notebook)}")

        notebook_context = "\n".join(
            [f"- {item}" for item in notebook]) if notebook else "No information collected yet."

        # 定义 GLM 风格的 Reasoning Prompt
        prompt_content = f"""
        你是一个基于图谱的推理专家 (R-Agent)。

        【图谱 Schema (地图)】:
        {self.schema}

        【用户问题】: 
        {question}

        【Notebook (已知信息)】:
        {notebook_context}

        请思考：基于 Schema 和 Notebook 中的已知信息，是否足以回答用户问题？

        - 如果足够，请直接给出最终答案。
        - 如果不足，请规划 **下一步** 需要 A-Agent (代码代理) 去执行的具体任务。
          注意：任务必须能够通过编写 Python 代码查询图谱来实现 (如查询邻居、属性排序等)。

        请仅返回 JSON 格式，格式如下：
        {{
            "status": "CONTINUE" | "FINISH",
            "content": "如果是 CONTINUE，这里写给 A-Agent 的具体指令（如：查询 'GraphRAG' 节点的邻居）；如果是 FINISH，这里写最终答案。"
        }}
        """

        try:
            # 这里不使用复杂的 Chain，直接调用 invoke 保持灵活性
            response = self.llm.invoke([HumanMessage(content=prompt_content)])

            # 简单的清洗逻辑，防止 LLM 返回 Markdown 包裹
            content = response.content.replace("```json", "").replace("```", "").strip()
            plan = json.loads(content)

            return plan
        except Exception as e:
            logger.error(f"❌ [R-Agent] Planning failed: {e}")
            return {"status": "FINISH", "content": "I encountered an error during reasoning."}

    def update_notebook(self, notebook: List[str], new_finding: str) -> List[str]:
        """
        辅助函数：更新 Notebook
        """
        if new_finding and "No result" not in new_finding:
            notebook.append(new_finding)
        return notebook