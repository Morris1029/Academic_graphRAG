import json
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger


class RAgent:
    def __init__(self, schema_text: str = ""):
        self.llm = LLMCompletionCall().langchain_model

    def plan(self, question: str, notebook: List[str]) -> Dict[str, Any]:
        logger.info(f"🧠 [R-Agent] Planning... (Notebook entries: {len(notebook)})")

        notebook_str = "\n".join([f"[{i + 1}] {item}" for i, item in enumerate(notebook)]) if notebook else "Empty"

        prompt = ChatPromptTemplate.from_template("""
        你是一个图谱推理规划师 (R-Agent)。你的目标是通过**分步骤**的探索来回答复杂问题。

        【用户问题】: {question}

        【已知信息 (Notebook)】: 
        {notebook}

        【思考流程】
        1. 分析用户问题包含的实体和关系。
        2. 检查 Notebook 中是否已经有了这些信息。
        3. 如果 Notebook 中的信息**完全足够**回答问题，且逻辑闭环，输出 FINISH。
        4. 如果不够，**只规划当前最需要的一步**。不要试图一次性解决所有问题。

        【决策逻辑示例】
        问题："A的老师的妻子的职业是什么？"
        - Notebook为空 -> 指令："查询 A 的老师是谁" (CONTINUE)
        - Notebook有"A的老师是B" -> 指令："查询 B 的妻子是谁" (CONTINUE)
        - Notebook有"B的妻子是C" -> 指令："查询 C 的职业" (CONTINUE)
        - Notebook有"C是律师" -> (FINISH)

        请返回 JSON:
        {{
            "status": "CONTINUE" | "FINISH",
            "thought": "我现在的思考...",
            "instruction": "给 A-Agent 的具体指令。如果是 CONTINUE，请描述要查询什么；如果是 FINISH，请留空或写'生成答案'。"
        }}
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({"question": question, "notebook": notebook_str})
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return {"status": "CONTINUE", "instruction": f"搜索: {question}", "thought": "解析失败，尝试直接搜索"}