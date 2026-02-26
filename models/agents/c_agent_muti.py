import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger


class CAgent:
    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model

        self.prompt = ChatPromptTemplate.from_template("""
        你是一个意图识别专家。分析用户问题并分类。

        【分类标准】
        1. "SIMPLE": 简单实体查询。如 "Who is Messi?", "Where is Barcelona?"。
           -> 策略：单次检索即可。
        2. "COMPLEX": 多跳推理、比较、时序分析、条件筛选。如 "Compare A and B", "When did X do Y after Z?".
           -> 策略：需要分解问题，多步检索 (CoT)。
        3. "SURVEY": 综述、总结类。
           -> 策略：广度检索。

        【用户问题】: {question}

        返回 JSON:
        {{
            "intent": "SIMPLE" | "COMPLEX" | "SURVEY",
            "reason": "..."
        }}
        """)
        self.parser = JsonOutputParser()

    def route(self, question: str) -> Dict[str, Any]:
        try:
            result = (self.prompt | self.llm | self.parser).invoke({"question": question})
            logger.info(f"🤖 [C-Agent] Intent: {result.get('intent')}")
            return result
        except:
            return {"intent": "COMPLEX"}  # 默认走复杂流程