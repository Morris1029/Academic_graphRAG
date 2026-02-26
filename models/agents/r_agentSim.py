import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.call_llm_api import LLMCompletionCall


class RAgent:
    def __init__(self, schema_text=""):
        self.llm = LLMCompletionCall().langchain_model

    def plan(self, question: str, notebook: list) -> dict:
        notebook_str = "\n".join(notebook) if notebook else "Empty"

        prompt = ChatPromptTemplate.from_template("""
        你是一个图谱检索规划师。基于已知信息，规划下一步检索。

        【用户问题】: {question}
        【已知信息 (Notebook)】: {notebook}

        【决策】
        1. 如果已知信息足够回答问题，返回 FINISH。
        2. 如果不够，生成一个 **优化的检索 Query**。
           - 不要写 Python 代码。
           - Query 应该是自然语言，描述你想找什么。
           - 例如：想知道 Messi 的进球，Query="Messi Copa del Rey goal comparison"

        返回 JSON:
        {{
            "status": "CONTINUE" | "FINISH",
            "content": "检索 Query 或 最终答案"
        }}
        """)

        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({"question": question, "notebook": notebook_str})
        except Exception:
            # 兜底：直接搜原问题
            return {"status": "CONTINUE", "content": question}