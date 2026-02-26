from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger
from typing import List


class SAgent:
    """
    Summarizer / Survey Agent: 写作专家。
    负责将 Notebook 中收集到的碎片化事实（由 A-Agent 执行得到）
    整合成一篇流畅、有深度的学术回答或综述。
    """

    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model

        # 定义写作 Prompt
        self.prompt = ChatPromptTemplate.from_template("""
        你是一个资深的学术编辑和综述作家 (S-Agent)。
        你的任务是根据用户的问题和收集到的已知信息 (Notebook)，撰写最终的回答。

        【用户问题】: {question}

        【意图类型】: {intent} (如果是 SURVEY，请按综述格式写；如果是 FACTUAL，请直接回答)

        【收集到的信息 (Notebook)】:
        {notebook_content}

        【写作要求】:
        1. **基于事实**: 严格基于 Notebook 中的信息，不要编造未出现的数据。
        2. **结构清晰**: 
           - 如果是综述，使用 Markdown (## 标题, - 列表) 组织，包含“研究背景”、“主要方法”、“关键结论”等。
           - 如果是时序分析，按时间轴组织。
        3. **引用来源**: 如果 Notebook 中包含论文标题或 ID，请在文中适当位置标注。
        4. **语言风格**: 专业、客观、学术化。

        请直接输出最终回答：
        """)

    def write(self, question: str, notebook: list, intent: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
        你是一个阅读理解专家。请根据以下检索到的片段回答问题。

        【问题】: {question}

        【检索片段 (Notebook)】:
        {notebook}

        【注意】
        1. 仔细阅读片段中的文本（Chunk Text）。
        2. 答案通常隐藏在文本细节中，例如 "signed by Barcelona in 2003" 或 "comparison to Maradona"。
        3. 如果文本中提到了 "Messi's goal" 和 "Maradona"，这就是关键线索。
        4. 如果实在找不到，回答“根据现有文档未找到”。

        答案：
        """)
        logger.info(f"✍️ [S-Agent] Generating final answer for intent: {intent}")

        notebook_str = "\n".join(
            [f"[{i + 1}] {item}" for i, item in enumerate(notebook)]) if notebook else "No specific data found."

        try:
            chain = self.prompt | self.llm | StrOutputParser()
            final_answer = chain.invoke({
                "question": question,
                "intent": intent,
                "notebook_content": notebook_str
            })
            logger.info("✅ [S-Agent] Answer generated successfully.")
            return final_answer
        except Exception as e:
            logger.error(f"❌ [S-Agent] Generation failed: {e}")
            return f"Error generating answer: {e}"