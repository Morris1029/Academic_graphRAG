from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger


class SAgent:
    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model

        self.prompt = ChatPromptTemplate.from_template("""
        你是一个学术助手。请根据以下收集到的证据链 (Notebook)，回答用户问题。

        【用户问题】: {question}

        【证据链 (Notebook)】:
        {notebook}

        【要求】
        1. **综合推理**：串联证据链中的信息。例如，如果 Step 1 找到了人名，Step 2 找到了该人的时间，请将它们结合。
        2. **引用事实**：回答必须基于 Notebook 中的 Triples 或 Chunks。如果 Notebook 里没有，就说不知道。
        3. **逻辑清晰**：先直接回答结论，再解释推理过程。

        回答：
        """)

    def write(self, question: str, notebook: list) -> str:
        logger.info("✍️ [S-Agent] Synthesizing final answer...")
        notebook_str = "\n".join([f"--- Evidence {i + 1} ---\n{item}" for i, item in enumerate(notebook)])

        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "notebook": notebook_str})