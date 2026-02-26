from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.call_llm_api import LLMCompletionCall
from models.agents.tools import graph_retrieval


class AAgent:
    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model
        # 绑定工具
        self.llm_with_tools = self.llm.bind_tools([graph_retrieval])

    def execute(self, instruction: str) -> str:
        # 这里实际上简化了，可以直接让 LLM 决定是否调用工具
        # 或者强制调用
        try:
            # 模拟：直接执行 graph_retrieval
            # 在更复杂的 LangGraph 中，这应该由 LLM 自动选择
            # 这里为了确保融合，我们强制执行检索
            result = graph_retrieval.invoke(instruction)
            return f"Search Query: {instruction}\nResult:\n{result}"
        except Exception as e:
            return f"Retrieval failed: {e}"