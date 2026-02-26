from utils.call_llm_api import LLMCompletionCall
from models.agents.tools import graph_retrieval
from utils.logger import logger


class AAgent:
    def __init__(self):
        # A-Agent 现在是一个纯粹的执行器
        pass

    def execute(self, query: str) -> str:
        logger.info(f"🛠️ [A-Agent] Executing: {query}")
        try:
            # 调用工具
            result = graph_retrieval.invoke(query)

            # 在日志中打印结果摘要，方便调试
            preview = result[:200].replace('\n', ' ')
            logger.info(f"✅ [A-Agent] Retrieved data preview: {preview}...")

            return f"Query: {query}\n{result}"
        except Exception as e:
            logger.error(f"Execution Failed: {e}")
            return f"Search failed: {e}"