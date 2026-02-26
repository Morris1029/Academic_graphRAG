from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger
from models.agents.tools import graph_retrieval, python_interpreter


class AAgent:
    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model

        # 定义 A-Agent 的 Prompt，教它如何使用工具
        self.prompt = ChatPromptTemplate.from_template("""
        你是一个执行助手 (A-Agent)。
        你接收 R-Agent 的指令，并利用工具完成任务。

        【可用工具】
        1. `graph_retrieval(query)`: 搜索图谱和文档。用于获取信息。
        2. `python_interpreter(code)`: 执行 Python 代码。用于处理逻辑（如排序、提取）。

        【指令】: {instruction}

        【执行策略】
        - 如果指令需要获取新信息（如“查询某人”、“搜索某事”），请生成 Python 代码调用 `graph_retrieval`。
        - 如果指令是逻辑处理（如“比较日期”），请生成 Python 代码进行计算。
        - **必须**使用 print() 输出结果。

        【示例】
        指令: "查询 Messi 的进球"
        代码:
        ```python
        print(graph_retrieval.invoke("Messi goals"))
        ```

        请直接返回可执行的 Python 代码：
        """)

    def execute(self, instruction: str) -> str:
        logger.info(f"🛠️ [A-Agent] Processing: {instruction}")
        try:
            # 1. 生成调用代码
            chain = self.prompt | self.llm | StrOutputParser()
            code = chain.invoke({"instruction": instruction})
            code = code.replace("```python", "").replace("```", "").strip()

            # 2. 在沙箱中执行
            # 我们构建一个包含工具的上下文
            local_scope = {
                "graph_retrieval": graph_retrieval,
                "python_interpreter": python_interpreter,
                "print": print
            }

            import io
            import contextlib
            output_capture = io.StringIO()
            with contextlib.redirect_stdout(output_capture):
                exec(code, local_scope)

            result = output_capture.getvalue()
            logger.info(f"✅ [A-Agent] Result length: {len(result)}")
            return result if result else "Executed but no output."

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return f"Error: {e}"