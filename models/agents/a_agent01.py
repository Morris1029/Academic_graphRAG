from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger
from models.agents.tools import execute_python_analysis  # 导入你定义的工具


class AAgent:
    """
    Action Agent: 代码执行者。
    接收 R-Agent 的指令，生成 Python 代码，并调用 tools.py 中的沙箱执行。
    """

    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model

        # 定义代码生成 Prompt
        # 明确告诉它有哪些 API 可用 (对应 models/retriever/enhanced_kt_retriever.py 的新增接口)
        self.prompt = ChatPromptTemplate.from_template("""
        你是一个 Python 代码生成助手 (A-Agent)。你的任务是将自然语言指令转化为可执行的 Python 代码。

        【可用工具环境】
        环境中预置了以下函数，你可以直接调用：
        1. `search_entity(query: str)` -> str: 
           搜索实体，返回 ID 和属性。
        2. `get_neighbors(node_id: str)` -> str: 
           获取邻居节点和关系。
        3. `print(content)`: 
           必须使用 print 输出结果，否则无法被捕获。
        4. 标准 Python 库: len, sorted, set, list 等。

        【指令】: {instruction}

        【要求】:
        1. 代码必须简洁，只做指令要求的事。
        2. 如果需要查找实体，先调用 search_entity，解析出 ID 后再调用 get_neighbors。
        3. **必须**把关键结果 print 出来。
        4. 不要包含 ```python 标记，只返回代码本身。

        【代码示例】:
        # 查找 A 的邻居
        info = search_entity("Paper A")
        # 假设 info 包含 ID: 123
        # 你需要编写解析逻辑(简单字符串处理)或直接观察输出
        # 为简化，假设你先查一下
        print(info)

        代码:
        """)

    def execute(self, instruction: str) -> str:
        """
        生成代码 -> 执行代码 -> 返回结果
        """
        logger.info(f"🛠️ [A-Agent] Receiving instruction: {instruction}")

        try:
            # 1. 生成代码
            chain = self.prompt | self.llm | StrOutputParser()
            code_snippet = chain.invoke({"instruction": instruction})

            # 清洗代码 (防止 markdown)
            code_snippet = code_snippet.replace("```python", "").replace("```", "").strip()
            logger.debug(f"💻 [A-Agent] Generated Code:\n{code_snippet}")

            # 2. 调用 tools.py 中的沙箱执行
            # 注意：execute_python_analysis 是一个 LangChain Tool，我们需要手动调用其 func 或者直接调用逻辑
            # 如果 tools.py 中是 @tool 装饰器，可以直接调用 func(code_snippet) 或者 invoke
            execution_result = execute_python_analysis.invoke(code_snippet)

            logger.info(f"✅ [A-Agent] Execution Result: {str(execution_result)[:100]}...")
            return f"Executed Code:\n{code_snippet}\n\nResult:\n{execution_result}"

        except Exception as e:
            logger.error(f"❌ [A-Agent] Execution failed: {e}")
            return f"Error executing instruction: {e}"