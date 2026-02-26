from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger
from models.agents.tools import execute_python_analysis


class AAgent:
    def __init__(self, retriever=None):  # 兼容接口，retriever在tools里单例获取
        self.llm = LLMCompletionCall().langchain_model

        self.prompt = ChatPromptTemplate.from_template("""
        你是一个 Python 代码生成助手 (A-Agent)。

        【可用工具】
        1. `search_entity(query)`: 搜索实体ID。
        2. `get_neighbors(node_id)`: 获取图关系。
        3. `read_node_content(node_id)`: 【关键】读取节点关联的原始文本。如果 get_neighbors 返回空或找不到答案，必须调用此函数阅读原文。

        【指令】: {instruction}

        【编写规则】
        1. 优先尝试图遍历，如果通过 `get_neighbors` 没找到想要的信息，立即对关键节点调用 `read_node_content`。
        2. 代码必须包含 `print(...)` 来输出你的发现，或者将结果赋值给 `result` 变量。
        3. 处理 "No neighbors found" 等空情况，不要报错。
        4. 不要使用 markdown 标记，直接返回代码。

        【示例：结合图与文本】
        # 找 Messi 的进球信息
        messi_info = search_entity("Lionel Messi")
        print(messi_info)
        # 假设解析出 ID 为 entity_123
        # 尝试找关系
        rels = get_neighbors("entity_123")
        print(rels)
        # 如果关系里没写进球细节，读文本
        text = read_node_content("entity_123")
        print(text)

        代码:
        """)

    def execute(self, instruction: str) -> str:
        logger.info(f"🛠️ [A-Agent] Instruction: {instruction[:100]}...")
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            code_snippet = chain.invoke({"instruction": instruction})
            code_snippet = code_snippet.replace("```python", "").replace("```", "").strip()

            # logger.debug(f"💻 Generated Code:\n{code_snippet}")

            execution_result = execute_python_analysis.invoke(code_snippet)

            logger.info(f"✅ [A-Agent] Result: {execution_result}")

            return f"Executed Code:\n{code_snippet}\n\nExecution Output:\n{execution_result}"
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            return f"Error: {e}"