import json
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger


class RAgent:
    def __init__(self, schema_text: str = ""):
        self.llm = LLMCompletionCall().langchain_model
        self.schema = schema_text

    def plan(self, question: str, notebook: List[str]) -> Dict[str, Any]:
        notebook_context = "\n".join([f"- {item}" for item in notebook]) if notebook else "Empty"

        prompt_content = f"""
        你是一个图谱推理专家 (R-Agent)。

        【任务】
        基于 Schema 和 Notebook 中的已知信息，解决用户问题："{question}"

        【当前状态 (Notebook)】
        {notebook_context}

        【决策指南】
        1. **检查完整性**: 如果 Notebook 中的信息足以回答问题，返回 FINISH。
        2. **处理图稀疏性**: 之前的 Action 如果返回了 "No neighbors found" 或空结果，说明图连接缺失。
           此时，**必须** 指示 A-Agent 使用 `read_node_content` 工具读取关键实体（如 "Messi" 或 "Copa del Rey"）的原始文本内容。
           GraphRAG 的优势在于阅读原文，不仅仅是跳图！
        3. **规划下一步**: 生成给 A-Agent 的 Python 编写指令。

        请仅返回 JSON:
        {{
            "status": "CONTINUE" | "FINISH",
            "content": "给 A-Agent 的指令 (例如：'获取 Messi 节点的文本内容并查找关于 Copa del Rey 的描述') 或 最终答案"
        }}
        """

        try:
            response = self.llm.invoke([HumanMessage(content=prompt_content)])
            content = response.content.replace("```json", "").replace("```", "").strip()
            # 简单的清理
            if content.startswith("{") and content.endswith("}"):
                return json.loads(content)
            # 容错处理
            import re
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError("Invalid JSON format")

        except Exception as e:
            logger.error(f"❌ Planning failed: {e}")
            # 兜底策略：如果规划挂了，尝试让 A-Agent 读原文
            return {
                "status": "CONTINUE",
                "content": "使用 read_node_content 工具读取查询中涉及到的关键实体的文本内容，寻找答案。"
            }