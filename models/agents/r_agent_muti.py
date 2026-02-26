from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger

class RAgent:
    def __init__(self):
        self.llm = LLMCompletionCall().langchain_model

    def plan(self, question: str, notebook: list, step_count: int, query_history: list) -> dict:
        # 格式化 Notebook
        context_str = "\n".join([f"[Info {i+1}]: {item}" for i, item in enumerate(notebook)]) if notebook else "(Empty)"
        # 格式化历史查询
        history_str = "\n".join([f"- {q}" for q in query_history]) if query_history else "(None)"

        prompt = ChatPromptTemplate.from_template("""
        你是一个图谱推理规划师 (R-Agent)。基于迭代检索解决复杂问题。

        【用户问题】: {question}

        【已收集信息 (Notebook)】:
        {context}

        【已执行过的查询 (History)】:
        {history}

        【决策规则 - 严禁死循环】:
        1. **检查 Notebook**: 如果 Notebook 中已经包含了问题的核心答案（例如具体的比赛名称、时间、人物），立即输出 "FINISH"。
        2. **禁止重复**: 绝对不要生成与【已执行过的查询】语义相似的 Search Query。
           - 错误示例: 已查 "Messi goal comparison"，再次查 "Messi goal compared to Maradona"。
           - 正确策略: 如果上次查询结果为空或不相关，必须**更换关键词**或**拆解问题**。
        3. **负反馈处理**: 如果上一条记录显示 "No relevant info found"，请尝试更宽泛的词（去掉修饰语）或查询关联实体。

        请返回 JSON:
        {{
            "thought": "分析当前状态，说明为什么需要/不需要继续检索...",
            "action": "SEARCH" | "FINISH",
            "search_query": "新的检索词 (仅当 action=SEARCH)"
        }}
        """)

        chain = prompt | self.llm | JsonOutputParser()

        try:
            plan = chain.invoke({
                "question": question,
                "context": context_str,
                "history": history_str
            })

            logger.info(f"🧠 [R-Agent] Thought: {plan.get('thought')}")
            if plan.get('action') == "SEARCH":
                q = plan.get('search_query')
                # 二次校验：简单的去重逻辑
                if q in history_str:
                    logger.warning(f"⚠️ Detected loop query '{q}', forcing modification.")
                    plan['search_query'] = q + " details info"  # 简单加扰动，或者应该在 Prompt 里解决

            return plan
        except Exception as e:
            logger.error(f"Planning Error: {e}")
            return {"action": "FINISH", "thought": "Error in planning."}