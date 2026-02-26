from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger

class RAgent:
    def __init__(self, schema_text: str = ""):
        self.llm = LLMCompletionCall().langchain_model
        # 确保 Schema 不为空，如果为空给一个警告提示
        if not schema_text:
            logger.warning("⚠️ RAgent initialized with EMPTY schema! Hallucination risk high.")
            self.schema = "Schema definition missing. Please rely on general knowledge but prefer graph search."
        else:
            self.schema = schema_text

    def plan(self, question: str, notebook: list, step_count: int, query_history: list) -> dict:
        # 1. 压缩 Notebook：只保留最近的由 S-Agent 总结过的 Insight (如果有)，或者摘要
        # 这里为了简单，我们只提取 Notebook 条目的第一行（Query）和结果摘要
        # 实际生产中可以使用 LLM 对 Notebook 进行 Summary
        # 格式化 Notebook
        context_str = "\n".join(
            [f"[{i + 1}] {item}" for i, item in enumerate(notebook)]) if notebook else "(No information yet)"
        history_str = "\n".join([f"- {q}" for q in query_history]) if query_history else "(None)"



        prompt = ChatPromptTemplate.from_template("""
        你是一个图谱推理规划师 (R-Agent)。基于迭代检索解决复杂问题。

        【Schema】:
        {schema}
        
        【用户问题】: {question}

        【 Notebook(已知信息)】:
        {history}

        【已执行过的查询 (History)】:
        {history}
        
                【当前状态分析】:
        请先进行自我反思 (Self-Reflection)：
        1. 之前检索到的信息是否已经部分回答了问题？
        2. 之前的检索是否存在失败（No result）？如果是，必须换一种表达方式（如同义词替换、不仅查关系也查属性）。
        3. 还没解决的**核心子问题**是什么？

        【决策规则 - 严禁死循环】:
        1.  **核心目标**: 仅当 Notebook 缺少回答问题所需的**关键事实**时，才进行检索
        2. **禁止重复**: 严禁生成与【已执行过的查询】语义相似的 Search Query。
           - 错误示例: 已查 "Messi goal comparison"，再次查 "Messi goal compared to Maradona"。
           - 正确策略: 如果上次查询结果为空或不相关，必须**更换关键词**或**拆解问题**。
        3. **负反馈处理**: 如果上一条记录显示 "No relevant info found"，请尝试更宽泛的词或查询关联实体。
        4. **幻觉抑制**: 如果 Notebook 为空，先从问题中的核心实体（如 "Lionel Messi"）开始查。不要假设任何未检索到的事实。
        5. **基于 Schema 拆解**: 你的查询词必须尽量贴合 Schema 中的实体和关系。不要生成图谱中不可能存在的复杂长句。
        6. **原子化查询**: 严禁生成像 "Lionel Messi official debut date 2004..." 这样的一坨关键词。
           - 正确: "Lionel Messi debut date", "Lionel Messi belongs_to"
           - 错误: "Messi early career challenges and achievements" 
        7. **结束条件**: 如果你能根据 Notebook 推断出答案（ Notebook 中已经包含了问题的核心答案），或者连续两次查询无果，请输出 "FINISH"。
           
        请返回 JSON:
        {{
            "thought": "分析当前状态，说明为什么需要/不需要继续检索...",
            "action": "SEARCH" | "FINISH",
            "search_query": "简短、精确的检索词 (仅当 action=SEARCH)"
        }}
        """)

        chain = prompt | self.llm | JsonOutputParser()

        # 我们只把 Notebook 的 header 传进去，具体内容让 LLM 相信 Orchestrator 的去重
        # context_summary = []
        # for i, item in enumerate(notebook):
        #     lines = item.split('\n')
        #     query_line = lines[0] if lines else "Unknown Step"
        #     summary_line = lines[1] if len(lines) > 1 else ""  # Usually "Found: ..." or "Result: ..."
        #     context_summary.append(f"[{i + 1}] {query_line} | Status: {summary_line[:50]}...")
        #
        # context_str = "\n".join(context_summary)

        try:
            plan = chain.invoke({
                "schema": self.schema,
                "question": question,
                "context": context_str,
                "history": history_str
            })

            logger.info(f"🧠 [R-Agent] Thought: {plan.get('thought')}")
            if plan.get('action') == "SEARCH":
                q = plan.get('search_query', '').strip()
                if any(q in old_q or old_q in q for old_q in query_history):
                    logger.warning(f"⚠️ Loop query detected: '{q}'. Modifying strategy...")
                    # 策略回退：如果重复，尝试查询实体的属性
                    plan['search_query'] = q + " attributes"

            return plan
        except Exception as e:
            logger.error(f"Planning Error: {e}")
            return {"action": "FINISH", "thought": "Error in planning."}