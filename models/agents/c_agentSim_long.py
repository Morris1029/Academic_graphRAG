import json
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.call_llm_api import LLMCompletionCall
from utils.logger import logger


class CAgent:
    """
    Controller/Classifier Agent: 负责用户意图识别和路由。
    对应 GLM 论文中的 Classification Agent。
    """

    def __init__(self):
        # 获取封装了 LangChain 的 LLM 客户端
        self.llm = LLMCompletionCall().langchain_model

        # # 定义路由 Prompt
        # self.prompt = ChatPromptTemplate.from_template("""
        # 你是一个学术问答系统的路由专家。请分析用户的输入，将其分类为以下几种意图之一：
        #
        # 1. "FACTUAL": 简单的事实查询，只需查找实体属性或直接邻居即可回答。（例如："Transformer论文的作者是谁？"）
        # 2. "COMPLEX": 复杂的多跳推理问题，需要多步查找和逻辑推断。（例如："DeepSeek引用的论文中，哪些使用了RLHF技术？"）
        # 3. "TEMPORAL": 涉及时间演变、历史脉络、趋势分析的问题。（例如："图神经网络在2020年后的发展趋势是什么？"）
        # 4. "SURVEY": 需要生成综述、总结某领域研究现状的问题。（例如："请总结GraphRAG技术的主要流派。"）
        #
        # 用户输入: {question}
        #
        # 请仅返回一个 JSON 对象，格式如下，不要包含 markdown 格式：
        # {{
        #     "intent": "FACTUAL" | "COMPLEX" | "TEMPORAL" | "SURVEY",
        #     "reason": "简短的分类理由"
        # }}
        # """)

        self.prompt = ChatPromptTemplate.from_template("""
        你是一个学术问答系统的路由专家。你的任务是分析用户的意图，并将其分类。

        【分类定义】
        1. "FACTUAL": 简单事实问题。只需查找实体属性或直接邻居即可回答。不需要多步跳跃。
        2. "COMPLEX": 复杂的多跳推理问题。需要多步推理、比较、条件筛选或处理嵌套关系的问题。
           - 特征词："...的人" (nested clause), "比较", "共同点", "之前/之后", "原因"。
        3. "TEMPORAL": 涉及时间演变、历史脉络、趋势分析的问题。
        4. "SURVEY": 需要生成综述、总结某领域研究现状的问题。

        【示例分析】
        Q: "Transformer论文的作者是谁？" 
        A: FACTUAL (直接查询属性)

        Q: "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?"
        A: COMPLEX (原因：这是一个嵌套问题。必须先找到"那个被比较的人"是谁，然后再查"他"什么时候签的巴萨。不能一步查到。)

        Q: "列出GCN和GAT在2020年的引用差异。"
        A: COMPLEX (需要查询两个实体并进行数值比较)

        【当前用户问题】: {question}

        请先思考问题的结构，然后返回 JSON，格式如下：
        {{
            "intent": "FACTUAL" | "COMPLEX" | "TEMPORAL" | "SURVEY",
            "reason": "简短的分析理由"
        }}
        """)

        self.parser = JsonOutputParser()

    def route(self, question: str) -> Dict[str, Any]:
        """
        执行路由逻辑
        Returns: {'intent': '...', 'reason': '...'}
        """
        logger.info(f"🤖 [C-Agent] Analyzing intent for: {question[:50]}...")
        try:
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({"question": question})
            logger.info(f"✅ [C-Agent] Routed to: {result.get('intent')}")
            return result
        except Exception as e:
            logger.error(f"❌ [C-Agent] Routing failed: {e}")
            # 兜底策略：默认为复杂推理
            return {"intent": "COMPLEX", "reason": "Routing failed, fallback to default."}