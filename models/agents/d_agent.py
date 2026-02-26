from typing import List, Dict, Any
from models.retriever.agentic_decomposer import GraphQ
from config import get_config
from utils.logger import logger


class DAgent:
    """
    Decomposer Agent: 负责利用原有的 GraphQ 模块进行问题拆解。
    这是 "Global Planning" 的第一步。
    """

    def __init__(self, dataset_name: str):
        self.config = get_config()
        # 复用原有的 GraphQ 类
        self.decomposer = GraphQ(dataset_name, config=self.config)
        # 获取 schema 路径
        self.dataset_cfg = self.config.get_dataset_config(dataset_name)
        self.schema_path = self.dataset_cfg.schema_path

    def decompose(self, question: str) -> List[str]:
        """
        将复杂问题拆解为子问题列表。
        """
        logger.info(f"🧩 [D-Agent] Decomposing: {question}")
        try:
            # 调用底层 agentic_decomposer 的逻辑
            result = self.decomposer.decompose(question, self.schema_path)

            # 解析结果
            sub_questions_objs = result.get("sub_questions", [])

            # 提取纯文本问题
            sub_questions = []
            for item in sub_questions_objs:
                if isinstance(item, dict):
                    sub_questions.append(item.get("sub-question", ""))
                elif isinstance(item, str):
                    sub_questions.append(item)

            # 限制子问题数量，防止步骤过多 (控制在 3-4 个以内)
            if len(sub_questions) > 4:
                logger.warning("Too many sub-questions, truncating to 4.")
                sub_questions = sub_questions[:4]

            logger.info(f"✅ [D-Agent] Plan: {sub_questions}")
            return sub_questions

        except Exception as e:
            logger.error(f"❌ Decomposition failed: {e}")
            # 兜底：如果拆解失败，返回原问题作为唯一的子问题
            return [question]