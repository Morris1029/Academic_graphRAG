import os
import re
from typing import Optional

# 引入 LangChain 的 OpenAI 集成
# 请确保已安装: pip install langchain-openai langchain-core
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from utils.logger import logger

load_dotenv()


class LLMCompletionCall:
    def __init__(self):
        """
        初始化 LLM 客户端。
        内部现在使用 LangChain 的 ChatModel 接口，
        但对外保持原有兼容性，同时提供获取 LangChain 对象的能力。
        """
        self.llm_model_name = os.getenv("LLM_MODEL", "deepseek-chat")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")

        if not self.llm_api_key:
            raise ValueError("LLM API key not provided")

        self.openai_provider = os.getenv("OPENAI_PROVIDER", "openai").lower()

        # --- LangChain 对象初始化 ---
        # 统一设置 temperature=0.3 以保持与原项目一致的生成稳定性
        if self.openai_provider == "azure":
            self.api_version = os.getenv("API_VERSION", "2025-01-01-preview")
            self._lc_model = AzureChatOpenAI(
                azure_endpoint=self.llm_base_url,
                api_key=self.llm_api_key,
                api_version=self.api_version,
                deployment_name=self.llm_model_name,
                temperature=0.3
            )
        else:
            # 适用于 OpenAI 以及兼容 OpenAI 接口的模型（如 DeepSeek, Moonshot 等）
            self._lc_model = ChatOpenAI(
                model=self.llm_model_name,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                temperature=0.3
            )

    def call_api(self, content: str) -> str:
        """
        [兼容接口] 供原有 Youtu-GraphRAG 代码调用。
        输入 Prompt 字符串，返回清洗后的文本字符串。

        Args:
            content: Prompt content

        Returns:
            Generated text response
        """
        try:
            # 使用 LangChain 的 invoke 方法
            # LangChain 会自动将 str 转换为 HumanMessage
            response = self._lc_model.invoke(content)

            # 获取 content 内容 (可能是 str 或 list，通常 text 模型返回 str)
            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)

            clean_completion = self._clean_llm_content(raw_content)
            return clean_completion

        except Exception as e:
            logger.error(f"LLM api calling failed via LangChain. Error: {e}")
            raise e

    @property
    def langchain_model(self):
        """
        [新接口] 供新的 Agent 代码获取 LangChain ChatModel 对象。
        用于 LangGraph 编排、Tool Binding 等高级功能。

        Usage:
            llm_client = LLMCompletionCall()
            agent_model = llm_client.langchain_model
            # agent_model.bind_tools(...)
        """
        return self._lc_model

    def _clean_llm_content(self, text: str) -> str:
        """
        [保持不变] 保持原有的清洗逻辑，确保下游 JSON 解析不报错。
        """
        if not isinstance(text, str):
            return ""
        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)

        # 移除 Markdown 代码块包裹
        fence_re = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE)
        m = fence_re.match(t)
        if m:
            t = m.group("body").strip()
        else:
            if t.startswith("```") and t.endswith("```") and len(t) >= 6:
                t = t[3:-3].strip()

        # 移除可能存在的 json\n 前缀
        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()

        return t