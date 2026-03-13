import os
import re

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from utils.logger import logger

load_dotenv()


class LLMCompletionCall:
    def __init__(self, scope: str = "default"):
        """
        Initialize an LLM client for a specific workflow scope.

        Supported scopes:
        - default: legacy single-model path
        - kg: graph extraction and community summaries
        - rag: question decomposition and retrieval QA
        """
        self.scope = (scope or "default").lower()
        self.llm_model_name, self.llm_base_url, self.llm_api_key = self._resolve_config(self.scope)

        if not self.llm_api_key:
            raise ValueError(f"{self.scope.upper()} LLM API key not provided")

        self.openai_provider = os.getenv("OPENAI_PROVIDER", "openai").lower()
        self._lc_model = self._build_model()

    def _resolve_config(self, scope: str):
        if scope == "kg":
            return (
                os.getenv("KG_LLM_MODEL", "qwen3-max"),
                os.getenv("KG_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                os.getenv("KG_LLM_API_KEY", ""),
            )

        if scope == "rag":
            return (
                os.getenv("RAG_LLM_MODEL", "deepseek-chat"),
                os.getenv("RAG_LLM_BASE_URL", "https://api.deepseek.com"),
                os.getenv("RAG_LLM_API_KEY", ""),
            )

        return (
            os.getenv("LLM_MODEL", "deepseek-chat"),
            os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            os.getenv("LLM_API_KEY", ""),
        )

    def _build_model(self):
        if self.scope in {"kg", "rag"}:
            return ChatOpenAI(
                model=self.llm_model_name,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
                temperature=0.3,
            )

        if self.openai_provider == "azure":
            api_version = os.getenv("API_VERSION", "2025-01-01-preview")
            return AzureChatOpenAI(
                azure_endpoint=self.llm_base_url,
                api_key=self.llm_api_key,
                api_version=api_version,
                deployment_name=self.llm_model_name,
                temperature=0.3,
            )

        return ChatOpenAI(
            model=self.llm_model_name,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            temperature=0.3,
        )

    def call_api(self, content: str) -> str:
        """
        Keep the original project-facing API stable.
        """
        try:
            response = self._lc_model.invoke(content)
            raw_content = response.content
            if not isinstance(raw_content, str):
                raw_content = str(raw_content)
            return self._clean_llm_content(raw_content)
        except Exception as e:
            logger.error(f"LLM api calling failed via LangChain. scope={self.scope} error={e}")
            raise e

    @property
    def langchain_model(self):
        return self._lc_model

    def _clean_llm_content(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)

        fence_re = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE)
        m = fence_re.match(t)
        if m:
            t = m.group("body").strip()
        else:
            if t.startswith("```") and t.endswith("```") and len(t) >= 6:
                t = t[3:-3].strip()

        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()

        return t
