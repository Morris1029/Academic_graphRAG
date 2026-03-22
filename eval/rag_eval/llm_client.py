from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from utils.logger import logger


@dataclass
class LLMProfile:
    name: str
    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    timeout_seconds: Optional[float] = None
    api_version: Optional[str] = None


def load_eval_env(env_path: str) -> None:
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        logger.info(f"Loaded evaluation env file: {env_path}")
    else:
        logger.warning(f"Evaluation env file not found: {env_path}")


def _resolve_value(profile_cfg: Dict, key: str, env_key_field: str, default=None):
    env_name = profile_cfg.get(env_key_field)
    if env_name:
        return os.getenv(env_name, default)
    return profile_cfg.get(key, default)


def resolve_model_profile(
    model_name: str,
    models_cfg: Dict[str, Dict[str, Any]],
    role_cfg: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> LLMProfile:
    if model_name not in models_cfg:
        raise ValueError(f"LLM model '{model_name}' not found in config")

    model_cfg = models_cfg[model_name]
    merged_cfg: Dict[str, Any] = dict(model_cfg)
    if role_cfg:
        merged_cfg.update(role_cfg)
    if overrides:
        merged_cfg.update({key: value for key, value in overrides.items() if value is not None})

    provider = str(
        _resolve_value(merged_cfg, "provider", "provider_env", "openai")
    ).lower()
    model = _resolve_value(merged_cfg, "model", "model_env")
    base_url = _resolve_value(merged_cfg, "base_url", "base_url_env")
    api_key = _resolve_value(merged_cfg, "api_key", "api_key_env")
    api_version = _resolve_value(merged_cfg, "api_version", "api_version_env")
    temperature = float(merged_cfg.get("temperature", 0.0))
    timeout_seconds = merged_cfg.get("timeout_seconds")
    timeout_seconds = float(timeout_seconds) if timeout_seconds is not None else None

    if not model:
        raise ValueError(f"Model '{model_name}' is missing model/model_env")
    if not base_url:
        raise ValueError(f"Model '{model_name}' is missing base_url/base_url_env")
    if not api_key:
        raise ValueError(f"Model '{model_name}' is missing api_key/api_key_env")

    return LLMProfile(
        name=model_name,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        api_version=api_version,
    )


class StandaloneLLMClient:
    def __init__(self, profile: LLMProfile):
        self.profile = profile
        self._lc_model = self._build_model()

    def _build_model(self):
        if self.profile.provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=self.profile.base_url,
                api_key=self.profile.api_key,
                api_version=self.profile.api_version or "2025-01-01-preview",
                deployment_name=self.profile.model,
                temperature=self.profile.temperature,
                timeout=self.profile.timeout_seconds,
            )

        return ChatOpenAI(
            model=self.profile.model,
            base_url=self.profile.base_url,
            api_key=self.profile.api_key,
            temperature=self.profile.temperature,
            timeout=self.profile.timeout_seconds,
        )

    def call_api(self, content: str) -> str:
        response = self._lc_model.invoke(content)
        raw_content = response.content
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)
        return self._clean_llm_content(raw_content)

    def _clean_llm_content(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
        if t.startswith("```") and t.endswith("```") and len(t) >= 6:
            t = t[3:-3].strip()
        if t.lower().startswith("json\n"):
            t = t.split("\n", 1)[1].strip()
        return t


@contextmanager
def temporary_rag_env(profile: LLMProfile) -> Iterator[None]:
    env_updates = {
        "RAG_LLM_MODEL": profile.model,
        "RAG_LLM_BASE_URL": profile.base_url,
        "RAG_LLM_API_KEY": profile.api_key,
        "RAG_OPENAI_PROVIDER": profile.provider,
    }
    if profile.api_version:
        env_updates["RAG_API_VERSION"] = profile.api_version

    previous_values = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            if value is not None:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
