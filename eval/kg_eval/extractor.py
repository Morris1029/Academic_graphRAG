from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from ..rag_eval.llm_client import StandaloneLLMClient, resolve_model_profile

from .construction import ConstructionBridge
from .data_models import ExtractionRunResult


class ExtractionService:
    def __init__(
        self,
        bridge: ConstructionBridge,
        models_cfg: Dict[str, Dict[str, Any]],
        roles_cfg: Dict[str, Dict[str, Any]],
        cache_dir: Optional[str] = None,
    ):
        self.bridge = bridge
        self.models_cfg = models_cfg
        self.roles_cfg = roles_cfg
        self._clients: Dict[Tuple[str, str], StandaloneLLMClient] = {}
        self.cache_dir = cache_dir or "eval/results/kg_eval/.extract_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_client(self, model_name: str, role_name: str) -> StandaloneLLMClient:
        key = (model_name, role_name)
        if key not in self._clients:
            role_cfg = self.roles_cfg.get(role_name, {})
            profile = resolve_model_profile(model_name, self.models_cfg, role_cfg=role_cfg)
            self._clients[key] = StandaloneLLMClient(profile)
        return self._clients[key]

    def _cache_key(self, sample: Dict[str, Any], model_name: str) -> str:
        sample_id = str(sample.get("id", "")).strip()
        raw = f"{sample_id}::{model_name}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, sample: Dict[str, Any], model_name: str) -> str:
        return os.path.join(self.cache_dir, f"{self._cache_key(sample, model_name)}.json")

    def _load_cache(self, sample: Dict[str, Any], model_name: str) -> Optional[ExtractionRunResult]:
        path = self._cache_path(sample, model_name)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            extraction = self.bridge.parse_response(data.get("raw_response", ""))
            return ExtractionRunResult(
                model_name=data.get("model_name", model_name),
                prompt=data.get("prompt", ""),
                raw_response=data.get("raw_response", ""),
                extraction=extraction,
                error=data.get("error"),
            )
        except Exception:
            return None

    def _save_cache(self, sample: Dict[str, Any], result: ExtractionRunResult) -> None:
        path = self._cache_path(sample, result.model_name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": result.model_name,
                        "prompt": result.prompt,
                        "raw_response": result.raw_response,
                        "error": result.error,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception:
            pass  # 缓存写入失败不影响主流程

    def extract(self, sample: Dict[str, Any], model_name: str, role_name: str) -> ExtractionRunResult:
        # 尝试从磁盘缓存加载（仅 candidate 角色缓存，gold 生成时始终重新请求）
        if role_name == "candidate":
            cached = self._load_cache(sample, model_name)
            if cached is not None:
                return cached

        prompt = self.bridge.build_prompt(sample)
        client = self._get_client(model_name, role_name)
        raw_response = ""

        try:
            raw_response = client.call_api(prompt)
            extraction = self.bridge.parse_response(raw_response)
            result = ExtractionRunResult(
                model_name=model_name,
                prompt=prompt,
                raw_response=raw_response,
                extraction=extraction,
            )
        except Exception as exc:
            result = ExtractionRunResult(
                model_name=model_name,
                prompt=prompt,
                raw_response=raw_response,
                extraction=self.bridge.parse_response(raw_response),
                error=f"{type(exc).__name__}: {exc}",
            )

        # 写入缓存（仅 candidate 角色）
        if role_name == "candidate":
            self._save_cache(sample, result)
        return result

    def build_gold_payload(self, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        result = self.extract(sample, model_name, role_name="gold")
        notes = sample.get("kg_eval", {}).get("gold", {}).get("review_notes", "")
        if result.error:
            auto_note = f"AUTO_ERROR: {result.error}"
            notes = f"{notes}\n{auto_note}".strip()

        return {
            "status": "draft",
            "generator_model": model_name,
            "reviewer": sample.get("kg_eval", {}).get("gold", {}).get("reviewer", ""),
            "review_notes": notes,
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "extraction": result.extraction.to_dict(),
        }
