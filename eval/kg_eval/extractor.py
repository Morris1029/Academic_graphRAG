from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

from ..rag_eval.llm_client import StandaloneLLMClient, resolve_model_profile

from .construction import ConstructionBridge
from .data_models import ExtractionRunResult


class ExtractionService:
    def __init__(
        self,
        bridge: ConstructionBridge,
        models_cfg: Dict[str, Dict[str, Any]],
        roles_cfg: Dict[str, Dict[str, Any]],
    ):
        self.bridge = bridge
        self.models_cfg = models_cfg
        self.roles_cfg = roles_cfg
        self._clients: Dict[Tuple[str, str], StandaloneLLMClient] = {}

    def _get_client(self, model_name: str, role_name: str) -> StandaloneLLMClient:
        key = (model_name, role_name)
        if key not in self._clients:
            role_cfg = self.roles_cfg.get(role_name, {})
            profile = resolve_model_profile(model_name, self.models_cfg, role_cfg=role_cfg)
            self._clients[key] = StandaloneLLMClient(profile)
        return self._clients[key]

    def extract(self, sample: Dict[str, Any], model_name: str, role_name: str) -> ExtractionRunResult:
        prompt = self.bridge.build_prompt(sample)
        client = self._get_client(model_name, role_name)
        raw_response = ""

        try:
            raw_response = client.call_api(prompt)
            extraction = self.bridge.parse_response(raw_response)
            return ExtractionRunResult(
                model_name=model_name,
                prompt=prompt,
                raw_response=raw_response,
                extraction=extraction,
            )
        except Exception as exc:
            return ExtractionRunResult(
                model_name=model_name,
                prompt=prompt,
                raw_response=raw_response,
                extraction=self.bridge.parse_response(raw_response),
                error=f"{type(exc).__name__}: {exc}",
            )

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
