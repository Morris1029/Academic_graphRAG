from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from ..rag_eval.llm_client import StandaloneLLMClient, resolve_profile

from .construction import ConstructionBridge
from .data_models import ExtractionRunResult


class ExtractionService:
    def __init__(self, bridge: ConstructionBridge, profiles_cfg: Dict[str, Dict[str, Any]]):
        self.bridge = bridge
        self.profiles_cfg = profiles_cfg
        self._clients: Dict[str, StandaloneLLMClient] = {}

    def _get_client(self, profile_name: str) -> StandaloneLLMClient:
        if profile_name not in self._clients:
            profile = resolve_profile(profile_name, self.profiles_cfg)
            self._clients[profile_name] = StandaloneLLMClient(profile)
        return self._clients[profile_name]

    def extract(self, sample: Dict[str, Any], profile_name: str) -> ExtractionRunResult:
        prompt = self.bridge.build_prompt(sample)
        client = self._get_client(profile_name)
        raw_response = ""

        try:
            raw_response = client.call_api(prompt)
            extraction = self.bridge.parse_response(raw_response)
            return ExtractionRunResult(
                profile_name=profile_name,
                prompt=prompt,
                raw_response=raw_response,
                extraction=extraction,
            )
        except Exception as exc:
            return ExtractionRunResult(
                profile_name=profile_name,
                prompt=prompt,
                raw_response=raw_response,
                extraction=self.bridge.parse_response(raw_response),
                error=f"{type(exc).__name__}: {exc}",
            )

    def build_gold_payload(self, sample: Dict[str, Any], profile_name: str) -> Dict[str, Any]:
        result = self.extract(sample, profile_name)
        notes = sample.get("kg_eval", {}).get("gold", {}).get("review_notes", "")
        if result.error:
            auto_note = f"AUTO_ERROR: {result.error}"
            notes = f"{notes}\n{auto_note}".strip()

        return {
            "status": "draft",
            "generator_profile": profile_name,
            "reviewer": sample.get("kg_eval", {}).get("gold", {}).get("reviewer", ""),
            "review_notes": notes,
            "updated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "extraction": result.extraction.to_dict(),
        }
