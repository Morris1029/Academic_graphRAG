from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import json_repair

from utils.logger import logger

from .llm_client import LLMProfile, StandaloneLLMClient
from .models import EvaluationSample, JudgmentResult, QAPrediction


class LLMJudge:
    def __init__(
        self,
        profile: LLMProfile,
        prompt_path: str,
        dimensions: Dict[str, Dict],
        max_attempts: int = 2,
    ):
        self.profile = profile
        self.client = StandaloneLLMClient(profile)
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.dimensions = dimensions
        self.max_attempts = max(1, int(max_attempts))

    def _dimension_guide(self) -> str:
        lines = []
        for name, spec in self.dimensions.items():
            description = spec.get("description", "")
            weight = spec.get("weight", 0.0)
            lines.append(f"- {name}: {description} 权重={weight}")
        return "\n".join(lines)

    def _build_prompt(self, sample: EvaluationSample, prediction: QAPrediction) -> str:
        payload = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "question": sample.question,
            "reference_answer": sample.reference_answer,
            "eval_focus": sample.eval_focus,
            "predicted_answer": prediction.answer,
            "retrieved_triples": prediction.retrieved_triples[:25],
            "retrieved_chunks": [chunk[:1200] for chunk in prediction.retrieved_chunks[:12]],
        }

        prompt = self.prompt_template
        prompt = prompt.replace("{{DIMENSION_GUIDE}}", self._dimension_guide())
        prompt = prompt.replace(
            "{{DIMENSION_KEYS}}",
            json.dumps(list(self.dimensions.keys()), ensure_ascii=False),
        )
        prompt = prompt.replace(
            "{{INPUT_PAYLOAD}}",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        return prompt

    def _normalize_list(self, value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return [str(value).strip()]

    def _validate_result(self, data: Dict, question_id: str) -> JudgmentResult:
        raw_scores = data.get("scores", {})
        if not isinstance(raw_scores, dict):
            raise ValueError("Judge output 'scores' must be a JSON object")

        scores: Dict[str, float] = {}
        for name in self.dimensions:
            if name not in raw_scores:
                raise ValueError(f"Judge output missing score for '{name}'")
            score = float(raw_scores[name])
            if score < 1 or score > 5:
                raise ValueError(f"Score for '{name}' must be between 1 and 5")
            scores[name] = score

        weighted_score = 0.0
        for name, score in scores.items():
            weighted_score += score * float(self.dimensions[name].get("weight", 0.0))
        weighted_score = round(weighted_score, 4)

        confidence = float(data.get("judge_confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)

        return JudgmentResult(
            question_id=str(data.get("question_id", question_id)),
            scores=scores,
            weighted_score=weighted_score,
            verdict=str(data.get("verdict", "scored")).strip() or "scored",
            strengths=self._normalize_list(data.get("strengths")),
            weaknesses=self._normalize_list(data.get("weaknesses")),
            missing_points=self._normalize_list(data.get("missing_points")),
            hallucination_flags=self._normalize_list(data.get("hallucination_flags")),
            judge_confidence=confidence,
            raw_response=json.dumps(data, ensure_ascii=False),
        )

    def judge(self, sample: EvaluationSample, prediction: QAPrediction) -> JudgmentResult:
        prompt = self._build_prompt(sample, prediction)
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_response = self.client.call_api(prompt)
                parsed = json_repair.loads(raw_response)
                result = self._validate_result(parsed, sample.question_id)
                result.raw_response = raw_response
                return result
            except Exception as exc:
                last_error = exc
                logger.error(
                    f"Judge failed for question_id={sample.question_id} attempt={attempt}: {exc}"
                )
                time.sleep(min(2.0 * attempt, 5.0))

        return JudgmentResult(
            question_id=sample.question_id,
            scores={name: 0.0 for name in self.dimensions},
            weighted_score=0.0,
            verdict="judge_error",
            judge_confidence=0.0,
            error=str(last_error) if last_error else "Judge failed",
        )
