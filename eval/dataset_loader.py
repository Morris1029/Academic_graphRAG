from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from utils.logger import logger

from .models import EvaluationSample

REQUIRED_JSON_FIELDS = ("question_id", "question", "reference_answer")


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_question_set_from_json(
    json_path: str,
    max_samples: Optional[int] = None,
) -> List[EvaluationSample]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Question set JSON not found: {json_path}. "
            "Please run `python -m eval.dataset.excel_to_json` first."
        )

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Question set JSON must be a list: {json_path}")

    samples: List[EvaluationSample] = []
    seen_ids = set()

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Question set row {index} must be an object")

        for field_name in REQUIRED_JSON_FIELDS:
            if field_name not in record:
                raise ValueError(
                    f"Question set row {index} is missing required field '{field_name}'"
                )

        question_id = _normalize_text(record.get("question_id"))
        question = _normalize_text(record.get("question"))
        reference_answer = _normalize_text(record.get("reference_answer"))

        if not question_id:
            raise ValueError(f"Question set row {index} has empty question_id")
        if not question:
            raise ValueError(f"Question set row {index} has empty question")
        if not reference_answer:
            raise ValueError(f"Question set row {index} has empty reference_answer")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id '{question_id}' in {json_path}")
        seen_ids.add(question_id)

        samples.append(
            EvaluationSample(
                question_id=question_id,
                question_type=_normalize_text(record.get("question_type")),
                question=question,
                reference_answer=reference_answer,
                eval_focus=_normalize_text(record.get("eval_focus")),
                source_sheet=_normalize_text(record.get("source_sheet")),
                row_number=int(record.get("row_number", 0) or 0),
            )
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    logger.info(f"Loaded {len(samples)} evaluation samples from JSON {json_path}")
    return samples


def load_question_set(
    question_set_path: str,
    max_samples: Optional[int] = None,
) -> List[EvaluationSample]:
    return load_question_set_from_json(
        json_path=question_set_path,
        max_samples=max_samples,
    )
