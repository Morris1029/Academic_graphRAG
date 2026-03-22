from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List


def default_gold_payload() -> Dict[str, Any]:
    return {
        "status": "pending",
        "generator_model": "",
        "reviewer": "",
        "review_notes": "",
        "updated_at": "",
        "extraction": {
            "entity_types": {},
            "triples": [],
            "attributes": {},
        },
    }


def _ensure_gold_structure(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(record)
    kg_eval = normalized.setdefault("kg_eval", {})
    if not isinstance(kg_eval, dict):
        kg_eval = {}
        normalized["kg_eval"] = kg_eval

    gold = kg_eval.setdefault("gold", default_gold_payload())
    if not isinstance(gold, dict):
        gold = default_gold_payload()
        kg_eval["gold"] = gold

    default_gold = default_gold_payload()
    for key, default_value in default_gold.items():
        if key not in gold:
            gold[key] = copy.deepcopy(default_value)

    extraction = gold.get("extraction", {})
    if not isinstance(extraction, dict):
        extraction = copy.deepcopy(default_gold["extraction"])
        gold["extraction"] = extraction

    for key in ("entity_types", "attributes"):
        value = extraction.get(key)
        if not isinstance(value, dict):
            extraction[key] = {}

    triples = extraction.get("triples")
    if not isinstance(triples, list):
        extraction["triples"] = []

    return normalized


def _validate_record(record: Dict[str, Any], index: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Sample #{index} is not an object")
    if not str(record.get("id", "")).strip():
        raise ValueError(f"Sample #{index} is missing id")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Sample #{index} is missing meta")
    if not str(meta.get("title", "")).strip():
        raise ValueError(f"Sample #{index} is missing meta.title")


def load_samples(sample_path: str) -> List[Dict[str, Any]]:
    path = Path(sample_path)
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Sample file must contain a list: {sample_path}")

    records: List[Dict[str, Any]] = []
    for index, record in enumerate(payload, start=1):
        _validate_record(record, index)
        records.append(_ensure_gold_structure(record))
    return records


def save_samples(sample_path: str, records: List[Dict[str, Any]]) -> None:
    normalized = [_ensure_gold_structure(record) for record in records]
    path = Path(sample_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def count_gold_statuses(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        status = str(record.get("kg_eval", {}).get("gold", {}).get("status", "pending")).strip() or "pending"
        counts[status] = counts.get(status, 0) + 1
    return counts
