from __future__ import annotations

from typing import Any, Dict, Iterable, Set, Tuple

from .construction import ConstructionBridge
from .data_models import MetricCounts


def _attribute_accuracy(counts: MetricCounts) -> float:
    denom = counts.tp + counts.fp + counts.fn
    return counts.tp / denom if denom else 0.0


def _compare_sets(gold_set: Set[Tuple[str, ...]], candidate_set: Set[Tuple[str, ...]]) -> Dict[str, Any]:
    tp_items = sorted(gold_set & candidate_set)
    fp_items = sorted(candidate_set - gold_set)
    fn_items = sorted(gold_set - candidate_set)
    counts = MetricCounts(tp=len(tp_items), fp=len(fp_items), fn=len(fn_items))
    return {
        "counts": counts,
        "tp_items": tp_items,
        "fp_items": fp_items,
        "fn_items": fn_items,
    }


def compare_extractions(
    bridge: ConstructionBridge,
    gold_extraction: Any,
    candidate_extraction: Any,
) -> Dict[str, Any]:
    normalized_gold = bridge.normalize_extraction(gold_extraction)
    normalized_candidate = bridge.normalize_extraction(candidate_extraction)

    entity_result = _compare_sets(normalized_gold["entities"], normalized_candidate["entities"])
    relation_result = _compare_sets(normalized_gold["triples"], normalized_candidate["triples"])
    attribute_result = _compare_sets(normalized_gold["attributes"], normalized_candidate["attributes"])

    return {
        "normalized_gold": {
            "entities": sorted(normalized_gold["entities"]),
            "triples": sorted(normalized_gold["triples"]),
            "attributes": sorted(normalized_gold["attributes"]),
        },
        "normalized_candidate": {
            "entities": sorted(normalized_candidate["entities"]),
            "triples": sorted(normalized_candidate["triples"]),
            "attributes": sorted(normalized_candidate["attributes"]),
        },
        "entity_metrics": {
            **entity_result["counts"].to_dict(),
            "tp_items": entity_result["tp_items"],
            "fp_items": entity_result["fp_items"],
            "fn_items": entity_result["fn_items"],
        },
        "relation_metrics": {
            **relation_result["counts"].to_dict(),
            "tp_items": relation_result["tp_items"],
            "fp_items": relation_result["fp_items"],
            "fn_items": relation_result["fn_items"],
        },
        "attribute_metrics": {
            "tp": attribute_result["counts"].tp,
            "fp": attribute_result["counts"].fp,
            "fn": attribute_result["counts"].fn,
            "accuracy": _attribute_accuracy(attribute_result["counts"]),
            "tp_items": attribute_result["tp_items"],
            "fp_items": attribute_result["fp_items"],
            "fn_items": attribute_result["fn_items"],
        },
    }


def aggregate_metric_blocks(per_sample_blocks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    entity_counts = MetricCounts()
    relation_counts = MetricCounts()
    attribute_counts = MetricCounts()

    for block in per_sample_blocks:
        entity_counts.tp += int(block["entity_metrics"]["tp"])
        entity_counts.fp += int(block["entity_metrics"]["fp"])
        entity_counts.fn += int(block["entity_metrics"]["fn"])

        relation_counts.tp += int(block["relation_metrics"]["tp"])
        relation_counts.fp += int(block["relation_metrics"]["fp"])
        relation_counts.fn += int(block["relation_metrics"]["fn"])

        attribute_counts.tp += int(block["attribute_metrics"]["tp"])
        attribute_counts.fp += int(block["attribute_metrics"]["fp"])
        attribute_counts.fn += int(block["attribute_metrics"]["fn"])

    return {
        "entity_prf": entity_counts.to_dict(),
        "relation_prf": relation_counts.to_dict(),
        "attribute_accuracy": {
            "tp": attribute_counts.tp,
            "fp": attribute_counts.fp,
            "fn": attribute_counts.fn,
            "accuracy": _attribute_accuracy(attribute_counts),
        },
    }
