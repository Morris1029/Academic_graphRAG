from __future__ import annotations

from typing import Any, Dict, Iterable, Set, Tuple

from .construction import ConstructionBridge
from .data_models import MetricCounts

STRUCTURED_RELATIONS = {"撰写", "隶属", "发表于"}
STRUCTURED_ATTR_PREFIXES = ("年份:", "关键词:", "来源:")



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
    gold_title: Optional[str] = None,
) -> Dict[str, Any]:
    # 如果没有显式提供 gold_title，尝试从 gold_extraction 中提取
    if not gold_title and isinstance(gold_extraction, dict):
        etypes = gold_extraction.get("entity_types", {})
        for name, etype in etypes.items():
            if bridge.normalize_entity_type(etype) == "论文":
                gold_title = name
                break

    normalized_gold = bridge.normalize_extraction(gold_extraction, gold_title=gold_title)
    normalized_candidate = bridge.normalize_extraction(candidate_extraction, gold_title=gold_title)

    gold_struct_triples = {t for t in normalized_gold["triples"] if t[1] in STRUCTURED_RELATIONS}
    cand_struct_triples = {t for t in normalized_candidate["triples"] if t[1] in STRUCTURED_RELATIONS}
    gold_unstruct_triples = normalized_gold["triples"] - gold_struct_triples
    cand_unstruct_triples = normalized_candidate["triples"] - cand_struct_triples

    gold_struct_attrs = {a for a in normalized_gold["attributes"] if any(a[1].startswith(p) for p in STRUCTURED_ATTR_PREFIXES)}
    cand_struct_attrs = {a for a in normalized_candidate["attributes"] if any(a[1].startswith(p) for p in STRUCTURED_ATTR_PREFIXES)}
    gold_unstruct_attrs = normalized_gold["attributes"] - gold_struct_attrs
    cand_unstruct_attrs = normalized_candidate["attributes"] - cand_struct_attrs

    entity_result = _compare_sets(normalized_gold["entities"], normalized_candidate["entities"])
    
    relation_overall = _compare_sets(normalized_gold["triples"], normalized_candidate["triples"])
    relation_struct = _compare_sets(gold_struct_triples, cand_struct_triples)
    relation_unstruct = _compare_sets(gold_unstruct_triples, cand_unstruct_triples)

    attribute_overall = _compare_sets(normalized_gold["attributes"], normalized_candidate["attributes"])
    attribute_struct = _compare_sets(gold_struct_attrs, cand_struct_attrs)
    attribute_unstruct = _compare_sets(gold_unstruct_attrs, cand_unstruct_attrs)

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
            "overall": relation_overall["counts"].to_dict(),
            "structured": relation_struct["counts"].to_dict(),
            "unstructured": relation_unstruct["counts"].to_dict(),
            "tp_items": relation_overall["tp_items"],
            "fp_items": relation_overall["fp_items"],
            "fn_items": relation_overall["fn_items"],
        },
        "attribute_metrics": {
            "overall": {
                "tp": attribute_overall["counts"].tp,
                "fp": attribute_overall["counts"].fp,
                "fn": attribute_overall["counts"].fn,
                "accuracy": _attribute_accuracy(attribute_overall["counts"])
            },
            "structured": {
                "tp": attribute_struct["counts"].tp,
                "fp": attribute_struct["counts"].fp,
                "fn": attribute_struct["counts"].fn,
                "accuracy": _attribute_accuracy(attribute_struct["counts"])
            },
            "unstructured": {
                "tp": attribute_unstruct["counts"].tp,
                "fp": attribute_unstruct["counts"].fp,
                "fn": attribute_unstruct["counts"].fn,
                "accuracy": _attribute_accuracy(attribute_unstruct["counts"])
            },
            "tp_items": attribute_overall["tp_items"],
            "fp_items": attribute_overall["fp_items"],
            "fn_items": attribute_overall["fn_items"],
        },
    }


def aggregate_metric_blocks(per_sample_blocks: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    entity_counts = MetricCounts()
    
    relation_counts_overall = MetricCounts()
    relation_counts_struct = MetricCounts()
    relation_counts_unstruct = MetricCounts()

    attribute_counts_overall = MetricCounts()
    attribute_counts_struct = MetricCounts()
    attribute_counts_unstruct = MetricCounts()

    for block in per_sample_blocks:
        entity_counts.tp += int(block["entity_metrics"]["tp"])
        entity_counts.fp += int(block["entity_metrics"]["fp"])
        entity_counts.fn += int(block["entity_metrics"]["fn"])

        rm = block["relation_metrics"]
        relation_counts_overall.tp += int(rm["overall"]["tp"])
        relation_counts_overall.fp += int(rm["overall"]["fp"])
        relation_counts_overall.fn += int(rm["overall"]["fn"])
        relation_counts_struct.tp += int(rm["structured"]["tp"])
        relation_counts_struct.fp += int(rm["structured"]["fp"])
        relation_counts_struct.fn += int(rm["structured"]["fn"])
        relation_counts_unstruct.tp += int(rm["unstructured"]["tp"])
        relation_counts_unstruct.fp += int(rm["unstructured"]["fp"])
        relation_counts_unstruct.fn += int(rm["unstructured"]["fn"])

        am = block["attribute_metrics"]
        attribute_counts_overall.tp += int(am["overall"]["tp"])
        attribute_counts_overall.fp += int(am["overall"]["fp"])
        attribute_counts_overall.fn += int(am["overall"]["fn"])
        attribute_counts_struct.tp += int(am["structured"]["tp"])
        attribute_counts_struct.fp += int(am["structured"]["fp"])
        attribute_counts_struct.fn += int(am["structured"]["fn"])
        attribute_counts_unstruct.tp += int(am["unstructured"]["tp"])
        attribute_counts_unstruct.fp += int(am["unstructured"]["fp"])
        attribute_counts_unstruct.fn += int(am["unstructured"]["fn"])

    return {
        "entity_prf": entity_counts.to_dict(),
        "relation_prf": relation_counts_overall.to_dict(),
        "relation_prf_structured": relation_counts_struct.to_dict(),
        "relation_prf_unstructured": relation_counts_unstruct.to_dict(),
        "attribute_accuracy": {
            "tp": attribute_counts_overall.tp,
            "fp": attribute_counts_overall.fp,
            "fn": attribute_counts_overall.fn,
            "accuracy": _attribute_accuracy(attribute_counts_overall),
        },
        "attribute_accuracy_structured": {
            "tp": attribute_counts_struct.tp,
            "fp": attribute_counts_struct.fp,
            "fn": attribute_counts_struct.fn,
            "accuracy": _attribute_accuracy(attribute_counts_struct),
        },
        "attribute_accuracy_unstructured": {
            "tp": attribute_counts_unstruct.tp,
            "fp": attribute_counts_unstruct.fp,
            "fn": attribute_counts_unstruct.fn,
            "accuracy": _attribute_accuracy(attribute_counts_unstruct),
        },
    }
