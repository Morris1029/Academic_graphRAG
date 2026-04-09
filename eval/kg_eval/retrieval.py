from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import reload_config
from models.retriever.enhanced_kt_retriever import KTRetriever

from .construction import ConstructionBridge, ensure_dataset_registered
from .data_models import RetrievalMetricSummary


RELATION_QUERY_TEMPLATES = {
    "撰写": lambda paper: f"{paper} 的作者是谁",
    "发表于": lambda paper: f"{paper} 发表于什么期刊",
    "应用于": lambda paper: f"{paper} 应用于什么场景",
    "聚焦": lambda paper: f"{paper} 聚焦什么研究主题",
    "采用": lambda paper: f"{paper} 采用了什么研究方法",
    "提出": lambda paper: f"{paper} 提出了什么",
}


def build_retriever(
    dataset_name: str,
    main_config_path: str,
    top_k: int,
    override_json_path: Optional[str] = None,
    override_cache_dir: Optional[str] = None,
) -> KTRetriever:
    config = reload_config(main_config_path)
    dataset_config = ensure_dataset_registered(config, dataset_name)
    retriever = KTRetriever(
        dataset_name,
        json_path=override_json_path or dataset_config.graph_output,
        recall_paths=config.retrieval.recall_paths,
        schema_path=dataset_config.schema_path,
        top_k=top_k,
        mode=config.triggers.mode,
        config=config,
        cache_dir=override_cache_dir or config.retrieval.cache_dir,
    )
    retriever.build_indices()
    return retriever


def _node_display_name(retriever: Any, node_id: str) -> str:
    graph = getattr(retriever, "graph", None)
    if graph is None or node_id not in graph.nodes:
        return str(node_id)
    props = graph.nodes[node_id].get("properties", {}) or {}
    return str(props.get("name", node_id)).strip()


def _node_schema_type(retriever: Any, node_id: str) -> str:
    graph = getattr(retriever, "graph", None)
    if graph is None or node_id not in graph.nodes:
        return ""
    props = graph.nodes[node_id].get("properties", {}) or {}
    return str(props.get("schema_type", "")).strip()


def _parse_entity_with_type(raw: str) -> Tuple[str, str]:
    text = str(raw or "").strip()
    match = re.match(r"^(.*?)(?:\s*\[schema_type:\s*([^\]]+)\])?$", text)
    if not match:
        return text, ""
    return match.group(1).strip(), str(match.group(2) or "").strip()


def _parse_formatted_triple(triple_text: str, bridge: ConstructionBridge) -> Optional[Tuple[str, str, str]]:
    text = str(triple_text or "").strip()
    if not text:
        return None

    text = re.sub(r"\s+\[score:\s*[-+]?[0-9]*\.?[0-9]+\]\s*$", "", text)
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]

    parts = [part.strip() for part in text.split(",", 2)]
    if len(parts) != 3:
        return None

    head_text, head_type = _parse_entity_with_type(parts[0])
    tail_text, tail_type = _parse_entity_with_type(parts[2])
    head = bridge.normalize_entity_name(head_text, head_type)
    relation = bridge.normalize_relation_name(parts[1])
    tail = bridge.normalize_entity_name(tail_text, tail_type)
    if not head or not relation or not tail:
        return None
    return (head, relation, tail)


def _paper_name_for_triple(
    sample: Dict[str, Any],
    triple: List[str],
    entity_types: Dict[str, str],
    bridge: ConstructionBridge,
) -> Optional[str]:
    sample_title = str(sample.get("meta", {}).get("title", "")).strip()
    normalized_sample_title = bridge.normalize_entity_name(sample_title, "论文")
    head, _, tail = triple
    head_type = entity_types.get(head, "")
    tail_type = entity_types.get(tail, "")

    if bridge.is_paper_type(head_type) or bridge.normalize_entity_name(head, head_type) == normalized_sample_title:
        return head
    if bridge.is_paper_type(tail_type) or bridge.normalize_entity_name(tail, tail_type) == normalized_sample_title:
        return tail
    return None


def build_gold_triple_query(
    sample: Dict[str, Any],
    triple: List[str],
    entity_types: Dict[str, str],
    bridge: ConstructionBridge,
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(triple, list) or len(triple) < 3:
        return None, "invalid_triple"

    paper_name = _paper_name_for_triple(sample, triple, entity_types, bridge)
    if not paper_name:
        return None, "not_paper_centered"

    relation = bridge.normalize_relation_name(triple[1])
    template = RELATION_QUERY_TEMPLATES.get(relation)
    if template is None:
        return None, f"missing_template:{relation}"
    return template(paper_name), None


def evaluate_sample_retrieval(
    sample: Dict[str, Any],
    retriever: Any,
    bridge: ConstructionBridge,
    paper_top_k: int,
    triple_top_k: int,
) -> Dict[str, Any]:
    title = str(sample.get("meta", {}).get("title", "")).strip()
    normalized_title = bridge.normalize_entity_name(title, "论文")
    _, retrieve_result = retriever.retrieve(title)
    top_nodes = retrieve_result.get("path1_results", {}).get("top_nodes", []) or []
    top_node_names = [
        bridge.normalize_entity_name(_node_display_name(retriever, node_id), _node_schema_type(retriever, node_id))
        for node_id in top_nodes[:paper_top_k]
    ]
    paper_hit = normalized_title in top_node_names

    gold = sample.get("kg_eval", {}).get("gold", {}) or {}
    extraction = gold.get("extraction", {}) or {}
    entity_types = extraction.get("entity_types", {}) or {}

    triple_details: List[Dict[str, Any]] = []
    eligible = 0
    hit_count = 0
    skipped = 0

    for triple in extraction.get("triples", []) or []:
        query, skip_reason = build_gold_triple_query(sample, triple, entity_types, bridge)
        normalized_gold_triple = None
        if isinstance(triple, list) and len(triple) >= 3:
            normalized_gold_triple = (
                bridge.normalize_entity_name(triple[0], entity_types.get(triple[0], "")),
                bridge.normalize_relation_name(triple[1]),
                bridge.normalize_entity_name(triple[2], entity_types.get(triple[2], "")),
            )

        if not query:
            skipped += 1
            triple_details.append(
                {
                    "gold_triple": triple,
                    "query": None,
                    "skip_reason": skip_reason,
                    "hit": False,
                    "retrieved_triples": [],
                }
            )
            continue

        eligible += 1
        retrieval_results, _ = retriever.process_retrieval_results(query, top_k=triple_top_k)
        retrieved_triples = []
        for triple_text in retrieval_results.get("triples", [])[:triple_top_k]:
            parsed = _parse_formatted_triple(triple_text, bridge)
            if parsed:
                retrieved_triples.append(parsed)

        hit = normalized_gold_triple in retrieved_triples if normalized_gold_triple else False
        if hit:
            hit_count += 1

        triple_details.append(
            {
                "gold_triple": triple,
                "query": query,
                "skip_reason": None,
                "hit": hit,
                "retrieved_triples": retrieved_triples,
            }
        )

    return {
        "paper_node_hit@5": paper_hit,
        "paper_top_nodes": top_node_names,
        "gold_triple_total": eligible,
        "gold_triple_hit_count": hit_count,
        "skipped_triples": skipped,
        "gold_triple_hit_rate": hit_count / eligible if eligible else 0.0,
        "triple_details": triple_details,
    }


def aggregate_retrieval_results(per_sample_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    summary = RetrievalMetricSummary()
    for result in per_sample_results:
        summary.paper_node_total += 1
        if result.get("paper_node_hit@5"):
            summary.paper_node_hit_count += 1
        summary.gold_triple_total += int(result.get("gold_triple_total", 0))
        summary.gold_triple_hit_count += int(result.get("gold_triple_hit_count", 0))
        summary.skipped_gold_triples += int(result.get("skipped_triples", 0))
    return summary.to_dict()
