from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from utils.dataset_audit import audit_dataset

from .construction import ConstructionBridge


def summarize_consistency_audit(dataset_name: str) -> Dict[str, Any]:
    raw = audit_dataset(dataset_name)
    return {
        "dataset_name": dataset_name,
        "source_docs_total": int(raw.get("source_docs", 0)),
        "chunk_total": int(raw.get("chunk_records", 0)),
        "paper_node_total": int(raw.get("graph_papers", 0)),
        "missing_doc_uids_in_graph": list(raw.get("missing_doc_uids_in_graph", [])),
        "missing_doc_uids_in_chunks": list(raw.get("missing_doc_uids_in_chunks", [])),
        "raw_audit": raw,
    }


def _load_relationships(graph_path: str) -> List[Dict[str, Any]]:
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Graph payload must be a list: {graph_path}")
    return payload


def _node_fingerprint(node: Dict[str, Any]) -> str:
    payload = {
        "label": node.get("label", ""),
        "properties": node.get("properties", {}) or {},
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def reconstruct_graph_from_relationships(graph_path: str) -> Tuple[nx.MultiDiGraph, List[Dict[str, Any]]]:
    relationships = _load_relationships(graph_path)
    graph = nx.MultiDiGraph()

    for relationship in relationships:
        start_node = relationship.get("start_node", {}) or {}
        end_node = relationship.get("end_node", {}) or {}
        relation = str(relationship.get("relation", "")).strip()
        edge_properties = relationship.get("edge_properties", {}) or {}
        if not isinstance(edge_properties, dict):
            edge_properties = {}

        for node in (start_node, end_node):
            node_id = _node_fingerprint(node)
            if node_id not in graph:
                graph.add_node(
                    node_id,
                    label=node.get("label", ""),
                    properties=node.get("properties", {}) or {},
                )

        graph.add_edge(
            _node_fingerprint(start_node),
            _node_fingerprint(end_node),
            relation=relation,
            **edge_properties,
        )

    return graph, relationships


def evaluate_graph_structure(
    graph_path: str,
    bridge: ConstructionBridge,
    cross_doc_precision: float | None = None,
) -> Dict[str, Any]:
    graph, relationships = reconstruct_graph_from_relationships(graph_path)
    primary_nodes = []
    duplicate_groups: Dict[Tuple[str, str], List[str]] = {}

    for node_id, node_data in graph.nodes(data=True):
        label = str(node_data.get("label", "")).strip().lower()
        props = node_data.get("properties", {}) or {}
        schema_type = bridge.normalize_entity_type(props.get("schema_type", ""))
        if label != "entity" and not schema_type:
            continue

        primary_nodes.append(node_id)
        key = (
            schema_type,
            bridge.normalize_entity_name(props.get("name", "")),
        )
        duplicate_groups.setdefault(key, []).append(node_id)

    duplicate_extra = sum(max(0, len(group) - 1) for group in duplicate_groups.values() if len(group) > 1)
    duplicate_node_rate = duplicate_extra / len(primary_nodes) if primary_nodes else 0.0

    isolated_nodes = [node_id for node_id in primary_nodes if graph.degree(node_id) == 0]
    isolated_node_rate = len(isolated_nodes) / len(primary_nodes) if primary_nodes else 0.0

    paper_nodes = []
    paper_nodes_with_author_edge = 0
    for node_id in primary_nodes:
        node_props = graph.nodes[node_id].get("properties", {}) or {}
        if not bridge.is_paper_type(node_props.get("schema_type", "")):
            continue
        paper_nodes.append(node_id)
        has_author_edge = False

        for source, _, edge_data in graph.in_edges(node_id, data=True):
            relation = bridge.normalize_relation_name(edge_data.get("relation", ""))
            source_type = bridge.normalize_entity_type(graph.nodes[source].get("properties", {}).get("schema_type", ""))
            if relation == "撰写" and bridge.is_author_type(source_type):
                has_author_edge = True
                break

        if not has_author_edge:
            for _, target, edge_data in graph.out_edges(node_id, data=True):
                relation = bridge.normalize_relation_name(edge_data.get("relation", ""))
                target_type = bridge.normalize_entity_type(graph.nodes[target].get("properties", {}).get("schema_type", ""))
                if relation == "撰写" and bridge.is_author_type(target_type):
                    has_author_edge = True
                    break

        if has_author_edge:
            paper_nodes_with_author_edge += 1

    paper_author_edge_coverage = (
        paper_nodes_with_author_edge / len(paper_nodes) if paper_nodes else 0.0
    )

    cross_doc_edge_total = sum(
        1
        for relationship in relationships
        if str((relationship.get("edge_properties", {}) or {}).get("relation_origin", "")).strip() == "cross_doc"
    )

    return {
        "primary_node_total": len(primary_nodes),
        "duplicate_node_rate": duplicate_node_rate,
        "isolated_node_rate": isolated_node_rate,
        "paper_node_total": len(paper_nodes),
        "paper_author_edge_coverage": paper_author_edge_coverage,
        "cross_doc_edge_total": cross_doc_edge_total,
        "cross_doc_precision": cross_doc_precision,
    }


def build_cross_doc_review_template(
    graph_path: str,
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    _, relationships = reconstruct_graph_from_relationships(graph_path)
    review_rows: List[Dict[str, Any]] = []
    dedupe = set()

    for relationship in relationships:
        edge_properties = relationship.get("edge_properties", {}) or {}
        if str(edge_properties.get("relation_origin", "")).strip() != "cross_doc":
            continue

        start_props = (relationship.get("start_node", {}) or {}).get("properties", {}) or {}
        end_props = (relationship.get("end_node", {}) or {}).get("properties", {}) or {}
        row = {
            "edge_id": hashlib.sha1(
                json.dumps(relationship, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "head": str(start_props.get("name", "")).strip(),
            "relation": str(relationship.get("relation", "")).strip(),
            "tail": str(end_props.get("name", "")).strip(),
            "confidence": edge_properties.get("confidence"),
            "source_paper_ids": edge_properties.get("source_paper_ids", []),
            "evidence_chunk_ids": edge_properties.get("evidence_chunk_ids", []),
            "reason": str(edge_properties.get("reason", "")).strip(),
            "verdict": "",
            "reviewer": "",
            "notes": "",
        }
        row_key = (row["head"], row["relation"], row["tail"], tuple(row["source_paper_ids"]))
        if row_key in dedupe:
            continue
        dedupe.add(row_key)
        review_rows.append(row)

    rng = random.Random(seed)
    if sample_size > 0 and len(review_rows) > sample_size:
        review_rows = rng.sample(review_rows, sample_size)

    review_rows.sort(key=lambda item: (item["head"], item["relation"], item["tail"]))
    return review_rows


def score_cross_doc_reviews(review_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reviewed = 0
    accepted = 0
    rejected = 0

    for row in review_rows:
        verdict = str(row.get("verdict", "")).strip().casefold()
        if verdict not in {"accepted", "rejected"}:
            continue
        reviewed += 1
        if verdict == "accepted":
            accepted += 1
        else:
            rejected += 1

    precision = accepted / reviewed if reviewed else None
    return {
        "reviewed": reviewed,
        "accepted": accepted,
        "rejected": rejected,
        "precision": precision,
        "pending": max(0, len(review_rows) - reviewed),
    }
