from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def create_run_dir(base_dir: str, dataset_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_id}_{dataset_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_markdown_report(summary: Dict[str, Any], audit_payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# KG Eval Report")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Dataset: {summary.get('dataset_name', '')}")
    lines.append(f"- Sample file: {summary.get('sample_path', '')}")
    lines.append(f"- Approved samples: {summary.get('approved_sample_total', 0)}")
    lines.append(f"- Candidate model: {summary.get('candidate_model', '')}")
    lines.append(f"- Evaluation Mode: {summary.get('eval_mode', 'standard')}")
    if summary.get('eval_mode') == 'shadow_evaluation':
        lines.append("  > [!NOTE]\n  > This run used **Shadow Evaluation**. Retrieval metrics are calculated by building a temporary index from the model's own extractions for these samples, ensuring metrics reflect model-specific quality.")
    lines.append("")

    graph_quality = audit_payload.get("graph_quality", {})

    lines.append("## Extraction Quality")
    extraction = summary.get("extraction_metrics", {})
    lines.append(f"- Entity P/R/F1: {extraction.get('entity_prf', {})}")
    lines.append(f"- Relation P/R/F1 (Overall): {extraction.get('relation_prf', {})}")
    lines.append(f"  - Structured: {extraction.get('relation_prf_structured', {})}")
    lines.append(f"  - Unstructured: {extraction.get('relation_prf_unstructured', {})}")
    lines.append(f"- Attribute accuracy (Overall): {extraction.get('attribute_accuracy', {})}")
    lines.append(f"  - Structured: {extraction.get('attribute_accuracy_structured', {})}")
    lines.append(f"  - Unstructured: {extraction.get('attribute_accuracy_unstructured', {})}")
    lines.append("")

    lines.append("## Graph Quality")
    lines.append(f"- primary_node_total (节点总数): {graph_quality.get('primary_node_total')}")
    lines.append(f"- paper_node_total (论文总数): {graph_quality.get('paper_node_total')}")
    lines.append(f"- duplicate_node_rate (重复节点率): {graph_quality.get('duplicate_node_rate')}")
    lines.append(f"- isolated_node_rate (孤立节点率): {graph_quality.get('isolated_node_rate')}")
    lines.append(f"- paper_author_edge_coverage (作者边覆盖率): {graph_quality.get('paper_author_edge_coverage')}")
    lines.append(f"- cross_doc_edge_total (跨文档边总数): {graph_quality.get('cross_doc_edge_total')}")
    lines.append(f"- cross_doc_precision (跨文档边准确率): {graph_quality.get('cross_doc_precision')}")
    lines.append("")

    lines.append("## Retrieval Readiness (Isolated)")
    retrieval = summary.get("retrieval_metrics", {})
    lines.append(f"- paper_node_hit@5: {retrieval.get('paper_node_hit@5', {})}")
    lines.append(f"- gold_triple_hit@10: {retrieval.get('gold_triple_hit@10', {})}")
    lines.append("")

    return "\n".join(lines)
