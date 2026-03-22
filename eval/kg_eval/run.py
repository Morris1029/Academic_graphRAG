from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from ..rag_eval.llm_client import load_eval_env
    from .audit import (
        build_cross_doc_review_template,
        evaluate_graph_structure,
        score_cross_doc_reviews,
        summarize_consistency_audit,
    )
    from .construction import ConstructionBridge
    from .extractor import ExtractionService
    from .loader import count_gold_statuses, load_samples, save_samples
    from .metrics import aggregate_metric_blocks, compare_extractions
    from .reporter import build_markdown_report, create_run_dir, write_json, write_jsonl
    from .retrieval import aggregate_retrieval_results, build_retriever, evaluate_sample_retrieval
except ImportError:  # pragma: no cover - fallback for direct script execution
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from eval.rag_eval.llm_client import load_eval_env
    from eval.kg_eval.audit import (
        build_cross_doc_review_template,
        evaluate_graph_structure,
        score_cross_doc_reviews,
        summarize_consistency_audit,
    )
    from eval.kg_eval.construction import ConstructionBridge
    from eval.kg_eval.extractor import ExtractionService
    from eval.kg_eval.loader import count_gold_statuses, load_samples, save_samples
    from eval.kg_eval.metrics import aggregate_metric_blocks, compare_extractions
    from eval.kg_eval.reporter import build_markdown_report, create_run_dir, write_json, write_jsonl
    from eval.kg_eval.retrieval import (
        aggregate_retrieval_results,
        build_retriever,
        evaluate_sample_retrieval,
    )

from utils.logger import logger


def load_runtime_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_defaults(config_path: str) -> Dict[str, Any]:
    runtime_config = load_runtime_config(config_path)
    env_path = runtime_config.get("env_path", "eval/.env")
    load_eval_env(env_path)
    return runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge graph construction evaluation CLI")
    parser.add_argument("--config", default="eval/kg_eval/config.yaml", help="Path to kg_eval config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_gold = subparsers.add_parser("generate_gold", help="Generate or refresh draft gold annotations")
    generate_gold.add_argument("--sample-path", help="Override sample JSON path")
    generate_gold.add_argument("--gold-model", help="Override gold model name")
    generate_gold.add_argument("--max-samples", type=int, help="Only process the first N samples")

    run_eval = subparsers.add_parser("run", help="Run candidate extraction and score approved gold samples")
    run_eval.add_argument("--sample-path", help="Override sample JSON path")
    run_eval.add_argument("--candidate-model", help="Override candidate model name")
    run_eval.add_argument("--max-samples", type=int, help="Only score the first N approved samples")
    run_eval.add_argument("--review-file", help="Optional reviewed cross-doc template JSONL path")

    cross_doc_review = subparsers.add_parser("cross_doc_review", help="Export or score cross-document review samples")
    cross_doc_review.add_argument("--review-file", help="Reviewed JSONL path. If omitted, export a fresh template.")
    cross_doc_review.add_argument("--output-dir", help="Output directory for exported review template")
    cross_doc_review.add_argument("--sample-size", type=int, help="Override review sample size")
    cross_doc_review.add_argument("--seed", type=int, help="Override sampling seed")

    return parser


def _dataset_paths(bridge: ConstructionBridge) -> Dict[str, str]:
    dataset_config = bridge.dataset_config
    return {
        "graph_path": dataset_config.graph_output,
        "schema_path": dataset_config.schema_path,
        "chunk_path": f"output/chunks/{bridge.dataset_name}.txt",
    }


def _format_duration(seconds: float) -> str:
    return f"{max(0.0, float(seconds)):.1f}s"


def command_generate_gold(args: argparse.Namespace, runtime_config: Dict[str, Any]) -> None:
    defaults = runtime_config.get("defaults", {})
    dataset_name = defaults.get("dataset_name", "AIGC-EDU")
    sample_path = args.sample_path or defaults.get("sample_path", "eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json")
    gold_model = args.gold_model or defaults.get("gold_model")
    if not gold_model:
        raise ValueError("gold_model must be configured for generate_gold")

    bridge = ConstructionBridge(dataset_name, defaults.get("main_config_path", "config/base_config.yaml"))
    models_cfg = runtime_config.get("models", {})
    roles_cfg = runtime_config.get("roles", {})
    service = ExtractionService(bridge, models_cfg, roles_cfg)

    all_records = load_samples(sample_path)
    target_records = all_records[: max(0, args.max_samples)] if args.max_samples is not None else all_records
    total_targets = len(target_records)
    updated = 0
    skipped = 0
    total_start = time.perf_counter()

    logger.info(
        "Starting gold draft generation | dataset=%s sample_path=%s gold_model=%s max_samples=%s target_total=%d",
        dataset_name,
        sample_path,
        gold_model,
        args.max_samples if args.max_samples is not None else "all",
        total_targets,
    )

    for index, record in enumerate(target_records, start=1):
        sample_id = str(record.get("id", "")).strip() or f"sample_{index}"
        logger.info("[%d/%d] generating gold for sample_id=%s", index, total_targets, sample_id)
        sample_start = time.perf_counter()
        gold = record.get("kg_eval", {}).get("gold", {}) or {}
        if str(gold.get("status", "")).strip() == "approved":
            skipped += 1
            now = time.perf_counter()
            elapsed = now - sample_start
            avg = (now - total_start) / index if index else 0.0
            eta = avg * max(total_targets - index, 0)
            logger.info(
                "[%d/%d] finished sample_id=%s status=skipped_approved elapsed=%s avg=%s eta=%s auto_error=%s",
                index,
                total_targets,
                sample_id,
                _format_duration(elapsed),
                _format_duration(avg),
                _format_duration(eta),
                "false",
            )
            continue

        gold_payload = service.build_gold_payload(record, gold_model)
        record.setdefault("kg_eval", {})["gold"] = gold_payload
        updated += 1
        review_notes = str(gold_payload.get("review_notes", "")).strip()
        auto_error = "AUTO_ERROR:" in review_notes
        now = time.perf_counter()
        elapsed = now - sample_start
        avg = (now - total_start) / index if index else 0.0
        eta = avg * max(total_targets - index, 0)
        logger.info(
            "[%d/%d] finished sample_id=%s status=updated elapsed=%s avg=%s eta=%s auto_error=%s",
            index,
            total_targets,
            sample_id,
            _format_duration(elapsed),
            _format_duration(avg),
            _format_duration(eta),
            "true" if auto_error else "false",
        )

    save_samples(sample_path, all_records)
    total_elapsed = time.perf_counter() - total_start
    avg_elapsed = total_elapsed / total_targets if total_targets else 0.0
    logger.info(
        "Gold draft generation finished | sample_path=%s updated=%d skipped_approved=%d total_elapsed=%s avg_per_sample=%s",
        sample_path,
        updated,
        skipped,
        _format_duration(total_elapsed),
        _format_duration(avg_elapsed),
    )


def _load_review_rows(review_file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(review_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_audit_bundle(
    bridge: ConstructionBridge,
    review_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    dataset_paths = _dataset_paths(bridge)
    cross_doc_summary = score_cross_doc_reviews(review_rows or []) if review_rows else {}
    graph_quality = evaluate_graph_structure(
        dataset_paths["graph_path"],
        bridge,
        cross_doc_precision=cross_doc_summary.get("precision") if cross_doc_summary else None,
    )
    return {
        "consistency_audit": summarize_consistency_audit(bridge.dataset_name),
        "graph_quality": graph_quality,
    }


def command_run(args: argparse.Namespace, runtime_config: Dict[str, Any]) -> None:
    defaults = runtime_config.get("defaults", {})
    dataset_name = defaults.get("dataset_name", "AIGC-EDU")
    sample_path = args.sample_path or defaults.get("sample_path", "eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json")
    candidate_model = args.candidate_model or defaults.get("candidate_model")
    if not candidate_model:
        raise ValueError("candidate_model must be configured for run")

    bridge = ConstructionBridge(dataset_name, defaults.get("main_config_path", "config/base_config.yaml"))
    models_cfg = runtime_config.get("models", {})
    roles_cfg = runtime_config.get("roles", {})
    service = ExtractionService(bridge, models_cfg, roles_cfg)
    records = load_samples(sample_path)
    status_counts = count_gold_statuses(records)
    approved_records = [
        record for record in records
        if str(record.get("kg_eval", {}).get("gold", {}).get("status", "")).strip() == "approved"
    ]
    if args.max_samples is not None:
        approved_records = approved_records[: max(0, args.max_samples)]

    review_rows = _load_review_rows(args.review_file) if args.review_file else None
    audit_bundle = _build_audit_bundle(bridge, review_rows=review_rows)

    if not approved_records:
        logger.warning("No approved gold samples found in %s", sample_path)

    retriever = build_retriever(
        dataset_name,
        defaults.get("main_config_path", "config/base_config.yaml"),
        top_k=max(int(defaults.get("gold_triple_top_k", 10)), int(defaults.get("paper_node_top_k", 5))),
    ) if approved_records else None

    per_sample_rows: List[Dict[str, Any]] = []
    metric_blocks: List[Dict[str, Any]] = []
    retrieval_blocks: List[Dict[str, Any]] = []

    for record in approved_records:
        candidate_result = service.extract(record, candidate_model, role_name="candidate")
        comparison = compare_extractions(
            bridge,
            record.get("kg_eval", {}).get("gold", {}).get("extraction", {}),
            candidate_result.extraction.to_dict(),
        )
        retrieval_result = evaluate_sample_retrieval(
            record,
            retriever,
            bridge,
            paper_top_k=int(defaults.get("paper_node_top_k", 5)),
            triple_top_k=int(defaults.get("gold_triple_top_k", 10)),
        ) if retriever is not None else {}

        metric_blocks.append(comparison)
        retrieval_blocks.append(retrieval_result)
        per_sample_rows.append(
            {
                "id": record["id"],
                "title": record.get("meta", {}).get("title", ""),
                "gold_status": record.get("kg_eval", {}).get("gold", {}).get("status", ""),
                "gold_extraction": record.get("kg_eval", {}).get("gold", {}).get("extraction", {}),
                "candidate_result": candidate_result.to_dict(),
                **comparison,
                "retrieval_metrics": retrieval_result,
            }
        )

    extraction_summary = aggregate_metric_blocks(metric_blocks)
    retrieval_summary = aggregate_retrieval_results(retrieval_blocks)

    cross_doc_template = build_cross_doc_review_template(
        _dataset_paths(bridge)["graph_path"],
        sample_size=int(defaults.get("cross_doc_review_sample_size", 50)),
        seed=int(defaults.get("cross_doc_review_seed", 42)),
    )
    cross_doc_summary = score_cross_doc_reviews(review_rows or []) if review_rows else {
        "reviewed": 0,
        "accepted": 0,
        "rejected": 0,
        "precision": None,
        "pending": len(cross_doc_template),
    }
    audit_bundle["graph_quality"]["cross_doc_precision"] = cross_doc_summary.get("precision")

    summary = {
        "dataset_name": dataset_name,
        "sample_path": sample_path,
        "status_counts": status_counts,
        "approved_sample_total": len(approved_records),
        "candidate_model": candidate_model,
        "extraction_metrics": extraction_summary,
        "retrieval_metrics": retrieval_summary,
        "cross_doc_review": cross_doc_summary,
    }

    run_dir = create_run_dir(defaults.get("results_dir", "eval/results/kg_eval"), dataset_name)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "audit.json", audit_bundle)
    write_jsonl(run_dir / "per_sample.jsonl", per_sample_rows)
    write_jsonl(run_dir / "cross_doc_review_template.jsonl", cross_doc_template)
    (run_dir / "report.md").write_text(
        build_markdown_report(summary, audit_bundle),
        encoding="utf-8",
    )
    logger.info("KG evaluation run completed. Results saved to %s", run_dir)


def command_cross_doc_review(args: argparse.Namespace, runtime_config: Dict[str, Any]) -> None:
    defaults = runtime_config.get("defaults", {})
    dataset_name = defaults.get("dataset_name", "AIGC-EDU")
    bridge = ConstructionBridge(dataset_name, defaults.get("main_config_path", "config/base_config.yaml"))
    graph_path = _dataset_paths(bridge)["graph_path"]

    if args.review_file:
        review_rows = _load_review_rows(args.review_file)
        summary = score_cross_doc_reviews(review_rows)
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.review_file).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "cross_doc_review_summary.json", summary)
        logger.info("Cross-doc review scoring completed. Summary saved to %s", output_dir)
        return

    output_dir = Path(args.output_dir) if args.output_dir else create_run_dir(
        defaults.get("results_dir", "eval/results/kg_eval"),
        f"{dataset_name}_cross_doc_review",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_cross_doc_review_template(
        graph_path,
        sample_size=int(args.sample_size or defaults.get("cross_doc_review_sample_size", 50)),
        seed=int(args.seed or defaults.get("cross_doc_review_seed", 42)),
    )
    write_jsonl(output_dir / "cross_doc_review_template.jsonl", rows)
    logger.info("Cross-doc review template exported to %s", output_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runtime_config = resolve_defaults(args.config)

    if args.command == "generate_gold":
        command_generate_gold(args, runtime_config)
        return
    if args.command == "run":
        command_run(args, runtime_config)
        return
    if args.command == "cross_doc_review":
        command_cross_doc_review(args, runtime_config)
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
