from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .models import EvaluationSample, JudgmentResult, QAPrediction


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _refusal_detected(answer: str) -> bool:
    normalized = (answer or "").lower()
    refusal_markers = [
        "无法回答",
        "不能回答",
        "现有资料不足",
        "insufficient",
        "cannot answer",
        "not enough information",
    ]
    return any(marker in normalized for marker in refusal_markers)


def build_summary(rows: List[Dict], dimensions: Dict[str, Dict]) -> Dict:
    total = len(rows)
    if total == 0:
        return {
            "sample_count": 0,
            "overall_score": 0.0,
            "per_dimension_avg": {},
            "per_question_type_avg": {},
            "failure_rate": 0.0,
            "refusal_rate": 0.0,
            "hallucination_rate": 0.0,
        }

    per_dimension_sum = defaultdict(float)
    per_type_sum = defaultdict(float)
    per_type_count = defaultdict(int)
    failures = 0
    refusals = 0
    hallucinations = 0

    for row in rows:
        weighted_score = float(row.get("weighted_score", 0.0) or 0.0)
        question_type = row.get("question_type", "") or "未分类"

        for dimension_name in dimensions:
            per_dimension_sum[dimension_name] += float(row.get(dimension_name, 0.0) or 0.0)

        per_type_sum[question_type] += weighted_score
        per_type_count[question_type] += 1

        if row.get("qa_error") or row.get("judge_error"):
            failures += 1
        if _refusal_detected(row.get("predicted_answer", "")):
            refusals += 1
        if row.get("hallucination_flags"):
            hallucinations += 1

    overall_score = round(
        sum(float(row.get("weighted_score", 0.0) or 0.0) for row in rows) / total,
        4,
    )

    return {
        "sample_count": total,
        "overall_score": overall_score,
        "per_dimension_avg": {
            name: round(per_dimension_sum[name] / total, 4) for name in dimensions
        },
        "per_question_type_avg": {
            name: {
                "count": per_type_count[name],
                "avg_weighted_score": round(per_type_sum[name] / per_type_count[name], 4),
            }
            for name in per_type_sum
        },
        "failure_rate": round(failures / total, 4),
        "refusal_rate": round(refusals / total, 4),
        "hallucination_rate": round(hallucinations / total, 4),
    }


def render_report_markdown(run_meta: Dict, summary: Dict, rows: List[Dict], dimensions: Dict[str, Dict]) -> str:
    lines = [
        "# GraphRAG LLM 自动化评估报告",
        "",
        "## Run Meta",
        f"- run_id: `{run_meta['run_id']}`",
        f"- dataset_name: `{run_meta['dataset_name']}`",
        f"- question_set_path: `{run_meta['question_set_path']}`",
        f"- qa_mode: `{run_meta['qa_mode']}`",
        f"- answer_profile: `{run_meta['answer_profile']}`",
        f"- judge_profile: `{run_meta['judge_profile']}`",
        f"- sample_count: `{summary['sample_count']}`",
        "",
        "## Summary",
        f"- overall_score: `{summary['overall_score']}`",
        f"- failure_rate: `{summary['failure_rate']}`",
        f"- refusal_rate: `{summary['refusal_rate']}`",
        f"- hallucination_rate: `{summary['hallucination_rate']}`",
        "",
        "## Dimension Averages",
    ]

    for name in dimensions:
        lines.append(f"- {name}: `{summary['per_dimension_avg'].get(name, 0.0)}`")

    lines.extend(["", "## Lowest Scored Samples"])
    for row in sorted(rows, key=lambda item: item.get("weighted_score", 0.0))[:5]:
        lines.append(
            f"- {row['question_id']} | {row.get('question_type', '')} | score={row.get('weighted_score', 0.0)} | "
            f"verdict={row.get('verdict', '')} | question={row.get('question', '')[:80]}"
        )

    lines.extend(["", "## Highest Scored Samples"])
    for row in sorted(rows, key=lambda item: item.get("weighted_score", 0.0), reverse=True)[:5]:
        lines.append(
            f"- {row['question_id']} | {row.get('question_type', '')} | score={row.get('weighted_score', 0.0)} | "
            f"verdict={row.get('verdict', '')} | question={row.get('question', '')[:80]}"
        )

    return "\n".join(lines) + "\n"


def save_run_outputs(
    output_dir: str,
    run_meta: Dict,
    samples: List[EvaluationSample],
    predictions: List[QAPrediction],
    judgments: List[JudgmentResult],
    dimensions: Dict[str, Dict],
    save_raw_context: bool = True,
) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    judgment_rows = []
    combined_rows = []

    for sample, prediction, judgment in zip(samples, predictions, judgments):
        prediction_row = {
            **sample.to_dict(),
            **prediction.to_dict(),
            "answer_profile": run_meta["answer_profile"],
            "qa_mode": run_meta["qa_mode"],
        }
        if not save_raw_context:
            prediction_row["retrieved_triples"] = []
            prediction_row["retrieved_chunks"] = []
            prediction_row["reasoning_steps"] = []

        judgment_row = {
            **sample.to_dict(),
            **judgment.to_dict(),
            "judge_profile": run_meta["judge_profile"],
        }
        combined_row = {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "question": sample.question,
            "reference_answer": sample.reference_answer,
            "eval_focus": sample.eval_focus,
            "predicted_answer": prediction.answer,
            "latency_seconds": prediction.latency_seconds,
            "qa_error": prediction.error,
            "judge_error": judgment.error,
            "weighted_score": judgment.weighted_score,
            "verdict": judgment.verdict,
            "judge_confidence": judgment.judge_confidence,
            "hallucination_flags": "; ".join(judgment.hallucination_flags),
            "strengths": " | ".join(judgment.strengths),
            "weaknesses": " | ".join(judgment.weaknesses),
            "missing_points": " | ".join(judgment.missing_points),
        }
        for dimension_name in dimensions:
            combined_row[dimension_name] = judgment.scores.get(dimension_name, 0.0)

        prediction_rows.append(prediction_row)
        judgment_rows.append(judgment_row)
        combined_rows.append(combined_row)

    summary = build_summary(combined_rows, dimensions)
    report_markdown = render_report_markdown(run_meta, summary, combined_rows, dimensions)

    _write_jsonl(output_path / "predictions.jsonl", prediction_rows)
    _write_jsonl(output_path / "judgments.jsonl", judgment_rows)

    summary_payload = {
        "run_meta": run_meta,
        "summary": summary,
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_path / "report.md").write_text(report_markdown, encoding="utf-8")

    fieldnames = list(combined_rows[0].keys()) if combined_rows else [
        "question_id",
        "question_type",
        "question",
        "reference_answer",
        "predicted_answer",
        "weighted_score",
    ]
    with (output_path / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    return summary_payload
