from __future__ import annotations

import argparse
import os
from datetime import datetime

import yaml

from utils.logger import logger

from .dataset_loader import load_question_set
from .judge import LLMJudge
from .llm_client import load_eval_env, resolve_profile, temporary_rag_env
from .models import JudgmentResult, QAPrediction
from .qa_runner import DatasetValidationError, OfflineGraphRAGRunner
from .reporter import append_sample_outputs, init_run_outputs, write_summary_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Independent GraphRAG evaluation runner")
    parser.add_argument("--config", default="eval/rag_eval/config.yaml", help="Path to evaluation config")
    parser.add_argument("--dataset", help="Dataset name override")
    parser.add_argument("--answer-profile", help="Answer model profile override")
    parser.add_argument("--judge-profile", help="Judge model profile override")
    parser.add_argument("--qa-mode", choices=["agent", "noagent"], help="QA mode override")
    parser.add_argument("--max-samples", type=int, help="Only evaluate the first N samples")
    return parser.parse_args()


def load_runtime_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def zero_judgment(sample_id: str, dimensions: dict, verdict: str, error: str) -> JudgmentResult:
    return JudgmentResult(
        question_id=sample_id,
        scores={name: 0.0 for name in dimensions},
        weighted_score=0.0,
        verdict=verdict,
        judge_confidence=0.0,
        error=error,
    )


def main():
    args = parse_args()
    eval_config = load_runtime_config(args.config)

    defaults = eval_config.get("defaults", {})
    env_path = eval_config.get("env_path", "eval/.env")
    load_eval_env(env_path)

    dataset_name = args.dataset or defaults.get("dataset_name", "AIGC-EDU-test")
    answer_profile_name = args.answer_profile or defaults.get("answer_profile")
    judge_profile_name = args.judge_profile or defaults.get("judge_profile")
    qa_mode = args.qa_mode or defaults.get("qa_mode", "agent")
    max_samples = args.max_samples if args.max_samples is not None else defaults.get("max_samples")
    concurrency = int(defaults.get("concurrency", 1) or 1)
    save_raw_context = bool(defaults.get("save_raw_context", True))
    question_set_path = defaults.get("question_set_path", "eval/dataset/sheet1_questions.json")

    if not answer_profile_name:
        raise ValueError("answer_profile must be provided via config or CLI")
    if not judge_profile_name:
        raise ValueError("judge_profile must be provided via config or CLI")
    if concurrency != 1:
        logger.warning(
            "Current evaluation runner executes sequentially; 'concurrency' is reserved for future expansion."
        )

    dimensions = eval_config.get("evaluation", {}).get("dimensions", {})
    retry_policy = eval_config.get("runtime", {}).get("retry_policy", {})
    profiles_cfg = eval_config.get("profiles", {})
    answer_profiles_cfg = profiles_cfg.get("answer_profiles") or profiles_cfg.get("answer", {})
    judge_profiles_cfg = profiles_cfg.get("judge_profiles") or profiles_cfg.get("judge", {})

    samples = load_question_set(
        question_set_path=question_set_path,
        max_samples=max_samples,
    )

    answer_profile = resolve_profile(answer_profile_name, answer_profiles_cfg)
    judge_profile = resolve_profile(judge_profile_name, judge_profiles_cfg)
    logger.info(
        f"Evaluation profiles | answer_profile={answer_profile_name} judge_profile={judge_profile_name} "
        "Judge is fixed per run; answer profile is the compared model."
    )
    logger.info(f"Question set path: {question_set_path}")

    with temporary_rag_env(answer_profile):
        try:
            runner = OfflineGraphRAGRunner(
                config_path=defaults.get("main_config_path", "config/base_config.yaml"),
                dataset_name=dataset_name,
                qa_mode=qa_mode,
                answer_max_attempts=int(retry_policy.get("answer_max_attempts", 2)),
            )
        except DatasetValidationError as exc:
            logger.error(f"Run blocked [{exc.status}]: {exc}")
            raise SystemExit(1) from None

        judge = LLMJudge(
            profile=judge_profile,
            prompt_path=defaults.get("prompt_path", "eval/prompts/judge_prompt.txt"),
            dimensions=dimensions,
            max_attempts=int(retry_policy.get("judge_max_attempts", 2)),
        )

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            defaults.get("results_dir", "eval/results"),
            f"{run_id}_{dataset_name}_{answer_profile_name}",
        )
        run_meta = {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "question_set_path": question_set_path,
            "qa_mode": qa_mode,
            "answer_profile": answer_profile_name,
            "judge_profile": judge_profile_name,
            "dataset_audit": runner.dataset_audit.to_dict(),
        }
        init_run_outputs(output_dir, dimensions)

        predictions: list[QAPrediction] = []
        judgments: list[JudgmentResult] = []
        combined_rows: list[dict] = []
        summary_payload = write_summary_outputs(
            output_dir=output_dir,
            run_meta=run_meta,
            combined_rows=combined_rows,
            dimensions=dimensions,
        )

        for index, sample in enumerate(samples, start=1):
            logger.info(f"[{index}/{len(samples)}] Evaluating question_id={sample.question_id}")
            prediction = runner.answer_question(sample.question_id, sample.question)
            predictions.append(prediction)

            if prediction.error:
                judgments.append(
                    zero_judgment(
                        sample.question_id,
                        dimensions,
                        "qa_error",
                        prediction.error,
                    )
                )
            else:
                judgments.append(judge.judge(sample, prediction))

            latest_judgment = judgments[-1]
            combined_rows.append(
                append_sample_outputs(
                    output_dir=output_dir,
                    run_meta=run_meta,
                    sample=sample,
                    prediction=prediction,
                    judgment=latest_judgment,
                    dimensions=dimensions,
                    save_raw_context=save_raw_context,
                )
            )
            summary_payload = write_summary_outputs(
                output_dir=output_dir,
                run_meta=run_meta,
                combined_rows=combined_rows,
                dimensions=dimensions,
            )
            logger.info(
                f"Checkpoint saved after question_id={sample.question_id} "
                f"({len(combined_rows)}/{len(samples)})"
            )

    logger.info(f"Evaluation completed. Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_payload['summary']}")


if __name__ == "__main__":
    main()
