from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml

from utils.logger import logger

from .dataset_loader import load_question_set
from .judge import LLMJudge
from .llm_client import load_eval_env, resolve_model_profile, temporary_rag_env
from .models import EvaluationSample, JudgmentResult, QAPrediction
from .prediction_cache import PredictionCache
from .qa_runner import DatasetValidationError, OfflineGraphRAGRunner
from .reporter import append_sample_outputs, init_run_outputs, write_summary_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Independent GraphRAG evaluation runner")
    parser.add_argument("--config", default="eval/rag_eval/config.yaml", help="Path to evaluation config")
    parser.add_argument("--dataset", help="Dataset name override")
    parser.add_argument("--answer-model", help="Answer model override")
    parser.add_argument("--judge-model", help="Judge model override")
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


# ---------------------------------------------------------------------------
# Progress Tracking
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, total: int, phase_name: str):
        self.total = total
        self.phase_name = phase_name
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def format_seconds(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def update(self) -> str:
        with self.lock:
            self.completed += 1
            elapsed = time.time() - self.start_time
            avg_time = elapsed / self.completed
            remaining = avg_time * (self.total - self.completed)
            return f"({self.format_seconds(elapsed)} | ETA {self.format_seconds(remaining)})"


# ---------------------------------------------------------------------------
# Phase-1: 并发 Retrieval & Answer
# ---------------------------------------------------------------------------

def _answer_one(
    runner: OfflineGraphRAGRunner,
    cache: PredictionCache,
    sample: EvaluationSample,
    tracker: ProgressTracker,
    total: int,
) -> Tuple[EvaluationSample, QAPrediction]:
    """在线程池中执行单问题的检索+回答，命中缓存则直接返回。"""
    cached = cache.get(sample.question_id)
    if cached is not None:
        stats = tracker.update()
        logger.info(
            f"[{tracker.completed}/{total}] {stats} ⚡ 缓存命中 question_id={sample.question_id}"
        )
        return sample, cached

    # logger.info(f"[{index}/{total}] 🔍 正在检索 question_id={sample.question_id}")
    prediction = runner.answer_question(sample.question_id, sample.question)
    cache.save(prediction)
    
    stats = tracker.update()
    logger.info(f"[{tracker.completed}/{total}] {stats} ✅ 检索回答完成 question_id={sample.question_id}")
    return sample, prediction


def run_retrieval_phase(
    runner: OfflineGraphRAGRunner,
    cache: PredictionCache,
    samples: List[EvaluationSample],
    concurrency: int,
) -> List[Tuple[EvaluationSample, QAPrediction]]:
    """
    Phase-1：并发执行所有问题的检索+回答。

    Returns
    -------
    按 samples 原始顺序排列的 (sample, prediction) 列表。
    """
    total = len(samples)
    futures_map: Dict = {}

    logger.info(
        f"🚀 Phase-1 开始：并发检索回答，共 {total} 题，并发数={concurrency}"
    )

    tracker = ProgressTracker(total, "Phase-1")
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="rag-retrieval") as executor:
        for index, sample in enumerate(samples, start=1):
            future = executor.submit(_answer_one, runner, cache, sample, tracker, total)
            futures_map[future] = index - 1  # 记录原始下标

    # 按原始顺序重组结果
    results: List[Optional[Tuple[EvaluationSample, QAPrediction]]] = [None] * total
    for future, orig_idx in futures_map.items():
        try:
            results[orig_idx] = future.result()
        except Exception as exc:
            sample = samples[orig_idx]
            logger.error(f"Phase-1 异常 question_id={sample.question_id}: {exc}")
            results[orig_idx] = (
                sample,
                QAPrediction(
                    question_id=sample.question_id,
                    answer="",
                    schema_path_used=runner.schema_path,
                    latency_seconds=0.0,
                    error=str(exc),
                ),
            )

    logger.info("✅ Phase-1 完成：全部问题检索回答结束")
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Phase-2: 并发 Judge
# ---------------------------------------------------------------------------

def _judge_one(
    judge: LLMJudge,
    dimensions: dict,
    sample: EvaluationSample,
    prediction: QAPrediction,
    tracker: ProgressTracker,
    total: int,
) -> Tuple[EvaluationSample, QAPrediction, JudgmentResult]:
    """在线程池中执行单问题的 Judge 评分。"""
    if prediction.error:
        judgment = zero_judgment(sample.question_id, dimensions, "qa_error", prediction.error)
    else:
        judgment = judge.judge(sample, prediction)
    
    stats = tracker.update()
    logger.info(f"[{tracker.completed}/{total}] {stats} ⚖️  Judge 评分完成 question_id={sample.question_id}")
    return sample, prediction, judgment


def run_judge_phase(
    judge: LLMJudge,
    dimensions: dict,
    retrieval_results: List[Tuple[EvaluationSample, QAPrediction]],
    concurrency: int,
) -> List[Tuple[EvaluationSample, QAPrediction, JudgmentResult]]:
    """
    Phase-2：并发执行全部 Judge 评分。

    Returns
    -------
    按输入顺序排列的 (sample, prediction, judgment) 列表。
    """
    total = len(retrieval_results)
    futures_map: Dict = {}

    logger.info(
        f"🚀 Phase-2 开始：并发 Judge 评分，共 {total} 题，并发数={concurrency}"
    )

    tracker = ProgressTracker(total, "Phase-2")
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="rag-judge") as executor:
        for index, (sample, prediction) in enumerate(retrieval_results, start=1):
            future = executor.submit(
                _judge_one, judge, dimensions, sample, prediction, tracker, total
            )
            futures_map[future] = index - 1

    results: List[Optional[Tuple]] = [None] * total
    for future, orig_idx in futures_map.items():
        try:
            results[orig_idx] = future.result()
        except Exception as exc:
            sample, prediction = retrieval_results[orig_idx]
            logger.error(f"Phase-2 Judge 异常 question_id={sample.question_id}: {exc}")
            results[orig_idx] = (
                sample,
                prediction,
                zero_judgment(sample.question_id, dimensions, "judge_error", str(exc)),
            )

    logger.info("✅ Phase-2 完成：全部 Judge 评分结束")
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    eval_config = load_runtime_config(args.config)

    defaults = eval_config.get("defaults", {})
    env_path = eval_config.get("env_path", "eval/.env")
    load_eval_env(env_path)

    dataset_name = args.dataset or defaults.get("dataset_name", "AIGC-EDU-test")
    answer_model_name = args.answer_model or defaults.get("answer_model")
    judge_model_name = args.judge_model or defaults.get("judge_model")
    qa_mode = args.qa_mode or defaults.get("qa_mode", "agent")
    max_samples = args.max_samples if args.max_samples is not None else defaults.get("max_samples")
    concurrency = int(defaults.get("concurrency", 1) or 1)
    save_raw_context = bool(defaults.get("save_raw_context", True))
    prediction_cache_enabled = bool(defaults.get("prediction_cache", True))
    question_set_path = defaults.get("question_set_path", "eval/dataset/sheet1_questions.json")

    if not answer_model_name:
        raise ValueError("answer_model must be provided via config or CLI")
    if not judge_model_name:
        raise ValueError("judge_model must be provided via config or CLI")

    logger.info(
        f"评估配置 | dataset={dataset_name} qa_mode={qa_mode} "
        f"concurrency={concurrency} prediction_cache={prediction_cache_enabled}"
    )

    dimensions = eval_config.get("evaluation", {}).get("dimensions", {})
    retry_policy = eval_config.get("runtime", {}).get("retry_policy", {})
    models_cfg = eval_config.get("models", {})
    roles_cfg = eval_config.get("roles", {})

    samples = load_question_set(
        question_set_path=question_set_path,
        max_samples=max_samples,
    )

    answer_model_profile = resolve_model_profile(
        answer_model_name,
        models_cfg,
        role_cfg=roles_cfg.get("answer", {}),
    )
    judge_model_profile = resolve_model_profile(
        judge_model_name,
        models_cfg,
        role_cfg=roles_cfg.get("judge", {}),
    )
    logger.info(
        f"评估模型 | answer_model={answer_model_name} judge_model={judge_model_name}"
    )
    logger.info(f"问题集路径: {question_set_path}")

    # -----------------------------------------------------------------------
    # 初始化输出目录（在 temporary_rag_env 外先确定 run_id，确保缓存路径一致）
    # -----------------------------------------------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        defaults.get("results_dir", "eval/results/rag_eval"),
        f"{run_id}_{dataset_name}_{answer_model_name}",
    )

    # 若本次是续跑同一任务，可通过命令行传入已有 output_dir 覆盖（预留扩展点）
    # 此处默认每次新建目录；缓存文件放在同一 output_dir 下。
    init_run_outputs(output_dir, dimensions)

    with temporary_rag_env(answer_model_profile):
        # -------------------------------------------------------------------
        # 初始化 Runner & Judge
        # -------------------------------------------------------------------
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
            profile=judge_model_profile,
            prompt_path=defaults.get("prompt_path", "eval/rag_eval/prompts/judge_prompt.txt"),
            dimensions=dimensions,
            max_attempts=int(retry_policy.get("judge_max_attempts", 2)),
        )

        run_meta = {
            "run_id": run_id,
            "dataset_name": dataset_name,
            "question_set_path": question_set_path,
            "qa_mode": qa_mode,
            "answer_model": answer_model_name,
            "judge_model": judge_model_name,
            "concurrency": concurrency,
            "dataset_audit": runner.dataset_audit.to_dict(),
        }

        # -------------------------------------------------------------------
        # 初始化预测缓存（支持断点续跑）
        # -------------------------------------------------------------------
        cache = PredictionCache(output_dir=output_dir, enabled=prediction_cache_enabled)
        if cache.cached_ids:
            logger.info(
                f"📦 断点续跑：发现 {len(cache)} 条已缓存预测，将跳过对应检索"
            )

        # -------------------------------------------------------------------
        # Phase-1：并发检索 & 回答
        # -------------------------------------------------------------------
        retrieval_results = run_retrieval_phase(
            runner=runner,
            cache=cache,
            samples=samples,
            concurrency=concurrency,
        )

        # -------------------------------------------------------------------
        # Phase-2：并发 Judge 评分
        # -------------------------------------------------------------------
        judged_results = run_judge_phase(
            judge=judge,
            dimensions=dimensions,
            retrieval_results=retrieval_results,
            concurrency=concurrency,
        )

        # -------------------------------------------------------------------
        # Phase-3：汇总写报告（顺序执行，确保输出文件顺序一致）
        # -------------------------------------------------------------------
        logger.info("📝 Phase-3 开始：汇总写入评估报告")
        combined_rows: List[dict] = []

        for sample, prediction, judgment in judged_results:
            combined_rows.append(
                append_sample_outputs(
                    output_dir=output_dir,
                    run_meta=run_meta,
                    sample=sample,
                    prediction=prediction,
                    judgment=judgment,
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

    logger.info(f"🎉 评估完成，结果保存至: {output_dir}")
    logger.info(f"总体评分摘要: {summary_payload['summary']}")


if __name__ == "__main__":
    main()
