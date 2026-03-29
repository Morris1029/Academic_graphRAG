"""
prediction_cache.py
====================
线程安全的 QAPrediction 磁盘缓存模块。

缓存文件格式：JSONL（每行一个 QAPrediction.to_dict()），存放在
  {output_dir}/predictions_cache.jsonl

用于支持：
  1. 断点续跑：已完成的题目直接从缓存加载，跳过检索+回答阶段。
  2. 阶段分离：Retrieval 阶段全部跑完后，再批量进入 Judge 阶段。
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from utils.logger import logger

from .models import QAPrediction

_CACHE_FILENAME = "predictions_cache.jsonl"


class PredictionCache:
    """
    线程安全的预测结果磁盘缓存。

    Parameters
    ----------
    output_dir:
        评估结果目录，缓存文件将写入该目录下的 predictions_cache.jsonl。
    enabled:
        是否启用缓存。若为 False，所有操作均为 no-op。
    """

    def __init__(self, output_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache_path = Path(output_dir) / _CACHE_FILENAME
        self._lock = threading.Lock()
        self._cache: Dict[str, QAPrediction] = {}

        if self.enabled:
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, question_id: str) -> Optional[QAPrediction]:
        """从内存缓存中查找，命中则返回，否则返回 None。"""
        if not self.enabled:
            return None
        return self._cache.get(question_id)

    def save(self, prediction: QAPrediction) -> None:
        """
        将 prediction 追加写入磁盘并更新内存缓存（线程安全）。
        若该 question_id 已在缓存中，则跳过（幂等）。
        """
        if not self.enabled:
            return
        if prediction.question_id in self._cache:
            return

        with self._lock:
            # 双重检查，防止并发写入重复条目
            if prediction.question_id in self._cache:
                return
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                row = asdict(prediction)
                with self._cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                self._cache[prediction.question_id] = prediction
            except Exception as exc:
                logger.warning(f"[PredictionCache] 写入缓存失败 question_id={prediction.question_id}: {exc}")

    @property
    def cached_ids(self) -> set:
        """返回当前已缓存的 question_id 集合。"""
        return set(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """启动时从磁盘加载已有缓存（断点续跑）。"""
        if not self._cache_path.exists():
            return

        loaded = 0
        skipped = 0
        try:
            with self._cache_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        qid = data.get("question_id", "")
                        if not qid:
                            skipped += 1
                            continue
                        prediction = QAPrediction(
                            question_id=qid,
                            answer=data.get("answer", ""),
                            sub_questions=data.get("sub_questions", []),
                            retrieved_triples=data.get("retrieved_triples", []),
                            retrieved_chunks=data.get("retrieved_chunks", []),
                            reasoning_steps=data.get("reasoning_steps", []),
                            decompose_fallback=bool(data.get("decompose_fallback", False)),
                            decompose_error=data.get("decompose_error"),
                            schema_path_used=data.get("schema_path_used"),
                            latency_seconds=float(data.get("latency_seconds", 0.0)),
                            error=data.get("error"),
                        )
                        self._cache[qid] = prediction
                        loaded += 1
                    except Exception as exc:
                        logger.warning(f"[PredictionCache] 解析缓存行失败: {exc}")
                        skipped += 1
        except Exception as exc:
            logger.warning(f"[PredictionCache] 读取缓存文件失败: {exc}")
            return

        if loaded:
            logger.info(
                f"[PredictionCache] 从缓存恢复 {loaded} 条预测结果，跳过重新检索"
                + (f"（{skipped} 行解析失败）" if skipped else "")
            )
