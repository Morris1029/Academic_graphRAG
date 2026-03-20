from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationSample:
    question_id: str
    question_type: str
    question: str
    reference_answer: str
    eval_focus: str = ""
    source_sheet: str = ""
    row_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetAuditResult:
    dataset_name: str
    graph_path: str
    graph_relation_count: int = 0
    chunk_path: str = ""
    chunk_file_size: int = 0
    chunk_line_count: int = 0
    schema_path: str = ""
    dataset_ready: bool = False
    error: Optional[str] = None
    recommended_dataset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QAPrediction:
    question_id: str
    answer: str
    sub_questions: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_triples: List[str] = field(default_factory=list)
    retrieved_chunks: List[str] = field(default_factory=list)
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    decompose_fallback: bool = False
    decompose_error: Optional[str] = None
    schema_path_used: Optional[str] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgmentResult:
    question_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    verdict: str = "unknown"
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    missing_points: List[str] = field(default_factory=list)
    hallucination_flags: List[str] = field(default_factory=list)
    judge_confidence: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
