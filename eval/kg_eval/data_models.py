from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KGExtraction:
    entity_types: Dict[str, str] = field(default_factory=dict)
    triples: List[List[str]] = field(default_factory=list)
    attributes: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoldRecord:
    status: str = "pending"
    generator_model: str = ""
    reviewer: str = ""
    review_notes: str = ""
    updated_at: str = ""
    extraction: KGExtraction = field(default_factory=KGExtraction)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["extraction"] = self.extraction.to_dict()
        return data


@dataclass
class ExtractionRunResult:
    model_name: str
    prompt: str
    raw_response: str
    extraction: KGExtraction
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["extraction"] = self.extraction.to_dict()
        return payload


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
        }


@dataclass
class RetrievalMetricSummary:
    paper_node_hit_count: int = 0
    paper_node_total: int = 0
    gold_triple_hit_count: int = 0
    gold_triple_total: int = 0
    skipped_gold_triples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        paper_rate = self.paper_node_hit_count / self.paper_node_total if self.paper_node_total else 0.0
        triple_rate = self.gold_triple_hit_count / self.gold_triple_total if self.gold_triple_total else 0.0
        return {
            "paper_node_hit@5": {
                "hits": self.paper_node_hit_count,
                "total": self.paper_node_total,
                "rate": paper_rate,
            },
            "gold_triple_hit@10": {
                "hits": self.gold_triple_hit_count,
                "total": self.gold_triple_total,
                "rate": triple_rate,
                "skipped_triples": self.skipped_gold_triples,
            },
        }
