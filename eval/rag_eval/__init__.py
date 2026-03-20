"""Independent evaluation package for GraphRAG answer quality assessment."""

from importlib import import_module

from .dataset_loader import load_question_set
from .judge import LLMJudge
from .models import EvaluationSample

__all__ = ["EvaluationSample", "LLMJudge", "OfflineGraphRAGRunner", "load_question_set"]


def __getattr__(name):
    if name == "OfflineGraphRAGRunner":
        return import_module(".qa_runner", __name__).OfflineGraphRAGRunner
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
