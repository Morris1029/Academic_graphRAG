from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from config import reload_config
from models.retriever import agentic_decomposer as decomposer
from models.retriever import enhanced_kt_retriever as retriever
from utils.logger import logger

from .models import DatasetAuditResult, QAPrediction


def rerank_chunks_by_keywords(chunks: List[str], question: str, top_k: int) -> List[str]:
    if len(chunks) <= top_k:
        return chunks

    question_keywords = set(question.lower().split())
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in question_keywords if keyword in chunk_lower)
        scored_chunks.append((chunk, score))
    scored_chunks.sort(key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_k]]


def deduplicate_triples(triples: List[str]) -> List[str]:
    return list(dict.fromkeys(triples))


def merge_chunk_contents(chunk_ids: List[str], chunk_contents_dict: Dict[str, str]) -> List[str]:
    return [
        chunk_contents_dict.get(chunk_id, f"[Missing content for chunk {chunk_id}]")
        for chunk_id in chunk_ids
    ]


class DatasetValidationError(RuntimeError):
    def __init__(self, message: str, audit: DatasetAuditResult):
        super().__init__(message)
        self.audit = audit
        self.status = "dataset_invalid"


class OfflineGraphRAGRunner:
    def __init__(
        self,
        config_path: str,
        dataset_name: str,
        qa_mode: str = "agent",
        answer_max_attempts: int = 2,
    ):
        self.config = reload_config(config_path)
        self.config.override_config({"triggers": {"mode": qa_mode}})
        self.dataset_name = dataset_name
        self.qa_mode = qa_mode
        self.answer_max_attempts = max(1, int(answer_max_attempts))

        self.graph_path, self.chunk_path, self.schema_path = self._resolve_dataset_paths(dataset_name)
        self.dataset_audit = self._audit_dataset(dataset_name)
        self._log_dataset_audit(self.dataset_audit)
        if not self.dataset_audit.dataset_ready:
            raise DatasetValidationError(
                self._format_dataset_error(self.dataset_audit),
                self.dataset_audit,
            )

        self.graphq = decomposer.GraphQ(dataset_name, config=self.config)
        self.kt_retriever = retriever.KTRetriever(
            dataset_name,
            self.graph_path,
            recall_paths=self.config.retrieval.recall_paths,
            schema_path=self.schema_path,
            top_k=self.config.retrieval.top_k_filter,
            mode=qa_mode,
            config=self.config,
        )

        logger.info(
            f"Building retrieval indices for evaluation dataset={dataset_name} mode={qa_mode}"
        )
        self.kt_retriever.build_indices()

    def _resolve_dataset_paths(self, dataset_name: str) -> Tuple[str, str, str]:
        graph_path = f"output/graphs/{dataset_name}_new.json"
        chunk_path = f"output/chunks/{dataset_name}.txt"
        schema_path = f"schemas/{dataset_name}.json"

        if dataset_name in self.config.datasets:
            dataset_config = self.config.get_dataset_config(dataset_name)
            graph_path = dataset_config.graph_output or graph_path
            schema_path = dataset_config.schema_path or schema_path

        if not os.path.exists(schema_path) and os.path.exists("schemas/demo.json"):
            schema_path = "schemas/demo.json"

        return graph_path, chunk_path, schema_path

    def _count_graph_relations(self, graph_path: str) -> int:
        if not os.path.exists(graph_path) or os.path.getsize(graph_path) == 0:
            return 0

        with open(graph_path, "r", encoding="utf-8", errors="replace") as handle:
            data = json.load(handle)

        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            if isinstance(data.get("edges"), list):
                return len(data["edges"])
            if isinstance(data.get("links"), list):
                return len(data["links"])
        return 0

    def _count_chunk_lines(self, chunk_path: str) -> int:
        if not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
            return 0

        line_count = 0
        with open(chunk_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    line_count += 1
        return line_count

    def _find_recommended_dataset(self) -> Optional[str]:
        preferred = "AIGC-EDU"
        preferred_graph = f"output/graphs/{preferred}_new.json"
        preferred_chunk = f"output/chunks/{preferred}.txt"
        preferred_schema = f"schemas/{preferred}.json"
        if (
            os.path.exists(preferred_graph)
            and os.path.exists(preferred_chunk)
            and os.path.exists(preferred_schema)
        ):
            if self._count_graph_relations(preferred_graph) > 0 and os.path.getsize(preferred_chunk) > 0:
                return preferred
        return None

    def _audit_dataset(self, dataset_name: str) -> DatasetAuditResult:
        graph_relation_count = 0
        chunk_file_size = os.path.getsize(self.chunk_path) if os.path.exists(self.chunk_path) else 0
        chunk_line_count = 0
        error_messages: List[str] = []

        if not os.path.exists(self.graph_path):
            error_messages.append(f"graph file not found: {self.graph_path}")
        else:
            try:
                graph_relation_count = self._count_graph_relations(self.graph_path)
                if graph_relation_count <= 0:
                    error_messages.append(f"graph file is empty: {self.graph_path}")
            except Exception as exc:
                error_messages.append(f"graph file is unreadable: {self.graph_path} ({exc})")

        if not os.path.exists(self.chunk_path):
            error_messages.append(f"chunk file not found: {self.chunk_path}")
        else:
            try:
                chunk_line_count = self._count_chunk_lines(self.chunk_path)
                if chunk_file_size <= 0 or chunk_line_count <= 0:
                    error_messages.append(f"chunk file is empty: {self.chunk_path}")
            except Exception as exc:
                error_messages.append(f"chunk file is unreadable: {self.chunk_path} ({exc})")

        if not os.path.exists(self.schema_path):
            error_messages.append(f"schema file not found: {self.schema_path}")

        return DatasetAuditResult(
            dataset_name=dataset_name,
            graph_path=self.graph_path,
            graph_relation_count=graph_relation_count,
            chunk_path=self.chunk_path,
            chunk_file_size=chunk_file_size,
            chunk_line_count=chunk_line_count,
            schema_path=self.schema_path,
            dataset_ready=not error_messages,
            error="; ".join(error_messages) if error_messages else None,
            recommended_dataset=self._find_recommended_dataset(),
        )

    def _log_dataset_audit(self, audit: DatasetAuditResult) -> None:
        logger.info(
            "Dataset audit | "
            f"dataset={audit.dataset_name} ready={audit.dataset_ready} "
            f"graph_path={audit.graph_path} graph_relation_count={audit.graph_relation_count} "
            f"chunk_path={audit.chunk_path} chunk_file_size={audit.chunk_file_size} "
            f"chunk_line_count={audit.chunk_line_count} schema_path={audit.schema_path}"
        )

    def _format_dataset_error(self, audit: DatasetAuditResult) -> str:
        message = (
            f"Dataset '{audit.dataset_name}' is not ready for evaluation. "
            f"{audit.error or 'Dataset validation failed.'}"
        )
        if audit.recommended_dataset and audit.recommended_dataset != audit.dataset_name:
            message += (
                f" Recommended dataset: '{audit.recommended_dataset}'. "
                f"Try: python -m eval.run --config eval/config.yaml --sheet Sheet3 "
                f"--dataset {audit.recommended_dataset} --answer-profile deepseek_rag"
            )
        else:
            message += " Please rebuild the target dataset graph and chunk files before evaluation."
        return message

    def _call_answer_model(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.answer_max_attempts + 1):
            try:
                answer = self.kt_retriever.generate_answer(prompt)
                if answer and answer.strip():
                    return answer
            except Exception as exc:
                last_error = exc
                logger.error(f"Answer generation failed on attempt {attempt}: {exc}")
                time.sleep(min(2.0 * attempt, 5.0))

        if last_error:
            raise last_error
        return ""

    def _initial_question_decomposition(self, question: str) -> Dict:
        all_triples = set()
        all_chunk_ids = set()
        all_chunk_contents: Dict[str, str] = {}
        total_time = 0.0

        decomposition_result = {
            "sub_questions": [{"sub-question": question}],
            "involved_types": {"nodes": [], "relations": [], "attributes": []},
        }
        decompose_fallback = False
        decompose_error = None

        try:
            decomposition_result = self.graphq.decompose(question, self.schema_path)
            sub_questions = decomposition_result.get("sub_questions", []) or [{"sub-question": question}]
            involved_types = decomposition_result.get(
                "involved_types",
                {"nodes": [], "relations": [], "attributes": []},
            )
        except Exception as exc:
            logger.error(f"Question decomposition failed: {exc}")
            sub_questions = [{"sub-question": question}]
            involved_types = {"nodes": [], "relations": [], "attributes": []}
            decompose_fallback = True
            decompose_error = str(exc)

        sub_question_results: List[Dict] = []
        for index, sub_question in enumerate(sub_questions, start=1):
            sub_question_text = sub_question.get("sub-question", question)
            try:
                retrieval_results, time_taken = self.kt_retriever.process_retrieval_results(
                    sub_question_text,
                    top_k=self.config.retrieval.top_k_filter,
                    involved_types=involved_types,
                )
                total_time += time_taken
                triples = retrieval_results.get("triples", []) or []
                chunk_ids = retrieval_results.get("chunk_ids", []) or []
                chunk_contents = retrieval_results.get("chunk_contents", []) or []
                diagnostics = retrieval_results.get("diagnostics", {}) or {}

                all_triples.update(triples)
                all_chunk_ids.update(str(chunk_id) for chunk_id in chunk_ids)

                if isinstance(chunk_contents, dict):
                    for chunk_id, content in chunk_contents.items():
                        all_chunk_contents[str(chunk_id)] = content
                else:
                    for chunk_index, chunk_id in enumerate(chunk_ids):
                        if chunk_index < len(chunk_contents):
                            all_chunk_contents[str(chunk_id)] = chunk_contents[chunk_index]

                sub_question_results.append(
                    {
                        "type": "sub_question",
                        "index": index,
                        "question": sub_question_text,
                        "triples": triples[:10],
                        "triples_count": len(triples),
                        "chunks_count": len(chunk_ids),
                        "processing_time": time_taken,
                        "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
                        "chunk_contents": merge_chunk_contents(
                            [str(chunk_id) for chunk_id in chunk_ids],
                            all_chunk_contents,
                        )[:5],
                        "retrieval_diagnostics": diagnostics,
                    }
                )
            except Exception as exc:
                logger.error(f"Retrieval failed for sub-question {index}: {exc}")
                sub_question_results.append(
                    {
                        "type": "sub_question",
                        "index": index,
                        "question": sub_question_text,
                        "triples": [],
                        "triples_count": 0,
                        "chunks_count": 0,
                        "processing_time": 0.0,
                        "chunk_ids": [],
                        "chunk_contents": [],
                        "retrieval_diagnostics": {},
                        "error": str(exc),
                    }
                )

        dedup_triples = deduplicate_triples(list(all_triples))
        dedup_chunk_ids = list(dict.fromkeys(all_chunk_ids))
        dedup_chunk_contents = merge_chunk_contents(dedup_chunk_ids, all_chunk_contents)

        if len(dedup_triples) > self.config.retrieval.top_k_filter:
            question_keywords = set(question.lower().split())
            scored_triples = []
            for triple in dedup_triples:
                triple_lower = triple.lower()
                score = sum(1 for keyword in question_keywords if keyword in triple_lower)
                scored_triples.append((triple, score))
            scored_triples.sort(key=lambda item: item[1], reverse=True)
            dedup_triples = [triple for triple, _ in scored_triples[: self.config.retrieval.top_k_filter]]

        if len(dedup_chunk_contents) > self.config.retrieval.top_k_filter:
            dedup_chunk_contents = rerank_chunks_by_keywords(
                dedup_chunk_contents,
                question,
                self.config.retrieval.top_k_filter,
            )

        context = "=== Triples ===\n" + "\n".join(dedup_triples)
        context += "\n=== Chunks ===\n" + "\n".join(dedup_chunk_contents)
        prompt = self.kt_retriever.generate_prompt(question, context)
        initial_answer = self._call_answer_model(prompt)

        return {
            "decomposition_result": decomposition_result,
            "sub_questions": sub_questions,
            "involved_types": involved_types,
            "triples": dedup_triples,
            "chunk_ids": dedup_chunk_ids,
            "chunk_contents": dedup_chunk_contents,
            "sub_question_results": sub_question_results,
            "initial_answer": initial_answer,
            "total_time": total_time,
            "decompose_fallback": decompose_fallback,
            "decompose_error": decompose_error,
        }

    def _run_noagent(self, question_id: str, question: str) -> QAPrediction:
        start_time = time.time()
        result = self._initial_question_decomposition(question)
        return QAPrediction(
            question_id=question_id,
            answer=result["initial_answer"],
            sub_questions=result["sub_questions"],
            retrieved_triples=result["triples"],
            retrieved_chunks=result["chunk_contents"],
            reasoning_steps=result["sub_question_results"],
            decompose_fallback=result["decompose_fallback"],
            decompose_error=result["decompose_error"],
            schema_path_used=self.schema_path,
            latency_seconds=time.time() - start_time,
        )

    def _run_agent(self, question_id: str, question: str) -> QAPrediction:
        start_time = time.time()
        initial_result = self._initial_question_decomposition(question)

        all_triples = set(initial_result["triples"])
        all_chunk_ids = set(initial_result["chunk_ids"])
        all_chunk_contents = {
            str(chunk_id): content
            for chunk_id, content in zip(
                initial_result["chunk_ids"],
                initial_result["chunk_contents"],
            )
        }

        reasoning_steps = list(initial_result["sub_question_results"])
        thoughts = [f"Initial: {initial_result['initial_answer'][:200]}"]
        current_query = question
        final_answer = initial_result["initial_answer"]
        max_steps = int(getattr(self.config.retrieval.agent, "max_steps", 3))

        for step in range(1, max_steps + 1):
            loop_triples = deduplicate_triples(list(all_triples))
            loop_chunk_ids = list(dict.fromkeys(all_chunk_ids))
            loop_chunk_contents = merge_chunk_contents(loop_chunk_ids, all_chunk_contents)
            loop_context = "=== Triples ===\n" + "\n".join(loop_triples[:20])
            loop_context += "\n=== Chunks ===\n" + "\n".join(loop_chunk_contents[:10])

            loop_prompt = f"""
You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.
Current Question: {question}
Current Iteration Query: {current_query}
Knowledge Context:
{loop_context}
Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}
Instructions:
1. If enough information is available, answer with: So the answer is: <answer>
2. Otherwise propose a new retrieval query with: The new query is: <query>
Your reasoning:
"""
            reasoning = self._call_answer_model(loop_prompt)
            thoughts.append(reasoning[:400])
            reasoning_steps.append(
                {
                    "type": "ircot_step",
                    "step": step,
                    "question": current_query,
                    "triples_count": len(loop_triples),
                    "chunks_count": len(loop_chunk_ids),
                    "thought": reasoning[:300],
                }
            )

            if "So the answer is:" in reasoning:
                match = re.search(
                    r"So the answer is:\s*(.*)",
                    reasoning,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                final_answer = match.group(1).strip() if match else reasoning
                break

            if "The new query is:" not in reasoning:
                final_answer = final_answer or reasoning
                break

            new_query = reasoning.split("The new query is:", 1)[1].strip().splitlines()[0]
            if not new_query or new_query == current_query:
                final_answer = final_answer or reasoning
                break

            current_query = new_query
            retrieval_results, _ = self.kt_retriever.process_retrieval_results(
                current_query,
                top_k=self.config.retrieval.top_k_filter,
            )
            new_triples = retrieval_results.get("triples", []) or []
            new_chunk_ids = retrieval_results.get("chunk_ids", []) or []
            new_chunk_contents = retrieval_results.get("chunk_contents", []) or []

            all_triples.update(new_triples)
            all_chunk_ids.update(str(chunk_id) for chunk_id in new_chunk_ids)

            if isinstance(new_chunk_contents, dict):
                for chunk_id, content in new_chunk_contents.items():
                    all_chunk_contents[str(chunk_id)] = content
            else:
                for chunk_index, chunk_id in enumerate(new_chunk_ids):
                    if chunk_index < len(new_chunk_contents):
                        all_chunk_contents[str(chunk_id)] = new_chunk_contents[chunk_index]

        final_triples = deduplicate_triples(list(all_triples))[:20]
        final_chunk_ids = list(dict.fromkeys(all_chunk_ids))
        final_chunk_contents = merge_chunk_contents(final_chunk_ids, all_chunk_contents)[:10]

        return QAPrediction(
            question_id=question_id,
            answer=final_answer,
            sub_questions=initial_result["sub_questions"],
            retrieved_triples=final_triples,
            retrieved_chunks=final_chunk_contents,
            reasoning_steps=reasoning_steps,
            decompose_fallback=initial_result["decompose_fallback"],
            decompose_error=initial_result["decompose_error"],
            schema_path_used=self.schema_path,
            latency_seconds=time.time() - start_time,
        )

    def answer_question(self, question_id: str, question: str) -> QAPrediction:
        try:
            if self.qa_mode == "noagent":
                return self._run_noagent(question_id, question)
            return self._run_agent(question_id, question)
        except Exception as exc:
            logger.error(f"Offline QA failed for question_id={question_id}: {exc}")
            return QAPrediction(
                question_id=question_id,
                answer="",
                schema_path_used=self.schema_path,
                latency_seconds=0.0,
                error=str(exc),
            )
