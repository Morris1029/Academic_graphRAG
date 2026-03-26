import json
import os
import threading
import time
import hashlib
import re
from concurrent import futures
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import nanoid
import networkx as nx
import tiktoken
import json_repair

from config import get_config
from utils import call_llm_api, graph_processor, tree_comm
from utils.entity_normalizer import EntityNormalizer
from utils.logger import logger

class KTBuilder:
    def __init__(self, dataset_name, schema_path=None, mode=None, config=None):
        if config is None:
            config = get_config()

        self.config = config
        self.dataset_name = dataset_name
        self.schema = self.load_schema(schema_path or config.get_dataset_config(dataset_name).schema_path)
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.datasets_no_chunk = config.construction.datasets_no_chunk
        self.token_len = 0
        self.lock = threading.Lock()
        self.llm_client = call_llm_api.LLMCompletionCall(
            scope="kg",
            timeout_seconds=float(getattr(config.construction, "llm_timeout_seconds", 90)),
        )
        self.all_chunks = {}
        self.mode = mode or config.construction.mode
        self.doc_meta_by_chunk_id: Dict[str, Dict[str, Any]] = {}
        self.chunk_audit_records: Dict[str, Dict[str, Any]] = {}
        self._entity_name_index: Dict[Tuple[str, str], str] = {}
        self.construction_cache_root = os.path.join(
            getattr(self.config.construction, "extraction_cache_dir", "output/construction_cache"),
            self.dataset_name,
        )
        self.bridge_cache_dir = self.construction_cache_root
        self.doc_extraction_cache_dir = os.path.join(self.construction_cache_root, "doc_extraction")
        os.makedirs(self.bridge_cache_dir, exist_ok=True)
        os.makedirs(self.doc_extraction_cache_dir, exist_ok=True)
        self.resume_enabled = bool(getattr(self.config.construction, "resume_enabled", True))
        self.replay_cached_extractions = bool(getattr(self.config.construction, "replay_cached_extractions", True))
        self.max_concurrent_llm_requests = max(
            1, int(getattr(self.config.construction, "max_concurrent_llm_requests", 4))
        )
        self.requests_per_minute = max(0, int(getattr(self.config.construction, "requests_per_minute", 120)))
        self.tokens_per_minute_budget = max(
            0, int(getattr(self.config.construction, "tokens_per_minute_budget", 0))
        )
        self.retry_attempts = max(1, int(getattr(self.config.construction, "retry_attempts", 3)))
        self.retry_backoff_base_seconds = max(
            0.1, float(getattr(self.config.construction, "retry_backoff_base_seconds", 2.0))
        )
        self.retry_backoff_max_seconds = max(
            self.retry_backoff_base_seconds,
            float(getattr(self.config.construction, "retry_backoff_max_seconds", 20.0)),
        )
        self.llm_timeout_seconds = max(
            1.0, float(getattr(self.config.construction, "llm_timeout_seconds", 90))
        )
        self.llm_request_semaphore = threading.Semaphore(self.max_concurrent_llm_requests)
        self.rate_limit_lock = threading.Lock()
        self.request_times = deque()
        self.request_token_events = deque()
        self.duplicate_doc_ids: Set[str] = set()
        self.schema_type_aliases = {
            "paper": "论文",
            "author": "作者",
            "organization": "机构",
            "institution": "机构",
            "journal": "期刊",
            "technology": "技术",
            "technique": "技术",
            "method": "研究方法",
            "framework": "研究方法",
            "topic": "研究主题",
            "theme": "研究主题",
            "scenario": "教学场景",
            "field": "教育领域",
            "教学模式": "研究方法",
            "人才培养模式": "研究方法",
            "研究理论": "研究主题",
            "教育理念": "研究主题",
        }
        self.generic_entity_name_blacklist = {
            "教学模式",
            "研究理论",
            "教育理念",
            "人才培养模式",
        }
        self.metadata_author_blacklist_tokens = (
            "编辑部",
            "本刊",
            "记者",
            "评论员",
            "导读",
            "选题",
            "活动",
            "指南",
        )
        self.allowed_schema_node_types = {
            str(node_type).strip()
            for node_type in self.schema.get("Nodes", [])
            if str(node_type).strip()
        }
        self.entity_normalizer = EntityNormalizer(
            schema_type_aliases=self.schema_type_aliases,
            config_path=getattr(self.config.construction, "entity_aliases_path", "config/entity_aliases.yaml"),
        )
        self.entity_alias_audit: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.cross_doc_stats = {
            "cross_doc_enabled": bool(getattr(self.config.construction, "cross_doc_enabled", True)),
            "bridge_batches_total": 0,
            "bridge_batches_cached": 0,
            "bridge_batches_failed": 0,
            "bridge_edges_added": 0,
            "bridge_edges_fallback": 0,
            "documents_total": 0,
            "documents_extracted": 0,
            "documents_loaded_from_cache": 0,
            "documents_failed": 0,
            "documents_applied": 0,
            "documents_apply_failed": 0,
            "llm_retries": 0,
            "parse_retries": 0,
        }

    def _get_doc_meta(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        meta = doc.get("meta", {})
        return meta if isinstance(meta, dict) else {}

    def _get_document_uid(self, doc: Dict[str, Any]) -> str:
        base_id = str(doc.get("id", "") or nanoid.generate(size=8))
        if base_id not in self.duplicate_doc_ids:
            return base_id

        meta = self._get_doc_meta(doc)
        signature = "|".join(
            [
                base_id,
                str(meta.get("title", "")).strip(),
                str(meta.get("source", "")).strip(),
                str(meta.get("year", "")).strip(),
                str(meta.get("authors", "")).strip(),
            ]
        )
        digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
        return f"{base_id}__{digest}"

    def _build_paper_anchor_key(self, entity_name: Any, chunk_id: Any) -> str:
        if not chunk_id:
            return ""
        meta = self.doc_meta_by_chunk_id.get(str(chunk_id), {})
        if not isinstance(meta, dict):
            return ""
        entity_key = self._normalize_entity_name(entity_name)
        title_key = self._normalize_entity_name(meta.get("title", ""))
        doc_uid = str(meta.get("doc_uid", "")).strip()
        if entity_key and title_key and entity_key == title_key and doc_uid:
            return f"paper_doc::{doc_uid}"
        return ""

    def _register_chunk_audit_entry(self, meta: Dict[str, Any], chunk_text: str) -> None:
        doc_uid = str(meta.get("doc_uid", "")).strip()
        if not doc_uid:
            return
        self.chunk_audit_records[doc_uid] = {
            "doc_uid": doc_uid,
            "source_doc_id": str(meta.get("source_doc_id", "")).strip(),
            "title": str(meta.get("title", "")).strip(),
            "year": str(meta.get("year", "")).strip(),
            "source": str(meta.get("source", "")).strip(),
            "authors": str(meta.get("authors", "")).strip(),
            "chunk_written": bool(chunk_text),
            "chunk_char_count": len(str(chunk_text or "")),
        }

    def _save_chunk_audit_file(self) -> None:
        os.makedirs("output/chunks", exist_ok=True)
        audit_path = f"output/chunks/{self.dataset_name}_chunk_audit.jsonl"
        with open(audit_path, "w", encoding="utf-8") as f:
            for record in sorted(self.chunk_audit_records.values(), key=lambda item: item.get("doc_uid", "")):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Chunk audit saved to {audit_path} ({len(self.chunk_audit_records)} records)")

    def _canonicalize_entity_type(self, entity_type: Any, entity_name: str = "") -> str:
        raw = str(entity_type or "").strip()
        if not raw:
            return ""

        lowered = raw.lower()
        mapped = self.schema_type_aliases.get(lowered, self.schema_type_aliases.get(raw, raw))
        if mapped in self.allowed_schema_node_types:
            return mapped

        logger.info(
            "Dropping schema-outside entity type '%s' for entity '%s' in dataset '%s'",
            raw,
            str(entity_name or "").strip(),
            self.dataset_name,
        )
        return ""

    def _should_skip_entity_name(self, entity_name: Any) -> bool:
        name = str(entity_name or "").strip()
        return bool(name and name in self.generic_entity_name_blacklist)

    def load_schema(self, schema_path) -> Dict[str, Any]:
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                return schema
        except FileNotFoundError:
            return dict()
    # 2. 优化论文元数据提取逻辑：这里是抽取环节
    def _generate_llm_input(self, chunk: Dict) -> str:
        """
        将结构化的论文 JSON 转为 LLM 易理解的字符串格式，
        用于图谱构建，包含完整元数据信息。
        """
        if not isinstance(chunk, dict):
            return str(chunk)

        # 兼容数据结构：
        # chunk 可能是包含 "meta" 的字典，也可能本身就是 meta 字典
        data_source = chunk.get("meta", chunk)

        title = data_source.get("title", "")
        abstract = data_source.get("abstract", "")

        prompt_text = (
            f"文献标题: {title}\n"
            f"摘要: {abstract}\n"
        )
        return prompt_text

    def chunk_text(self, doc: Dict) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        处理论文 JSON 结构。
        Chunk 保存逻辑：仅保留 Title + Abstract，用于检索。
        """
        source_doc_id = str(doc.get("id", nanoid.generate(size=8)))
        doc_id = self._get_document_uid(doc)

        meta = dict(self._get_doc_meta(doc))
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")
        chunk_content_text = f"Title: {title}\nAbstract: {abstract}"
        meta["source_doc_id"] = source_doc_id
        meta["doc_uid"] = doc_id

        chunk = {
            "id": doc_id,
            "text": chunk_content_text,
            "meta": meta,
        }

        with self.lock:
            self.all_chunks[doc_id] = chunk_content_text
            self.doc_meta_by_chunk_id[doc_id] = meta
            self._register_chunk_audit_entry(meta, chunk_content_text)

        return [chunk], {doc_id: chunk}

    def _clean_text(self, text: str) -> str:
        if not text:
            return "[EMPTY_TEXT]"

        if self.dataset_name == "graphrag-bench":
            safe_chars = {
                *" .:,!?()-+=[]{}()\\/|_^~<>*&%$#@!;\"'`"
            }
            cleaned = "".join(
                char for char in text
                if char.isalnum() or char.isspace() or char in safe_chars
            ).strip()
        else:
            safe_chars = {
                *" .:,!?()-+="
            }
            cleaned = "".join(
                char for char in text
                if char.isalnum() or char.isspace() or char in safe_chars
            ).strip()

        return cleaned if cleaned else "[EMPTY_AFTER_CLEANING]"

    def save_chunks_to_file(self):
        os.makedirs("output/chunks", exist_ok=True)
        chunk_file = f"output/chunks/{self.dataset_name}.txt"

        # 简单写入逻辑，使用覆盖模式避免重复
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in self.all_chunks.items():
                # 移除换行符以避免破坏格式
                clean_text = str(chunk_text).replace('\n', ' \\n ')
                f.write(f"id: {chunk_id}\tChunk: {clean_text}\n")

        logger.info(
            f"Chunk data saved to {chunk_file} ({len(self.all_chunks)} unique chunks, "
            f"{len(self.duplicate_doc_ids)} duplicated source ids normalized)"
        )
        self._save_chunk_audit_file()

    def _get_doc_cache_path(self, doc_id: str) -> str:
        safe_doc_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(doc_id or "unknown"))
        return os.path.join(self.doc_extraction_cache_dir, f"{safe_doc_id}.json")

    def _load_doc_extraction_cache(self, doc_id: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_doc_cache_path(doc_id)
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else None
        except Exception as e:
            logger.warning(f"Failed to load extraction cache for {doc_id}: {type(e).__name__}: {e}")
            return None

    def _save_doc_extraction_cache(self, doc_id: str, payload: Dict[str, Any]) -> None:
        cache_path = self._get_doc_cache_path(doc_id)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _build_doc_cache_payload(self, doc: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_doc_id = str(doc.get("id", "unknown"))
        doc_id = self._get_document_uid(doc) if doc else str(chunks[0].get("id", "unknown") if chunks else "unknown")
        return {
            "doc_id": doc_id,
            "source_doc_id": source_doc_id,
            "meta": doc.get("meta", {}),
            "prompt_version": getattr(self.config.construction, "prompt_version", "v1"),
            "status": "pending",
            "timestamp": time.time(),
            "chunks": [
                {
                    "chunk_id": str(chunk.get("id", "")),
                    "chunk_text": chunk.get("text", ""),
                    "status": "pending",
                    "attempt_count": 0,
                    "parsed_response": None,
                    "error": None,
                }
                for chunk in chunks
            ],
        }

    def _all_cached_chunks_successful(self, payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(payload, dict):
            return False
        chunks = payload.get("chunks", [])
        return bool(chunks) and all(
            isinstance(item, dict) and item.get("status") == "success" and item.get("parsed_response")
            for item in chunks
        )

    def _prune_rate_limit_windows(self, now: float) -> None:
        while self.request_times and now - self.request_times[0] >= 60:
            self.request_times.popleft()
        while self.request_token_events and now - self.request_token_events[0][0] >= 60:
            self.request_token_events.popleft()

    def _wait_for_llm_slot(self, prompt_tokens: int) -> None:
        while True:
            sleep_for = 0.0
            with self.rate_limit_lock:
                now = time.monotonic()
                self._prune_rate_limit_windows(now)
                req_ok = self.requests_per_minute <= 0 or len(self.request_times) < self.requests_per_minute
                token_sum = sum(tokens for _, tokens in self.request_token_events)
                token_ok = (
                    self.tokens_per_minute_budget <= 0
                    or token_sum + prompt_tokens <= self.tokens_per_minute_budget
                )
                if req_ok and token_ok:
                    self.request_times.append(now)
                    if self.tokens_per_minute_budget > 0:
                        self.request_token_events.append((now, prompt_tokens))
                    return

                candidate_delays = []
                if self.requests_per_minute > 0 and self.request_times:
                    candidate_delays.append(max(0.05, 60 - (now - self.request_times[0])))
                if self.tokens_per_minute_budget > 0 and self.request_token_events:
                    candidate_delays.append(max(0.05, 60 - (now - self.request_token_events[0][0])))
                sleep_for = min(candidate_delays) if candidate_delays else 0.1
            time.sleep(sleep_for)

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        message = str(error).lower()
        retry_markers = [
            "429",
            "rate limit",
            "too many requests",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "connection",
            "reset by peer",
            "overloaded",
        ]
        return any(marker in message for marker in retry_markers)

    def _backoff_seconds(self, attempt: int) -> float:
        raw = self.retry_backoff_base_seconds * (2 ** max(0, attempt - 1))
        return min(self.retry_backoff_max_seconds, raw)

    def extract_with_llm(self, prompt: str):
        prompt_tokens = self.token_cal(prompt)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._wait_for_llm_slot(prompt_tokens)
                with self.llm_request_semaphore:
                    response = self.llm_client.call_api(
                        prompt,
                        timeout_seconds=self.llm_timeout_seconds,
                    )
                self.token_len += self.token_cal(prompt + str(response))
                print("\n" + "=" * 50)
                print(f"DEBUG: LLM Response for Prompt (first 100 chars): {prompt[:100]}...")
                print(f"DEBUG: Raw Response: {response}")
                print("=" * 50 + "\n")
                return response
            except Exception as e:
                last_error = e
                if attempt >= self.retry_attempts or not self._is_retryable_llm_error(e):
                    logger.error(
                        f"LLM extraction failed after {attempt} attempt(s): {type(e).__name__}: {e}"
                    )
                    raise
                self.cross_doc_stats["llm_retries"] += 1
                backoff = self._backoff_seconds(attempt)
                logger.warning(
                    f"Retryable LLM error on attempt {attempt}/{self.retry_attempts}: "
                    f"{type(e).__name__}: {e}. Backing off {backoff:.1f}s"
                )
                time.sleep(backoff)

        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM extraction failed without response")

    def token_cal(self, text: str):
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(str(text)))
        except Exception:
            return 0

    def _get_construction_prompt(self, chunk: Any) -> str:
        """Get the appropriate construction prompt based on dataset name and mode (agent/noagent)."""
        recommend_schema = json.dumps(self.schema, ensure_ascii=False, indent=2)
        # 将 Chunk 数据转换为 LLM 易读的字符串
        chunk_str = self._generate_llm_input(chunk)

        # 当前统一使用 general/general_agent，可按数据集再扩展
        prompt_type = "general"
        if self.mode == "agent":
            prompt_type = f"{prompt_type}_agent"
        prompt = self.config.get_prompt_formatted(
            "construction",
            prompt_type,
            schema=recommend_schema,
            chunk=chunk_str,
        )
        prompt += (
            "\n\n[Entity naming normalization rule]\n"
            "For the same technology concept, abbreviations, full names, and Chinese-English aliases "
            "must be unified to one entity name."
        )
        prompt += (
            "\n\n[Attribute formatting rule]\n"
            "Keyword attributes must be split into separate items, for example: "
            '["关键词: A", "关键词: B", "关键词: C"].'
        )
        return prompt

        # base_prompt_type = prompt_type_map.get(self.dataset_name, "general")

        # Add agent suffix if in agent mode
        # if self.mode == "agent":
        #     prompt_type = f"{base_prompt_type}_agent"
        # else:
        #     prompt_type = base_prompt_type

        # return self.config.get_prompt_formatted("construction", prompt_type, schema=recommend_schema, chunk=chunk)

    # def _validate_and_parse_llm_response(self, prompt: str, llm_response: str) -> dict:
    #     """Validate and parse LLM response, returning None if invalid."""
    #     if llm_response is None:
    #         return None
    #
    #     try:
    #         self.token_len += self.token_cal(prompt + llm_response)
    #         return json_repair.loads(llm_response)
    #     except Exception as e:
    #         llm_response_str = str(llm_response) if llm_response is not None else "None"
    #         return None

    # ---  完善解析逻辑，去掉内部的 Token 统计 ---
    def _validate_and_parse_llm_response(self, llm_response: str) -> dict:
        """Parse and validate JSON returned by the LLM."""
        if not llm_response:
            return None
        try:
            return json_repair.loads(llm_response)
        except Exception as e:
            logger.error(f"JSON Repair failed: {e}")
            return None

    def _normalize_entity_name(self, name: Any) -> str:
        """Normalize entity names for stable indexing and deduplication."""
        return self.entity_normalizer.normalize_name_key(name)

    def _normalize_relation_name(self, relation: Any) -> str:
        """Normalize relation names while keeping the original Chinese labels."""
        raw = str(relation or "").strip()
        if not raw:
            return raw

        synonym_map = {
            "提出了": "提出",
            "提出的": "提出",
            "发表于于": "发表于",
            "发布于": "发表于",
            "应用于": "应用于",
            "用于": "应用于",
            "基于": "基于",
        }
        return synonym_map.get(raw, raw)

    def _is_paper_schema_type(self, schema_type: Any) -> bool:
        """Detect whether a schema type represents a paper node."""
        text = str(schema_type or "").strip().lower()
        return text in {"论文", "paper"}

    def _get_node_schema_type(self, node_id: str) -> str:
        props = self.graph.nodes[node_id].get("properties", {})
        if not isinstance(props, dict):
            return ""
        return str(props.get("schema_type", "")).strip()

    def _get_node_chunk_id(self, node_id: str) -> str:
        props = self.graph.nodes[node_id].get("properties", {})
        if not isinstance(props, dict):
            return ""
        chunk_id = props.get("chunk_id", props.get("chunk id", ""))
        return str(chunk_id).strip() if chunk_id is not None else ""

    def _split_metadata_values(self, raw_value: Any) -> List[str]:
        text = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        if not text:
            return []
        if any(sep in text for sep in [";", "；", "\n"]):
            parts = re.split(r"[;；\n]+", text)
        else:
            parts = re.split(r"[、,，]+", text)

        values: List[str] = []
        seen: Set[str] = set()
        for part in parts:
            clean = str(part or "").strip(" \t\r\n;；,，、")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            values.append(clean)
        return values

    def _is_valid_metadata_author_name(self, author_name: Any) -> bool:
        name = str(author_name or "").strip()
        if not name or self._should_skip_entity_name(name):
            return False
        if "《" in name or "》" in name:
            return False
        if any(token in name for token in self.metadata_author_blacklist_tokens):
            return False
        return True

    def _normalize_primary_organization_name(self, org_name: Any) -> str:
        name = re.sub(r"\s+", " ", str(org_name or "")).strip()
        if not name:
            return ""

        suffixes = (
            "大学",
            "学院",
            "中学",
            "小学",
            "学校",
            "研究院",
            "研究所",
            "图书馆",
            "出版社",
            "医院",
            "中心",
            "实验室",
        )
        for suffix in suffixes:
            idx = name.find(suffix)
            if idx != -1:
                return name[: idx + len(suffix)]
        return name

    def _edge_exists(self, src_id: str, tgt_id: str, relation: str) -> bool:
        normalized_relation = self._normalize_relation_name(relation)
        if src_id not in self.graph.nodes or tgt_id not in self.graph.nodes:
            return False
        for _, neighbor, data in self.graph.out_edges(src_id, data=True):
            if neighbor == tgt_id and self._normalize_relation_name(data.get("relation", "")) == normalized_relation:
                return True
        return False

    def _has_outgoing_relation_to_type(self, src_id: str, relation: str, schema_type: str) -> bool:
        normalized_relation = self._normalize_relation_name(relation)
        expected_type = str(schema_type or "").strip()
        if src_id not in self.graph.nodes:
            return False
        for _, neighbor, data in self.graph.out_edges(src_id, data=True):
            if self._normalize_relation_name(data.get("relation", "")) != normalized_relation:
                continue
            if self._get_node_schema_type(neighbor) == expected_type:
                return True
        return False

    def _resolve_canonical_entity_name(self, entity_name: Any, entity_type: Any = None) -> Tuple[str, str]:
        return self.entity_normalizer.resolve(entity_name, entity_type)

    def _record_entity_alias(self, entity_name: Any, canonical_name: str, entity_type: Any = None) -> None:
        alias_text = str(entity_name or "").strip()
        canonical_text = str(canonical_name or "").strip()
        normalized_type = self._canonicalize_entity_type(entity_type, canonical_text)
        if not alias_text or not canonical_text:
            return
        bucket = self.entity_alias_audit[(normalized_type, canonical_text)]
        bucket.add(canonical_text)
        bucket.add(alias_text)

    def _collect_entity_aliases(
        self,
        entity_name: Any,
        canonical_name: str,
        entity_type: Any = None,
        chunk_id: Any = None,
    ) -> List[str]:
        aliases: List[str] = []
        raw_name = str(entity_name or "").strip()
        if raw_name and raw_name != canonical_name:
            aliases.append(raw_name)
        if self._is_paper_schema_type(entity_type):
            meta = self.doc_meta_by_chunk_id.get(str(chunk_id), {})
            title = str(meta.get("title", "")).strip() if isinstance(meta, dict) else ""
            if title and title != canonical_name:
                aliases.append(title)
        deduped: List[str] = []
        for alias in aliases:
            if alias and alias != canonical_name and alias not in deduped:
                deduped.append(alias)
        return deduped

    def _apply_entity_aliases_to_properties(
        self,
        properties: Dict[str, Any],
        entity_name: Any,
        canonical_name: str,
        entity_type: Any = None,
        chunk_id: Any = None,
    ) -> None:
        alias_candidates = self._collect_entity_aliases(entity_name, canonical_name, entity_type, chunk_id)
        if not alias_candidates:
            self._record_entity_alias(entity_name, canonical_name, entity_type)
            return
        aliases = properties.setdefault("aliases", [])
        if not isinstance(aliases, list):
            aliases = [str(aliases)]
            properties["aliases"] = aliases
        for alias in alias_candidates:
            if alias not in aliases:
                aliases.append(alias)
        self._record_entity_alias(entity_name, canonical_name, entity_type)

    def _ensure_entity_alias_on_node(
        self,
        entity_node_id: str,
        entity_name: Any,
        canonical_name: str,
        entity_type: Any = None,
        chunk_id: Any = None,
        nodes_to_add: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        if entity_node_id in self.graph.nodes:
            node_props = self.graph.nodes[entity_node_id].setdefault("properties", {})
            if isinstance(node_props, dict):
                self._apply_entity_aliases_to_properties(
                    node_props,
                    entity_name,
                    canonical_name,
                    entity_type,
                    chunk_id,
                )
            return

        if nodes_to_add is not None:
            for node_id, node_data in nodes_to_add:
                if node_id != entity_node_id:
                    continue
                node_props = node_data.setdefault("properties", {})
                if isinstance(node_props, dict):
                    self._apply_entity_aliases_to_properties(
                        node_props,
                        entity_name,
                        canonical_name,
                        entity_type,
                        chunk_id,
                    )
                return

        self._record_entity_alias(entity_name, canonical_name, entity_type)

    def _register_entity_index(self, entity_name: Any, node_id: str, entity_type: Any = None, chunk_id: Any = None):
        canonical_name, canonical_key = self._resolve_canonical_entity_name(entity_name, entity_type)
        if not canonical_key:
            return
        raw_key = self._normalize_entity_name(entity_name)
        type_key = str(entity_type or "").strip().lower()
        self._entity_name_index[(canonical_key, type_key)] = node_id
        self._entity_name_index[(canonical_key, "")] = node_id
        if raw_key:
            self._entity_name_index[(raw_key, type_key)] = node_id
            self._entity_name_index[(raw_key, "")] = node_id
        # 同名无类型索引，便于兜底检索
        if self._is_paper_schema_type(entity_type):
            paper_anchor = self._build_paper_anchor_key(canonical_name, chunk_id)
            if paper_anchor:
                self._entity_name_index[(paper_anchor, type_key)] = node_id
                self._entity_name_index[(paper_anchor, "")] = node_id

    def _lookup_entity_node_id(self, entity_name: Any, entity_type: Any = None, chunk_id: Any = None) -> Optional[str]:
        canonical_name, canonical_key = self._resolve_canonical_entity_name(entity_name, entity_type)
        raw_key = self._normalize_entity_name(entity_name)
        if not canonical_key and not raw_key:
            return None
        type_key = str(entity_type or "").strip().lower()
        if self._is_paper_schema_type(entity_type):
            paper_anchor = self._build_paper_anchor_key(canonical_name, chunk_id)
            if paper_anchor:
                anchored = self._entity_name_index.get((paper_anchor, type_key))
                if anchored:
                    return anchored
                anchored = self._entity_name_index.get((paper_anchor, ""))
                if anchored:
                    return anchored
        for key in filter(None, [canonical_key, raw_key]):
            if type_key:
                exact = self._entity_name_index.get((key, type_key))
                if exact:
                    return exact
            fallback = self._entity_name_index.get((key, ""))
            if fallback:
                fallback_type = self._get_node_schema_type(fallback).strip().lower()
                if not type_key or not fallback_type or fallback_type == type_key:
                    return fallback
        return None

    def _add_edge_with_metadata(
        self,
        src_id: str,
        tgt_id: str,
        relation: str,
        relation_origin: str,
        confidence: float,
        evidence_chunk_ids: Optional[List[str]] = None,
        source_paper_ids: Optional[List[str]] = None,
        reason: str = "",
    ):
        """Add an edge with optional metadata while keeping legacy readers compatible."""
        edge_payload = {
            "relation": self._normalize_relation_name(relation),
            "relation_origin": relation_origin,
            "confidence": float(confidence),
            "evidence_chunk_ids": sorted({str(x) for x in (evidence_chunk_ids or []) if str(x)}),
            "source_paper_ids": sorted({str(x) for x in (source_paper_ids or []) if str(x)}),
        }
        if reason:
            edge_payload["reason"] = str(reason)
        self.graph.add_edge(src_id, tgt_id, **edge_payload)

    def _find_or_create_entity(self, entity_name: str, chunk_id: int, nodes_to_add: list,
                               entity_type: str = None) -> str:
        """Find existing entity or create a new one, returning the entity node ID."""
        entity_type = self._canonicalize_entity_type(entity_type, entity_name)
        canonical_name, _ = self._resolve_canonical_entity_name(entity_name, entity_type)
        with self.lock:
            entity_node_id = self._lookup_entity_node_id(entity_name, entity_type, chunk_id)
            if not entity_node_id:
                if self._is_paper_schema_type(entity_type):
                    current_doc_uid = str(self.doc_meta_by_chunk_id.get(str(chunk_id), {}).get("doc_uid", "")).strip()
                    entity_node_id = next(
                        (
                            n
                            for n, d in self.graph.nodes(data=True)
                            if d.get("label") == "entity"
                            and d["properties"]["name"] == canonical_name
                            and str(d.get("properties", {}).get("doc_uid", "")).strip() == current_doc_uid
                        ),
                        None,
                    )
                else:
                    entity_node_id = next(
                        (
                            n
                            for n, d in self.graph.nodes(data=True)
                            if d.get("label") == "entity"
                            and d["properties"]["name"] == canonical_name
                            and (
                                not entity_type
                                or not str(d.get("properties", {}).get("schema_type", "")).strip()
                                or str(d.get("properties", {}).get("schema_type", "")).strip() == entity_type
                            )
                        ),
                        None,
                    )

            if not entity_node_id:
                entity_node_id = f"entity_{self.node_counter}"
                properties = {"name": canonical_name, "chunk id": chunk_id}
                if entity_type:
                    properties["schema_type"] = entity_type
                if self._is_paper_schema_type(entity_type):
                    meta = self.doc_meta_by_chunk_id.get(str(chunk_id), {})
                    properties["doc_uid"] = str(meta.get("doc_uid", "")).strip()
                    properties["source_doc_id"] = str(meta.get("source_doc_id", "")).strip()
                self._apply_entity_aliases_to_properties(
                    properties,
                    entity_name,
                    canonical_name,
                    entity_type,
                    chunk_id,
                )

                nodes_to_add.append((
                    entity_node_id,
                    {
                        "label": "entity",
                        "properties": properties,
                        "level": 2
                    }
                ))
                self._register_entity_index(entity_name, entity_node_id, entity_type, chunk_id)
                self.node_counter += 1
            else:
                self._ensure_entity_alias_on_node(
                    entity_node_id,
                    entity_name,
                    canonical_name,
                    entity_type,
                    chunk_id,
                    nodes_to_add=nodes_to_add,
                )
                self._register_entity_index(entity_name, entity_node_id, entity_type, chunk_id)

        return entity_node_id

    def _validate_triple_format(self, triple: list) -> tuple:
        """Validate and normalize triple format, returning (subject, predicate, object) or None."""
        try:
            if len(triple) > 3:
                triple = triple[:3]
            elif len(triple) < 3:
                return None

            return tuple(triple)
        except Exception as e:
            return None

    def _split_keyword_attribute(self, attr_text: str) -> List[str]:
        text = str(attr_text or "").strip()
        if not text:
            return []

        match = re.match(r"^(关键词|关键字|keywords?)\s*[:：]\s*(.+)$", text, flags=re.IGNORECASE)
        if not match:
            return [text]

        prefix = "关键词"
        raw_values = str(match.group(2) or "").strip()
        if not raw_values:
            return [f"{prefix}: "]

        parts = [
            part.strip()
            for part in re.split(r"[;；,，/、]+", raw_values)
            if part and str(part).strip()
        ]
        if not parts:
            return [f"{prefix}: {raw_values}"]
        return [f"{prefix}: {part}" for part in parts]

    def _normalize_attribute_values(self, attributes: Any) -> List[str]:
        normalized: List[str] = []
        seen: Set[str] = set()

        if not isinstance(attributes, list):
            attributes = [attributes] if attributes is not None else []

        for attr in attributes:
            safe_attr = str(attr) if isinstance(attr, dict) else str(attr or "")
            for normalized_attr in self._split_keyword_attribute(safe_attr):
                clean_attr = re.sub(r"\s+", " ", str(normalized_attr or "")).strip()
                if not clean_attr or clean_attr in seen:
                    continue
                seen.add(clean_attr)
                normalized.append(clean_attr)

        return normalized

    def _process_attributes(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """Process extracted attributes and return nodes and edges to add."""
        nodes_to_add = []
        edges_to_add = []

        for entity, attributes in extracted_attr.items():
            if self._should_skip_entity_name(entity):
                continue
            normalized_attributes = self._normalize_attribute_values(attributes)
            for attr in normalized_attributes:
                # Create attribute node
                attr_node_id = f"attr_{self.node_counter}"
                nodes_to_add.append((
                    attr_node_id,
                    {
                        "label": "attribute",
                        "properties": {"name": attr, "chunk id": chunk_id},
                        "level": 1,
                    }
                ))
                self.node_counter += 1

                entity_type = self._canonicalize_entity_type(entity_types.get(entity), entity) if entity_types else None
                entity_node_id = self._find_or_create_entity(entity, chunk_id, nodes_to_add, entity_type)
                edges_to_add.append((entity_node_id, attr_node_id, "has_attribute"))

        return nodes_to_add, edges_to_add

    def _process_triples(self, extracted_triples: list, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """Process extracted triples and return nodes and edges to add."""
        nodes_to_add = []
        edges_to_add = []

        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue

            subj, pred, obj = validated_triple
            if self._should_skip_entity_name(subj) or self._should_skip_entity_name(obj):
                continue

            subj_type = self._canonicalize_entity_type(entity_types.get(subj), subj) if entity_types else None
            obj_type = self._canonicalize_entity_type(entity_types.get(obj), obj) if entity_types else None

            subj_node_id = self._find_or_create_entity(subj, chunk_id, nodes_to_add, subj_type)
            obj_node_id = self._find_or_create_entity(obj, chunk_id, nodes_to_add, obj_type)

            edges_to_add.append((subj_node_id, obj_node_id, pred))

        return nodes_to_add, edges_to_add

    def process_level1_level2(self, chunk: str, id: int):
        """Process attributes (level 1) and triples (level 2) with optimized structure."""
        prompt = self._get_construction_prompt(chunk)
        llm_response = self.extract_with_llm(prompt)

        # Validate and parse response
        parsed_response = self._validate_and_parse_llm_response(llm_response)
        if not parsed_response:
            return

        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})

        # Process attributes and triples
        attr_nodes, attr_edges = self._process_attributes(extracted_attr, id, entity_types)
        triple_nodes, triple_edges = self._process_triples(extracted_triples, id, entity_types)

        all_nodes = attr_nodes + triple_nodes
        all_edges = attr_edges + triple_edges

        with self.lock:
            for node_id, node_data in all_nodes:
                self.graph.add_node(node_id, **node_data)

            for u, v, relation in all_edges:
                self._add_edge_with_metadata(
                    u,
                    v,
                    relation,
                    relation_origin="doc_local",
                    confidence=0.7,
                    evidence_chunk_ids=[str(id)],
                    source_paper_ids=[str(id)],
                )

    def _apply_parsed_response(self, parsed_response: Dict[str, Any], chunk_id: str) -> None:
        if not parsed_response:
            return

        new_schema_types = parsed_response.get("new_schema_types", {})
        if new_schema_types:
            logger.info(
                "Ignoring LLM-proposed schema extensions for dataset '%s': %s",
                self.dataset_name,
                json.dumps(new_schema_types, ensure_ascii=False),
            )

        entity_types = parsed_response.get("entity_types", {})
        attributes = parsed_response.get("attributes", {})
        triples = parsed_response.get("triples", [])

        with self.lock:
            for entity, attrs in attributes.items():
                if self._should_skip_entity_name(entity):
                    continue
                entity_type = self._canonicalize_entity_type(entity_types.get(entity), entity)
                entity_id = self._find_or_create_entity_direct(entity, chunk_id, entity_type)
                normalized_attributes = self._normalize_attribute_values(attrs)
                for attr in normalized_attributes:
                    attr_id = f"attr_{self.node_counter}"
                    self.graph.add_node(
                        attr_id,
                        label="attribute",
                        properties={"name": attr, "chunk_id": chunk_id},
                        level=1,
                    )
                    self._add_edge_with_metadata(
                        entity_id,
                        attr_id,
                        "has_attribute",
                        relation_origin="doc_local",
                        confidence=0.7,
                        evidence_chunk_ids=[str(chunk_id)],
                        source_paper_ids=[str(chunk_id)],
                    )
                    self.node_counter += 1

            for triple in triples:
                if len(triple) < 3:
                    continue

                src, rel, tgt = triple[0], triple[1], triple[2]
                if self._should_skip_entity_name(src) or self._should_skip_entity_name(tgt):
                    continue
                src_type = self._canonicalize_entity_type(entity_types.get(src), src)
                tgt_type = self._canonicalize_entity_type(entity_types.get(tgt), tgt)
                src_id = self._find_or_create_entity_direct(src, chunk_id, src_type)
                tgt_id = self._find_or_create_entity_direct(tgt, chunk_id, tgt_type)
                self._add_edge_with_metadata(
                    src_id,
                    tgt_id,
                    rel,
                    relation_origin="doc_local",
                    confidence=0.75,
                    evidence_chunk_ids=[str(chunk_id)],
                    source_paper_ids=[str(chunk_id)],
                )

    def _find_or_create_entity_direct(self, entity_name: str, chunk_id: int, entity_type: str = None) -> str:
        """Find existing entity or create a new one directly in graph (for agent mode)."""
        entity_type = self._canonicalize_entity_type(entity_type, entity_name)
        canonical_name, _ = self._resolve_canonical_entity_name(entity_name, entity_type)
        entity_node_id = self._lookup_entity_node_id(entity_name, entity_type, chunk_id)
        if not entity_node_id:
            if self._is_paper_schema_type(entity_type):
                current_doc_uid = str(self.doc_meta_by_chunk_id.get(str(chunk_id), {}).get("doc_uid", "")).strip()
                entity_node_id = next(
                    (
                        n
                        for n, d in self.graph.nodes(data=True)
                        if d["properties"].get("name") == canonical_name
                        and str(d.get("properties", {}).get("doc_uid", "")).strip() == current_doc_uid
                    ),
                    None,
                )
            else:
                entity_node_id = next(
                    (
                        n
                        for n, d in self.graph.nodes(data=True)
                        if d["properties"].get("name") == canonical_name
                        and (
                            not entity_type
                            or not str(d.get("properties", {}).get("schema_type", "")).strip()
                            or str(d.get("properties", {}).get("schema_type", "")).strip() == entity_type
                        )
                    ),
                    None,
                )

        if not entity_node_id:
            entity_node_id = f"entity_{self.node_counter}"
            properties = {
                "name": canonical_name,
                "chunk id": chunk_id
            }
            # # 鏍稿績淇敼锛氬鏋?entity_type 瀛樺湪锛屽氨鐢ㄥ畠鍋?label锛屽惁鍒欐墠鐢?"entity"
            # display_label = entity_type if entity_type else "entity"
            # properties = {"name": entity_name, "chunk id": chunk_id,"schema_type": entity_type}
            if entity_type:
                properties["schema_type"] = entity_type
            if self._is_paper_schema_type(entity_type):
                meta = self.doc_meta_by_chunk_id.get(str(chunk_id), {})
                properties["doc_uid"] = str(meta.get("doc_uid", "")).strip()
                properties["source_doc_id"] = str(meta.get("source_doc_id", "")).strip()
            self._apply_entity_aliases_to_properties(
                properties,
                entity_name,
                canonical_name,
                entity_type,
                chunk_id,
            )
            self.graph.add_node(
                entity_node_id,
                label="entity",
                properties=properties,
                level=2
            )
            self._register_entity_index(entity_name, entity_node_id, entity_type, chunk_id)
            self.node_counter += 1
        else:
            # 如果节点已存在但标签仍是通用类型，尝试补充更具体的 schema_type
            node_props = self.graph.nodes[entity_node_id]["properties"]
            if entity_type and "schema_type" not in node_props:
                node_props["schema_type"] = entity_type
            if self._is_paper_schema_type(entity_type):
                meta = self.doc_meta_by_chunk_id.get(str(chunk_id), {})
                node_props["doc_uid"] = str(meta.get("doc_uid", node_props.get("doc_uid", ""))).strip()
                node_props["source_doc_id"] = str(
                    meta.get("source_doc_id", node_props.get("source_doc_id", ""))
                ).strip()
            self._apply_entity_aliases_to_properties(
                node_props,
                entity_name,
                canonical_name,
                node_props.get("schema_type", entity_type),
                chunk_id,
            )
            self._register_entity_index(
                entity_name,
                entity_node_id,
                node_props.get("schema_type", entity_type),
                chunk_id,
            )

        return entity_node_id

    def _process_attributes_agent(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None):
        """Process extracted attributes in agent mode (direct graph operations)."""
        for entity, attributes in extracted_attr.items():
            if self._should_skip_entity_name(entity):
                continue
            normalized_attributes = self._normalize_attribute_values(attributes)
            for attr in normalized_attributes:
                # Create attribute node
                attr_node_id = f"attr_{self.node_counter}"
                self.graph.add_node(
                    attr_node_id,
                    label="attribute",
                    properties={
                        "name": attr,
                        "chunk id": chunk_id
                    },
                    level=1,
                )
                self.node_counter += 1

                entity_type = self._canonicalize_entity_type(entity_types.get(entity), entity) if entity_types else None
                entity_node_id = self._find_or_create_entity_direct(entity, chunk_id, entity_type)
                self._add_edge_with_metadata(
                    entity_node_id,
                    attr_node_id,
                    "has_attribute",
                    relation_origin="doc_local",
                    confidence=0.7,
                    evidence_chunk_ids=[str(chunk_id)],
                    source_paper_ids=[str(chunk_id)],
                )

    def _process_triples_agent(self, extracted_triples: list, chunk_id: int, entity_types: dict = None):
        """Process extracted triples in agent mode (direct graph operations)."""
        for triple in extracted_triples:
            validated_triple = self._validate_triple_format(triple)
            if not validated_triple:
                continue

            subj, pred, obj = validated_triple
            if self._should_skip_entity_name(subj) or self._should_skip_entity_name(obj):
                continue

            subj_type = self._canonicalize_entity_type(entity_types.get(subj), subj) if entity_types else None
            obj_type = self._canonicalize_entity_type(entity_types.get(obj), obj) if entity_types else None

            # Find or create subject and object entities
            subj_node_id = self._find_or_create_entity_direct(subj, chunk_id, subj_type)
            obj_node_id = self._find_or_create_entity_direct(obj, chunk_id, obj_type)

            self._add_edge_with_metadata(
                subj_node_id,
                obj_node_id,
                pred,
                relation_origin="doc_local",
                confidence=0.75,
                evidence_chunk_ids=[str(chunk_id)],
                source_paper_ids=[str(chunk_id)],
            )

    def process_level1_level2_agent(self, chunk: Dict, chunk_id: str):
        """核心处理流程"""
        # 1. 生成 Prompt
        prompt = self._get_construction_prompt(chunk)

        # 2. 调用 LLM
        # print(f"DEBUG: Processing {chunk_id}...")  # 调试用
        llm_response = self.extract_with_llm(prompt)

        # 3. 解析结果
        parsed_response = self._validate_and_parse_llm_response(llm_response)

        if not parsed_response:
            logger.warning(f"Failed to parse LLM response for chunk {chunk_id}")
            return
        self._apply_parsed_response(parsed_response, str(chunk_id))
        #
        # with self.lock:
        #     # 防御性处理：确保属性值不是 dict，避免可视化报错
        #     for entity, attrs in attributes.items():
        #         entity_type = entity_types.get(entity)
        #         entity_id = self._find_or_create_entity_direct(entity, chunk_id, entity_type)
        #         if isinstance(attrs, list):
        #             for attr in attrs:
        #                 # 强制转为字符串，避免 unhashable dict 错误
        #                 safe_attr = str(attr) if isinstance(attr, dict) else attr
        #                 attr_id = f"attr_{self.node_counter}"
        #                 self.graph.add_node(attr_id, label="attribute",
        #                                     properties={"name": safe_attr, "chunk_id": chunk_id},
        #                                     level=1)
        #                 self.graph.add_edge(entity_id, attr_id, relation="has_attribute")
        #                 self.node_counter += 1
        #     # 处理三元组（Level 2）
        #     for triple in triples:
        #         if len(triple) < 3: continue
        #         src, rel, tgt = triple[0], triple[1], triple[2]
        #         # src_id = self._find_or_create_entity_direct(src, chunk_id)
        #         # tgt_id = self._find_or_create_entity_direct(tgt, chunk_id)
        #         # 获取具体的类型
        #         src_type = entity_types.get(src)
        #         tgt_type = entity_types.get(tgt)
        #
        #         src_id = self._find_or_create_entity_direct(src, chunk_id, src_type)  # 传入类型
        #         tgt_id = self._find_or_create_entity_direct(tgt, chunk_id, tgt_type)  # 传入类型
        #         self.graph.add_edge(src_id, tgt_id, relation=rel)

    def _update_schema_with_new_types(self, new_schema_types: Dict[str, List[str]]):
        """Update the schema file with new types discovered by the agent.

        This method processes schema evolution suggestions from the LLM and updates
        the corresponding schema file with new node types, relations, and attributes.
        Only adds types that don't already exist in the current schema.

        Args:
            new_schema_types: Dictionary containing 'nodes', 'relations', and 'attributes' lists
        """
        try:
            schema_paths = {
                "hotpot": "schemas/hotpot.json",
                "2wiki": "schemas/2wiki.json",
                "musique": "schemas/musique.json",
                "novel": "schemas/novels_chs.json",
                "graphrag-bench": "schemas/graphrag-bench.json"
            }

            schema_path = schema_paths.get(self.dataset_name)
            if not schema_path:
                return

            with open(schema_path, 'r', encoding='utf-8') as f:
                current_schema = json.load(f)

            updated = False

            if "nodes" in new_schema_types:
                for new_node in new_schema_types["nodes"]:
                    if new_node not in current_schema.get("Nodes", []):
                        current_schema.setdefault("Nodes", []).append(new_node)
                        updated = True

            if "relations" in new_schema_types:
                for new_relation in new_schema_types["relations"]:
                    if new_relation not in current_schema.get("Relations", []):
                        current_schema.setdefault("Relations", []).append(new_relation)
                        updated = True

            if "attributes" in new_schema_types:
                for new_attribute in new_schema_types["attributes"]:
                    if new_attribute not in current_schema.get("Attributes", []):
                        current_schema.setdefault("Attributes", []).append(new_attribute)
                        updated = True

            # Save updated schema back to file
            if updated:
                with open(schema_path, 'w', encoding='utf-8') as f:
                    json.dump(current_schema, f, ensure_ascii=False, indent=2)

                # Update the in-memory schema
                self.schema = current_schema

        except Exception as e:
            logger.error(f"Failed to update schema for dataset '{self.dataset_name}': {type(e).__name__}: {e}")

    def _build_paper_node_indexes(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        paper_by_chunk: Dict[str, str] = {}
        paper_by_title: Dict[str, str] = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if not self._is_paper_schema_type(self._get_node_schema_type(node_id)):
                continue
            props = node_data.get("properties", {})
            if not isinstance(props, dict):
                continue
            title = str(props.get("name", "")).strip()
            if title:
                paper_by_title[self._normalize_entity_name(title)] = node_id
            aliases = props.get("aliases", []) or []
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_text = str(alias).strip()
                    if alias_text:
                        paper_by_title[self._normalize_entity_name(alias_text)] = node_id
            chunk_id = self._get_node_chunk_id(node_id)
            if chunk_id:
                paper_by_chunk[str(chunk_id)] = node_id
        return paper_by_chunk, paper_by_title

    def _supplement_metadata_edges(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        logger.info("Supplementing metadata-driven paper/author/journal/organization edges...")
        paper_by_chunk, paper_by_title = self._build_paper_node_indexes()
        edge_counts: Dict[str, int] = defaultdict(int)

        for doc in documents:
            doc_id = self._get_document_uid(doc)
            meta = self._get_doc_meta(doc)
            title = str(meta.get("title", "")).strip()
            if not title:
                continue

            title_key = self._normalize_entity_name(title)
            paper_node_id = paper_by_chunk.get(doc_id) or paper_by_title.get(title_key)
            if not paper_node_id:
                paper_node_id = self._find_or_create_entity_direct(title, doc_id, "论文")
                paper_by_chunk[str(doc_id)] = paper_node_id
                paper_by_title[title_key] = paper_node_id

            authors = [
                author
                for author in self._split_metadata_values(meta.get("authors", ""))
                if self._is_valid_metadata_author_name(author)
            ][:3]

            institutions: List[str] = []
            for org in self._split_metadata_values(meta.get("organ", "")):
                primary_name = self._normalize_primary_organization_name(org)
                if primary_name and primary_name not in institutions:
                    institutions.append(primary_name)

            journal_name = str(meta.get("source", "")).strip()
            if journal_name:
                journal_node_id = self._find_or_create_entity_direct(journal_name, doc_id, "期刊")
                if not self._edge_exists(paper_node_id, journal_node_id, "发表于"):
                    self._add_edge_with_metadata(
                        paper_node_id,
                        journal_node_id,
                        "发表于",
                        relation_origin="metadata",
                        confidence=0.95,
                        evidence_chunk_ids=[str(doc_id)],
                        source_paper_ids=[str(doc_id)],
                    )
                    edge_counts["paper_journal"] += 1

            for author_idx, author_name in enumerate(authors):
                author_node_id = self._find_or_create_entity_direct(author_name, doc_id, "作者")
                if not self._edge_exists(author_node_id, paper_node_id, "撰写"):
                    self._add_edge_with_metadata(
                        author_node_id,
                        paper_node_id,
                        "撰写",
                        relation_origin="metadata",
                        confidence=0.95,
                        evidence_chunk_ids=[str(doc_id)],
                        source_paper_ids=[str(doc_id)],
                    )
                    edge_counts["author_paper"] += 1

                if not institutions or self._has_outgoing_relation_to_type(author_node_id, "隶属", "机构"):
                    continue

                if len(institutions) == 1:
                    candidate_orgs = institutions
                elif author_idx < len(institutions):
                    candidate_orgs = [institutions[author_idx]]
                else:
                    candidate_orgs = [institutions[0]]

                for institution_name in candidate_orgs:
                    institution_node_id = self._find_or_create_entity_direct(institution_name, doc_id, "机构")
                    if self._edge_exists(author_node_id, institution_node_id, "隶属"):
                        continue
                    self._add_edge_with_metadata(
                        author_node_id,
                        institution_node_id,
                        "隶属",
                        relation_origin="metadata",
                        confidence=0.9,
                        evidence_chunk_ids=[str(doc_id)],
                        source_paper_ids=[str(doc_id)],
                    )
                    edge_counts["author_org"] += 1

            # 补充1：自动抽取 Keywords 作为属性节点
            keywords = self._extract_keywords(meta.get("keywords", ""))
            for kw in keywords:
                if kw:
                    attr_id = f"attr_{self.node_counter}"
                    self.graph.add_node(attr_id, label="attribute",
                                        properties={"name": f"关键词: {kw}", "chunk_id": doc_id},
                                        level=1)
                    if not self._edge_exists(paper_node_id, attr_id, "has_attribute"):
                        self._add_edge_with_metadata(
                            paper_node_id,
                            attr_id,
                            "has_attribute",
                            relation_origin="metadata",
                            confidence=0.95,
                            evidence_chunk_ids=[str(doc_id)],
                            source_paper_ids=[str(doc_id)],
                        )
                        self.node_counter += 1
                        edge_counts["paper_keyword"] += 1
            
            # 补充2：自动抽取 Year
            year = str(meta.get("year", "")).strip()
            if year:
                attr_id = f"attr_{self.node_counter}"
                self.graph.add_node(attr_id, label="attribute",
                                    properties={"name": f"年份: {year}", "chunk_id": doc_id},
                                    level=1)
                if not self._edge_exists(paper_node_id, attr_id, "has_attribute"):
                    self._add_edge_with_metadata(
                        paper_node_id,
                        attr_id,
                        "has_attribute",
                        relation_origin="metadata",
                        confidence=0.95,
                        evidence_chunk_ids=[str(doc_id)],
                        source_paper_ids=[str(doc_id)],
                    )
                    self.node_counter += 1
                    edge_counts["paper_year"] += 1

        logger.info(
            "Metadata edge supplement added %d author-paper, %d paper-journal, %d author-organization, %d paper-keyword, %d paper-year edges",
            edge_counts["author_paper"],
            edge_counts["paper_journal"],
            edge_counts["author_org"],
            edge_counts["paper_keyword"],
            edge_counts["paper_year"],
        )

    def _extract_keywords(self, raw_keywords: Any) -> List[str]:
        if isinstance(raw_keywords, list):
            return [str(x).strip() for x in raw_keywords if str(x).strip()]
        text = str(raw_keywords or "")
        parts = re.split(r"[;,锛岋紱|/\\\\]+", text)
        return [x.strip() for x in parts if x.strip()]

    def _build_document_fact_units(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paper_by_chunk, paper_by_title = self._build_paper_node_indexes()
        units: List[Dict[str, Any]] = []
        for doc in documents:
            doc_id = self._get_document_uid(doc)
            meta = self._get_doc_meta(doc)
            title = str(meta.get("title", "")).strip()
            title_key = self._normalize_entity_name(title)
            paper_node = paper_by_chunk.get(doc_id) or paper_by_title.get(title_key)
            keywords = self._extract_keywords(meta.get("keywords", ""))
            summary = str(meta.get("abstract", "")).strip()[:220]

            facts: List[str] = []
            if paper_node and paper_node in self.graph.nodes:
                for _, tgt, edge_data in self.graph.out_edges(paper_node, data=True):
                    relation = edge_data.get("relation", "")
                    tgt_name = self.graph.nodes[tgt].get("properties", {}).get("name", tgt)
                    if relation:
                        facts.append(f"{title} --{relation}--> {tgt_name}")
                    if len(facts) >= 8:
                        break
                if len(facts) < 8:
                    for src, _, edge_data in self.graph.in_edges(paper_node, data=True):
                        relation = edge_data.get("relation", "")
                        src_name = self.graph.nodes[src].get("properties", {}).get("name", src)
                        if relation:
                            facts.append(f"{src_name} --{relation}--> {title}")
                        if len(facts) >= 8:
                            break

            units.append(
                {
                    "paper_id": doc_id,
                    "title": title,
                    "keywords": keywords,
                    "summary": summary,
                    "facts": facts,
                }
            )
        return units

    def _cluster_fact_units(self, units: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for unit in units:
            keywords = unit.get("keywords", []) or []
            key = str(keywords[0]).strip().lower() if keywords else "misc"
            clusters[key].append(unit)
        return [cluster for _, cluster in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)]

    def _split_batches_by_budget(self, units: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        batch_size = max(1, int(getattr(self.config.construction, "batch_size_docs", 24)))
        token_budget = max(500, int(getattr(self.config.construction, "token_budget_per_batch", 12000)))

        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []

        def estimate_tokens(items: List[Dict[str, Any]]) -> int:
            return self.token_cal(json.dumps(items, ensure_ascii=False))

        for unit in units:
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

            trial = current_batch + [unit]
            if current_batch and estimate_tokens(trial) > token_budget:
                batches.append(current_batch)
                current_batch = [unit]
            else:
                current_batch = trial

        if current_batch:
            batches.append(current_batch)
        return batches

    def _resolve_bridge_entity_node(self, entity_name: str, default_chunk_id: str = "") -> str:
        entity_name = str(entity_name or "").strip()
        existing = self._lookup_entity_node_id(entity_name, None)
        if existing:
            return existing

        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("properties", {}).get("name") == entity_name:
                self._register_entity_index(entity_name, node_id, self._get_node_schema_type(node_id))
                return node_id

        paper_titles = {
            self._normalize_entity_name(meta.get("title", ""))
            for meta in self.doc_meta_by_chunk_id.values()
        }
        entity_type = "论文" if self._normalize_entity_name(entity_name) in paper_titles else None
        return self._find_or_create_entity_direct(entity_name, default_chunk_id or "unknown", entity_type)
    def _resolve_existing_bridge_paper_node(
        self,
        entity_name: str,
        paper_by_title: Dict[str, str],
    ) -> Optional[str]:
        entity_key = self._normalize_entity_name(entity_name)
        if not entity_key:
            return None
        return paper_by_title.get(entity_key)


    def _get_cross_doc_prompt(self, batch_facts: List[Dict[str, Any]]) -> str:
        bridge_relations = getattr(self.config.construction, "bridge_relations", [])
        payload = json.dumps(batch_facts, ensure_ascii=False, indent=2)
        try:
            return self.config.get_prompt_formatted(
                "construction",
                "cross_doc_agent",
                bridge_relations=", ".join(bridge_relations),
                batch_facts=payload,
            )
        except Exception:
            return (
                "Only return JSON with the top-level key cross_doc_triples.\n"
                "Each item must include head, relation, tail, source_paper_ids, "
                "evidence_chunk_ids, confidence, and reason.\n"
                f"Allowed relations: {', '.join(bridge_relations)}\n"
                f"Batch facts:\n{payload}"
            )

    def _load_bridge_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_path = os.path.join(self.bridge_cache_dir, f"{cache_key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.cross_doc_stats["bridge_batches_cached"] += 1
            return payload.get("parsed")
        except Exception:
            return None

    def _save_bridge_cache(self, cache_key: str, raw_response: str, parsed: Dict[str, Any]) -> None:
        cache_path = os.path.join(self.bridge_cache_dir, f"{cache_key}.json")
        payload = {
            "raw": raw_response,
            "parsed": parsed,
            "timestamp": time.time(),
            "prompt_version": getattr(self.config.construction, "prompt_version", "v1"),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _fallback_cross_doc_triples(self, batch_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fallback: List[Dict[str, Any]] = []
        relations = getattr(self.config.construction, "bridge_relations", []) or []
        dense_relation = relations[0] if relations else "\u6269\u5c55"
        sparse_relation = relations[1] if len(relations) > 1 else (relations[0] if relations else "\u5bf9\u6bd4")

        for i in range(len(batch_facts)):
            for j in range(i + 1, len(batch_facts)):
                left = batch_facts[i]
                right = batch_facts[j]
                left_kw = set(k.lower() for k in left.get("keywords", []))
                right_kw = set(k.lower() for k in right.get("keywords", []))
                overlap = left_kw.intersection(right_kw)
                if not overlap:
                    continue
                relation = dense_relation if len(overlap) >= 2 else sparse_relation
                fallback.append(
                    {
                        "head": left.get("title", ""),
                        "relation": relation,
                        "tail": right.get("title", ""),
                        "source_paper_ids": [left.get("paper_id", ""), right.get("paper_id", "")],
                        "evidence_chunk_ids": [left.get("paper_id", ""), right.get("paper_id", "")],
                        "confidence": 0.55,
                        "reason": f"keyword_overlap={','.join(list(overlap)[:3])}",
                    }
                )
                if len(fallback) >= 4:
                    return fallback
        return fallback

    def _request_cross_doc_triples(self, batch_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload_str = json.dumps(batch_facts, ensure_ascii=False, sort_keys=True)
        prompt_version = getattr(self.config.construction, "prompt_version", "v1")
        cache_key = hashlib.sha1(
            f"{self.dataset_name}|cross_doc|{prompt_version}|{payload_str}".encode("utf-8")
        ).hexdigest()

        cached = self._load_bridge_cache(cache_key)
        if isinstance(cached, dict):
            triples = cached.get("cross_doc_triples", [])
            if isinstance(triples, list):
                return triples

        prompt = self._get_cross_doc_prompt(batch_facts)
        raw_response = self.extract_with_llm(prompt)
        parsed = self._validate_and_parse_llm_response(raw_response) or {}
        self._save_bridge_cache(cache_key, str(raw_response), parsed)
        triples = parsed.get("cross_doc_triples", [])
        return triples if isinstance(triples, list) else []

    def _apply_cross_doc_triples(
        self,
        triples: List[Dict[str, Any]],
        fallback: bool = False,
        paper_bridge_counts: Optional[Dict[str, int]] = None,
        seen_paper_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> int:
        if not triples:
            return 0
        allowed_relations: Set[str] = set(getattr(self.config.construction, "bridge_relations", []))
        min_confidence = float(getattr(self.config.construction, "bridge_min_confidence", 0.75))
        max_edges_per_paper = max(1, int(getattr(self.config.construction, "bridge_max_edges_per_paper", 2)))
        paper_bridge_counts = paper_bridge_counts if paper_bridge_counts is not None else defaultdict(int)
        seen_paper_pairs = seen_paper_pairs if seen_paper_pairs is not None else set()
        _, paper_by_title = self._build_paper_node_indexes()
        added = 0

        for item in triples:
            if isinstance(item, list) and len(item) >= 3:
                item = {
                    "head": item[0],
                    "relation": item[1],
                    "tail": item[2],
                    "source_paper_ids": [],
                    "evidence_chunk_ids": [],
                    "confidence": 0.55 if fallback else 0.65,
                    "reason": "list_format",
                }
            if not isinstance(item, dict):
                continue

            head = str(item.get("head", "")).strip()
            tail = str(item.get("tail", "")).strip()
            relation = self._normalize_relation_name(item.get("relation", ""))
            if not head or not tail or not relation or head == tail:
                continue
            if allowed_relations and relation not in allowed_relations:
                continue

            confidence = float(item.get("confidence", 0.55 if fallback else 0.65))
            if confidence < min_confidence:
                continue

            source_paper_ids = sorted({str(x) for x in item.get("source_paper_ids", []) if str(x)})
            if len(source_paper_ids) < 2:
                continue
            evidence_chunk_ids = [str(x) for x in item.get("evidence_chunk_ids", []) if str(x)]

            head_id = self._resolve_existing_bridge_paper_node(head, paper_by_title)
            tail_id = self._resolve_existing_bridge_paper_node(tail, paper_by_title)
            if not head_id or not tail_id or head_id == tail_id:
                continue

            pair_key = tuple(sorted((head_id, tail_id)))
            if pair_key in seen_paper_pairs:
                continue
            if paper_bridge_counts[head_id] >= max_edges_per_paper or paper_bridge_counts[tail_id] >= max_edges_per_paper:
                continue

            self._add_edge_with_metadata(
                head_id,
                tail_id,
                relation,
                relation_origin="cross_doc",
                confidence=confidence,
                evidence_chunk_ids=evidence_chunk_ids,
                source_paper_ids=source_paper_ids,
                reason=str(item.get("reason", "")),
            )
            seen_paper_pairs.add(pair_key)
            paper_bridge_counts[head_id] += 1
            paper_bridge_counts[tail_id] += 1
            added += 1
        return added

    def _build_cross_document_bridges(self, documents: List[Dict[str, Any]]) -> None:
        if not getattr(self.config.construction, "cross_doc_enabled", True):
            return

        fact_units = self._build_document_fact_units(documents)
        if len(fact_units) < 2:
            return

        total_added = 0
        paper_bridge_counts: Dict[str, int] = defaultdict(int)
        seen_paper_pairs: Set[Tuple[str, str]] = set()
        bridge_enable_fallback = bool(getattr(self.config.construction, "bridge_enable_fallback", False))
        for cluster_units in self._cluster_fact_units(fact_units):
            for batch_facts in self._split_batches_by_budget(cluster_units):
                if len(batch_facts) < 2:
                    continue
                self.cross_doc_stats["bridge_batches_total"] += 1
                try:
                    triples = self._request_cross_doc_triples(batch_facts)
                    added = self._apply_cross_doc_triples(
                        triples,
                        fallback=False,
                        paper_bridge_counts=paper_bridge_counts,
                        seen_paper_pairs=seen_paper_pairs,
                    )
                    if added == 0 and bridge_enable_fallback:
                        fallback_triples = self._fallback_cross_doc_triples(batch_facts)
                        fallback_added = self._apply_cross_doc_triples(
                            fallback_triples,
                            fallback=True,
                            paper_bridge_counts=paper_bridge_counts,
                            seen_paper_pairs=seen_paper_pairs,
                        )
                        total_added += fallback_added
                        self.cross_doc_stats["bridge_edges_fallback"] += fallback_added
                    else:
                        total_added += added
                except Exception as e:
                    self.cross_doc_stats["bridge_batches_failed"] += 1
                    logger.warning(f"Cross-doc bridge batch failed: {type(e).__name__}: {e}")

        self.cross_doc_stats["bridge_edges_added"] += total_added
        logger.info(
            "Cross-doc bridge summary: "
            f"batches={self.cross_doc_stats['bridge_batches_total']} "
            f"cached={self.cross_doc_stats['bridge_batches_cached']} "
            f"failed={self.cross_doc_stats['bridge_batches_failed']} "
            f"edges_added={self.cross_doc_stats['bridge_edges_added']} "
            f"fallback_edges={self.cross_doc_stats['bridge_edges_fallback']}"
        )

    def process_level4(self):
        """Process communities using Tree-Comm algorithm"""
        level2_nodes = [n for n, d in self.graph.nodes(data=True) if d['level'] == 2]
        start_comm = time.time()
        _tree_comm = tree_comm.FastTreeComm(
            self.graph,
            embedding_model=self.config.tree_comm.embedding_model,
            struct_weight=self.config.tree_comm.struct_weight,
            config=self.config,
        )
        comm_to_nodes = _tree_comm.detect_communities(level2_nodes)

        # create super nodes (level 4 communities)
        _tree_comm.create_super_nodes(comm_to_nodes, level=4)
        end_comm = time.time()
        logger.info(f"Community Indexing Time: {end_comm - start_comm}s")

    def _connect_keywords_to_communities(self):
        """Deprecated: keyword community nodes are no longer generated."""
        return

    def _extract_chunk_with_parse_retry(self, chunk: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
        prompt = self._get_construction_prompt(chunk)
        last_raw_response = ""

        for attempt in range(1, self.retry_attempts + 1):
            raw_response = self.extract_with_llm(prompt)
            last_raw_response = str(raw_response)
            parsed_response = self._validate_and_parse_llm_response(raw_response)
            if parsed_response:
                return {
                    "status": "success",
                    "attempt_count": attempt,
                    "parsed_response": parsed_response,
                    "error": None,
                }

            if attempt < self.retry_attempts:
                self.cross_doc_stats["parse_retries"] += 1
                backoff = self._backoff_seconds(attempt)
                logger.warning(
                    f"Parse failed for chunk {chunk_id} on attempt {attempt}/{self.retry_attempts}. "
                    f"Retrying in {backoff:.1f}s"
                )
                time.sleep(backoff)

        return {
            "status": "failed",
            "attempt_count": self.retry_attempts,
            "parsed_response": None,
            "error": f"Failed to parse LLM response after {self.retry_attempts} attempts",
            "raw_response": last_raw_response[:1000],
        }

    def process_document_extraction(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return {"doc_id": "unknown", "status": "failed", "error": "Empty document"}

        chunks, _ = self.chunk_text(doc)
        doc_id = self._get_document_uid(doc)
        cached_payload = self._load_doc_extraction_cache(doc_id) if self.resume_enabled else None

        if self.resume_enabled and self.replay_cached_extractions and self._all_cached_chunks_successful(cached_payload):
            self.cross_doc_stats["documents_loaded_from_cache"] += 1
            return {"doc_id": doc_id, "status": "cached"}

        payload = cached_payload if isinstance(cached_payload, dict) else self._build_doc_cache_payload(doc, chunks)
        chunk_map = {str(item.get("chunk_id", "")): item for item in payload.get("chunks", [])}

        payload["status"] = "processing"
        payload["timestamp"] = time.time()
        self._save_doc_extraction_cache(doc_id, payload)

        doc_failed = False
        for chunk in chunks:
            chunk_id = str(chunk.get("id", ""))
            chunk_entry = chunk_map.get(chunk_id)
            if chunk_entry is None:
                chunk_entry = {
                    "chunk_id": chunk_id,
                    "chunk_text": chunk.get("text", ""),
                    "status": "pending",
                    "attempt_count": 0,
                    "parsed_response": None,
                    "error": None,
                }
                payload.setdefault("chunks", []).append(chunk_entry)
                chunk_map[chunk_id] = chunk_entry

            if (
                self.resume_enabled
                and self.replay_cached_extractions
                and chunk_entry.get("status") == "success"
                and chunk_entry.get("parsed_response")
            ):
                continue

            try:
                result = self._extract_chunk_with_parse_retry(chunk, chunk_id)
                chunk_entry.update(result)
                if result["status"] != "success":
                    doc_failed = True
            except Exception as e:
                chunk_entry.update(
                    {
                        "status": "failed",
                        "attempt_count": self.retry_attempts,
                        "parsed_response": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                doc_failed = True

            payload["timestamp"] = time.time()
            payload["status"] = "failed" if doc_failed else "processing"
            self._save_doc_extraction_cache(doc_id, payload)

        payload["status"] = "failed" if doc_failed else "success"
        payload["timestamp"] = time.time()
        self._save_doc_extraction_cache(doc_id, payload)

        if doc_failed:
            self.cross_doc_stats["documents_failed"] += 1
            return {"doc_id": doc_id, "status": "failed"}

        self.cross_doc_stats["documents_extracted"] += 1
        return {"doc_id": doc_id, "status": "success"}

    def process_document(self, doc: Dict[str, Any]) -> None:
        if not doc:
            return
        chunks, _ = self.chunk_text(doc)
        doc_id = self._get_document_uid(doc)
        payload = self._load_doc_extraction_cache(doc_id)
        if not isinstance(payload, dict):
            logger.warning(f"No extraction cache found for document {doc_id}, skipping graph replay")
            return

        applied_any = False
        for chunk_entry in payload.get("chunks", []):
            if chunk_entry.get("status") != "success" or not chunk_entry.get("parsed_response"):
                continue
            self._apply_parsed_response(chunk_entry.get("parsed_response", {}), str(chunk_entry.get("chunk_id", doc_id)))
            applied_any = True

        if applied_any:
            self.cross_doc_stats["documents_applied"] += 1
        else:
            self.cross_doc_stats["documents_apply_failed"] += 1

    def process_all_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Extract documents to cache first, then replay cached extractions into the graph."""

        max_workers = min(self.config.construction.max_workers, (os.cpu_count() or 1) + 4)
        start_construct = time.time()
        total_docs = len(documents)
        self.cross_doc_stats["documents_total"] = total_docs
        doc_id_counts = defaultdict(int)
        for doc in documents:
            doc_id_counts[str(doc.get("id", ""))] += 1
        self.duplicate_doc_ids = {doc_id for doc_id, count in doc_id_counts.items() if doc_id and count > 1}

        logger.info(
            f"Starting extraction for {total_docs} documents with {max_workers} workers "
            f"(max concurrent LLM requests: {self.max_concurrent_llm_requests})..."
        )
        if self.duplicate_doc_ids:
            logger.warning(
                "Detected %d duplicated source ids in dataset '%s'; duplicate documents will use hashed doc_uids.",
                len(self.duplicate_doc_ids),
                self.dataset_name,
            )

        completed_count = 0
        failed_count = 0

        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            all_futures = [executor.submit(self.process_document_extraction, doc) for doc in documents]

            for future in futures.as_completed(all_futures):
                try:
                    result = future.result()
                    completed_count += 1
                    if result.get("status") == "failed":
                        failed_count += 1

                    if completed_count % 10 == 0 or completed_count == total_docs:
                        elapsed_time = time.time() - start_construct
                        avg_time_per_doc = elapsed_time / completed_count if completed_count > 0 else 0
                        remaining_docs = total_docs - completed_count
                        estimated_remaining_time = remaining_docs * avg_time_per_doc
                        logger.info(
                            f"Extraction progress: {completed_count}/{total_docs} "
                            f"({completed_count / total_docs * 100:.1f}%) "
                            f"[{failed_count} failed, {self.cross_doc_stats['documents_loaded_from_cache']} cache hits] "
                            f"ETA: {estimated_remaining_time / 60:.1f} minutes"
                        )
                except Exception as e:
                    completed_count += 1
                    failed_count += 1
                    self.cross_doc_stats["documents_failed"] += 1
                    logger.error(f"Document extraction future failed: {type(e).__name__}: {e}", exc_info=True)

        logger.info("Replaying cached extractions into graph...")
        for doc in documents:
            try:
                self.process_document(doc)
            except Exception as e:
                self.cross_doc_stats["documents_apply_failed"] += 1
                logger.error(f"Error replaying document into graph: {type(e).__name__}: {e}", exc_info=True)

        end_construct = time.time()
        logger.info(f"Construction Time: {end_construct - start_construct}s")
        logger.info(
            f"Extraction summary: success={self.cross_doc_stats['documents_extracted']} "
            f"cache_hits={self.cross_doc_stats['documents_loaded_from_cache']} "
            f"failed={self.cross_doc_stats['documents_failed']}"
        )
        logger.info(
            f"Replay summary: applied={self.cross_doc_stats['documents_applied']} "
            f"apply_failed={self.cross_doc_stats['documents_apply_failed']}"
        )

        self._supplement_metadata_edges(documents)

        if getattr(self.config.construction, "cross_doc_enabled", True):
            logger.info("Starting cross-document bridge stage...")
            bridge_start = time.time()
            self._build_cross_document_bridges(documents)
            logger.info(f"Cross-document bridge stage finished in {time.time() - bridge_start:.2f}s")

        logger.info(f"{'Processing Level 3 and 4':^30}")
        logger.info("-" * 20)
        self.triple_deduplicate()
        self.process_level4()

    def triple_deduplicate(self):
        """deduplicate triples in lv1 and lv2"""
        new_graph = nx.MultiDiGraph()

        for node, node_data in self.graph.nodes(data=True):
            new_graph.add_node(node, **node_data)

        seen_triples = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            relation = data.get('relation')
            if (u, v, relation) not in seen_triples:
                seen_triples.add((u, v, relation))
                new_graph.add_edge(u, v, **data)
        self.graph = new_graph

    def _save_entity_alias_audit_file(self) -> None:
        output_path = f"output/graphs/{self.dataset_name}_entity_alias_audit.json"
        rows = []
        for (schema_type, canonical_name), aliases in sorted(
            self.entity_alias_audit.items(),
            key=lambda item: (item[0][0], item[0][1]),
        ):
            if len(aliases) <= 1:
                continue
            rows.append(
                {
                    "schema_type": schema_type,
                    "canonical_name": canonical_name,
                    "aliases": sorted(aliases),
                }
            )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        logger.info("Entity alias audit saved to %s (%d groups)", output_path, len(rows))

    def format_output(self) -> List[Dict[str, Any]]:
        """convert graph to specified output format"""
        output = []

        for u, v, data in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]

            relationship = {
                "start_node": {
                    "label": u_data["label"],
                    "properties": u_data["properties"],
                },
                "relation": data["relation"],
                "end_node": {
                    "label": v_data["label"],
                    "properties": v_data["properties"],
                },
            }
            edge_properties = {k: v for k, v in data.items() if k != "relation"}
            if edge_properties:
                relationship["edge_properties"] = edge_properties
            output.append(relationship)

        return output

    def save_graphml(self, output_path: str):
        graph_processor.save_graph(self.graph, output_path)

    def build_knowledge_graph(self, corpus):
        logger.info(f"========{'Start Building':^20}========")
        logger.info("-" * 30)

        with open(corpus, 'r', encoding='utf-8') as f:
            documents = json_repair.load(f)

        self.process_all_documents(documents)

        logger.info(f"All Process finished, token cost: {self.token_len}")

        self.save_chunks_to_file()

        output = self.format_output()

        json_output_path = f"output/graphs/{self.dataset_name}_new.json"
        os.makedirs("output/graphs", exist_ok=True)
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved to {json_output_path}")
        self._save_entity_alias_audit_file()

        stats_path = f"output/graphs/{self.dataset_name}_construction_stats.json"
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.cross_doc_stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Construction stats saved to {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save construction stats: {type(e).__name__}: {e}")

        return output


