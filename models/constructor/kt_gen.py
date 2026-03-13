import json
import os
import threading
import time
import hashlib
import re
from concurrent import futures
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import nanoid
import networkx as nx
import tiktoken
import json_repair

from config import get_config
from utils import call_llm_api, graph_processor, tree_comm
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
        self.llm_client = call_llm_api.LLMCompletionCall(scope="kg")
        self.all_chunks = {}
        self.mode = mode or config.construction.mode
        self.doc_meta_by_chunk_id: Dict[str, Dict[str, Any]] = {}
        self._entity_name_index: Dict[Tuple[str, str], str] = {}
        self.bridge_cache_dir = os.path.join("output", "construction_cache", self.dataset_name)
        os.makedirs(self.bridge_cache_dir, exist_ok=True)
        self.cross_doc_stats = {
            "cross_doc_enabled": bool(getattr(self.config.construction, "cross_doc_enabled", True)),
            "bridge_batches_total": 0,
            "bridge_batches_cached": 0,
            "bridge_batches_failed": 0,
            "bridge_edges_added": 0,
            "bridge_edges_fallback": 0,
        }

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
        用于图谱构建（包含完整元数据）。
        """
        if not isinstance(chunk, dict):
            return str(chunk)

        # 兼容数据结构：
        # chunk 可能是包含 "meta" 的字典，也可能本身就是 meta 字典
        data_source = chunk.get("meta", chunk)

        title = data_source.get("title", "")
        authors = data_source.get("authors", "")
        organ = data_source.get("organ", "")
        keywords = data_source.get("keywords", "")
        abstract = data_source.get("abstract", "")
        source = data_source.get("source", "")
        year = data_source.get("year", "")

        prompt_text = (
            f"文献标题: {title}\n"
            f"作者: {authors}\n"
            f"机构: {organ}\n"
            f"来源: {source}\n"
            f"关键词: {keywords}\n"
            f"摘要: {abstract}\n"
            f"年份: {year}\n"
        )
        return prompt_text

    def chunk_text(self, doc: Dict) -> Tuple[List[Dict], Dict[str, Dict]]:
        """
        处理论文 JSON 结构。
        Chunk 保存逻辑：仅保留 Title + Abstract（用于检索）。
        """
        doc_id = str(doc.get("id", nanoid.generate(size=8)))

        meta = doc.get("meta", {})
        title = meta.get("title", "")
        abstract = meta.get("abstract", "")
        chunk_content_text = f"Title: {title}\nAbstract: {abstract}"

        chunk = {
            "id": doc_id,
            "text": chunk_content_text,
            "meta": meta,
        }

        with self.lock:
            self.all_chunks[doc_id] = chunk_content_text
            self.doc_meta_by_chunk_id[doc_id] = meta

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

        # 简单的写入逻辑，覆盖模式以避免重复
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk_id, chunk_text in self.all_chunks.items():
                # 移除换行符以免破坏格式
                clean_text = str(chunk_text).replace('\n', ' \\n ')
                f.write(f"id: {chunk_id}\tChunk: {clean_text}\n")

        logger.info(f"Chunk data saved to {chunk_file} ({len(self.all_chunks)} chunks)")

    def extract_with_llm(self, prompt: str):
        response = self.llm_client.call_api(prompt)
        # 增加 Token 计算
        self.token_len += self.token_cal(prompt + str(response))
        # --- 打印 LLM 返回结果到控制台 ---
        print("\n" + "=" * 50)
        print(f"DEBUG: LLM Response for Prompt (first 100 chars): {prompt[:100]}...")
        print(f"DEBUG: Raw Response: {response}")
        print("=" * 50 + "\n")
        return response


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

        # 强制使用 'academic' 或 'general'，你可以根据需要修改
        # 如果配置文件中有 'academic'，这里最好写死或者动态判断
        prompt_type = "general"
        if self.mode == "agent":
            prompt_type = f"{prompt_type}_agent"
        return self.config.get_prompt_formatted("construction", prompt_type, schema=recommend_schema, chunk=chunk_str)

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
        """解析并校验 LLM 返回的 JSON 字符串"""
        if not llm_response:
            return None
        try:
            # 移除之前的 prompt 参数，这里只负责解析
            return json_repair.loads(llm_response)
        except Exception as e:
            logger.error(f"JSON Repair failed: {e}")
            return None

    def _normalize_entity_name(self, name: Any) -> str:
        """规范化实体名，便于同名去重与索引。"""
        text = str(name or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text.lower()

    def _normalize_relation_name(self, relation: Any) -> str:
        """
        关系词最小归一化：保留原中文关系，仅做必要同义合并。
        不做中文关系英文化，避免语义漂移。
        """
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
        """识别论文节点类型，兼容中文与英文。"""
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

    def _register_entity_index(self, entity_name: Any, node_id: str, entity_type: Any = None):
        name_key = self._normalize_entity_name(entity_name)
        if not name_key:
            return
        type_key = str(entity_type or "").strip().lower()
        self._entity_name_index[(name_key, type_key)] = node_id
        # 同名无类型索引，便于兜底检索
        self._entity_name_index[(name_key, "")] = node_id

    def _lookup_entity_node_id(self, entity_name: Any, entity_type: Any = None) -> Optional[str]:
        name_key = self._normalize_entity_name(entity_name)
        if not name_key:
            return None
        type_key = str(entity_type or "").strip().lower()
        if type_key:
            exact = self._entity_name_index.get((name_key, type_key))
            if exact:
                return exact
        return self._entity_name_index.get((name_key, ""))

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
        """统一写边并附加可选元数据，保证旧读取路径兼容。"""
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
        with self.lock:
            entity_node_id = self._lookup_entity_node_id(entity_name, entity_type)
            if not entity_node_id:
                entity_node_id = next(
                    (
                        n
                        for n, d in self.graph.nodes(data=True)
                        if d.get("label") == "entity" and d["properties"]["name"] == entity_name
                    ),
                    None,
                )

            if not entity_node_id:
                entity_node_id = f"entity_{self.node_counter}"
                properties = {"name": entity_name, "chunk id": chunk_id}
                if entity_type:
                    properties["schema_type"] = entity_type
                if self._is_paper_schema_type(entity_type):
                    aliases = [str(entity_name)]
                    title = self.doc_meta_by_chunk_id.get(str(chunk_id), {}).get("title")
                    if title and str(title) not in aliases:
                        aliases.append(str(title))
                    properties["aliases"] = aliases

                nodes_to_add.append((
                    entity_node_id,
                    {
                        "label": "entity",
                        "properties": properties,
                        "level": 2
                    }
                ))
                self._register_entity_index(entity_name, entity_node_id, entity_type)
                self.node_counter += 1
            else:
                self._register_entity_index(entity_name, entity_node_id, entity_type)

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

    def _process_attributes(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None) -> tuple[list, list]:
        """Process extracted attributes and return nodes and edges to add."""
        nodes_to_add = []
        edges_to_add = []

        for entity, attributes in extracted_attr.items():
            for attr in attributes:
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

                entity_type = entity_types.get(entity) if entity_types else None
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

            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None

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

    def _find_or_create_entity_direct(self, entity_name: str, chunk_id: int, entity_type: str = None) -> str:
        """Find existing entity or create a new one directly in graph (for agent mode)."""
        entity_node_id = self._lookup_entity_node_id(entity_name, entity_type)
        if not entity_node_id:
            entity_node_id = next(
                (
                    n
                    for n, d in self.graph.nodes(data=True)
                    if d["properties"].get("name") == entity_name
                ),
                None,
            )

        if not entity_node_id:
            entity_node_id = f"entity_{self.node_counter}"
            properties = {
                "name": entity_name,
                "chunk id": chunk_id
            }
            # # 核心修改：如果 entity_type 存在，就用它做 label，否则才用 "entity"
            # display_label = entity_type if entity_type else "entity"
            # properties = {"name": entity_name, "chunk id": chunk_id,"schema_type": entity_type}
            if entity_type:
                properties["schema_type"] = entity_type
            if self._is_paper_schema_type(entity_type):
                aliases = [str(entity_name)]
                title = self.doc_meta_by_chunk_id.get(str(chunk_id), {}).get("title")
                if title and str(title) not in aliases:
                    aliases.append(str(title))
                properties["aliases"] = aliases
            self.graph.add_node(
                entity_node_id,
                label="entity",
                properties=properties,
                level=2
            )
            self._register_entity_index(entity_name, entity_node_id, entity_type)
            self.node_counter += 1
        else:
            # 如果节点已存在但标签是通用的，尝试更新它
            node_props = self.graph.nodes[entity_node_id]["properties"]
            if entity_type and "schema_type" not in node_props:
                node_props["schema_type"] = entity_type
            aliases = node_props.setdefault("aliases", [])
            if entity_name not in aliases:
                aliases.append(entity_name)
            self._register_entity_index(entity_name, entity_node_id, node_props.get("schema_type", entity_type))

        return entity_node_id

    def _process_attributes_agent(self, extracted_attr: dict, chunk_id: int, entity_types: dict = None):
        """Process extracted attributes in agent mode (direct graph operations)."""
        for entity, attributes in extracted_attr.items():
            for attr in attributes:
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

                entity_type = entity_types.get(entity) if entity_types else None
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

            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None

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
        # print(f"DEBUG: Processing {chunk_id}...") # 调试用
        llm_response = self.extract_with_llm(prompt)

        # 3. 解析结果
        parsed_response = self._validate_and_parse_llm_response(llm_response)

        if not parsed_response:
            logger.warning(f"Failed to parse LLM response for chunk {chunk_id}")
            return

        # 4. 参照示例：处理 Schema 演变
        new_schema_types = parsed_response.get("new_schema_types", {})
        if new_schema_types:
            self._update_schema_with_new_types(new_schema_types)

        entity_types = parsed_response.get("entity_types", {})
        attributes = parsed_response.get("attributes", {})
        triples = parsed_response.get("triples", [])

        # 6. 使用专门的 agent 方法进行图构建
        with self.lock:
            # 5. 处理属性 (Level 1)
            for entity, attrs in attributes.items():
                # 在创建实体时传入从 entity_types 中获取的具体类型
                entity_type = entity_types.get(entity)
                entity_id = self._find_or_create_entity_direct(entity, chunk_id, entity_type)

                if isinstance(attrs, list):
                    for attr in attrs:
                        # 强制转为字符串，避免 unhashable dict 错误，确保可视化不报错
                        safe_attr = str(attr) if isinstance(attr, dict) else attr
                        attr_id = f"attr_{self.node_counter}"
                        self.graph.add_node(attr_id,
                                            label="attribute",
                                            properties={"name": safe_attr, "chunk_id": chunk_id},
                                            level=1)
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

            # 6. 处理三元组 (Level 2)
            for triple in triples:
                if len(triple) < 3:
                    continue

                src, rel, tgt = triple[0], triple[1], triple[2]

                # 从映射表中获取具体的类型
                src_type = entity_types.get(src)
                tgt_type = entity_types.get(tgt)

                # 传入具体的类型以确保 properties.schema_type 被正确填充
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
        #
        # with self.lock:
        #     # 防御性处理：确保属性值不是 dict (防止可视化报错)
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
        #     # 处理三元组 (Level 2)
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

    def _extract_keywords(self, raw_keywords: Any) -> List[str]:
        if isinstance(raw_keywords, list):
            return [str(x).strip() for x in raw_keywords if str(x).strip()]
        text = str(raw_keywords or "")
        parts = re.split(r"[;,，；|/\\\\]+", text)
        return [x.strip() for x in parts if x.strip()]

    def _build_document_fact_units(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paper_by_chunk, paper_by_title = self._build_paper_node_indexes()
        units: List[Dict[str, Any]] = []
        for doc in documents:
            doc_id = str(doc.get("id", ""))
            meta = doc.get("meta", doc if isinstance(doc, dict) else {})
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
                "只返回 JSON，主键为 cross_doc_triples。"
                "每项字段：head, relation, tail, source_paper_ids, evidence_chunk_ids, confidence, reason。\n"
                f"允许关系：{', '.join(bridge_relations)}\n"
                f"批次事实：\n{payload}"
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
        dense_relation = relations[3] if len(relations) > 3 else (relations[0] if relations else "同任务不同方法")
        sparse_relation = relations[1] if len(relations) > 1 else (relations[0] if relations else "对比")

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

    def _apply_cross_doc_triples(self, triples: List[Dict[str, Any]], fallback: bool = False) -> int:
        if not triples:
            return 0
        allowed_relations: Set[str] = set(getattr(self.config.construction, "bridge_relations", []))
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

            source_paper_ids = [str(x) for x in item.get("source_paper_ids", []) if str(x)]
            evidence_chunk_ids = [str(x) for x in item.get("evidence_chunk_ids", []) if str(x)]
            default_chunk_id = source_paper_ids[0] if source_paper_ids else (
                evidence_chunk_ids[0] if evidence_chunk_ids else ""
            )

            head_id = self._resolve_bridge_entity_node(head, default_chunk_id=default_chunk_id)
            tail_id = self._resolve_bridge_entity_node(tail, default_chunk_id=default_chunk_id)
            if not head_id or not tail_id or head_id == tail_id:
                continue

            self._add_edge_with_metadata(
                head_id,
                tail_id,
                relation,
                relation_origin="cross_doc",
                confidence=float(item.get("confidence", 0.55 if fallback else 0.65)),
                evidence_chunk_ids=evidence_chunk_ids,
                source_paper_ids=source_paper_ids,
                reason=str(item.get("reason", "")),
            )
            added += 1
        return added

    def _build_cross_document_bridges(self, documents: List[Dict[str, Any]]) -> None:
        if not getattr(self.config.construction, "cross_doc_enabled", True):
            return

        fact_units = self._build_document_fact_units(documents)
        if len(fact_units) < 2:
            return

        total_added = 0
        for cluster_units in self._cluster_fact_units(fact_units):
            for batch_facts in self._split_batches_by_budget(cluster_units):
                if len(batch_facts) < 2:
                    continue
                self.cross_doc_stats["bridge_batches_total"] += 1
                try:
                    triples = self._request_cross_doc_triples(batch_facts)
                    added = self._apply_cross_doc_triples(triples, fallback=False)
                    if added == 0:
                        fallback_triples = self._fallback_cross_doc_triples(batch_facts)
                        fallback_added = self._apply_cross_doc_triples(fallback_triples, fallback=True)
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
        )
        comm_to_nodes = _tree_comm.detect_communities(level2_nodes)

        # create super nodes (level 4 communities)
        _tree_comm.create_super_nodes_with_keywords(comm_to_nodes, level=4)
        # _tree_comm.add_keywords_to_level3(comm_to_nodes)
        # connect keywords to communities (optional)
        self._connect_keywords_to_communities()
        end_comm = time.time()
        logger.info(f"Community Indexing Time: {end_comm - start_comm}s")

    def _connect_keywords_to_communities(self):
        """Connect relevant keywords to communities"""
        # comm_names = [self.graph.nodes[n]['properties']['name'] for n, d in self.graph.nodes(data=True) if d['level'] == 4]
        comm_nodes = [n for n, d in self.graph.nodes(data=True) if d['level'] == 4]
        kw_nodes = [n for n, d in self.graph.nodes(data=True) if d['label'] == 'keyword']
        with self.lock:
            for comm in comm_nodes:
                comm_name = self.graph.nodes[comm]['properties']['name'].lower()
                for kw in kw_nodes:
                    kw_name = self.graph.nodes[kw]['properties']['name'].lower()
                    if kw_name in comm_name or comm_name in kw_name:
                        self._add_edge_with_metadata(
                            kw,
                            comm,
                            "describes",
                            relation_origin="derived",
                            confidence=0.6,
                        )

    # 修复 process_document 中的 ID 匹配崩溃
    def process_document(self, doc: Dict[str, Any]) -> None:
        try:
            if not doc: return

            # 使用修改后的 chunk_text
            chunks, _ = self.chunk_text(doc)

            for chunk in chunks:
                chunk_id = chunk.get("id")
                # 无论是否 agent 模式，核心逻辑是一样的，这里简化调用统一逻辑
                self.process_level1_level2_agent(chunk, chunk_id)

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)

    def process_all_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Process all documents with high concurrency and pass results to process_level4."""

        max_workers = min(self.config.construction.max_workers, (os.cpu_count() or 1) + 4)
        start_construct = time.time()
        total_docs = len(documents)

        logger.info(f"Starting processing {total_docs} documents with {max_workers} workers...")

        all_futures = []
        processed_count = 0
        failed_count = 0

        try:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all documents for processing and store futures
                all_futures = [executor.submit(self.process_document, doc) for doc in documents]

                for i, future in enumerate(futures.as_completed(all_futures)):
                    try:
                        future.result()
                        processed_count += 1

                        if processed_count % 10 == 0 or processed_count == total_docs:
                            elapsed_time = time.time() - start_construct
                            avg_time_per_doc = elapsed_time / processed_count if processed_count > 0 else 0
                            remaining_docs = total_docs - processed_count
                            estimated_remaining_time = remaining_docs * avg_time_per_doc

                            logger.info(f"Progress: {processed_count}/{total_docs} documents processed "
                                        f"({processed_count / total_docs * 100:.1f}%) "
                                        f"[{failed_count} failed] "
                                        f"ETA: {estimated_remaining_time / 60:.1f} minutes")

                    except Exception as e:
                        failed_count += 1

        except Exception as e:
            return

        end_construct = time.time()
        logger.info(f"Construction Time: {end_construct - start_construct}s")
        logger.info(f"Successfully processed: {processed_count}/{total_docs} documents")
        logger.info(f"Failed: {failed_count} documents")

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

        stats_path = f"output/graphs/{self.dataset_name}_construction_stats.json"
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.cross_doc_stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Construction stats saved to {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to save construction stats: {type(e).__name__}: {e}")

        return output
