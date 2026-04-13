from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from config import DatasetConfig, reload_config
from models.constructor.kt_gen import KTBuilder
from utils.entity_normalizer import EntityNormalizer, normalize_text

from .data_models import KGExtraction


def sanitize_extraction_payload(payload: Any) -> KGExtraction:
    extraction = KGExtraction()
    if not isinstance(payload, dict):
        return extraction

    entity_types = payload.get("entity_types", {})
    if isinstance(entity_types, dict):
        for name, entity_type in entity_types.items():
            clean_name = str(name or "").strip()
            clean_type = str(entity_type or "").strip()
            if clean_name:
                extraction.entity_types[clean_name] = clean_type

    attributes = payload.get("attributes", {})
    if isinstance(attributes, dict):
        for entity_name, values in attributes.items():
            clean_entity = str(entity_name or "").strip()
            if not clean_entity:
                continue
            if isinstance(values, list):
                clean_values = [str(value).strip() for value in values if str(value).strip()]
            elif values is None:
                clean_values = []
            else:
                clean_values = [str(values).strip()] if str(values).strip() else []
            extraction.attributes[clean_entity] = clean_values

    triples = payload.get("triples", [])
    if isinstance(triples, list):
        for triple in triples:
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            head = str(triple[0] or "").strip()
            relation = str(triple[1] or "").strip()
            tail = str(triple[2] or "").strip()
            if head and relation and tail:
                extraction.triples.append([head, relation, tail])

    return extraction


class ConstructionBridge:
    """Reuse production normalization and prompts without requiring live KG construction."""

    def __init__(self, dataset_name: str, main_config_path: str):
        self.dataset_name = dataset_name
        self.config = reload_config(main_config_path)
        self.dataset_config = ensure_dataset_registered(self.config, dataset_name)
        self.schema_path = self.dataset_config.schema_path
        self.schema = json.loads(Path(self.schema_path).read_text(encoding="utf-8"))
        self.builder = self._build_prompt_bridge()
        self.entity_normalizer = self.builder.entity_normalizer

    def _build_prompt_bridge(self) -> KTBuilder:
        builder = KTBuilder.__new__(KTBuilder)
        builder.config = self.config
        builder.dataset_name = self.dataset_name
        builder.schema = self.schema
        builder.mode = self.config.construction.mode
        builder.schema_type_aliases = {
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
        builder.generic_entity_name_blacklist = {
            "教学模式",
            "研究理论",
            "教育理念",
            "人才培养模式",
        }
        builder.allowed_schema_node_types = {
            str(node_type).strip()
            for node_type in self.schema.get("Nodes", [])
            if str(node_type).strip()
        }
        builder.entity_normalizer = EntityNormalizer(
            schema_type_aliases=builder.schema_type_aliases,
            config_path=getattr(self.config.construction, "entity_aliases_path", "config/entity_aliases.yaml"),
        )
        return builder

    def build_prompt(self, sample: Dict[str, Any]) -> str:
        chunk = {"meta": sample.get("meta", {})}
        return self.builder._get_construction_prompt(chunk)

    def parse_response(self, raw_response: str) -> KGExtraction:
        parsed = self.builder._validate_and_parse_llm_response(raw_response)
        return sanitize_extraction_payload(parsed)

    def normalize_text(self, value: Any) -> str:
        return normalize_text(value).casefold()

    def normalize_entity_name(self, value: Any, entity_type: Any = "") -> str:
        if entity_type:
            _, normalized_key = self.entity_normalizer.resolve(value, entity_type)
            return normalized_key
        return self.entity_normalizer.normalize_name_key(value)

    def normalize_relation_name(self, value: Any) -> str:
        relation = normalize_text(value, strip_outer_brackets=False)
        relation = self.builder._normalize_relation_name(relation)
        return " ".join(str(relation).split()).strip()

    def normalize_entity_type(self, value: Any) -> str:
        normalized = self.entity_normalizer.normalize_entity_type(value)
        if normalized in self.builder.allowed_schema_node_types:
            return normalized
        return normalized

    def is_paper_type(self, value: Any) -> bool:
        return self.normalize_entity_type(value).casefold() in {"论文", "paper"}

    def is_author_type(self, value: Any) -> bool:
        return self.normalize_entity_type(value).casefold() in {"作者", "author"}

    def normalize_extraction(self, extraction: Any, gold_title: Optional[str] = None) -> Dict[str, Any]:
        clean_extraction = sanitize_extraction_payload(extraction).to_dict()
        entity_type_map: Dict[str, str] = {}
        raw_entity_types: Dict[str, str] = {}

        for entity_name, entity_type in clean_extraction["entity_types"].items():
            normalized_type = self.normalize_entity_type(entity_type)
            raw_entity_types[str(entity_name or "").strip()] = normalized_type
            
            # 强制对齐论文节点到 Gold Title
            current_entity_name = entity_name
            if gold_title and normalized_type == "论文":
                current_entity_name = gold_title

            normalized_name = self.normalize_entity_name(current_entity_name, normalized_type)
            if not normalized_name:
                continue
            entity_type_map[normalized_name] = normalized_type

        for entity_name in clean_extraction["attributes"].keys():
            raw_name = str(entity_name or "").strip()
            normalized_type = raw_entity_types.get(raw_name, "")

            # 属性对齐逻辑
            current_entity_name = entity_name
            if gold_title and normalized_type == "论文":
                current_entity_name = gold_title

            normalized_name = self.normalize_entity_name(current_entity_name, normalized_type)
            if normalized_name and normalized_name not in entity_type_map:
                entity_type_map[normalized_name] = normalized_type

        for head, _, tail in clean_extraction["triples"]:
            for entity_name in (head, tail):
                raw_name = str(entity_name or "").strip()
                normalized_type = raw_entity_types.get(raw_name, "")
                
                # 三元组节点对齐逻辑
                current_entity_name = entity_name
                if gold_title and normalized_type == "论文":
                    current_entity_name = gold_title

                normalized_name = self.normalize_entity_name(current_entity_name, normalized_type)
                if normalized_name and normalized_name not in entity_type_map:
                    entity_type_map[normalized_name] = normalized_type

        entity_set = {
            (name, entity_type_map.get(name, ""))
            for name in entity_type_map
            if name
        }

        triple_set = set()
        for head, relation, tail in clean_extraction["triples"]:
            head_type = raw_entity_types.get(str(head).strip(), "")
            tail_type = raw_entity_types.get(str(tail).strip(), "")

            norm_head_name = head
            if gold_title and head_type == "论文":
                norm_head_name = gold_title
                
            norm_tail_name = tail
            if gold_title and tail_type == "论文":
                norm_tail_name = gold_title

            normalized_head = self.normalize_entity_name(norm_head_name, head_type)
            normalized_relation = self.normalize_relation_name(relation)
            normalized_tail = self.normalize_entity_name(norm_tail_name, tail_type)
            if normalized_head and normalized_relation and normalized_tail:
                triple_set.add((normalized_head, normalized_relation, normalized_tail))

        attribute_pairs = set()
        for entity_name, values in clean_extraction["attributes"].items():
            ent_type = raw_entity_types.get(str(entity_name).strip(), "")
            norm_ent_name = entity_name
            if gold_title and ent_type == "论文":
                norm_ent_name = gold_title

            normalized_entity = self.normalize_entity_name(
                norm_ent_name,
                ent_type,
            )
            if not normalized_entity:
                continue
            for value in values:
                normalized_value = self.normalize_text(value)
                if normalized_value:
                    attribute_pairs.add((normalized_entity, normalized_value))

        return {
            "entity_type_map": entity_type_map,
            "entities": entity_set,
            "triples": triple_set,
            "attributes": attribute_pairs,
        }

    def normalize_items(self, items: Iterable[Tuple[str, ...]]) -> Set[Tuple[str, ...]]:
        normalized_items: Set[Tuple[str, ...]] = set()
        for item in items:
            normalized_items.add(tuple(self.normalize_text(part) for part in item))
        return normalized_items


def ensure_dataset_registered(config, dataset_name: str) -> DatasetConfig:
    try:
        return config.get_dataset_config(dataset_name)
    except ValueError:
        dataset_dir = Path("data") / "uploaded" / dataset_name
        fallback = DatasetConfig(
            corpus_path=str(dataset_dir / "corpus.json"),
            qa_path=str(dataset_dir / f"{dataset_name}.json"),
            schema_path=str(Path("schemas") / f"{dataset_name}.json"),
            graph_output=str(Path("output") / "graphs" / f"{dataset_name}_new.json"),
        )
        config.override_config({"datasets": {dataset_name: asdict(fallback)}})
        return config.get_dataset_config(dataset_name)
