from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from config import DatasetConfig, reload_config
from models.constructor.kt_gen import KTBuilder

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
    """Reuse the production construction prompt and normalization rules without requiring a KG API key."""

    def __init__(self, dataset_name: str, main_config_path: str):
        self.dataset_name = dataset_name
        self.config = reload_config(main_config_path)
        self.dataset_config = ensure_dataset_registered(self.config, dataset_name)
        self.schema_path = self.dataset_config.schema_path
        self.schema = json.loads(Path(self.schema_path).read_text(encoding="utf-8"))
        self.builder = self._build_prompt_bridge()

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
        return builder

    def build_prompt(self, sample: Dict[str, Any]) -> str:
        chunk = {"meta": sample.get("meta", {})}
        return self.builder._get_construction_prompt(chunk)

    def parse_response(self, raw_response: str) -> KGExtraction:
        parsed = self.builder._validate_and_parse_llm_response(raw_response)
        return sanitize_extraction_payload(parsed)

    def normalize_text(self, value: Any) -> str:
        text = unicodedata.normalize("NFKC", str(value or ""))
        text = text.replace("\u3000", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip("\"'“”‘’`")
        text = text.strip("()[]{}<>《》")
        return text.casefold()

    def normalize_entity_name(self, value: Any) -> str:
        return self.normalize_text(value)

    def normalize_relation_name(self, value: Any) -> str:
        relation = unicodedata.normalize("NFKC", str(value or "")).strip()
        relation = self.builder._normalize_relation_name(relation)
        return re.sub(r"\s+", " ", relation).strip()

    def normalize_entity_type(self, value: Any) -> str:
        raw = unicodedata.normalize("NFKC", str(value or "")).strip()
        if not raw:
            return ""
        lowered = raw.casefold()
        alias_hit = self.builder.schema_type_aliases.get(lowered, self.builder.schema_type_aliases.get(raw))
        normalized = alias_hit or raw
        if normalized in self.builder.allowed_schema_node_types:
            return normalized
        return normalized

    def is_paper_type(self, value: Any) -> bool:
        return self.normalize_entity_type(value).casefold() in {"论文", "paper"}

    def is_author_type(self, value: Any) -> bool:
        return self.normalize_entity_type(value).casefold() in {"作者", "author"}

    def normalize_extraction(self, extraction: Any) -> Dict[str, Any]:
        clean_extraction = sanitize_extraction_payload(extraction).to_dict()
        entity_type_map: Dict[str, str] = {}

        for entity_name, entity_type in clean_extraction["entity_types"].items():
            normalized_name = self.normalize_entity_name(entity_name)
            if not normalized_name:
                continue
            entity_type_map[normalized_name] = self.normalize_entity_type(entity_type)

        for entity_name in clean_extraction["attributes"].keys():
            normalized_name = self.normalize_entity_name(entity_name)
            if normalized_name and normalized_name not in entity_type_map:
                entity_type_map[normalized_name] = ""

        for head, _, tail in clean_extraction["triples"]:
            for entity_name in (head, tail):
                normalized_name = self.normalize_entity_name(entity_name)
                if normalized_name and normalized_name not in entity_type_map:
                    entity_type_map[normalized_name] = ""

        entity_set = {
            (name, entity_type_map.get(name, ""))
            for name in entity_type_map
            if name
        }

        triple_set = set()
        for head, relation, tail in clean_extraction["triples"]:
            normalized_head = self.normalize_entity_name(head)
            normalized_relation = self.normalize_relation_name(relation)
            normalized_tail = self.normalize_entity_name(tail)
            if normalized_head and normalized_relation and normalized_tail:
                triple_set.add((normalized_head, normalized_relation, normalized_tail))

        attribute_pairs = set()
        for entity_name, values in clean_extraction["attributes"].items():
            normalized_entity = self.normalize_entity_name(entity_name)
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
