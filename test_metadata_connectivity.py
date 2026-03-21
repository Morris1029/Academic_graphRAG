from __future__ import annotations

import threading
import unittest
from collections import defaultdict

import networkx as nx

from models.constructor.kt_gen import KTBuilder
from utils.entity_normalizer import EntityNormalizer


def build_builder() -> KTBuilder:
    builder = KTBuilder.__new__(KTBuilder)
    builder.dataset_name = "AIGC-EDU"
    builder.graph = nx.MultiDiGraph()
    builder.lock = threading.Lock()
    builder.node_counter = 1
    builder.duplicate_doc_ids = set()
    builder.doc_meta_by_chunk_id = {}
    builder._entity_name_index = {}
    builder.schema_type_aliases = {
        "paper": "论文",
        "author": "作者",
        "organization": "机构",
        "institution": "机构",
        "journal": "期刊",
    }
    builder.generic_entity_name_blacklist = set()
    builder.metadata_author_blacklist_tokens = (
        "编辑部",
        "本刊",
        "记者",
        "评论员",
        "导读",
        "选题",
        "活动",
        "指南",
    )
    builder.allowed_schema_node_types = {"论文", "作者", "机构", "期刊"}
    builder.entity_normalizer = EntityNormalizer(
        schema_type_aliases=builder.schema_type_aliases,
        config_path="config/entity_aliases.yaml",
    )
    builder.entity_alias_audit = defaultdict(set)
    return builder


class MetadataConnectivityTests(unittest.TestCase):
    def test_metadata_edges_are_supplemented_from_document_meta(self):
        builder = build_builder()
        doc = {
            "id": "doc_1",
            "meta": {
                "title": "生成式人工智能赋能教学研究",
                "authors": "张三;李四",
                "organ": "北京师范大学教育学部",
                "source": "电化教育研究",
            },
        }
        builder.doc_meta_by_chunk_id["doc_1"] = {
            "title": doc["meta"]["title"],
            "doc_uid": "doc_1",
            "source_doc_id": "doc_1",
            "source": doc["meta"]["source"],
            "authors": doc["meta"]["authors"],
        }
        builder.graph.add_node(
            "paper_1",
            label="entity",
            level=2,
            properties={
                "name": doc["meta"]["title"],
                "schema_type": "论文",
                "chunk id": "doc_1",
                "doc_uid": "doc_1",
                "source_doc_id": "doc_1",
            },
        )
        builder._register_entity_index(doc["meta"]["title"], "paper_1", "论文", "doc_1")

        builder._supplement_metadata_edges([doc])

        relation_rows = [
            (
                builder.graph.nodes[u]["properties"].get("name"),
                data.get("relation"),
                builder.graph.nodes[v]["properties"].get("name"),
            )
            for u, v, data in builder.graph.edges(data=True)
        ]

        self.assertIn(("张三", "撰写", doc["meta"]["title"]), relation_rows)
        self.assertIn(("李四", "撰写", doc["meta"]["title"]), relation_rows)
        self.assertIn((doc["meta"]["title"], "发表于", "电化教育研究"), relation_rows)
        self.assertIn(("张三", "隶属", "北京师范大学"), relation_rows)
        self.assertIn(("李四", "隶属", "北京师范大学"), relation_rows)


if __name__ == "__main__":
    unittest.main()
