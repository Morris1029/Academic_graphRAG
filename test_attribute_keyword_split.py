from __future__ import annotations

import unittest

from config import reload_config
from models.constructor.kt_gen import KTBuilder


class AttributeKeywordSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = reload_config("config/base_config.yaml")

    def setUp(self):
        self.builder = KTBuilder(
            "AIGC-EDU",
            schema_path="schemas/AIGC-EDU.json",
            config=self.config,
        )

    def test_split_keyword_attribute_with_semicolon(self):
        attrs = self.builder._normalize_attribute_values(["关键词: A;B;C"])
        self.assertEqual(attrs, ["关键词: A", "关键词: B", "关键词: C"])

    def test_split_keyword_attribute_with_chinese_delimiters(self):
        attrs = self.builder._normalize_attribute_values(["关键词：A，B，C"])
        self.assertEqual(attrs, ["关键词: A", "关键词: B", "关键词: C"])

    def test_deduplicate_repeated_keywords(self):
        attrs = self.builder._normalize_attribute_values(["关键词: A", "关键词: B", "关键词: A"])
        self.assertEqual(attrs, ["关键词: A", "关键词: B"])

    def test_keep_non_keyword_attributes_unchanged(self):
        attrs = self.builder._normalize_attribute_values(
            ["年份: 2024", "来源: 现代教育技术", "关键词: A;B"]
        )
        self.assertEqual(
            attrs,
            ["年份: 2024", "来源: 现代教育技术", "关键词: A", "关键词: B"],
        )

    def test_process_attributes_creates_one_node_per_keyword(self):
        nodes_to_add, edges_to_add = self.builder._process_attributes(
            {"论文A": ["关键词: A;B;C"]},
            chunk_id="chunk-1",
            entity_types={"论文A": "论文"},
        )

        attr_names = [
            node_data["properties"]["name"]
            for _, node_data in nodes_to_add
            if node_data.get("label") == "attribute"
        ]
        self.assertEqual(attr_names, ["关键词: A", "关键词: B", "关键词: C"])
        self.assertEqual(len(edges_to_add), 3)


if __name__ == "__main__":
    unittest.main()
