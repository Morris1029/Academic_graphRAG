from __future__ import annotations

import unittest

from config import reload_config
from eval.kg_eval.construction import ConstructionBridge
from models.constructor.kt_gen import KTBuilder
from utils.entity_normalizer import EntityNormalizer


class EntityNormalizerTests(unittest.TestCase):
    def test_technology_aliases_resolve_to_same_canonical_name(self):
        normalizer = EntityNormalizer()
        canonical_aigc, key_aigc = normalizer.resolve("AIGC", "技术")
        canonical_genai, key_genai = normalizer.resolve("GenAI", "技术")
        canonical_full, key_full = normalizer.resolve("生成式人工智能", "技术")

        self.assertEqual(canonical_aigc, "生成式人工智能")
        self.assertEqual(canonical_genai, "生成式人工智能")
        self.assertEqual(canonical_full, "生成式人工智能")
        self.assertEqual(key_aigc, key_genai)
        self.assertEqual(key_genai, key_full)


class BuilderEntityMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = reload_config("config/base_config.yaml")

    def test_same_technology_aliases_merge_to_one_node(self):
        builder = KTBuilder(
            "AIGC-EDU",
            schema_path="schemas/AIGC-EDU.json",
            config=self.config,
        )

        node_a = builder._find_or_create_entity_direct("AIGC", "chunk-1", "技术")
        node_b = builder._find_or_create_entity_direct("生成式人工智能", "chunk-2", "技术")
        node_c = builder._find_or_create_entity_direct("GenAI", "chunk-3", "技术")

        self.assertEqual(node_a, node_b)
        self.assertEqual(node_b, node_c)

        props = builder.graph.nodes[node_a]["properties"]
        self.assertEqual(props["name"], "生成式人工智能")
        self.assertIn("AIGC", props.get("aliases", []))
        self.assertIn("GenAI", props.get("aliases", []))

    def test_same_name_different_types_do_not_merge(self):
        builder = KTBuilder(
            "AIGC-EDU",
            schema_path="schemas/AIGC-EDU.json",
            config=self.config,
        )

        tech_node = builder._find_or_create_entity_direct("AIGC", "chunk-1", "技术")
        topic_node = builder._find_or_create_entity_direct("AIGC", "chunk-1", "研究主题")

        self.assertNotEqual(tech_node, topic_node)
        self.assertEqual(builder.graph.nodes[tech_node]["properties"]["schema_type"], "技术")
        self.assertEqual(builder.graph.nodes[topic_node]["properties"]["schema_type"], "研究主题")


class BridgeNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = ConstructionBridge("AIGC-EDU", "config/base_config.yaml")

    def test_bridge_normalizes_semantic_alias_duplicates(self):
        self.assertEqual(
            self.bridge.normalize_entity_name("AIGC", "技术"),
            self.bridge.normalize_entity_name("生成式人工智能", "技术"),
        )


if __name__ == "__main__":
    unittest.main()
