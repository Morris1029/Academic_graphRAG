from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx

from eval.kg_eval.audit import evaluate_graph_structure, score_cross_doc_reviews
from eval.kg_eval.construction import ConstructionBridge
from eval.kg_eval.loader import load_samples, save_samples
from eval.kg_eval.metrics import aggregate_metric_blocks, compare_extractions
from eval.kg_eval.retrieval import evaluate_sample_retrieval


class FakeRetriever:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("paper_1", properties={"name": "Paper A"})
        self.graph.add_node("author_1", properties={"name": "Author One"})

    def retrieve(self, question: str):
        return None, {"path1_results": {"top_nodes": ["paper_1", "author_1"]}}

    def process_retrieval_results(self, question: str, top_k: int = 10, involved_types=None):
        triples = [
            "(Paper A [schema_type: 论文], 撰写, Author One [schema_type: 作者]) [score: 0.900]",
        ]
        return {"triples": triples, "chunk_ids": [], "chunk_contents": []}, 0.0


class LoaderTests(unittest.TestCase):
    def test_load_samples_adds_default_gold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "samples.json"
            sample_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "paper_1",
                            "meta": {"title": "Paper A"},
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            records = load_samples(str(sample_path))
            self.assertEqual(records[0]["kg_eval"]["gold"]["status"], "pending")

            save_samples(str(sample_path), records)
            saved = json.loads(sample_path.read_text(encoding="utf-8"))
            self.assertIn("kg_eval", saved[0])


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = ConstructionBridge("AIGC-EDU", "config/base_config.yaml")

    def test_compare_extractions_uses_normalized_exact_matching(self):
        gold = {
            "entity_types": {"Paper A": "论文", "Author One": "作者"},
            "triples": [["Paper A", "撰写", "Author One"]],
            "attributes": {"Paper A": ["年份: 2024"]},
        }
        candidate = {
            "entity_types": {"paper a": "paper", "Author One": "作者"},
            "triples": [["paper a", "撰写", "author one"]],
            "attributes": {"Paper A": ["年份: 2024"]},
        }
        result = compare_extractions(self.bridge, gold, candidate)
        self.assertEqual(result["entity_metrics"]["f1"], 1.0)
        self.assertEqual(result["relation_metrics"]["f1"], 1.0)
        self.assertEqual(result["attribute_metrics"]["accuracy"], 1.0)

    def test_aggregate_metric_blocks(self):
        gold = {
            "entity_types": {"Paper A": "论文"},
            "triples": [["Paper A", "提出", "Method X"]],
            "attributes": {},
        }
        candidate = {
            "entity_types": {"Paper A": "论文"},
            "triples": [],
            "attributes": {},
        }
        block = compare_extractions(self.bridge, gold, candidate)
        summary = aggregate_metric_blocks([block])
        self.assertEqual(summary["entity_prf"]["precision"], 1.0)
        self.assertEqual(summary["relation_prf"]["recall"], 0.0)

    def test_attribute_accuracy_uses_tp_fp_fn_denominator(self):
        gold = {
            "entity_types": {"Paper A": "璁烘枃"},
            "triples": [],
            "attributes": {"Paper A": ["Year: 2024", "Venue: Journal X"]},
        }
        candidate = {
            "entity_types": {"Paper A": "璁烘枃"},
            "triples": [],
            "attributes": {"Paper A": ["Year: 2024", "Keyword: AIGC"]},
        }
        block = compare_extractions(self.bridge, gold, candidate)
        self.assertAlmostEqual(block["attribute_metrics"]["accuracy"], 1 / 3)

        summary = aggregate_metric_blocks([block])
        self.assertAlmostEqual(summary["attribute_accuracy"]["accuracy"], 1 / 3)


class GraphAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = ConstructionBridge("AIGC-EDU", "config/base_config.yaml")

    def test_graph_structure_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = Path(tmpdir) / "graph.json"
            relationships = [
                {
                    "start_node": {
                        "label": "entity",
                        "properties": {"name": "Paper A", "schema_type": "论文", "doc_uid": "doc-1"},
                    },
                    "relation": "撰写",
                    "end_node": {
                        "label": "entity",
                        "properties": {"name": "Author One", "schema_type": "作者"},
                    },
                },
                {
                    "start_node": {
                        "label": "entity",
                        "properties": {"name": "Paper A", "schema_type": "论文", "doc_uid": "doc-2"},
                    },
                    "relation": "聚焦",
                    "end_node": {
                        "label": "entity",
                        "properties": {"name": "Topic A", "schema_type": "研究主题"},
                    },
                },
            ]
            graph_path.write_text(json.dumps(relationships, ensure_ascii=False, indent=2), encoding="utf-8")
            metrics = evaluate_graph_structure(str(graph_path), self.bridge)
            self.assertGreater(metrics["duplicate_node_rate"], 0.0)
            self.assertEqual(metrics["paper_author_edge_coverage"], 0.5)

    def test_cross_doc_review_scoring(self):
        summary = score_cross_doc_reviews(
            [
                {"verdict": "accepted"},
                {"verdict": "rejected"},
                {"verdict": ""},
            ]
        )
        self.assertEqual(summary["reviewed"], 2)
        self.assertEqual(summary["precision"], 0.5)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = ConstructionBridge("AIGC-EDU", "config/base_config.yaml")

    def test_sample_retrieval_metrics(self):
        sample = {
            "id": "paper_1",
            "meta": {"title": "Paper A"},
            "kg_eval": {
                "gold": {
                    "status": "approved",
                    "extraction": {
                        "entity_types": {"Paper A": "论文", "Author One": "作者"},
                        "triples": [["Paper A", "撰写", "Author One"]],
                        "attributes": {},
                    },
                }
            },
        }
        result = evaluate_sample_retrieval(sample, FakeRetriever(), self.bridge, paper_top_k=5, triple_top_k=10)
        self.assertTrue(result["paper_node_hit@5"])
        self.assertEqual(result["gold_triple_hit_count"], 1)


if __name__ == "__main__":
    unittest.main()
