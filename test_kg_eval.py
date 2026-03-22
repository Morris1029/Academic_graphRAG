from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx

from eval.kg_eval.audit import evaluate_graph_structure, score_cross_doc_reviews
from eval.kg_eval.construction import ConstructionBridge
from eval.kg_eval.loader import load_samples, save_samples
from eval.kg_eval.metrics import aggregate_metric_blocks, compare_extractions
from eval.kg_eval.retrieval import evaluate_sample_retrieval
from eval.kg_eval.run import command_generate_gold


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


class GenerateGoldLoggingTests(unittest.TestCase):
    def _render_log_message(self, call) -> str:
        args = call.args
        if not args:
            return ""
        template = args[0]
        values = args[1:]
        return template % values if values else str(template)

    @patch("eval.kg_eval.run.save_samples")
    @patch("eval.kg_eval.run.load_samples")
    @patch("eval.kg_eval.run.ExtractionService")
    @patch("eval.kg_eval.run.ConstructionBridge")
    @patch("eval.kg_eval.run.logger.info")
    @patch("eval.kg_eval.run.time.perf_counter")
    def test_generate_gold_logs_per_sample_progress_and_summary(
        self,
        mock_perf_counter,
        mock_logger_info,
        mock_bridge_cls,
        mock_service_cls,
        mock_load_samples,
        mock_save_samples,
    ):
        records = [
            {"id": "paper_1", "meta": {"title": "Paper 1"}, "kg_eval": {"gold": {"status": "pending"}}},
            {"id": "paper_2", "meta": {"title": "Paper 2"}, "kg_eval": {"gold": {"status": "approved"}}},
            {"id": "paper_3", "meta": {"title": "Paper 3"}, "kg_eval": {"gold": {"status": "pending"}}},
        ]
        mock_load_samples.return_value = records
        mock_bridge_cls.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.build_gold_payload.side_effect = [
            {
                "status": "draft",
                "review_notes": "AUTO_ERROR: timeout",
                "updated_at": "2026-03-22T00:00:00Z",
                "extraction": {"entity_types": {}, "triples": [], "attributes": {}},
            },
            {
                "status": "draft",
                "review_notes": "",
                "updated_at": "2026-03-22T00:00:01Z",
                "extraction": {"entity_types": {}, "triples": [], "attributes": {}},
            },
        ]
        mock_service_cls.return_value = mock_service
        mock_perf_counter.side_effect = [100.0, 101.0, 104.0, 105.0, 106.0, 107.0, 109.0, 112.0]

        args = Namespace(sample_path="samples.json", gold_model="test_model", max_samples=3)
        runtime_config = {
            "defaults": {
                "dataset_name": "AIGC-EDU",
                "main_config_path": "config/base_config.yaml",
            },
            "models": {"test_model": {}},
            "roles": {"gold": {}},
        }

        command_generate_gold(args, runtime_config)

        self.assertEqual(mock_service.build_gold_payload.call_count, 2)
        mock_save_samples.assert_called_once_with("samples.json", records)
        self.assertEqual(records[0]["kg_eval"]["gold"]["review_notes"], "AUTO_ERROR: timeout")
        self.assertEqual(records[2]["kg_eval"]["gold"]["review_notes"], "")

        rendered_messages = [self._render_log_message(call) for call in mock_logger_info.call_args_list]
        self.assertIn(
            "Starting gold draft generation | dataset=AIGC-EDU sample_path=samples.json gold_model=test_model max_samples=3 target_total=3",
            rendered_messages[0],
        )
        self.assertIn("[1/3] generating gold for sample_id=paper_1", rendered_messages[1])
        self.assertIn("[1/3] finished sample_id=paper_1 status=updated", rendered_messages[2])
        self.assertIn("elapsed=", rendered_messages[2])
        self.assertIn("avg=", rendered_messages[2])
        self.assertIn("eta=", rendered_messages[2])
        self.assertIn("auto_error=true", rendered_messages[2])
        self.assertIn("[2/3] finished sample_id=paper_2 status=skipped_approved", rendered_messages[4])
        self.assertIn("auto_error=false", rendered_messages[4])
        self.assertIn("[3/3] finished sample_id=paper_3 status=updated", rendered_messages[6])
        self.assertIn("auto_error=false", rendered_messages[6])
        self.assertIn(
            "Gold draft generation finished | sample_path=samples.json updated=2 skipped_approved=1",
            rendered_messages[7],
        )
        self.assertIn("total_elapsed=", rendered_messages[7])
        self.assertIn("avg_per_sample=", rendered_messages[7])


if __name__ == "__main__":
    unittest.main()
