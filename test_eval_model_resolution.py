from __future__ import annotations

import os
import unittest

from eval.kg_eval.loader import default_gold_payload
from eval.rag_eval.llm_client import resolve_model_profile
from eval.rag_eval.reporter import render_report_markdown


class ResolveModelProfileTests(unittest.TestCase):
    def setUp(self):
        self.original_env = {
            "QWEN_MODEL": os.environ.get("QWEN_MODEL"),
            "QWEN_BASE_URL": os.environ.get("QWEN_BASE_URL"),
            "QWEN_API_KEY": os.environ.get("QWEN_API_KEY"),
        }
        os.environ["QWEN_MODEL"] = "qwen-max"
        os.environ["QWEN_BASE_URL"] = "https://example.test/v1"
        os.environ["QWEN_API_KEY"] = "secret"

    def tearDown(self):
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_resolve_model_profile_uses_shared_model_and_role_settings(self):
        profile = resolve_model_profile(
            "qwen",
            {
                "qwen": {
                    "provider": "openai",
                    "model_env": "QWEN_MODEL",
                    "base_url_env": "QWEN_BASE_URL",
                    "api_key_env": "QWEN_API_KEY",
                }
            },
            role_cfg={"temperature": 0.0, "timeout_seconds": 120},
        )

        self.assertEqual(profile.name, "qwen")
        self.assertEqual(profile.model, "qwen-max")
        self.assertEqual(profile.base_url, "https://example.test/v1")
        self.assertEqual(profile.api_key, "secret")
        self.assertEqual(profile.temperature, 0.0)
        self.assertEqual(profile.timeout_seconds, 120.0)

    def test_resolve_model_profile_allows_role_override(self):
        profile = resolve_model_profile(
            "qwen",
            {
                "qwen": {
                    "provider": "openai",
                    "model_env": "QWEN_MODEL",
                    "base_url_env": "QWEN_BASE_URL",
                    "api_key_env": "QWEN_API_KEY",
                    "temperature": 0.6,
                }
            },
            role_cfg={"temperature": 0.3},
            overrides={"temperature": 0.1},
        )

        self.assertEqual(profile.temperature, 0.1)

    def test_resolve_model_profile_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "LLM model 'missing' not found"):
            resolve_model_profile("missing", {})


class ReporterAndLoaderTests(unittest.TestCase):
    def test_report_markdown_uses_model_labels(self):
        markdown = render_report_markdown(
            run_meta={
                "run_id": "run_1",
                "dataset_name": "demo",
                "question_set_path": "dataset.json",
                "qa_mode": "agent",
                "answer_model": "deepseek",
                "judge_model": "qwen",
            },
            summary={
                "sample_count": 1,
                "overall_score": 4.0,
                "failure_rate": 0.0,
                "refusal_rate": 0.0,
                "hallucination_rate": 0.0,
                "per_dimension_avg": {"accuracy": 4.0},
            },
            rows=[],
            dimensions={"accuracy": {"weight": 1.0}},
        )

        self.assertIn("answer_model", markdown)
        self.assertIn("judge_model", markdown)

    def test_default_gold_payload_uses_generator_model(self):
        payload = default_gold_payload()
        self.assertIn("generator_model", payload)
        self.assertNotIn("generator_profile", payload)


if __name__ == "__main__":
    unittest.main()
