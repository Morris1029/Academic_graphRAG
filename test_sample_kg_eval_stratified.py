from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.utils.sample_kg_eval_stratified import (
    allocate_stratum_samples,
    build_sampling_payload,
    derive_report_path,
    get_length_bin,
    run_sampling,
)


def make_record(record_id: str, year: str, total_length: int) -> dict:
    title_length = min(20, total_length)
    abstract_length = max(0, total_length - title_length)
    return {
        "id": record_id,
        "meta": {
            "year": year,
            "title": "T" * title_length,
            "abstract": "A" * abstract_length,
        },
    }


class LengthBinTests(unittest.TestCase):
    def test_length_bins_cover_boundaries(self):
        self.assertEqual(get_length_bin(200), "<=200")
        self.assertEqual(get_length_bin(201), "201-400")
        self.assertEqual(get_length_bin(400), "201-400")
        self.assertEqual(get_length_bin(401), "401-800")
        self.assertEqual(get_length_bin(800), "401-800")
        self.assertEqual(get_length_bin(801), ">800")


class AllocationTests(unittest.TestCase):
    def test_proportional_allocation_keeps_exact_total(self):
        counts = {
            ("2023", "<=200"): 50,
            ("2024", "201-400"): 30,
            ("2025", "401-800"): 20,
        }
        allocations, meta = allocate_stratum_samples(counts, sample_size=10)
        self.assertEqual(sum(allocations.values()), 10)
        self.assertTrue(meta["guarantee_applied"])
        for key, count in counts.items():
            self.assertLessEqual(allocations[key], count)

    def test_sparse_strata_trigger_redistribution(self):
        counts = {
            ("2023", "<=200"): 2,
            ("2024", "201-400"): 2,
            ("2025", "401-800"): 96,
        }
        allocations, meta = allocate_stratum_samples(counts, sample_size=99)
        self.assertEqual(allocations[("2023", "<=200")], 2)
        self.assertEqual(allocations[("2024", "201-400")], 2)
        self.assertEqual(allocations[("2025", "401-800")], 95)
        self.assertEqual(sum(allocations.values()), 99)
        self.assertTrue(meta["redistribution_happened"])

    def test_sample_size_smaller_than_non_empty_strata_uses_pure_proportional(self):
        counts = {
            ("2023", "<=200"): 10,
            ("2023", "201-400"): 10,
            ("2024", "<=200"): 10,
            ("2024", "201-400"): 10,
            ("2025", "<=200"): 10,
        }
        allocations, meta = allocate_stratum_samples(counts, sample_size=3)
        self.assertEqual(sum(allocations.values()), 3)
        self.assertFalse(meta["guarantee_applied"])
        self.assertEqual(meta["allocation_mode"], "pure_proportional")


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for idx in range(15):
            self.records.append(make_record(f"2023-short-{idx}", "2023", 180))
            self.records.append(make_record(f"2024-mid-{idx}", "2024", 300))
            self.records.append(make_record(f"2025-long-{idx}", "2025", 500))
            self.records.append(make_record(f"2026-xl-{idx}", "2026", 900))

    def test_same_seed_is_stable_and_different_seed_changes_output(self):
        sample_a, report_a = build_sampling_payload(self.records, sample_size=20, seed=42)
        sample_b, report_b = build_sampling_payload(self.records, sample_size=20, seed=42)
        sample_c, report_c = build_sampling_payload(self.records, sample_size=20, seed=7)

        ids_a = [item["id"] for item in sample_a]
        ids_b = [item["id"] for item in sample_b]
        ids_c = [item["id"] for item in sample_c]

        self.assertEqual(ids_a, ids_b)
        self.assertNotEqual(ids_a, ids_c)
        self.assertEqual(len(ids_a), len(set(ids_a)))
        self.assertEqual(len(report_a["sample_ids"]), 20)
        self.assertEqual(report_a["sample_ids"], ids_a)
        self.assertEqual(report_b["sample_ids"], ids_b)
        self.assertEqual(report_c["sampled_record_count"], 20)


class IOTests(unittest.TestCase):
    def test_run_sampling_writes_sample_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_path = base / "input.json"
            output_path = base / "sample.json"
            report_path = derive_report_path(str(output_path))

            payload = [
                make_record("doc-1", "2023", 150),
                make_record("doc-2", "2024", 250),
                {"meta": {"year": "2025", "title": "bad", "abstract": "missing id"}},
                "not-an-object",
                make_record("doc-3", "2025", 450),
                make_record("doc-4", "2026", 850),
            ]
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            output, report, summary = run_sampling(
                input_path=str(input_path),
                output_path=str(output_path),
                sample_size=4,
                seed=42,
            )

            self.assertEqual(output, output_path)
            self.assertEqual(report, report_path)
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())
            self.assertEqual(summary["skipped_missing_id_count"], 1)
            self.assertEqual(summary["skipped_non_object_count"], 1)
            self.assertEqual(summary["sampled_record_count"], 4)

            sampled_payload = json.loads(output.read_text(encoding="utf-8"))
            report_payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(len(sampled_payload), 4)
            self.assertEqual(report_payload["sample_ids"], [item["id"] for item in sampled_payload])

    def test_existing_output_requires_overwrite_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_path = base / "input.json"
            output_path = base / "sample.json"
            input_path.write_text(
                json.dumps([make_record(f"doc-{idx}", "2025", 300) for idx in range(10)], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_path.write_text("[]", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_sampling(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    sample_size=5,
                    seed=42,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
