from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from utils.logger import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for direct script execution
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.logger import logger

LENGTH_BIN_LABELS = ("<=200", "201-400", "401-800", ">800")
StratumKey = Tuple[str, str]


def derive_report_path(output_path: str) -> Path:
    output = Path(output_path)
    return output.with_name(f"{output.stem}.sampling_report.json")


def get_length_bin(text_length: int) -> str:
    if text_length <= 200:
        return "<=200"
    if text_length <= 400:
        return "201-400"
    if text_length <= 800:
        return "401-800"
    return ">800"


def extract_year(record: Mapping[str, Any]) -> str:
    meta = record.get("meta", {}) if isinstance(record, dict) else {}
    year = str(meta.get("year", "") or "").strip()
    return year or "UNKNOWN"


def compute_text_length(record: Mapping[str, Any]) -> int:
    meta = record.get("meta", {}) if isinstance(record, dict) else {}
    title = str(meta.get("title", "") or "")
    abstract = str(meta.get("abstract", "") or "")
    return len(title) + len(abstract)


def get_stratum(record: Mapping[str, Any]) -> StratumKey:
    return extract_year(record), get_length_bin(compute_text_length(record))


def load_paper_records(input_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Input JSON must be a list of paper objects: {input_path}")

    valid_records: List[Dict[str, Any]] = []
    skipped_missing_id = 0
    skipped_non_object = 0

    for record in payload:
        if not isinstance(record, dict):
            skipped_non_object += 1
            continue
        if not str(record.get("id", "")).strip():
            skipped_missing_id += 1
            continue
        valid_records.append(record)

    return valid_records, {
        "input_total_records": len(payload),
        "valid_record_count": len(valid_records),
        "skipped_missing_id_count": skipped_missing_id,
        "skipped_non_object_count": skipped_non_object,
    }


def _year_sort_key(year: str) -> Tuple[int, str]:
    text = str(year or "").strip()
    if text.isdigit():
        return (0, f"{int(text):04d}")
    if text == "UNKNOWN":
        return (2, text)
    return (1, text)


def _length_bin_sort_key(length_bin: str) -> int:
    try:
        return LENGTH_BIN_LABELS.index(length_bin)
    except ValueError:
        return len(LENGTH_BIN_LABELS)


def sort_stratum_keys(keys: Sequence[StratumKey]) -> List[StratumKey]:
    return sorted(keys, key=lambda item: (_year_sort_key(item[0]), _length_bin_sort_key(item[1]), item))


def _capped_largest_remainder(
    weights: Mapping[StratumKey, int],
    total: int,
    capacities: Mapping[StratumKey, int],
) -> Tuple[Dict[StratumKey, int], Dict[str, Any]]:
    allocations = {key: 0 for key in weights}
    remaining_capacity = {key: max(0, int(capacities.get(key, 0))) for key in weights}
    remaining = total
    rounds = 0
    redistribution_happened = False
    reallocated_slots = 0

    while remaining > 0:
        eligible = {
            key: remaining_capacity[key]
            for key in sort_stratum_keys(list(weights.keys()))
            if remaining_capacity.get(key, 0) > 0 and int(weights.get(key, 0)) > 0
        }
        if not eligible:
            break

        rounds += 1
        total_weight = sum(int(weights[key]) for key in eligible)
        if total_weight <= 0:
            raise ValueError("No positive stratum weights available for allocation")

        quotas = {key: remaining * int(weights[key]) / total_weight for key in eligible}
        floor_allocations: Dict[StratumKey, int] = {}
        floor_total = 0
        capped_this_round = False

        for key in eligible:
            raw_floor = math.floor(quotas[key])
            allocation = min(raw_floor, remaining_capacity[key])
            if raw_floor > remaining_capacity[key]:
                capped_this_round = True
            floor_allocations[key] = allocation
            floor_total += allocation

        if floor_total:
            for key, allocation in floor_allocations.items():
                allocations[key] += allocation
                remaining_capacity[key] -= allocation
            remaining -= floor_total

        if remaining == 0:
            if rounds > 1:
                redistribution_happened = True
                reallocated_slots += floor_total
            if capped_this_round:
                redistribution_happened = True
            break

        remainder_candidates = []
        for key in eligible:
            if remaining_capacity[key] <= 0:
                continue
            remainder = quotas[key] - math.floor(quotas[key])
            remainder_candidates.append((remainder, int(weights[key]), key))

        remainder_candidates.sort(
            key=lambda item: (-item[0], -item[1], _year_sort_key(item[2][0]), _length_bin_sort_key(item[2][1]), item[2])
        )

        remainder_total = 0
        for _, _, key in remainder_candidates:
            if remaining == 0:
                break
            if remaining_capacity[key] <= 0:
                continue
            allocations[key] += 1
            remaining_capacity[key] -= 1
            remaining -= 1
            remainder_total += 1

        if rounds > 1:
            reallocated_slots += floor_total + remainder_total

        if capped_this_round or remaining > 0:
            redistribution_happened = redistribution_happened or capped_this_round or rounds > 1

        if floor_total == 0 and remainder_total == 0:
            break

    if remaining != 0:
        raise ValueError("Unable to satisfy requested sample size with available strata capacity")

    return allocations, {
        "redistribution_happened": redistribution_happened,
        "redistribution_rounds": max(0, rounds - 1),
        "reallocated_slots": reallocated_slots,
    }


def allocate_stratum_samples(
    stratum_counts: Mapping[StratumKey, int],
    sample_size: int,
) -> Tuple[Dict[StratumKey, int], Dict[str, Any]]:
    total_population = sum(int(count) for count in stratum_counts.values())
    non_empty_strata = {key: int(count) for key, count in stratum_counts.items() if int(count) > 0}

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size > total_population:
        raise ValueError(
            f"Requested sample_size={sample_size} exceeds valid population={total_population}"
        )
    if sample_size == 0:
        return {key: 0 for key in stratum_counts}, {
            "allocation_mode": "zero",
            "guarantee_applied": False,
            "redistribution_happened": False,
            "redistribution_rounds": 0,
            "reallocated_slots": 0,
        }

    if sample_size < len(non_empty_strata):
        allocations, meta = _capped_largest_remainder(
            weights=non_empty_strata,
            total=sample_size,
            capacities=non_empty_strata,
        )
        full_allocations = {key: allocations.get(key, 0) for key in stratum_counts}
        return full_allocations, {
            "allocation_mode": "pure_proportional",
            "guarantee_applied": False,
            **meta,
        }

    base_allocations = {key: (1 if key in non_empty_strata else 0) for key in stratum_counts}
    remaining = sample_size - len(non_empty_strata)
    if remaining == 0:
        return base_allocations, {
            "allocation_mode": "guaranteed_only",
            "guarantee_applied": True,
            "redistribution_happened": False,
            "redistribution_rounds": 0,
            "reallocated_slots": 0,
        }

    capacities = {key: max(0, count - 1) for key, count in non_empty_strata.items()}
    extra_allocations, meta = _capped_largest_remainder(
        weights=non_empty_strata,
        total=remaining,
        capacities=capacities,
    )
    full_allocations = {key: base_allocations.get(key, 0) + extra_allocations.get(key, 0) for key in stratum_counts}
    return full_allocations, {
        "allocation_mode": "guaranteed_plus_proportional",
        "guarantee_applied": True,
        **meta,
    }


def build_sampling_payload(
    records: Sequence[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(records) < sample_size:
        raise ValueError(
            f"Valid record count {len(records)} is smaller than requested sample_size {sample_size}"
        )

    population_by_stratum: Dict[StratumKey, List[Dict[str, Any]]] = defaultdict(list)
    population_year_counter: Counter[str] = Counter()
    population_length_counter: Counter[str] = Counter()

    for record in records:
        year, length_bin = get_stratum(record)
        population_by_stratum[(year, length_bin)].append(record)
        population_year_counter[year] += 1
        population_length_counter[length_bin] += 1

    stratum_counts = {key: len(items) for key, items in population_by_stratum.items()}
    allocations, allocation_meta = allocate_stratum_samples(stratum_counts, sample_size)

    draw_rng = random.Random(seed)
    sampled_records: List[Dict[str, Any]] = []
    actual_counts: Counter[StratumKey] = Counter()

    for stratum in sort_stratum_keys(list(stratum_counts.keys())):
        population = population_by_stratum[stratum]
        requested = allocations.get(stratum, 0)
        if requested <= 0:
            continue
        sampled = draw_rng.sample(population, requested)
        sampled_records.extend(sampled)
        actual_counts[stratum] += len(sampled)

    shuffle_rng = random.Random(seed)
    shuffle_rng.shuffle(sampled_records)

    sample_year_counter: Counter[str] = Counter()
    sample_length_counter: Counter[str] = Counter()
    for record in sampled_records:
        year, length_bin = get_stratum(record)
        sample_year_counter[year] += 1
        sample_length_counter[length_bin] += 1

    joint_rows = []
    for stratum in sort_stratum_keys(list(stratum_counts.keys())):
        joint_rows.append(
            {
                "year": stratum[0],
                "length_bin": stratum[1],
                "population_count": stratum_counts.get(stratum, 0),
                "target_quota": allocations.get(stratum, 0),
                "actual_sample_count": actual_counts.get(stratum, 0),
            }
        )

    report = {
        "seed": seed,
        "sample_size_requested": sample_size,
        "sampled_record_count": len(sampled_records),
        "allocation": allocation_meta,
        "year_distribution": {
            "population": {year: population_year_counter[year] for year in sorted(population_year_counter, key=_year_sort_key)},
            "sample": {year: sample_year_counter.get(year, 0) for year in sorted(population_year_counter, key=_year_sort_key)},
        },
        "length_distribution": {
            "population": {label: population_length_counter.get(label, 0) for label in LENGTH_BIN_LABELS},
            "sample": {label: sample_length_counter.get(label, 0) for label in LENGTH_BIN_LABELS},
        },
        "joint_strata": joint_rows,
        "sample_ids": [str(record.get("id", "")).strip() for record in sampled_records],
    }
    return sampled_records, report


def run_sampling(
    input_path: str,
    output_path: str,
    report_path: str | None = None,
    sample_size: int = 100,
    seed: int = 42,
    overwrite: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    output = Path(output_path)
    report = Path(report_path) if report_path else derive_report_path(output_path)

    if not overwrite:
        if output.exists():
            raise FileExistsError(f"Output file already exists: {output}")
        if report.exists():
            raise FileExistsError(f"Report file already exists: {report}")

    records, load_stats = load_paper_records(input_path)
    sampled_records, sampling_report = build_sampling_payload(records, sample_size=sample_size, seed=seed)
    final_report = {
        "input_path": str(Path(input_path)),
        "output_path": str(output),
        "report_path": str(report),
        **load_stats,
        **sampling_report,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sampled_records, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "Stratified sampling finished | input=%s valid=%d sampled=%d output=%s report=%s",
        input_path,
        load_stats["valid_record_count"],
        len(sampled_records),
        output,
        report,
    )
    return output, report, final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stratified sampler for KG evaluation paper samples")
    parser.add_argument(
        "--input-path",
        default="data/uploaded/AIGC-EDU/AIGC-EDU.json",
        help="Path to the source paper JSON file",
    )
    parser.add_argument(
        "--output-path",
        default="eval/kg_eval/dataset/AIGC-EDU-kgval.json",
        help="Path to the sampled output JSON file",
    )
    parser.add_argument(
        "--report-path",
        help="Optional path to the sampling report JSON file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of papers to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output and report files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_sampling(
        input_path=args.input_path,
        output_path=args.output_path,
        report_path=args.report_path,
        sample_size=args.sample_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
