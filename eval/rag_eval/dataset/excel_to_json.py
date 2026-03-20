from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_INPUT = "D:/STUDY MO/Research/Data/Q&A.xlsx"
DEFAULT_OUTPUT = "eval/dataset/sheet1_questions.json"
SHEET_NAME = "dataset"

SOURCE_COLUMNS = {
    "question_id": "序号",
    "question_type": "问题类型",
    "question": "问题",
    "reference_answer": "参考答案",
}


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def convert_sheet1_to_records(excel_path: str) -> List[dict]:
    df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)

    missing_columns = [
        column_name
        for column_name in SOURCE_COLUMNS.values()
        if column_name not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Sheet1 is missing required columns: {missing_columns}"
        )

    records: List[dict] = []
    seen_ids = set()

    for row_number, row in enumerate(df.to_dict(orient="records"), start=2):
        question = _normalize_text(row.get(SOURCE_COLUMNS["question"]))
        reference_answer = _normalize_text(row.get(SOURCE_COLUMNS["reference_answer"]))
        question_id = _normalize_text(row.get(SOURCE_COLUMNS["question_id"]))
        question_type = _normalize_text(row.get(SOURCE_COLUMNS["question_type"]))

        if not question:
            raise ValueError(f"Sheet1 row {row_number} has empty '问题'")
        if not reference_answer:
            raise ValueError(
                f"Sheet1 row {row_number} has empty '参考答案'"
            )
        if not question_id:
            question_id = f"Sheet1-{row_number}"
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id '{question_id}' in Sheet1")
        seen_ids.add(question_id)

        records.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "question": question,
                "reference_answer": reference_answer,
                "eval_focus": "",
                "source_sheet": SHEET_NAME,
                "row_number": row_number,
            }
        )

    return records


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Sheet1 from Excel to JSON question set")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input Excel path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    records = convert_sheet1_to_records(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Converted {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
