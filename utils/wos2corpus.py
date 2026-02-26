import json
import os
from typing import List, Dict, Any
import hashlib

INPUT_FILE = "D:/STUDY MO/Research/Data/Test-CNKI.txt"
OUTPUT_DIR = "data/uploaded/wos_paper_dataset"
OUTPUT_FILE = "cnki.json"


def normalize_id(text: str) -> str:
    """为 paper 生成稳定 ID（避免 Python hash 不稳定）"""
    return "paper_" + hashlib.md5(text.encode("utf-8")).hexdigest()


def parse_cnki_txt(file_path: str) -> List[Dict[str, Any]]:
    records = []
    current = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            field_map = {
                "SrcDatabase-来源库": "database",
                "Title-题名": "title",
                "Author-作者": "authors",
                "Organ-单位": "organ",
                "Source-文献来源": "source",
                "Keyword-关键词": "keywords",
                "Summary-摘要": "abstract",
                "PubTime-发表时间": "year",
            }

            if key in field_map:
                current[field_map[key]] = value

    if current:
        records.append(current)

    return records

def build_meta_only_corpus(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus = []

    for r in records:
        title = r.get("title", "").strip()
        abstract = r.get("abstract", "").strip()

        if not title:
            continue

        paper_id = normalize_id(title)

        corpus.append({
            "id": paper_id,
            "meta": {
                "title": title,
                "authors": r.get("authors", ""),
                "organ": r.get("organ", ""),
                "source": r.get("source", ""),
                "keywords": r.get("keywords", ""),
                "abstract": abstract,
                "year": r.get("year", ""),
                "database": r.get("database", "")
            }
        })

    return corpus


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Parsing input file: {INPUT_FILE}")
    records = parse_cnki_txt(INPUT_FILE)
    print(f"Loaded {len(records)} raw papers")

    corpus = build_meta_only_corpus(records)
    print(f"Built {len(corpus)} meta-only corpus items")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved corpus to {output_path}")


if __name__ == "__main__":
    main()
