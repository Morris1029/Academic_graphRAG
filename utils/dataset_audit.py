import hashlib
import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


PAPER_SCHEMA_TYPE = "\u8bba\u6587"
CHUNK_AUDIT_SUFFIX = "_chunk_audit.jsonl"


def _safe_load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _extract_doc_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    meta = doc.get("meta", {})
    return meta if isinstance(meta, dict) else {}


def build_document_uid(doc: Dict[str, Any], duplicate_doc_ids: Optional[Iterable[str]] = None) -> str:
    duplicate_doc_ids = set(duplicate_doc_ids or [])
    base_id = str(doc.get("id", "")).strip()
    if not base_id:
        meta = _extract_doc_meta(doc)
        title = str(meta.get("title", "")).strip()
        if title:
            digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
            return f"paper_{digest}"
        return "unknown"

    if base_id not in duplicate_doc_ids:
        return base_id

    meta = _extract_doc_meta(doc)
    signature = "|".join(
        [
            base_id,
            str(meta.get("title", "")).strip(),
            str(meta.get("source", "")).strip(),
            str(meta.get("year", "")).strip(),
            str(meta.get("authors", "")).strip(),
        ]
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
    return f"{base_id}__{digest}"


def get_dataset_paths(dataset_name: str) -> Dict[str, str]:
    base_uploaded = os.path.join("data", "uploaded", dataset_name)
    return {
        "corpus": os.path.join(base_uploaded, "corpus.json"),
        "chunk_text": os.path.join("output", "chunks", f"{dataset_name}.txt"),
        "chunk_audit": os.path.join("output", "chunks", f"{dataset_name}{CHUNK_AUDIT_SUFFIX}"),
        "graph": os.path.join("output", "graphs", f"{dataset_name}_new.json"),
        "stats": os.path.join("output", "graphs", f"{dataset_name}_construction_stats.json"),
    }


def load_chunk_records(chunk_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(chunk_path):
        return []

    records: List[Dict[str, str]] = []
    pattern = re.compile(r"^id:\s*(.*?)\tChunk:\s*(.*)$")
    title_pattern = re.compile(r"Title:\s*(.*?)(?:\\n|\n)\s*Abstract:", re.IGNORECASE | re.DOTALL)

    with open(chunk_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                records.append({"doc_uid": "", "title": "", "raw": line})
                continue
            doc_uid = match.group(1).strip()
            chunk_text = match.group(2)
            title_match = title_pattern.search(chunk_text)
            title = title_match.group(1).strip() if title_match else ""
            records.append({"doc_uid": doc_uid, "title": title, "raw": chunk_text})
    return records


def load_chunk_audit_records(chunk_audit_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(chunk_audit_path):
        return []

    records: List[Dict[str, Any]] = []
    with open(chunk_audit_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def collect_graph_paper_records(graph_path: str) -> List[Dict[str, Any]]:
    graph_data = _safe_load_json(graph_path)
    if not isinstance(graph_data, list):
        return []

    paper_records: List[Dict[str, Any]] = []
    seen = set()

    for item in graph_data:
        if not isinstance(item, dict):
            continue
        for side in ("start_node", "end_node"):
            node = item.get(side) or {}
            props = node.get("properties") or {}
            schema_type = str(props.get("schema_type", "")).strip()
            if schema_type != PAPER_SCHEMA_TYPE:
                continue
            name = str(props.get("name", "")).strip()
            doc_uid = str(props.get("doc_uid", props.get("chunk_id", props.get("chunk id", "")))).strip()
            source_doc_id = str(props.get("source_doc_id", "")).strip()
            identity = (doc_uid, name, source_doc_id)
            if identity in seen:
                continue
            seen.add(identity)
            paper_records.append(
                {
                    "doc_uid": doc_uid,
                    "title": name,
                    "source_doc_id": source_doc_id,
                    "schema_type": schema_type,
                }
            )
    return paper_records


def audit_dataset(dataset_name: str) -> Dict[str, Any]:
    paths = get_dataset_paths(dataset_name)
    corpus = _safe_load_json(paths["corpus"])
    source_docs = corpus if isinstance(corpus, list) else []

    source_id_counter = Counter(str(doc.get("id", "")).strip() for doc in source_docs if isinstance(doc, dict))
    duplicate_id_counts = {doc_id: count for doc_id, count in source_id_counter.items() if doc_id and count > 1}
    duplicate_doc_ids = set(duplicate_id_counts.keys())

    source_records: List[Dict[str, Any]] = []
    for doc in source_docs:
        if not isinstance(doc, dict):
            continue
        meta = _extract_doc_meta(doc)
        title = str(meta.get("title", "")).strip()
        source_doc_id = str(doc.get("id", "")).strip()
        doc_uid = build_document_uid(doc, duplicate_doc_ids)
        source_records.append(
            {
                "doc_uid": doc_uid,
                "source_doc_id": source_doc_id,
                "title": title,
                "year": str(meta.get("year", "")).strip(),
                "source": str(meta.get("source", "")).strip(),
                "authors": str(meta.get("authors", "")).strip(),
            }
        )

    chunk_records = load_chunk_records(paths["chunk_text"])
    chunk_audit_records = load_chunk_audit_records(paths["chunk_audit"])
    graph_paper_records = collect_graph_paper_records(paths["graph"])

    source_titles = [record["title"] for record in source_records if record["title"]]
    chunk_titles = [record["title"] for record in chunk_records if record.get("title")]
    graph_titles = [record["title"] for record in graph_paper_records if record.get("title")]

    source_title_set = set(source_titles)
    graph_title_set = set(graph_titles)
    chunk_doc_uids = {record["doc_uid"] for record in chunk_records if record.get("doc_uid")}
    chunk_audit_doc_uids = {str(record.get("doc_uid", "")).strip() for record in chunk_audit_records if record.get("doc_uid")}
    graph_doc_uids = {str(record.get("doc_uid", "")).strip() for record in graph_paper_records if record.get("doc_uid")}
    source_doc_uids = {record["doc_uid"] for record in source_records if record.get("doc_uid")}

    duplicate_title_counter = Counter(title for title in source_titles if title)
    duplicate_titles = {title: count for title, count in duplicate_title_counter.items() if count > 1}

    return {
        "dataset_name": dataset_name,
        "paths": paths,
        "source_docs": len(source_records),
        "chunk_records": len(chunk_records),
        "chunk_audit_records": len(chunk_audit_records),
        "graph_papers": len(graph_paper_records),
        "missing_titles": sorted(source_title_set - graph_title_set),
        "extra_titles": sorted(graph_title_set - source_title_set),
        "duplicate_ids": duplicate_id_counts,
        "duplicate_titles": duplicate_titles,
        "missing_doc_uids_in_chunks": sorted(source_doc_uids - chunk_doc_uids),
        "missing_doc_uids_in_chunk_audit": sorted(source_doc_uids - chunk_audit_doc_uids),
        "missing_doc_uids_in_graph": sorted(uid for uid in source_doc_uids - graph_doc_uids if uid),
        "graph_title_count": len(graph_title_set),
        "source_title_count": len(source_title_set),
        "chunk_title_count": len(set(chunk_titles)),
        "samples": {
            "missing_titles": sorted(source_title_set - graph_title_set)[:20],
            "extra_titles": sorted(graph_title_set - source_title_set)[:20],
            "missing_doc_uids_in_chunks": sorted(source_doc_uids - chunk_doc_uids)[:20],
            "missing_doc_uids_in_graph": sorted(uid for uid in source_doc_uids - graph_doc_uids if uid)[:20],
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit a Youtu-GraphRAG dataset.")
    parser.add_argument("dataset_name", help="Dataset name under data/uploaded/<dataset_name>")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.dataset_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
