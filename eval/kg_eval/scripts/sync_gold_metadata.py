import json
import os
from pathlib import Path

# Config
GOLD_FILE = Path("eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json")
BACKUP_FILE = GOLD_FILE.with_suffix(".json.bak")

STRUCTURED_RELATIONS = {"撰写", "隶属", "发表于"}
STRUCTURED_ATTR_PREFIXES = ("年份:", "关键词:", "来源:")

def sync_metadata():
    if not GOLD_FILE.exists():
        print(f"Error: {GOLD_FILE} not found.")
        return

    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Syncing {len(data)} records...")

    for record in data:
        meta = record.get("meta", {})
        title = meta.get("title", "").strip()
        authors_raw = meta.get("authors", "")
        organs_raw = meta.get("organ", "")
        source = meta.get("source", "").strip()
        year = meta.get("year", "").strip()
        keywords_raw = meta.get("keywords", "")

        # 1. Clean existing structured data
        ext = record.setdefault("kg_eval", {}).setdefault("gold", {}).setdefault("extraction", {})
        
        # Filter triples
        if "triples" in ext:
            ext["triples"] = [t for t in ext["triples"] if len(t) == 3 and t[1] not in STRUCTURED_RELATIONS]
        else:
            ext["triples"] = []
            
        # Filter attributes
        if "attributes" in ext:
            new_attrs = {}
            for node, attrs in ext["attributes"].items():
                filtered = [a for a in attrs if not any(a.startswith(p) for p in STRUCTURED_ATTR_PREFIXES)]
                if filtered:
                    new_attrs[node] = filtered
            ext["attributes"] = new_attrs
        else:
            ext["attributes"] = {}

        # Ensure entity_types exists
        entity_types = ext.setdefault("entity_types", {})

        # 2. Add New Structured Triples
        # Author -> Paper ("撰写")
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
        for a in authors:
            ext["triples"].append([a, "撰写", title])
            entity_types[a] = "作者"
        
        if title:
            entity_types[title] = "论文"

        # Author -> Organ ("隶属")
        organs = [o.strip() for o in organs_raw.split(";") if o.strip()]
        for idx, author_name in enumerate(authors):
            if not organs:
                continue
            
            if len(organs) == 1:
                target_orgs = organs
            elif idx < len(organs):
                target_orgs = [organs[idx]]
            else:
                target_orgs = [organs[0]]
            
            for org in target_orgs:
                ext["triples"].append([author_name, "隶属", org])
                entity_types[org] = "机构"

        # Paper -> Source ("发表于")
        if title and source:
            ext["triples"].append([title, "发表于", source])
            entity_types[source] = "期刊"

        # 3. Add New Structured Attributes
        # Year
        if title and year:
            node_attrs = ext["attributes"].setdefault(title, [])
            node_attrs.append(f"年份: {year}")

        # Keywords
        keywords = [k.strip() for k in keywords_raw.split(";") if k.strip()]
        for kw in keywords:
            if title:
                node_attrs = ext["attributes"].setdefault(title, [])
                if f"关键词: {kw}" not in node_attrs:
                    node_attrs.append(f"关键词: {kw}")

    # Save Updated Data
    with open(GOLD_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully updated {GOLD_FILE}")

if __name__ == "__main__":
    sync_metadata()
