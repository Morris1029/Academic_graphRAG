import json
import os
from pathlib import Path

# Paths
GOLD_PATH = Path(r"d:\WorkFile\PthonProject\Pycharm\academic-graphrag\eval\kg_eval\dataset\AIGC-EDU-kgval.gold.json")
OUTPUT_PATH = GOLD_PATH
BACKUP_PATH = GOLD_PATH.with_suffix(".bak_v2")

def refine_gold_standard():
    if not GOLD_PATH.exists():
        print(f"File not found: {GOLD_PATH}")
        return

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backup
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Created backup at {BACKUP_PATH}")

    stats = {
        "swapped": 0,
        "removed_triples": 0,
        "removed_attributes": 0,
        "normalized": 0
    }

    # Define swap rules: (H_type, R, T_type) -> (T, R, H)
    swap_rules = [
        ("研究方法", "采用", "论文"),
        ("研究主题", "提出", "论文"),
        ("研究主题", "聚焦", "论文"),
        ("机构", "隶属", "作者"),
        ("论文", "撰写", "作者"),
        ("期刊", "发表于", "论文"),
    ]

    for sample in data:
        if "kg_eval" not in sample or "gold" not in sample["kg_eval"]:
            continue
        
        extraction = sample["kg_eval"]["gold"].get("extraction", {})
        entity_types = extraction.get("entity_types", {})
        
        # 1. Normalization of entity names (minimal, mostly whitespace)
        # We assume the user has already handled punctuation manually, but we ensure consistency.
        
        # 2. Fix Triples
        original_triples = extraction.get("triples", [])
        new_triples = []
        for h, r, t in original_triples:
            # Check if entities exist in entity_types (Request 6 & 7)
            if h not in entity_types or t not in entity_types:
                stats["removed_triples"] += 1
                continue
            
            h_type = entity_types[h]
            t_type = entity_types[t]
            
            # Check for reversal (Request 5)
            should_swap = False
            for r_h_type, r_rel, r_t_type in swap_rules:
                if h_type == r_h_type and r == r_rel and t_type == r_t_type:
                    should_swap = True
                    break
            
            if should_swap:
                new_triples.append([t, r, h])
                stats["swapped"] += 1
            else:
                new_triples.append([h, r, t])
        
        extraction["triples"] = new_triples

        # 3. Clean Attributes
        original_attributes = extraction.get("attributes", {})
        new_attributes = {}
        for node, attrs in original_attributes.items():
            if node in entity_types:
                new_attributes[node] = attrs
            else:
                stats["removed_attributes"] += 1
        extraction["attributes"] = new_attributes

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\nRefinement Complete!")
    print(f"Swapped {stats['swapped']} reversed triples.")
    print(f"Removed {stats['removed_triples']} triples with missing nodes.")
    print(f"Removed {stats['removed_attributes']} attribute entries with missing nodes.")
    print(f"Updated gold standard saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    refine_gold_standard()
