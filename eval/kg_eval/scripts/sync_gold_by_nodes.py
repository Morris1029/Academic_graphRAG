import json
from pathlib import Path

# Gold Standard Path
GOLD_PATH = Path(r"d:\WorkFile\PthonProject\Pycharm\academic-graphrag\eval\kg_eval\dataset\AIGC-EDU-kgval.gold.json")
BACKUP_PATH = Path(r"d:\WorkFile\PthonProject\Pycharm\academic-graphrag\eval\kg_eval\dataset\AIGC-EDU-kgval.gold.json.sync_bak")

def sync_gold_by_entity_types():
    if not GOLD_PATH.exists():
        print(f"Error: {GOLD_PATH} not found.")
        return

    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backup before syncing
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Created pre-sync backup at: {BACKUP_PATH}")

    deleted_triples_count = 0
    deleted_attributes_count = 0

    for sample in data:
        if "kg_eval" not in sample or "gold" not in sample["kg_eval"]:
            continue
        
        extraction = sample["kg_eval"]["gold"].get("extraction", {})
        # Source of truth: entities defined in entity_types
        allowed_nodes = set(extraction.get("entity_types", {}).keys())
        
        # 1. Clean Triples
        original_triples = extraction.get("triples", [])
        filtered_triples = []
        for h, r, t in original_triples:
            if h in allowed_nodes and t in allowed_nodes:
                filtered_triples.append([h, r, t])
            else:
                deleted_triples_count += 1
        extraction["triples"] = filtered_triples

        # 2. Clean Attributes
        original_attributes = extraction.get("attributes", {})
        filtered_attributes = {}
        for node_name, attrs in original_attributes.items():
            if node_name in allowed_nodes:
                filtered_attributes[node_name] = attrs
            else:
                deleted_attributes_count += 1
        extraction["attributes"] = filtered_attributes

    # Write back to file
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print("Sync cleanup complete!")
    print(f"Removed {deleted_triples_count} invalid triples.")
    print(f"Removed {deleted_attributes_count} invalid attribute entries.")
    print("All entities in triples/attributes now match entity_types exactly.")

if __name__ == "__main__":
    sync_gold_by_entity_types()
