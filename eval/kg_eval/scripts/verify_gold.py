import json
from pathlib import Path

GOLD_PATH = Path(r"d:\WorkFile\PthonProject\Pycharm\academic-graphrag\eval\kg_eval\dataset\AIGC-EDU-kgval.gold.json")

def verify_gold():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    reversed_found = 0
    inconsistent_nodes = 0
    total_triples = 0

    swap_rules = [
        ("研究方法", "采用", "论文"),
        ("研究主题", "提出", "论文"),
        ("研究主题", "聚焦", "论文"),
        ("机构", "隶属", "作者"),
        ("论文", "撰写", "作者"),
        ("期刊", "发表于", "论文"),
    ]

    for sample in data:
        extraction = sample.get("kg_eval", {}).get("gold", {}).get("extraction", {})
        entity_types = extraction.get("entity_types", {})
        triples = extraction.get("triples", [])
        
        total_triples += len(triples)
        
        for h, r, t in triples:
            if h not in entity_types or t not in entity_types:
                inconsistent_nodes += 1
            else:
                h_type = entity_types[h]
                t_type = entity_types[t]
                for r_h_type, r_rel, r_t_type in swap_rules:
                    if h_type == r_h_type and r == r_rel and t_type == r_t_type:
                        reversed_found += 1
                        print(f"Found reversed: ({h}:{h_type}, {r}, {t}:{t_type})")

    print("-" * 30)
    print(f"Total Triples check: {total_triples}")
    print(f"Still Reversed: {reversed_found}")
    print(f"Inconsistent Nodes (h/t not in types): {inconsistent_nodes}")

if __name__ == "__main__":
    verify_gold()
