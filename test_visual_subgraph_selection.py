from __future__ import annotations

import unittest

from backend import select_connected_visual_subgraph


class VisualSubgraphSelectionTests(unittest.TestCase):
    def test_selection_prefers_structural_backbone_over_loose_nodes(self):
        nodes_dict = {
            "paper_1": {"id": "paper_1", "category": "论文", "name": "论文1"},
            "author_1": {"id": "author_1", "category": "作者", "name": "作者1"},
            "journal_1": {"id": "journal_1", "category": "期刊", "name": "期刊1"},
            "org_1": {"id": "org_1", "category": "机构", "name": "机构1"},
            "attr_1": {"id": "attr_1", "category": "属性", "name": "年份: 2025"},
            "topic_1": {"id": "topic_1", "category": "研究主题", "name": "研究主题1"},
        }
        links = [
            {"source": "author_1", "target": "paper_1", "name": "撰写", "value": 1},
            {"source": "paper_1", "target": "journal_1", "name": "发表于", "value": 1},
            {"source": "author_1", "target": "org_1", "name": "隶属", "value": 1},
            {"source": "paper_1", "target": "attr_1", "name": "has_attribute", "value": 1},
            {"source": "paper_1", "target": "topic_1", "name": "聚焦", "value": 1},
        ]
        node_degree = {
            "paper_1": 4,
            "author_1": 2,
            "journal_1": 1,
            "org_1": 1,
            "attr_1": 1,
            "topic_1": 1,
        }

        kept_nodes, kept_links = select_connected_visual_subgraph(
            nodes_dict,
            links,
            node_degree,
            max_nodes=4,
            max_links=3,
        )

        kept_ids = {node["id"] for node in kept_nodes}
        self.assertEqual(kept_ids, {"paper_1", "author_1", "journal_1", "org_1"})
        self.assertEqual(len(kept_links), 3)
        self.assertFalse(any(node["id"] == "attr_1" for node in kept_nodes))


if __name__ == "__main__":
    unittest.main()
