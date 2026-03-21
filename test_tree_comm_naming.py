from __future__ import annotations

import unittest

import networkx as nx

from utils.tree_comm import FastTreeComm


def build_tree_comm_for_test() -> FastTreeComm:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "paper_1",
        properties={"name": "生成式人工智能赋能课堂教学的价值透视与实践思考", "schema_type": "论文"},
    )
    graph.add_node(
        "paper_2",
        properties={"name": "AIGC时代课堂教学创新路径研究", "schema_type": "论文"},
    )
    graph.add_node(
        "tech_1",
        properties={"name": "生成式人工智能", "schema_type": "技术"},
    )
    graph.add_node(
        "topic_1",
        properties={"name": "课堂教学", "schema_type": "研究主题"},
    )

    tree_comm = FastTreeComm.__new__(FastTreeComm)
    tree_comm.graph = graph
    tree_comm.node_names = {node_id: graph.nodes[node_id]["properties"]["name"] for node_id in graph.nodes()}
    tree_comm.degree_cache = {node_id: graph.degree(node_id) for node_id in graph.nodes()}
    tree_comm.struct_weight = 0.3
    tree_comm.llm_client = None
    return tree_comm


class TreeCommNamingTests(unittest.TestCase):
    def test_fallback_name_prefers_concepts_over_paper_titles(self):
        tree_comm = build_tree_comm_for_test()
        members = ["paper_1", "tech_1", "topic_1", "paper_2"]

        name = tree_comm._build_community_display_name(members)

        self.assertEqual(name, "生成式人工智能 · 课堂教学")
        self.assertNotIn("价值透视", name)
        self.assertNotIn("创新路径研究", name)

    def test_valid_llm_name_is_used_for_community_node(self):
        tree_comm = build_tree_comm_for_test()
        tree_comm.llm_client = object()
        tree_comm._call_llm_api_batch = lambda _: [
            {
                "id": "7",
                "name": "生成式人工智能赋能课堂教学",
                "summary": "该社区聚焦生成式人工智能与课堂教学融合的主要路径和应用方向。",
            }
        ]

        tree_comm.create_super_nodes({"7": ["paper_1", "tech_1", "topic_1"]}, level=4)

        props = tree_comm.graph.nodes["comm_4_7"]["properties"]
        self.assertEqual(props["name"], "生成式人工智能赋能课堂教学")
        self.assertIn("课堂教学融合", props["description"])

    def test_invalid_llm_name_falls_back_to_filtered_concepts(self):
        tree_comm = build_tree_comm_for_test()
        tree_comm.llm_client = object()
        tree_comm._call_llm_api_batch = lambda _: [
            {
                "id": "9",
                "name": "生成式人工智能赋能课堂教学的价值透视与实践思考 / AIGC时代课堂教学创新路径研究",
                "summary": "",
            }
        ]

        tree_comm.create_super_nodes({"9": ["paper_1", "tech_1", "topic_1", "paper_2"]}, level=4)

        props = tree_comm.graph.nodes["comm_4_9"]["properties"]
        self.assertEqual(props["name"], "生成式人工智能 · 课堂教学")
        self.assertIn("生成式人工智能", props["description"])
        self.assertNotIn(" / ", props["name"])


if __name__ == "__main__":
    unittest.main()
