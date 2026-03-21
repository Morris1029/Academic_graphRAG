from __future__ import annotations

import unittest

import networkx as nx

from models.retriever.faiss_filter import DualFAISSRetriever
from utils.tree_comm import FastTreeComm


def build_tree_comm() -> FastTreeComm:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "paper_1",
        properties={"name": "Paper About Generative AI in Education", "schema_type": "论文"},
    )
    graph.add_node(
        "tech_1",
        properties={"name": "Generative AI", "schema_type": "技术"},
    )
    graph.add_node(
        "topic_1",
        properties={"name": "Classroom Teaching", "schema_type": "研究主题"},
    )

    tree_comm = FastTreeComm.__new__(FastTreeComm)
    tree_comm.graph = graph
    tree_comm.node_names = {node_id: graph.nodes[node_id]["properties"]["name"] for node_id in graph.nodes()}
    tree_comm.degree_cache = {node_id: graph.degree(node_id) for node_id in graph.nodes()}
    tree_comm.struct_weight = 0.3
    tree_comm.llm_client = None
    return tree_comm


def build_retriever() -> DualFAISSRetriever:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "entity_1",
        label="entity",
        level=2,
        community_l4="comm_4_1",
        properties={"name": "Generative AI", "description": "Core technology"},
    )
    graph.add_node(
        "attr_1",
        label="attribute",
        level=1,
        community_l4="comm_4_1",
        properties={"name": "关键词: learning outcome"},
    )
    graph.add_node(
        "comm_4_1",
        label="community",
        level=4,
        properties={"name": "AI Teaching", "description": "Focuses on classroom use"},
    )

    retriever = DualFAISSRetriever.__new__(DualFAISSRetriever)
    retriever.graph = graph
    return retriever


class RemoveKeywordNodesTests(unittest.TestCase):
    def test_create_super_nodes_with_keywords_keeps_backward_compatibility_without_keyword_nodes(self):
        tree_comm = build_tree_comm()

        super_nodes, keyword_mapping = tree_comm.create_super_nodes_with_keywords(
            {"1": ["paper_1", "tech_1", "topic_1"]},
            level=4,
        )

        self.assertIn("comm_4_1", super_nodes)
        self.assertEqual(keyword_mapping, {})
        self.assertFalse(any(data.get("label") == "keyword" for _, data in tree_comm.graph.nodes(data=True)))

    def test_retriever_formats_level1_attributes_as_attributes_not_keywords(self):
        retriever = build_retriever()

        text = retriever._nodes_to_text(["attr_1", "comm_4_1"])

        self.assertIn("=== Attribute Information ===", text)
        self.assertIn("关键词: learning outcome", text)
        self.assertNotIn("=== Keyword Information ===", text)


if __name__ == "__main__":
    unittest.main()
