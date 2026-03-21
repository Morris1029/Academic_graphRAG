from __future__ import annotations

import re
import threading
import time
import unittest
from collections import deque
from unittest.mock import patch

import networkx as nx

from config import ConfigManager
from utils.tree_comm import FastTreeComm


def build_tree_comm_for_test(
    *,
    llm_batch_size: int = 2,
    max_concurrent_llm_requests: int = 2,
    requests_per_minute: int = 120,
) -> FastTreeComm:
    graph = nx.MultiDiGraph()
    preferred_type = next(iter(FastTreeComm.COMMUNITY_NAMING_PREFERRED_TYPES))

    for index in range(1, 13):
        graph.add_node(
            f"node_{index}",
            properties={
                "name": f"concept_{index}",
                "schema_type": preferred_type,
            },
        )

    tree_comm = FastTreeComm.__new__(FastTreeComm)
    tree_comm.graph = graph
    tree_comm.config = None
    tree_comm.node_names = {
        node_id: graph.nodes[node_id]["properties"]["name"]
        for node_id in graph.nodes()
    }
    tree_comm.degree_cache = {node_id: graph.degree(node_id) for node_id in graph.nodes()}
    tree_comm.struct_weight = 0.3
    tree_comm.llm_client = object()
    tree_comm.community_llm_batch_size = llm_batch_size
    tree_comm.max_concurrent_llm_requests = max_concurrent_llm_requests
    tree_comm.requests_per_minute = requests_per_minute
    tree_comm.rate_limit_lock = threading.Lock()
    tree_comm.request_times = deque()
    return tree_comm


class TreeCommConcurrencyTests(unittest.TestCase):
    def test_tree_comm_config_parses_concurrency_defaults(self):
        config = ConfigManager("config/base_config.yaml")

        self.assertEqual(config.tree_comm.llm_batch_size, 8)
        self.assertEqual(config.tree_comm.max_concurrent_llm_requests, 4)
        self.assertEqual(config.tree_comm.requests_per_minute, 240)

    def test_create_super_nodes_runs_batches_concurrently_and_preserves_graph_integrity(self):
        tree_comm = build_tree_comm_for_test(
            llm_batch_size=2,
            max_concurrent_llm_requests=2,
            requests_per_minute=120,
        )
        communities = {
            "1": ["node_1", "node_2"],
            "2": ["node_3", "node_4"],
            "3": ["node_5", "node_6"],
            "4": ["node_7", "node_8"],
            "5": ["node_9", "node_10"],
            "6": ["node_11", "node_12"],
        }
        call_lock = threading.Lock()
        active_calls = 0
        peak_calls = 0

        def fake_call(content: str):
            nonlocal active_calls, peak_calls
            ids = re.findall(r'"id":\s*"([^"]+)"', content)
            with call_lock:
                active_calls += 1
                peak_calls = max(peak_calls, active_calls)
            try:
                time.sleep(0.08)
                return [
                    {"id": community_id, "name": f"community_{community_id}", "summary": f"summary_{community_id}"}
                    for community_id in ids
                ]
            finally:
                with call_lock:
                    active_calls -= 1

        tree_comm._call_llm_api_batch = fake_call

        super_nodes = tree_comm.create_super_nodes(communities, level=4, batch_size=None)

        self.assertEqual(peak_calls, 2)
        self.assertEqual(len(super_nodes), len(communities))
        for community_id, members in communities.items():
            super_node_id = f"comm_4_{community_id}"
            self.assertIn(super_node_id, tree_comm.graph.nodes)
            self.assertEqual(tree_comm.graph.nodes[super_node_id]["properties"]["name"], f"community_{community_id}")
            for member in members:
                self.assertTrue(tree_comm.graph.has_edge(member, super_node_id))

    def test_wait_for_llm_slot_respects_requests_per_minute(self):
        tree_comm = build_tree_comm_for_test(requests_per_minute=1)
        tree_comm.request_times = deque([0.0])

        with patch("utils.tree_comm.time.monotonic", side_effect=[0.1, 60.2]), patch(
            "utils.tree_comm.time.sleep"
        ) as sleep_mock:
            tree_comm._wait_for_llm_slot()

        self.assertTrue(sleep_mock.called)
        sleep_for = sleep_mock.call_args[0][0]
        self.assertGreater(sleep_for, 59.8)
        self.assertLess(sleep_for, 60.0)
        self.assertEqual(len(tree_comm.request_times), 1)
        self.assertAlmostEqual(tree_comm.request_times[0], 60.2, places=1)

    def test_failed_batch_uses_fallback_name_and_summary(self):
        tree_comm = build_tree_comm_for_test(
            llm_batch_size=1,
            max_concurrent_llm_requests=1,
            requests_per_minute=120,
        )
        tree_comm._call_llm_api_batch = lambda _: (_ for _ in ()).throw(RuntimeError("boom"))

        super_nodes = tree_comm.create_super_nodes(
            {"7": ["node_1", "node_2"]},
            level=4,
            batch_size=None,
        )

        self.assertIn("comm_4_7", super_nodes)
        props = tree_comm.graph.nodes["comm_4_7"]["properties"]
        self.assertIn("concept_1", props["name"])
        self.assertIn("concept_2", props["name"])
        self.assertTrue(props["description"])


if __name__ == "__main__":
    unittest.main()
