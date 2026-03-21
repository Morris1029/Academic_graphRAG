import json
import re
import threading
import time
import warnings
from collections import defaultdict, deque
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import json_repair
import os
# 设置镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from utils import call_llm_api
from utils.logger import logger


warnings.filterwarnings('ignore')

try:
    from config import get_config
except ImportError:
    get_config = None


class FastTreeComm:
    COMMUNITY_NAMING_PREFERRED_TYPES = {
        "技术",
        "研究主题",
        "教学场景",
        "教育领域",
        "研究方法",
    }
    COMMUNITY_NAMING_EXCLUDED_TYPES = {
        "论文",
        "作者",
        "机构",
        "期刊",
    }

    def __init__(self, graph, embedding_model="all-MiniLM-L6-v2", struct_weight=0.3, config=None):
        """
        :param graph: Input graph (NetworkX DiGraph)
        :param embedding_model: Sentence embedding model
        :param struct_weight: Structural similarity weight (float between 0 and 1)
        :param config: Configuration object (optional)
        """
        if config is None and get_config is not None:
            try:
                config = get_config()
            except:
                config = None
        self.config = config
        self.graph = graph

        if config:
            embedding_model = embedding_model or config.tree_comm.embedding_model
            struct_weight = struct_weight if struct_weight != 0.3 else config.tree_comm.struct_weight
        
        self.model = SentenceTransformer(embedding_model)
        self.triple_text_max_chars = max(
            64,
            int(getattr(config.tree_comm, "triple_text_max_chars", 360)) if config else 360,
        )
        self.max_triples_per_node = max(
            1,
            int(getattr(config.tree_comm, "max_triples_per_node", 8)) if config else 8,
        )
        self.community_llm_batch_size = max(
            1,
            int(getattr(config.tree_comm, "llm_batch_size", 8)) if config else 8,
        )
        self.max_concurrent_llm_requests = max(
            1,
            int(getattr(config.tree_comm, "max_concurrent_llm_requests", 4)) if config else 4,
        )
        self.requests_per_minute = max(
            0,
            int(getattr(config.tree_comm, "requests_per_minute", 240)) if config else 240,
        )
        self.rate_limit_lock = threading.Lock()
        self.request_times = deque()
        self.semantic_cache = {}
        self.struct_weight = struct_weight
        self.node_list = list(graph.nodes())
        self.node_names = {n: graph.nodes[n]["properties"]["name"] for n in graph.nodes()}
        self.neighbor_cache = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.edge_relations = {(u, v): data.get("relation", "related_to") 
                          for u, v, data in graph.edges(data=True)}
        
        self.triple_strings_cache = {}
        self.degree_cache = {n: self.graph.degree(n) for n in self.node_list}

        self.adjacency_sparse = self._build_sparse_adjacency()

        self._precompute_all_triples()
        
        self.llm_client = call_llm_api.LLMCompletionCall(scope="kg")

    def _build_sparse_adjacency(self):
        n = len(self.node_list)
        node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        row, col = [], []
        
        for node in self.node_list:
            i = node_to_idx[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor in node_to_idx:
                    j = node_to_idx[neighbor]
                    row.append(i)
                    col.append(j)
        
        data = [1] * len(row)
        return sp.csr_matrix((data, (row, col)), shape=(n, n))

    def _precompute_all_triples(self):
        for node_id in self.node_list:
            self.triple_strings_cache[node_id] = self._get_triple_strings(node_id)
        
        return

    def _get_triple_strings(self, node_id):
        """extract all neighbors for one node, enhance the structural perception with 1-hop neighbors"""
        if node_id in self.triple_strings_cache:
            return self.triple_strings_cache[node_id]
            
        node_name = self.graph.nodes[node_id]["properties"]["name"]
        triples = []
        
        for neighbor in self.graph.neighbors(node_id):
            rel = self.graph.edges[node_id, neighbor, 0].get("relation", "related_to")
            neighbor_name = self.graph.nodes[neighbor]["properties"]["name"]
            triples.append(f"{node_name} {rel} {neighbor_name}")
            
        result = list(set(triples))
        self.triple_strings_cache[node_id] = result
        return result

    def _prepare_triple_text_for_embedding(self, triples: List[str], fallback_name: str) -> Tuple[str, int, int]:
        """Build a bounded text for local embedding to avoid silent model truncation."""
        selected_triples = [str(triple).strip() for triple in (triples or []) if str(triple).strip()]
        selected_triples = selected_triples[:self.max_triples_per_node]
        text = " ".join(selected_triples) if selected_triples else str(fallback_name or "")
        text = " ".join(text.split())
        original_length = len(text)
        if original_length > self.triple_text_max_chars:
            text = text[: self.triple_text_max_chars].rstrip()
        return text or str(fallback_name or ""), original_length, len(text)

    def get_triple_embedding(self, node_id):
        """leverage triple-level embedding to represent one node"""
        if node_id not in self.semantic_cache:
            triples = self.triple_strings_cache.get(node_id, [])
            fallback_name = self.graph.nodes[node_id]["properties"]["name"]
            text, original_length, processed_length = self._prepare_triple_text_for_embedding(triples, fallback_name)
            if processed_length < original_length:
                logger.info(
                    "TreeComm truncated embedding text for node %s: %d -> %d chars",
                    node_id,
                    original_length,
                    processed_length,
                )
            self.semantic_cache[node_id] = self.model.encode(text)
        return self.semantic_cache[node_id]
    
    def get_triple_embeddings_batch(self, node_ids):
        """Batch processing for GPU acceleration with optimized caching"""
        uncached_ids = [nid for nid in node_ids if nid not in self.semantic_cache]
        
        if uncached_ids:
            texts = []
            truncated_count = 0
            for nid in uncached_ids:
                triples = self.triple_strings_cache.get(nid, [])
                text, original_length, processed_length = self._prepare_triple_text_for_embedding(
                    triples,
                    self.node_names[nid],
                )
                if processed_length < original_length:
                    truncated_count += 1
                texts.append(text)
            if truncated_count:
                logger.info(
                    "TreeComm truncated %d/%d embedding texts in batch (max_chars=%d, max_triples=%d)",
                    truncated_count,
                    len(uncached_ids),
                    self.triple_text_max_chars,
                    self.max_triples_per_node,
                )
            
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=128)
                
            for nid, emb in zip(uncached_ids, embeddings):
                self.semantic_cache[nid] = emb.cpu().numpy()
        return np.array([self.semantic_cache[nid] for nid in node_ids])

    def _compute_jaccard_matrix_vectorized(self, level_nodes):

        node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        level_indices = [node_to_idx[node] for node in level_nodes if node in node_to_idx]

        if not level_indices:
            return np.zeros((len(level_nodes), len(level_nodes)))

        sub_adj = self.adjacency_sparse[level_indices][:, level_indices]
        intersection = sub_adj.dot(sub_adj.T).toarray()
        row_sums = np.array(sub_adj.sum(axis=1)).flatten()

        union = row_sums[:, None] + row_sums - intersection
        jaccard_matrix = intersection / (union + 1e-9)
        np.fill_diagonal(jaccard_matrix, 1.0)

        return jaccard_matrix

    def _compute_sim_matrix(self, level_nodes):
        start_time = time.time()
        
        node_count = len(level_nodes)
        if node_count <= 1:
            return np.eye(node_count)

        embeddings = self.get_triple_embeddings_batch(level_nodes)
        
        embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        semantic_sim_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

        structural_sim_matrix = self._compute_jaccard_matrix_vectorized(level_nodes)
        
        sim_matrix = (self.struct_weight * structural_sim_matrix + 
                     (1 - self.struct_weight) * semantic_sim_matrix)
        return sim_matrix

    def _fast_clustering(self, level_nodes, n_clusters=None):
        if len(level_nodes) <= 2:
            return {0: level_nodes}
        
        if n_clusters is None:
            base_clusters = len(level_nodes) // 10
            n_clusters = min(max(2, base_clusters), len(level_nodes) // 2, 200)
        
        embeddings = self.get_triple_embeddings_batch(level_nodes)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for node, label in zip(level_nodes, cluster_labels):
            clusters[label].append(node)
        
        return dict(clusters)

    def detect_communities(self, level_nodes, max_iter=1, merge_threshold=0.5, max_total_communities=None):
        if len(level_nodes) <= 1:
            return {0: level_nodes} if level_nodes else {}

        # 从配置中读取 max_total_communities，如果没有配置则使用默认值
        if max_total_communities is None:
            if self.config and hasattr(self.config.tree_comm, 'max_total_communities'):
                max_total_communities = self.config.tree_comm.max_total_communities
            else:
                # 原有的默认逻辑：节点数的1/3，最少5个，最多200个
                max_total_communities = min(max(5, len(level_nodes) // 3), 200)

        initial_clusters = self._fast_clustering(level_nodes)
        final_communities = {}
        comm_id = 0
        
        # 按簇大小排序，优先处理大簇（确保大簇能得到细分机会）
        sorted_clusters = sorted(initial_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        processed_cluster_ids = set()
        
        for cluster_id, cluster_nodes in sorted_clusters:
            processed_cluster_ids.add(cluster_id)
            
            if len(cluster_nodes) <= 3:
                final_communities[comm_id] = cluster_nodes
                comm_id += 1
            else:
                # 检查是否还有剩余配额进行细分
                if len(final_communities) >= max_total_communities:
                    # 配额已满，将剩余簇直接作为社区，不再细分
                    final_communities[comm_id] = cluster_nodes
                    comm_id += 1
                else:
                    sub_communities = self._refine_cluster(cluster_nodes, max_iter, merge_threshold)
                    for sub_comm in sub_communities.values():
                        final_communities[comm_id] = sub_comm
                        comm_id += 1
                        
                        # 如果社区数量已经达到上限，停止细分
                        if len(final_communities) >= max_total_communities:
                            break
                    
                    # 如果达到上限，将剩余未处理的簇直接添加为社区
                    if len(final_communities) >= max_total_communities:
                        for remaining_cluster_id, remaining_nodes in sorted_clusters:
                            if remaining_cluster_id not in processed_cluster_ids:
                                if len(final_communities) < max_total_communities:
                                    final_communities[comm_id] = remaining_nodes
                                    comm_id += 1
                                else:
                                    break
                        break
        
        logger.info(f"Generated {len(final_communities)} communities from {len(level_nodes)} nodes")
        return final_communities

    def _refine_cluster(self, cluster_nodes, max_iter, merge_threshold):
        if len(cluster_nodes) <= 3:
            return {0: cluster_nodes}

        initial_clusters = self._fast_clustering(cluster_nodes)
        
        if len(initial_clusters) == 1:
            return initial_clusters
        
        cluster_centers = {}
        for cluster_id, nodes in initial_clusters.items():
            center = self._compute_community_center(nodes)
            cluster_centers[cluster_id] = center
        
        center_nodes = list(cluster_centers.values())
        center_sim_matrix = self._compute_sim_matrix(center_nodes)
        
        center_to_idx = {center: idx for idx, center in enumerate(center_nodes)}

        current_clusters = initial_clusters.copy()
        current_centers = cluster_centers.copy()
        
        for iteration in range(max_iter):
            changed = False
            
            cluster_ids = list(current_clusters.keys())
            n_clusters = len(cluster_ids)
            
            cluster_similarities = []
            
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    cluster1_id = cluster_ids[i]
                    cluster2_id = cluster_ids[j]
                    
                    center1 = current_centers[cluster1_id]
                    center2 = current_centers[cluster2_id]
                    idx1 = center_to_idx[center1]
                    idx2 = center_to_idx[center2]
                    center_sim = center_sim_matrix[idx1][idx2]
                    
                    if center_sim >= merge_threshold:
                        cluster_similarities.append({
                            'cluster1': cluster1_id,
                            'cluster2': cluster2_id,
                            'similarity': center_sim
                        })
            
            cluster_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            merged_clusters = set()
            new_clusters = {}
            new_centers = {}
            next_cluster_id = 0
            
            for sim_info in cluster_similarities:
                cluster1_id = sim_info['cluster1']
                cluster2_id = sim_info['cluster2']
                
                if cluster1_id not in merged_clusters and cluster2_id not in merged_clusters:

                    if self._should_merge_clusters(
                        current_clusters[cluster1_id], 
                        current_clusters[cluster2_id],
                        sim_info
                    ):
                        merged_nodes = current_clusters[cluster1_id] + current_clusters[cluster2_id]
                        new_clusters[next_cluster_id] = merged_nodes
                        
                        new_center = self._compute_community_center(merged_nodes)
                        new_centers[next_cluster_id] = new_center
                        center_to_idx[new_center] = len(center_to_idx)
                        
                        merged_clusters.add(cluster1_id)
                        merged_clusters.add(cluster2_id)
                        next_cluster_id += 1
                        changed = True
            
            for cluster_id, nodes in current_clusters.items():
                if cluster_id not in merged_clusters:
                    new_clusters[next_cluster_id] = nodes
                    new_centers[next_cluster_id] = current_centers[cluster_id]
                    next_cluster_id += 1
            
            if not changed:
                break
            
            current_clusters = new_clusters
            current_centers = new_centers
            
            if len(current_clusters) == 1:
                break
        
        return current_clusters
    
    def _should_merge_clusters(self, cluster1_nodes, cluster2_nodes, sim_info):

        if sim_info['similarity'] < 0.5:
            return False
        
        merged_size = len(cluster1_nodes) + len(cluster2_nodes)
        if merged_size > 100:
            return False
        
        return True

    def _compute_community_center(self, community_nodes):
        """Compute community center using the top keyword as the center node"""
        if len(community_nodes) == 1:
            return community_nodes[0]
        return self.extract_keywords_from_community(community_nodes)[0]

    def _get_node_schema_type(self, node_id: str) -> str:
        props = self.graph.nodes[node_id].get("properties", {}) or {}
        return str(props.get("schema_type", "")).strip()

    def _get_node_name(self, node_id: str) -> str:
        return str(self.node_names.get(node_id, "")).strip()

    def _rank_community_members(self, community_nodes: List[str], top_k: int = 5) -> List[str]:
        if not community_nodes:
            return []
        ranked = self.extract_keywords_from_community(community_nodes, top_k=min(top_k, len(community_nodes)))
        deduped = []
        for node_id in ranked:
            if node_id not in deduped:
                deduped.append(node_id)
        return deduped[:top_k]

    def _collect_community_context(
        self,
        members: List[str],
        concept_top_k: int = 4,
        paper_top_k: int = 3,
    ) -> Dict[str, Any]:
        preferred_nodes = [
            node_id for node_id in members
            if self._get_node_schema_type(node_id) in self.COMMUNITY_NAMING_PREFERRED_TYPES
        ]
        fallback_nodes = [
            node_id for node_id in members
            if self._get_node_schema_type(node_id) not in self.COMMUNITY_NAMING_EXCLUDED_TYPES
        ]
        paper_nodes = [
            node_id for node_id in members
            if self._get_node_schema_type(node_id) == "论文"
        ]

        concept_pool = preferred_nodes or fallback_nodes or paper_nodes or members
        concept_ids = self._rank_community_members(concept_pool, top_k=concept_top_k)
        paper_ids = self._rank_community_members(paper_nodes, top_k=paper_top_k) if paper_nodes else []

        concept_names = []
        for node_id in concept_ids:
            node_name = self._get_node_name(node_id)
            if node_name and node_name not in concept_names:
                concept_names.append(node_name)

        paper_titles = []
        for node_id in paper_ids:
            title = self._get_node_name(node_id)
            if title and title not in paper_titles:
                paper_titles.append(title)

        member_names = []
        for node_id in members:
            node_name = self._get_node_name(node_id)
            if node_name and node_name not in member_names:
                member_names.append(node_name)

        return {
            "concept_ids": concept_ids,
            "concept_names": concept_names,
            "paper_ids": paper_ids,
            "paper_titles": paper_titles,
            "member_names": member_names,
            "size": len(members),
        }

    def _sanitize_community_name(self, raw_name: Any) -> str:
        name = str(raw_name or "").strip()
        name = re.sub(r"^\s*主题社区\s*[:：-]\s*", "", name)
        name = name.strip("\"'“”‘’")
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _is_valid_community_name(self, candidate_name: Any, context: Dict[str, Any]) -> bool:
        name = self._sanitize_community_name(candidate_name)
        if not name:
            return False
        if len(name) > 32:
            return False
        lowered = name.casefold()
        member_names = {str(item).strip().casefold() for item in context.get("member_names", []) if str(item).strip()}
        paper_titles = {str(item).strip().casefold() for item in context.get("paper_titles", []) if str(item).strip()}
        if lowered in member_names:
            return False
        if lowered in paper_titles:
            return False
        if " / " in name or " | " in name:
            return False
        slash_parts = [part.strip() for part in re.split(r"[\\/]", name) if part.strip()]
        if len(slash_parts) >= 3:
            return False
        return True

    def _build_fallback_community_name(self, context: Dict[str, Any]) -> str:
        concept_names = list(context.get("concept_names", []))
        if concept_names:
            if len(concept_names) == 1:
                return concept_names[0]
            return " · ".join(concept_names[:2])

        paper_titles = list(context.get("paper_titles", []))
        if paper_titles:
            return "相关研究议题"
        return "相关主题"

    def _build_fallback_community_summary(self, context: Dict[str, Any], community_name: str) -> str:
        concept_names = list(context.get("concept_names", []))
        paper_titles = list(context.get("paper_titles", []))
        if concept_names and paper_titles:
            return (
                f"该社区围绕{', '.join(concept_names[:3])}等主题展开，"
                f"代表论文包括{', '.join(paper_titles[:2])}。"
            )
        if concept_names:
            return f"该社区主要聚焦{', '.join(concept_names[:3])}等相近主题与应用方向。"
        if paper_titles:
            return f"该社区汇聚了一组相近研究议题，代表论文包括{', '.join(paper_titles[:2])}。"
        return f"该社区聚焦与“{community_name}”相关的一组相近节点。"

    def _normalize_llm_community_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("communities", "results", "items", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _prune_rate_limit_window(self, now: float) -> None:
        while self.request_times and now - self.request_times[0] >= 60:
            self.request_times.popleft()

    def _wait_for_llm_slot(self) -> None:
        requests_per_minute = max(0, int(getattr(self, "requests_per_minute", 240)))
        if requests_per_minute <= 0:
            return

        if not hasattr(self, "rate_limit_lock"):
            self.rate_limit_lock = threading.Lock()
        if not hasattr(self, "request_times"):
            self.request_times = deque()

        while True:
            sleep_for = 0.0
            with self.rate_limit_lock:
                now = time.monotonic()
                self._prune_rate_limit_window(now)
                if len(self.request_times) < requests_per_minute:
                    self.request_times.append(now)
                    return
                sleep_for = max(0.05, 60 - (now - self.request_times[0]))
            time.sleep(sleep_for)

    def _resolve_community_batch_size(self, batch_size: Optional[int]) -> int:
        if batch_size is not None:
            return max(1, int(batch_size))
        return max(1, int(getattr(self, "community_llm_batch_size", 8)))

    def _build_community_batches(
        self,
        communities: List[Tuple[str, List[str]]],
        batch_size: int,
    ) -> List[List[Tuple[str, List[str]]]]:
        if not communities:
            return []
        return [
            communities[index:index + batch_size]
            for index in range(0, len(communities), batch_size)
        ]

    def _build_batch_prompt(self, community_batch):
        batch_data = []
        for comm_id, members in community_batch:
            context = self._collect_community_context(members)
            center_node = self._compute_community_center(members)
            center_name = self.node_names[center_node]
            
            comm_info = {
                "id": comm_id,
                "center": center_name,
                "concept_candidates": context.get("concept_names", [])[:4],
                "paper_evidence": context.get("paper_titles", [])[:3],
                "member_sample": context.get("member_names", [])[:8],
                "size": len(members),
                "fallback_name": self._build_fallback_community_name(context),
            }
            batch_data.append(comm_info)
        
        prompt = f"""Generate concise thematic names and summaries for the following {len(batch_data)} communities.
        Communities data: {json.dumps(batch_data, ensure_ascii=False)}
        
        For each community, follow these guidelines:
        1. **Naming Rules**:
           - Name the shared research theme, method, technology, application direction, or teaching problem
           - Prefer abstract thematic names over raw member names
           - Do not copy paper titles directly
           - Do not output prefixes like "主题社区:"
           - Keep the name short and suitable as a graph node label
        
        2. **Summary Requirements**:
           - 1-2 sentences, same language as the dominant member language
           - Explain what the community mainly focuses on and, when possible, its major scene or direction
        
        3. **Output Format** - return a JSON array:
        [
            {{"id": "community_id", "name": "community_name", "summary": "community summary"}},
            ...
        ]
        """
        return prompt

    def _call_llm_api_batch(self, content: str) -> List[Dict]:
        if not self.llm_client:
            return []
        response_text = self.llm_client.call_api(content)
        response_json = json_repair.loads(response_text)
        return self._normalize_llm_community_payload(response_json)

    def _request_community_batch(self, community_batch: List[Tuple[str, List[str]]]) -> Dict[str, Dict[str, Any]]:
        if not self.llm_client or not community_batch:
            return {}

        batch_prompt = self._build_batch_prompt(community_batch)
        self._wait_for_llm_slot()
        llm_results = self._call_llm_api_batch(batch_prompt)
        return {
            str(item.get("id", "")): item
            for item in llm_results
            if str(item.get("id", "")).strip()
        }

    def _apply_community_result(
        self,
        comm_id: str,
        members: List[str],
        llm_info: Optional[Dict[str, Any]],
        level: int,
        super_nodes: Dict[str, List[str]],
    ) -> None:
        context = self._collect_community_context(members)
        llm_info = llm_info or {}
        llm_name = self._sanitize_community_name(llm_info.get("name", ""))
        if self._is_valid_community_name(llm_name, context):
            comm_name = llm_name
        else:
            comm_name = self._build_fallback_community_name(context)

        llm_summary = str(llm_info.get("summary", "")).strip()
        comm_summary = llm_summary or self._build_fallback_community_summary(context, comm_name)

        super_node_id = f"comm_{level}_{comm_id}"
        member_names = [self.node_names[n] for n in members]

        self.graph.add_node(
            super_node_id,
            label="community",
            level=level,
            properties={
                "name": comm_name,
                "description": comm_summary,
                "members": member_names,
                "schema_type": "\u4e3b\u9898\u793e\u533a",
            }
        )

        for node in members:
            self.graph.add_edge(node, super_node_id, relation="member_of")

        super_nodes[super_node_id] = member_names

    def create_super_nodes(self, comm_to_nodes: Dict[str, List[str]], level: int = 4, batch_size: Optional[int] = None):
        super_nodes = {}
        communities = [
            (str(comm_id), members)
            for comm_id, members in comm_to_nodes.items()
            if len(members) >= 2
        ]
        resolved_batch_size = self._resolve_community_batch_size(batch_size)
        community_batches = self._build_community_batches(communities, resolved_batch_size)
        failed_batches = 0
        batch_results: Dict[int, Dict[str, Dict[str, Any]]] = {}
        naming_start = time.time()

        logger.info(
            "Starting community naming for %d communities across %d batches "
            "(batch_size=%d, max_concurrent_llm_requests=%d, requests_per_minute=%d)",
            len(communities),
            len(community_batches),
            resolved_batch_size,
            max(1, int(getattr(self, "max_concurrent_llm_requests", 4))),
            max(0, int(getattr(self, "requests_per_minute", 240))),
        )

        if self.llm_client and community_batches:
            max_workers = min(
                len(community_batches),
                max(1, int(getattr(self, "max_concurrent_llm_requests", 4))),
            )
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch_index = {
                    executor.submit(self._request_community_batch, batch): index
                    for index, batch in enumerate(community_batches)
                }
                for future in futures.as_completed(future_to_batch_index):
                    batch_index = future_to_batch_index[future]
                    try:
                        batch_results[batch_index] = future.result()
                    except Exception as e:
                        failed_batches += 1
                        batch_results[batch_index] = {}
                        logger.error(f"Community naming batch {batch_index} failed: {e}")
        elif not self.llm_client and communities:
            logger.info("TreeComm LLM client unavailable; using fallback naming for all communities")

        llm_dict: Dict[str, Dict[str, Any]] = {}
        for batch_index in range(len(community_batches)):
            llm_dict.update(batch_results.get(batch_index, {}))

        for comm_id, members in communities:
            try:
                self._apply_community_result(comm_id, members, llm_dict.get(comm_id), level, super_nodes)
            except Exception as e:
                logger.error(f"Error creating super node for community {comm_id}: {e}")

        naming_elapsed = time.time() - naming_start
        logger.info(
            "Community naming finished in %.2fs (communities=%d, batches=%d, batch_size=%d, failed_batches=%d)",
            naming_elapsed,
            len(communities),
            len(community_batches),
            resolved_batch_size,
            failed_batches,
        )
        logger.info(f"Created {len(super_nodes)} super nodes")
        return super_nodes

    def extract_keywords_from_community(self, community_nodes: List[str], top_k: int = 5) -> List[str]:
        if len(community_nodes) <= top_k:
            return community_nodes

        structural_scores = {node: self.degree_cache.get(node, 0) for node in community_nodes}
        
        node_embeddings = self.get_triple_embeddings_batch(community_nodes)
        avg_embedding = np.mean(node_embeddings, axis=0)
        
        semantic_scores = cosine_similarity(node_embeddings, [avg_embedding]).flatten()
        
        max_degree = max(structural_scores.values()) if structural_scores else 1
        norm_structural = {n: s / max_degree for n, s in structural_scores.items()}
        norm_semantic = dict(zip(community_nodes, semantic_scores))
        
        combined_scores = {
            node: (self.struct_weight * norm_structural[node] +
                   (1 - self.struct_weight) * norm_semantic[node])
            for node in community_nodes
        }
        
        top_nodes = sorted(community_nodes, key=lambda x: combined_scores[x], reverse=True)[:top_k]
        return top_nodes

    def _build_community_display_name(self, members: List[str]) -> str:
        """Build a fallback display name for a community super node."""
        context = self._collect_community_context(members)
        return self._build_fallback_community_name(context)

    def create_super_nodes_with_keywords(self, comm_to_nodes: Dict[str, List[str]], level: int = 4, batch_size: int = 5):
        """Backward-compatible wrapper without schema-external keyword node creation."""
        super_nodes = self.create_super_nodes(comm_to_nodes, level, batch_size)
        return super_nodes, {}

    
