from langchain_core.tools import tool
from models.retriever.enhanced_kt_retriever import KTRetriever
from config import get_config
from utils.logger import logger
import re

_RETRIEVER_INSTANCE = None


def get_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        cfg = get_config()
        _RETRIEVER_INSTANCE = KTRetriever(dataset=cfg.active_dataset, config=cfg, mode="agent")
        if not hasattr(_RETRIEVER_INSTANCE, 'node_index'):
            _RETRIEVER_INSTANCE.build_indices()
    return _RETRIEVER_INSTANCE


@tool
def graph_retrieval(query: str):
    """
    检索工具。返回图谱三元组和文本块。
    """
    logger.info(f"🔎 [Tool] Searching: '{query}'")
    retriever = get_retriever()

    # 核心检索: 包含 embedding search, graph traversal, reranking
    results, _ = retriever.process_retrieval_results(query, top_k=50)

    triples = results.get('triples', [])
    chunks_map = results.get('chunk_contents', {})  # 假设这是一个 dict: {id: content}
    # 如果是 list，这里需要兼容处理
    if isinstance(chunks_map, list):
        # 尝试从 results['chunk_ids'] 恢复 map，或者生成临时 ID
        chunk_ids = results.get('chunk_ids', [f"temp_{i}" for i in range(len(chunks_map))])
        chunks_map = dict(zip(chunk_ids, chunks_map))

    #  三元组去重
    unique_triples = []
    seen_triples = set()

    for t in triples:
        # 去掉分数部分进行比对: (A, r, B) [score: 0.9] -> (A, r, B)
        content_only = re.sub(r'\[score:.*?\]', '', t).strip()
        if content_only not in seen_triples:
            seen_triples.add(content_only)
            unique_triples.append(t)  # 保留原始带分数的，或者你可以选择只保留 content_only

    # 控制 Notebook 内容数量
    # 这里控制返回给 Agent 的数量，也是最终写入 Notebook 的最大数量
    # 三元组保留 Top 8-10，文本块保留 Top 2-3
    top_triples = unique_triples[:35]

    c_list = list(chunks_map.values()) if isinstance(chunks_map, dict) else chunks_map
    top_chunks = c_list[:10]

    # --- 1. 构造详细的日志输出 (Visualization) ---
    log_msg = f"\n{'=' * 20} RETRIEVAL RESULT {'=' * 20}\n"
    log_msg += f"Query: {query}\n"

    # Triples Visualization (Top 10)
    log_msg += "--- Top 5 Triples (Sub-graph) ---\n"
    for t in triples[:35]:
        log_msg += f"  {t}\n"

    # Chunks Visualization (Top 3)
    log_msg += "--- Top 3 Chunks (Source Text) ---\n"
    for i, c in enumerate(c_list[:10]):
        preview = c[:200].replace('\n', ' ') + "..."
        log_msg += f"  [Chunk {i}]: {preview}\n"

    log_msg += f"{'=' * 50}"
    logger.info(log_msg)  # 打印到控制台日志

    # --- 2. 构造返回给 Agent 的精简内容 ---
    # Agent 不需要看太长的日志，只需要核心信息
    agent_output = f"Search '{query}' Results:\n"
    #给agent看检索回来的top35条三元组
    if top_triples:
        # 修复：使用 final_triples 而不是原始 triples
        agent_output += "[Relationships]:\n" + "\n".join(top_triples) + "\n"
    else:
        agent_output += "[Relationships]: None\n"

    if chunks_map:
        agent_output += "[Context]:\n"
        count = 0
        for cid, content in chunks_map.items():
            if count >= 3: break  # 只返回 Top 3
            # 在文本前加上 [Source: ID] agent查看的chunks长度限制为1000
            clean_content = content[:1000].replace('\n', ' ')
            agent_output += f">> [Source: Chunk {cid}] {clean_content}...\n"
            count += 1
    else:
        agent_output += "[Context]: None\n"

    if not top_triples and not top_chunks:
        return "No information found."

    return agent_output


@tool
def get_graph_schema(dummy: str = ""):
    """
    获取知识图谱的 Schema 定义（节点类型、关系类型）。
    当你不知道该搜索什么类型的关系时，调用此工具。
    """
    retriever = get_retriever()
    # 假设 retriever 中加载了 schema
    if hasattr(retriever, 'schema') and retriever.schema:
        import json
        return json.dumps(retriever.schema, ensure_ascii=False)
    return "Schema info not available."


def get_all_tools():
    return [graph_retrieval, get_graph_schema]