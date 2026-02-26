from langchain_core.tools import tool
from models.retriever.enhanced_kt_retriever import KTRetriever
from config import get_config
from utils.logger import logger

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
    results, _ = retriever.process_retrieval_results(query, top_k=10)

    triples = results.get('triples', [])
    chunks = results.get('chunk_contents', [])

    # --- 1. 构造详细的日志输出 (Visualization) ---
    log_msg = f"\n{'=' * 20} RETRIEVAL RESULT {'=' * 20}\n"
    log_msg += f"Query: {query}\n"

    # Triples Visualization (Top 5)
    log_msg += "--- Top 5 Triples (Sub-graph) ---\n"
    for t in triples[:5]:
        log_msg += f"  {t}\n"

    # Chunks Visualization (Top 3)
    log_msg += "--- Top 3 Chunks (Source Text) ---\n"
    # chunk_contents 可能是 dict 或 list
    c_list = list(chunks.values()) if isinstance(chunks, dict) else chunks
    for i, c in enumerate(c_list[:3]):
        preview = c[:200].replace('\n', ' ') + "..."
        log_msg += f"  [Chunk {i}]: {preview}\n"

    log_msg += f"{'=' * 50}"
    logger.info(log_msg)  # 打印到控制台日志

    # --- 2. 构造返回给 Agent 的精简内容 ---
    # Agent 不需要看太长的日志，只需要核心信息
    agent_output = f"Search '{query}' Results:\n"
    if triples:
        agent_output += "[Relationships]:\n" + "\n".join(triples[:8]) + "\n"
    if c_list:
        agent_output += "[Context]:\n"
        for i, c in enumerate(c_list[:2]):  # 给 Agent 看前2个完整 Chunk
            agent_output += f">> {c[:800]}...\n"  # 稍微截断防止溢出

    if not triples and not c_list:
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