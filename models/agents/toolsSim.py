from langchain_core.tools import tool
from models.retriever.enhanced_kt_retriever import KTRetriever
from config import get_config

_RETRIEVER_INSTANCE = None


def get_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        cfg = get_config()
        # 强制单例初始化，避免重复加载索引
        _RETRIEVER_INSTANCE = KTRetriever(
            dataset=cfg.active_dataset,
            config=cfg,
            mode="agent"
        )
        if not hasattr(_RETRIEVER_INSTANCE, 'node_index'):
            _RETRIEVER_INSTANCE.build_indices()
    return _RETRIEVER_INSTANCE


@tool
def graph_retrieval(query: str):
    """
    【核心工具】执行图谱检索。
    输入自然语言查询（可以是原始问题，也可以是分解后的子问题）。
    返回：相关的三元组（Triples）和文本块（Chunks）。
    这是获取知识的主要途径，比微观搜索更高效。
    """
    retriever = get_retriever()
    # 复用原项目中强大的 process_retrieval_results
    # 它会自动做 Embedding 检索、双路召回、重排序
    results, _ = retriever.process_retrieval_results(query, top_k=10)

    triples = results.get('triples', [])
    chunks = results.get('chunk_contents', [])

    # 格式化输出，限制长度
    output = "=== Retrieved Triples ===\n" + "\n".join(triples[:10])
    output += "\n\n=== Retrieved Text Chunks ===\n"
    # 如果 chunks 是字典，转 list
    chunk_list = list(chunks.values()) if isinstance(chunks, dict) else chunks
    for i, c in enumerate(chunk_list[:5]):  # 只返回前5个最相关的 chunk，防止 Context 爆炸
        output += f"[Chunk {i + 1}]: {c[:500]}...\n"  # 截断

    return output


def get_all_tools():
    return [graph_retrieval]  # 只保留这就够了！