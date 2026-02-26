import io
import contextlib
from langchain_core.tools import tool
from models.retriever.enhanced_kt_retriever import KTRetriever
from config import get_config
from utils.logger import logger

_RETRIEVER_INSTANCE = None


def get_retriever():
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        print("🚀 Initializing Knowledge Graph Engine...")
        cfg = get_config()
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
    【检索工具】输入自然语言查询，返回相关的图谱三元组和文本片段。
    当你需要获取新知识时使用此工具。
    """
    logger.info(f"🔎 Retrieving: {query}")
    retriever = get_retriever()
    # 复用 Youtu 的核心检索逻辑，它包含 Vector Search + Graph Traversal
    results, _ = retriever.process_retrieval_results(query, top_k=10)

    triples = list(results.get('triples', []))
    chunks = results.get('chunk_contents', [])
    chunk_list = list(chunks.values()) if isinstance(chunks, dict) else chunks

    # 格式化输出给 LLM 看
    output = f"--- Search Results for '{query}' ---\n"
    if triples:
        output += "[Related Triples]:\n" + "\n".join(triples[:8]) + "\n"
    if chunk_list:
        output += "[Related Text Chunks]:\n"
        for i, c in enumerate(chunk_list[:2]):  # 限制数量防止 Token 爆炸
            output += f"Chunk_{i}: {c[:600]}...\n"  # 限制长度

    if not triples and not chunk_list:
        return "No relevant information found."

    return output


@tool
def python_interpreter(code: str):
    """
    【计算工具】执行 Python 代码。
    当你需要对已有的信息进行排序、计数、时间比较时使用。
    """
    local_vars = {}
    output_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code, {}, local_vars)
        return output_capture.getvalue()
    except Exception as e:
        return f"Error: {e}"


def get_all_tools():
    return [graph_retrieval, python_interpreter]