from langchain_core.tools import tool, StructuredTool
from models.retriever.enhanced_kt_retriever import KTRetriever
from config import get_config  # 假设你有配置加载器
from utils.logger import logger

# --- 1. 单例初始化 (Global Instance) ---
# 这样整个应用生命周期内，索引只加载一次
_RETRIEVER_INSTANCE = None


def get_retriever() -> KTRetriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        # logger.info("🚀 Initializing Knowledge Graph Engine...")
        print("🚀 Initializing Knowledge Graph Engine...")
        # 使用原有的配置加载逻辑
        cfg = get_config()

        # [修改点] 获取当前激活的数据集名称
        # 优先读取 active_dataset，如果没配置则默认由代码指定或报错
        # 注意：这里假设 ConfigManager 支持属性访问 (cfg.active_dataset)
        # 如果你的 ConfigManager 也是字典访问，请用 cfg['active_dataset']
        target_dataset = cfg.active_dataset

        if target_dataset not in cfg.datasets:
            raise ValueError(f"Active dataset '{target_dataset}' not defined in config.datasets")

        _RETRIEVER_INSTANCE = KTRetriever(
            dataset=target_dataset,  # 传入字符串
            config=cfg
        )
    return _RETRIEVER_INSTANCE


# --- 2. 定义 Tools (供 Agent 使用) ---

@tool
def search_entity(query: str):
    """
    根据名称或描述搜索知识图谱中的实体。
    返回实体的 ID、名称、描述和属性。
    当你想知道某个特定概念、论文或作者在图谱中的具体信息时使用。
    """
    retriever = get_retriever()
    # 调用底层引擎
    results = retriever.search_nodes_only(query, top_k=5)

    # 【关键】将结构化数据转为 Agent 易读的字符串，节省 Token
    if not results:
        return "No entities found."

    output = []
    for r in results:
        # 格式化一下，去掉不必要的字段
        output.append(f"ID: {r['id']} | Name: {r['name']} | Props: {r['attributes']}")
    return "\n".join(output)


@tool
def get_neighbors(node_id: str):
    """
    获取指定实体（Node ID）的所有邻居节点和关系。
    用于在图谱中进行单跳探索。
    输入必须是 search_entity 返回的 ID。
    """
    retriever = get_retriever()
    neighbors = retriever.get_node_neighbors(node_id)

    if not neighbors:
        return "No neighbors found."

    # 转换为简化的三元组字符串
    return str([(n['relation'], n['target_name'], n['target_id']) for n in neighbors])


@tool
def execute_python_analysis(code_snippet: str):
    """
    [这是 A-Agent 的核心工具]
    在一个拥有图谱查询能力的沙箱中执行 Python 代码。
    代码中可以直接调用 `search_entity` 和 `get_neighbors` 等函数。
    用于执行复杂的统计、排序、过滤任务。
    """
    # 这里实现一个简单的沙箱
    local_vars = {}
    tools_map = {
        "search_entity": search_entity.invoke,  # 允许代码中调用工具
        "get_neighbors": get_neighbors.invoke,
        "print": print,
        "sorted": sorted,
        "len": len
    }

    try:
        exec(code_snippet, tools_map, local_vars)
        return "Code executed. Output: " + str(local_vars.get("result", "No result variable set."))
    except Exception as e:
        return f"Error executing code: {e}"


# --- 3. 导出工具列表 ---
def get_all_tools():
    return [search_entity, get_neighbors, execute_python_analysis]