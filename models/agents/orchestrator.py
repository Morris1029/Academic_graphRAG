from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from models.agents.c_agent import CAgent
from models.agents.r_agent import RAgent
from models.agents.a_agent import AAgent
from models.agents.s_agent import SAgent
from config import get_config  # 导入配置加载
from utils.logger import logger
import re
import os
from difflib import SequenceMatcher

# --- 1. 状态机---
class AgentState(TypedDict):
    question: str
    intent: str
    notebook: List[str]
    raw_knowledge: Dict[str, set] # 新增：用于存储 raw triples 和 chunk ids 以便去重
    knowledge_hashes: set[str]
    current_plan: Dict
    step_count: int
    final_answer: str
    query_history: List[str]  #  记录历史查询词


class GLMOrchestrator:
    def __init__(self, schema_text=""):
        # --- 自动加载 Schema 逻辑 ---
        if not schema_text:
            try:
                cfg = get_config()
                # 获取当前活跃数据集的 schema 路径
                active_dataset = cfg.active_dataset
                schema_path = cfg.datasets[active_dataset].schema_path
                if os.path.exists(schema_path):
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_text = f.read()
                    logger.info(f"✅ Loaded schema from {schema_path}")
                else:
                    logger.warning(f"⚠️ Schema file not found at {schema_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load schema from config: {e}")

        self.c_agent = CAgent()
        self.r_agent = RAgent(schema_text=schema_text)
        self.a_agent = AAgent()
        self.s_agent = SAgent()
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("reasoner", self._reasoner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("writer", self._writer_node)

        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "reasoner")
        workflow.add_conditional_edges(
            "reasoner",
            self._check_plan,
            {"search": "executor", "answer": "writer"}
        )
        workflow.add_edge("executor", "reasoner")
        workflow.add_edge("writer", END)
        return workflow.compile()

    def _classifier_node(self, state: AgentState):
        res = self.c_agent.route(state["question"])
        # 初始化 query_history
        return {"intent": res.get("intent", "COMPLEX"), "notebook": [], "knowledge_hashes": set(), "step_count": 0, "query_history": []}

    def _reasoner_node(self, state: AgentState):
        # 传入 query_history
        plan = self.r_agent.plan(state["question"], state["notebook"], state["step_count"],
                                 state.get("query_history", []))
        return {"current_plan": plan}

    def _executor_node(self, state: AgentState):
        query = state["current_plan"].get("search_query", "")
        # 执行检索
        raw_result = self.a_agent.execute(query)

        # 初始化去重集合
        if "raw_knowledge" not in state or not state["raw_knowledge"]:
            known_triples = set()
            known_chunks = set()
        else:
            known_triples = state["raw_knowledge"].get("triples", set())
            known_chunks = state["raw_knowledge"].get("chunks", set())

        new_triples = []
        new_chunks = []

        # 解析 raw_result (假设格式为 Search '...' Results:\n[Relationships]:...)
        # 我们只提取非空行
        lines = raw_result.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line: continue

            if "[Relationships]:" in line:
                current_section = "triples"
                continue
            elif "[Context]:" in line:
                current_section = "chunks"
                continue
            elif "Search" in line and "Results" in line:
                continue

            if current_section == "triples":
                # 去除分数，进行纯内容去重
                content_only = re.sub(r'\[score:.*?\]', '', line).strip()
                if len(content_only) < 5: continue

                if content_only not in known_triples:
                    known_triples.add(content_only)
                    new_triples.append(line)  # 保留带分数的原行

            elif current_section == "chunks":
                # 假设格式: >> [Chunk ID] Content...
                # 提取内容部分进行相似度比对
                if line.startswith(">>"):
                    content = line[2:].strip()
                    # 简单查重：检查是否是已有 Chunk 的子串，或者相似度极高
                    is_duplicate = False
                    for existing in known_chunks:
                        # 如果相似度 > 0.8 则认为是重复
                        if SequenceMatcher(None, content, existing).ratio() > 0.8:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        known_chunks.add(content)
                        new_chunks.append(line)

        # 构建 Notebook 条目
        entry_content = ""
        if new_triples:
            entry_content += "New Triples:\n" + "\n".join(new_triples) + "\n"
        if new_chunks:
            entry_content += "New Context:\n" + "\n".join(new_chunks) + "\n"

        if not entry_content:
            entry_content = "No new relevant info found (Duplicates filtered)."

        new_entry = f"Step {state['step_count']} Query: '{query}'\nFound:\n{entry_content}"

        print(f"\n📘 [Notebook Update Step {state['step_count']}]")
        print(f"   Query: {query}")
        print(f"   New Triples: {len(new_triples)}, New Chunks: {len(new_chunks)}")

        return {
            "notebook": state["notebook"] + [new_entry],
            "raw_knowledge": {"triples": known_triples, "chunks": known_chunks},
            "step_count": state["step_count"] + 1,
            "query_history": state["query_history"] + [query]
        }


    def _writer_node(self, state: AgentState):
            ans = self.s_agent.write(state["question"], state["notebook"])
            return {"final_answer": ans}

    def _check_plan(self, state: AgentState) -> Literal["search", "answer"]:
        action = state["current_plan"].get("action", "FINISH")
        if state["step_count"] >= 6:
            logger.warning("⚠️ Max steps reached.")
            return "answer"
        if action == "SEARCH":
            return "search"
        return "answer"

    def run(self, question: str):
        logger.info(f"🚀 GLM Start: {question}")
        initial = {"question": question, "notebook": [],"knowledge_hashes": set(), "step_count": 0, "query_history": []}
        result = self.workflow.invoke(initial)
        # 打印完整 Notebook 供调试
        print("\n" + "=" * 40)
        print("📓 FINAL NOTEBOOK CONTENT (DEBUG)")
        print("=" * 40)
        for entry in result["notebook"]:
            print(entry)
            print("-" * 20)
        print("=" * 40 + "\n")
        return result["final_answer"]