from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from models.agents.c_agent import CAgent
from models.agents.r_agent import RAgent
from models.agents.a_agent import AAgent
from models.agents.s_agent import SAgent
from utils.logger import logger


# --- 1. 修改 State，增加 query_history ---
class AgentState(TypedDict):
    question: str
    intent: str
    notebook: List[str]
    current_plan: Dict
    step_count: int
    final_answer: str
    query_history: List[str]  #  记录历史查询词，防止重复


class GLMOrchestrator:
    def __init__(self, schema_text=""):
        self.c_agent = CAgent()
        self.r_agent = RAgent()
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
        return {"intent": res.get("intent", "COMPLEX"), "notebook": [], "step_count": 0, "query_history": []}

    def _reasoner_node(self, state: AgentState):
        # 传入 query_history
        plan = self.r_agent.plan(state["question"], state["notebook"], state["step_count"],
                                 state.get("query_history", []))
        return {"current_plan": plan}

    def _executor_node(self, state: AgentState):
        query = state["current_plan"].get("search_query", "")
        result = self.a_agent.execute(query)
        new_entry = f"Step {state['step_count']} Search: '{query}'\nResult Summary: {result[:500]}..."

        # 更新 notebook 和 query_history
        return {
            "notebook": state["notebook"] + [new_entry],
            "step_count": state["step_count"] + 1,
            "query_history": state["query_history"] + [query]
        }

    def _writer_node(self, state: AgentState):
        ans = self.s_agent.write(state["question"], state["notebook"])
        return {"final_answer": ans}

    def _check_plan(self, state: AgentState) -> Literal["search", "answer"]:
        action = state["current_plan"].get("action", "FINISH")
        if state["step_count"] >= 5:
            logger.warning("⚠️ Max steps reached.")
            return "answer"
        if action == "SEARCH":
            return "search"
        return "answer"

    def run(self, question: str):
        logger.info(f"🚀 GLM Start: {question}")
        initial = {"question": question, "notebook": [], "step_count": 0, "query_history": []}
        result = self.workflow.invoke(initial)
        return result["final_answer"]