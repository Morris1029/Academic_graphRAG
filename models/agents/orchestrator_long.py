from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from models.agents.c_agent import CAgent
from models.agents.r_agent import RAgent
from models.agents.a_agent import AAgent
from models.agents.s_agent import SAgent
from utils.logger import logger


class AgentState(TypedDict):
    question: str
    intent: str
    notebook: List[str]
    r_plan: Dict[str, Any]
    final_answer: str
    step_count: int


class GLMOrchestrator:
    def __init__(self, schema_text=""):
        self.c_agent = CAgent()
        self.r_agent = RAgent(schema_text)
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

        workflow.add_conditional_edges(
            "classifier",
            lambda x: "executor" if x["intent"] == "FACTUAL" else "reasoner",
            {"executor": "executor", "reasoner": "reasoner"}
        )

        workflow.add_conditional_edges(
            "reasoner",
            lambda x: "writer" if x["r_plan"]["status"] == "FINISH" else "executor",
            {"writer": "writer", "executor": "executor"}
        )

        workflow.add_edge("executor", "reasoner")  # 循环核心
        workflow.add_edge("writer", END)

        return workflow.compile()

    def _classifier_node(self, state: AgentState):
        res = self.c_agent.route(state["question"])
        return {"intent": res["intent"], "notebook": [], "step_count": 0}

    def _reasoner_node(self, state: AgentState):
        plan = self.r_agent.plan(state["question"], state["notebook"])
        # 死循环熔断机制
        if state["step_count"] > 5:
            plan = {"status": "FINISH", "thought": "Steps limit reached."}
        return {"r_plan": plan, "step_count": state["step_count"] + 1}

    def _executor_node(self, state: AgentState):
        instruction = state.get("r_plan", {}).get("instruction", state["question"])
        result = self.a_agent.execute(instruction)
        # 增量更新 Notebook
        new_entry = f"Step {state['step_count']} - Instruction: {instruction}\nResult: {result[:1000]}..."  # 限制长度
        return {"notebook": state["notebook"] + [new_entry]}

    def _writer_node(self, state: AgentState):
        ans = self.s_agent.write(state["question"], state["notebook"], state["intent"])
        return {"final_answer": ans}

    def run(self, question: str):
        logger.info(f"🚀 GLM Start: {question}")
        return self.workflow.invoke({"question": question})["final_answer"]