from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END

from models.agents.c_agent import CAgent
from models.agents.r_agent import RAgent
from models.agents.a_agent import AAgent
from models.agents.s_agent import SAgent
from utils.logger import logger


# --- 1. 定义状态 (State) ---
class AgentState(TypedDict):
    question: str  # 用户原始问题
    intent: str  # C-Agent 判断的意图 (FACTUAL, COMPLEX, SURVEY...)
    notebook: List[str]  # 共享记忆：记录 A-Agent 查到的结果
    r_plan: Dict[str, Any]  # R-Agent 的当前计划 (status, content)
    final_answer: str  # S-Agent 生成的最终答案
    step_count: int  # 防止无限循环的计数器


# --- 2. 定义编排器类 ---
class GLMOrchestrator:
    def __init__(self, schema_text: str = ""):
        # 初始化所有 Agent
        self.c_agent = CAgent()
        self.r_agent = RAgent(schema_text=schema_text)
        self.a_agent = AAgent()
        self.s_agent = SAgent()
        # 构建图
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # --- 添加节点 (Nodes) ---
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("reasoner", self._reasoner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("writer", self._writer_node)

        # --- 定义边 (Edges) ---
        # 1. 起点 -> 分类器
        workflow.set_entry_point("classifier")

        # 2. 分类器 -> (简单问题直接去执行) 或 (复杂问题去推理)
        workflow.add_conditional_edges(
            "classifier",
            self._route_logic,
            {
                "direct_execution": "executor",  # 简单事实，跳过推理规划，直接生成代码查询
                "reasoning": "reasoner"  # 复杂/综述/时序，需要 R-Agent 规划
            }
        )

        # 3. 推理器 -> (继续执行) 或 (结束去写作)
        workflow.add_conditional_edges(
            "reasoner",
            self._plan_check_logic,
            {
                "continue": "executor",
                "finish": "writer"
            }
        )

        # 4. 执行器 -> 回到推理器 (循环)
        # 注意：如果是从 direct_execution 过来的，执行完应该直接去 writer，这里做一个简单处理
        workflow.add_edge("executor", "reasoner")

        # 5. 写作器 -> 结束
        workflow.add_edge("writer", END)

        return workflow.compile()

    # --- 节点具体逻辑 (Node Functions) ---

    def _classifier_node(self, state: AgentState):
        result = self.c_agent.route(state["question"])
        return {"intent": result.get("intent", "COMPLEX"), "notebook": [], "step_count": 0}

    def _reasoner_node(self, state: AgentState):
        # 如果是 FACTUAL (简单题) 且已经执行过一次查询，R-Agent 应该倾向于结束
        # 如果是 COMPLEX，则进行正常的规划
        plan = self.r_agent.plan(state["question"], state["notebook"])

        # 强制防死循环机制
        if state["step_count"] > 5:
            logger.warning("⚠️ Max steps reached, forcing finish.")
            plan = {"status": "FINISH", "content": "Steps limit reached."}


    def _executor_node(self, state: AgentState):
        # 获取指令：来自 R-Agent 的计划，或者如果是简单题，直接用 Query
        instruction = state.get("r_plan", {}).get("content", "")

        if not instruction and state["intent"] == "FACTUAL":
            instruction = f"查询以下问题的答案: {state['question']}"

        # 执行代码
        result = self.a_agent.execute(instruction)

        # 更新 Notebook
        new_notebook = state["notebook"] + [f"Action: {instruction}\nResult: {result}"]
        return {"notebook": new_notebook}

    def _writer_node(self, state: AgentState):
        answer = self.s_agent.write(
            state["question"],
            state["notebook"],
            state["intent"]
        )
        return {"final_answer": answer}

    # --- 路由逻辑 (Conditional Logic) ---

    def _route_logic(self, state: AgentState) -> Literal["direct_execution", "reasoning"]:
        if state["intent"] == "FACTUAL":
            # 简单题，可以尝试直接让 A-Agent 写代码查
            # 为了简化流，这里统一先去 executor，但 executor 需要知道没有 plan 也要干活
            # 或者为了架构统一，简单题也走 reasoner，让 R-Agent 决定"查一下X"
            return "reasoning"
        return "reasoning"

    def _plan_check_logic(self, state: AgentState) -> Literal["continue", "finish"]:
        status = state["r_plan"].get("status", "FINISH")
        if status == "CONTINUE":
            return "continue"
        return "finish"

    # --- 公共接口 ---
    def run(self, question: str):
        """运行整个智能体流程"""
        initial_state = {"question": question, "notebook": [], "step_count": 0}
        logger.info(f"🚀 Starting GLM Orchestrator for: {question}")

        result = self.workflow.invoke(initial_state)
        return result["final_answer"]