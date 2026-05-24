# agents/factory.py — wires the full multi-agent system together
# api/main.py calls build_orchestrator() once at startup

from agents.llm_caller import LLMCaller
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.validator import ValidatorAgent
from agents.replan_engine import ReplanEngine
from agents.orchestrator import Orchestrator
from rag.pipeline import RAGPipeline


def build_orchestrator() -> Orchestrator:
    """
    Single composition root for the full multi-agent system.
    RAGPipeline handles FAISS index build/load on its own init.
    All agents share one LLMCaller instance — one Groq client, one rate limit bucket.
    """
    llm = LLMCaller()
    rag = RAGPipeline()

    planner = PlannerAgent(llm)
    executor = ExecutorAgent(llm, rag)
    validator = ValidatorAgent(llm)
    replan_engine = ReplanEngine(llm, planner)

    return Orchestrator(
        planner=planner,
        executor=executor,
        validator=validator,
        replan_engine=replan_engine
    )
