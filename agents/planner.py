# agents/planner.py — decomposes a user query into a structured TaskPlan
# single responsibility: produce a valid, ordered list of steps, nothing else

import json
from agents.llm_caller import LLMCaller
from agents.prompt import planner_prompt
from agents.contracts import TaskPlan, TaskStep
from config import MAX_PLAN_STEPS


class PlannerAgent:
    def __init__(self, llm: LLMCaller):
        self.llm = llm

    def plan(self, query: str) -> TaskPlan:
        messages = [
            {"role": "system", "content": planner_prompt()},
            {"role": "user", "content": query}
        ]
        raw = self.llm.call(messages, temperature=0.1)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # fallback: minimal plan that at least gets context and answers
            return self._fallback_plan(query)

        steps = [TaskStep(**s) for s in data.get("steps", [])][:MAX_PLAN_STEPS]
        return TaskPlan(
            query=query,
            steps=steps,
            strategy_rationale=data.get("strategy_rationale", "")
        )

    def _fallback_plan(self, query: str) -> TaskPlan:
        """Minimal 2-step plan when LLM output is unparseable."""
        return TaskPlan(
            query=query,
            steps=[
                TaskStep(step_id=1, description="Search float summaries for context",
                         tool="rag_search", params={"query": query}, depends_on=[]),
                TaskStep(step_id=2, description="Synthesise answer",
                         tool="final_answer", params={"text": "", "chart_spec": None}, depends_on=[1])
            ],
            strategy_rationale="fallback plan — LLM output was unparseable"
        )
