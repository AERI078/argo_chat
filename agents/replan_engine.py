# agents/replan_engine.py — generates a revised plan when PlanEvaluator signals replan
# chooses repair strategy: local fix (patch one step) or full regeneration

import json
from agents.llm_caller import LLMCaller
from agents.prompt import replan_prompt
from agents.contracts import TaskPlan, TaskStep, ValidationResult
from agents.planner import PlannerAgent
from config import MAX_REPLAN_ATTEMPTS


class ReplanEngine:
    def __init__(self, llm: LLMCaller, planner: PlannerAgent):
        self.llm = llm
        self.planner = planner
        self.attempts = 0

    def replan(self, query: str, original_plan: TaskPlan,
               failed_validation: ValidationResult) -> TaskPlan:

        self.attempts += 1

        # if we've already replanned too many times, fall back to fresh plan
        if self.attempts > MAX_REPLAN_ATTEMPTS:
            return self.planner.plan(query)

        failed_step = next(
            (s for s in original_plan.steps if s.step_id == failed_validation.step_id),
            None
        )
        if not failed_step:
            return self.planner.plan(query)

        messages = replan_prompt(
            query=query,
            original_plan=original_plan.model_dump(),
            failed_step=failed_step.model_dump(),
            reason=failed_validation.reason
        )
        raw = self.llm.call(messages, temperature=0.1)

        try:
            data = json.loads(raw)
            steps = [TaskStep(**s) for s in data.get("steps", [])]
            return TaskPlan(
                query=query,
                steps=steps,
                strategy_rationale=f"[Replanned] {data.get('strategy_rationale', '')}"
            )
        except (json.JSONDecodeError, Exception):
            # if replan LLM output is broken, fall back to fresh plan from planner
            return self.planner.plan(query)
