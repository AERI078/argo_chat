# agents/orchestrator.py — coordinates the full multi-agent loop
# only entry point the API calls: orchestrator.run(query)

import json
from agents.contracts import (
    OrchestratorResponse, PlanStatus, StepResult
)
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent
from agents.validator import ValidatorAgent
from agents.plan_evaluator import PlanEvaluator
from agents.replan_engine import ReplanEngine
from agents.logger import PipelineLogger


class Orchestrator:
    def __init__(self, planner: PlannerAgent, executor: ExecutorAgent,
                 validator: ValidatorAgent, replan_engine: ReplanEngine):
        self.planner = planner
        self.executor = executor
        self.validator = validator
        self.replan_engine = replan_engine

    def run(self, query: str) -> tuple[OrchestratorResponse, dict]:
        log = PipelineLogger(query=query)

        log.start("planner")
        plan = self.planner.plan(query)
        log.success("planner", {"steps": len(plan.steps), "rationale": plan.strategy_rationale[:80]})

        evaluator = PlanEvaluator()
        completed_ids: list[int] = []
        all_results: list[StepResult] = []
        context_parts: list[str] = []
        chart_spec = None
        validation_scores: list[float] = []

        for step in plan.steps:
            if any(dep not in completed_ids for dep in step.depends_on):
                log.failure("executor", error=f"Skipped step {step.step_id} — unmet dependencies",
                            step_id=step.step_id)
                continue

            context = "\n".join(context_parts) if context_parts else "No prior context."

            log.start("executor", step_id=step.step_id, tool=step.tool)
            result = self.executor.execute(step, context, query)
            all_results.append(result)

            if result.success:
                count = result.data.get("count", "") if result.data else ""
                log.success("executor", step_id=step.step_id, tool=step.tool,
                            data={"count": count} if count != "" else {})
            else:
                log.failure("executor", error=result.error or "unknown",
                            step_id=step.step_id, tool=step.tool)

            log.start("validator", step_id=step.step_id)
            validation = self.validator.validate(step, result)
            validation_scores.append(validation.score)

            if validation.passed:
                log.success("validator", step_id=step.step_id, data={"score": validation.score})
            else:
                log.failure("validator", error=validation.reason,
                            failure_type=validation.failure_type.value if validation.failure_type else None,
                            step_id=step.step_id)

            log.start("plan_evaluator", step_id=step.step_id)
            plan_eval = evaluator.evaluate(plan, validation, completed_ids)
            log.success("plan_evaluator", data={"status": plan_eval.status,
                                                 "failures": plan_eval.failure_counts})

            if plan_eval.status == PlanStatus.unrecoverable:
                avg_confidence = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
                log.finish(success=False)
                return OrchestratorResponse(
                    query=query,
                    answer="I encountered too many errors. Please try a more specific question.",
                    chart_spec=chart_spec, plan=plan, step_results=all_results, success=False,
                    confidence=avg_confidence
                ), log.trace()

            if plan_eval.status == PlanStatus.replan:
                log.replan(reason=plan_eval.reason, attempt=self.replan_engine.attempts + 1)
                plan = self.replan_engine.replan(query, plan, validation)
                evaluator = PlanEvaluator()
                completed_ids = []
                context_parts = []
                all_results = []
                validation_scores = []
                continue

            if result.success and result.data:
                context_parts.append(self._summarise_result(step.tool, result.data))

            if step.tool == "generate_chart" and result.success:
                chart_spec = result.data

            if step.tool == "final_answer" and result.success:
                answer = result.data.get("text", "") if result.data else ""
                avg_confidence = sum(validation_scores) / len(validation_scores) if validation_scores else 0.5
                log.finish(success=True, answer_length=len(answer))
                return OrchestratorResponse(
                    query=query, answer=answer,
                    chart_spec=result.data.get("chart_spec") or chart_spec,
                    plan=plan, step_results=all_results, success=True,
                    confidence=avg_confidence
                ), log.trace()

            completed_ids.append(step.step_id)

        log.finish(success=False)
        return OrchestratorResponse(
            query=query,
            answer="I completed the analysis but couldn't synthesise a final answer. Please try rephrasing.",
            chart_spec=chart_spec, plan=plan, step_results=all_results, success=False,
            confidence=avg_confidence
        ), log.trace()

    def _summarise_result(self, tool: str, data: dict) -> str:
        if tool == "rag_search":
            return "Retrieved context:\n" + "\n".join(data.get("docs", [])[:3])
        elif tool in ("fetch_region", "fetch_float", "db_query"):
            count = data.get("count", 0)
            rows = data.get("rows", [])[:3]
            return f"Data fetch returned {count} records. Sample: {json.dumps(rows)}"
        elif tool == "generate_chart":
            return f"Chart generated: {data.get('type', 'unknown')} chart."
        return str(data)[:300]