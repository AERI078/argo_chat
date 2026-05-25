# agents/orchestrator.py — coordinates the full multi-agent loop

import json
from agents.contracts import OrchestratorResponse, PlanStatus, StepResult
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
        log.success("planner", {"steps": len(plan.steps),
                                 "rationale": plan.strategy_rationale[:80]})

        evaluator = PlanEvaluator()
        completed_ids: list[int] = []
        all_results: list[StepResult] = []
        context_parts: list[str] = []
        chart_spec = None

        # track scores with their tool names for weighted confidence calc
        validation_scores: list[float] = []
        validation_tools: list[str] = []

        for step in plan.steps:
            if any(dep not in completed_ids for dep in step.depends_on):
                log.failure("executor",
                            error=f"Skipped step {step.step_id} — unmet dependencies",
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
            validation_tools.append(step.tool)

            if validation.passed:
                log.success("validator", step_id=step.step_id,
                            data={"score": validation.score})
            else:
                log.failure("validator", error=validation.reason,
                            failure_type=validation.failure_type.value if validation.failure_type else None,
                            step_id=step.step_id)

            log.start("plan_evaluator", step_id=step.step_id)
            plan_eval = evaluator.evaluate(plan, validation, completed_ids)
            log.success("plan_evaluator", data={"status": plan_eval.status,
                                                 "failures": plan_eval.failure_counts})

            if plan_eval.status == PlanStatus.unrecoverable:
                avg_confidence = _weighted_confidence(validation_scores, validation_tools, default=0.0)
                log.finish(success=False)
                return OrchestratorResponse(
                    query=query,
                    answer="I encountered too many errors. Please try a more specific question.",
                    chart_spec=chart_spec, plan=plan, step_results=all_results,
                    success=False, confidence=avg_confidence
                ), log.trace()

            if plan_eval.status == PlanStatus.replan:
                log.replan(reason=plan_eval.reason,
                           attempt=self.replan_engine.attempts + 1)
                plan = self.replan_engine.replan(query, plan, validation)
                evaluator = PlanEvaluator()
                completed_ids = []
                context_parts = []
                all_results = []
                validation_scores = []
                validation_tools = []
                continue

            if result.success and result.data:
                context_parts.append(self._summarise_result(step.tool, result.data))

            if step.tool == "generate_chart" and result.success:
                chart_spec = result.data

            if step.tool == "final_answer" and result.success:
                answer = result.data.get("text", "") if result.data else ""
                avg_confidence = _weighted_confidence(validation_scores, validation_tools, default=0.5)
                log.finish(success=True, answer_length=len(answer))
                return OrchestratorResponse(
                    query=query, answer=answer,
                    chart_spec=result.data.get("chart_spec") or chart_spec,
                    plan=plan, step_results=all_results,
                    success=True, confidence=avg_confidence
                ), log.trace()

            completed_ids.append(step.step_id)

        avg_confidence = _weighted_confidence(validation_scores, validation_tools, default=0.3)
        log.finish(success=False)
        return OrchestratorResponse(
            query=query,
            answer="I completed the analysis but couldn't synthesise a final answer. Please try rephrasing.",
            chart_spec=chart_spec, plan=plan, step_results=all_results,
            success=False, confidence=avg_confidence
        ), log.trace()

    def _summarise_result(self, tool: str, data: dict) -> str:
        """
        Converts a tool result into context for the next step.
        For rag_search, includes BOTH float summaries and knowledge docs.
        """
        if tool == "rag_search":
            parts = []
            docs = data.get("docs", [])
            knowledge = data.get("knowledge_docs", [])
            if docs:
                parts.append("Float data context:\n" + "\n".join(docs[:3]))
            if knowledge:
                parts.append("Ocean science context:\n" + "\n".join(knowledge[:2]))
            return "\n\n".join(parts)
        elif tool in ("fetch_region", "fetch_float", "db_query"):
            count = data.get("count", 0)
            rows = data.get("rows", [])[:3]
            return f"Data fetch returned {count} records. Sample: {json.dumps(rows)}"
        elif tool == "generate_chart":
            return f"Chart generated: {data.get('type', 'unknown')} chart."
        return str(data)[:300]


def _weighted_confidence(scores: list[float], tools: list[str], default: float) -> float:
    """
    Weights the final_answer score at 50% of the total confidence since it
    directly reflects answer quality. All other steps share the remaining 50%.
    Falls back to a simple average when there's only one score.
    """
    if not scores:
        return default

    if len(scores) == 1:
        return round(scores[0], 2)

    # find the final_answer score — it's the most meaningful signal
    final_answer_indices = [i for i, t in enumerate(tools) if t == "final_answer"]

    if not final_answer_indices:
        # no final_answer step scored — plain average
        return round(sum(scores) / len(scores), 2)

    final_idx = final_answer_indices[-1]
    final_score = scores[final_idx]

    # average of all non-final-answer steps
    other_scores = [s for i, s in enumerate(scores) if i != final_idx]
    other_avg = sum(other_scores) / len(other_scores) if other_scores else final_score

    # 50% weight to final_answer, 50% to everything else
    weighted = (final_score * 0.5) + (other_avg * 0.5)
    return round(weighted, 2)