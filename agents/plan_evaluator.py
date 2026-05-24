# agents/plan_evaluator.py — inspects global plan health after every step

from agents.contracts import (
    TaskPlan, ValidationResult, PlanEvaluation, PlanStatus, FailureType
)
from config import FAILURE_THRESHOLDS


class PlanEvaluator:
    def __init__(self):
        self.failure_counts: dict[str, int] = {ft.value: 0 for ft in FailureType}
        self.failed_ids: set[int] = set()

    def evaluate(self, plan: TaskPlan, latest_validation: ValidationResult,
                 completed_ids: list[int]) -> PlanEvaluation:

        current_step_passed = latest_validation.passed

        # record failure
        if not current_step_passed:
            self.failed_ids.add(latest_validation.step_id)
            if latest_validation.failure_type:
                self.failure_counts[latest_validation.failure_type.value] += 1

        # threshold check — only applies when step failed
        # count == threshold → replan, count > threshold → unrecoverable
        if not current_step_passed:
            for ft, count in self.failure_counts.items():
                threshold = FAILURE_THRESHOLDS.get(ft, 99)
                if count > threshold:
                    return PlanEvaluation(
                        status=PlanStatus.unrecoverable,
                        reason=f"'{ft}' failures ({count}) exceeded threshold ({threshold}). Giving up.",
                        failure_counts=self.failure_counts.copy()
                    )
                if count == threshold:
                    return PlanEvaluation(
                        status=PlanStatus.replan,
                        reason=f"'{ft}' failures hit threshold ({count}/{threshold}). Replanning.",
                        failure_counts=self.failure_counts.copy()
                    )
            # failure count below threshold — still coherent, will retry
            return PlanEvaluation(
                status=PlanStatus.coherent,
                reason="Failure below threshold — retrying.",
                failure_counts=self.failure_counts.copy()
            )

        # step passed — check if its dependencies were actually satisfied
        # catches cases where a step ran out of order or a dep was skipped
        completed_set = set(completed_ids)
        current_step = next(
            (s for s in plan.steps if s.step_id == latest_validation.step_id), None
        )
        if current_step:
            for dep_id in current_step.depends_on:
                if dep_id not in completed_set:
                    return PlanEvaluation(
                        status=PlanStatus.replan,
                        reason=f"Step {current_step.step_id} passed but its dependency (step {dep_id}) never completed.",
                        failure_counts=self.failure_counts.copy()
                    )

        return PlanEvaluation(
            status=PlanStatus.coherent,
            reason="Plan is on track.",
            failure_counts=self.failure_counts.copy()
        )