# agents/validator.py — inspects each StepResult and decides if it passes quality criteria
# single responsibility: score a result and classify any failure, nothing else

import json
from agents.llm_caller import LLMCaller
from agents.prompt import validator_prompt
from agents.contracts import TaskStep, StepResult, ValidationResult, FailureType


class ValidatorAgent:
    def __init__(self, llm: LLMCaller):
        self.llm = llm

    def validate(self, step: TaskStep, result: StepResult) -> ValidationResult:
        # if the tool itself returned an error, classify it without calling LLM
        if not result.success:
            return ValidationResult(
                step_id=step.step_id,
                passed=False,
                score=0.0,
                failure_type=FailureType.execution,
                reason=result.error or "Tool call failed."
            )

        # empty data from a fetch step means dependency or strategy issue
        if result.tool in ("fetch_region", "fetch_float", "db_query"):
            count = result.data.get("count", 0) if result.data else 0
            if count == 0:
                return ValidationResult(
                    step_id=step.step_id,
                    passed=False,
                    score=0.1,
                    failure_type=FailureType.strategy,
                    reason="Query returned no data. Region or date range may be wrong."
                )

        # for semantic quality checks, ask the LLM to score
        messages = validator_prompt(step.model_dump(), result.model_dump())
        raw = self.llm.call(messages, temperature=0.0)

        try:
            data = json.loads(raw)
            return ValidationResult(
                step_id=step.step_id,
                passed=data.get("passed", False),
                score=float(data.get("score", 0.5)),
                failure_type=FailureType(data["failure_type"]) if data.get("failure_type") else None,
                reason=data.get("reason", "")
            )
        except (json.JSONDecodeError, ValueError):
            # if validator LLM output is broken, pass with low confidence rather than crash
            return ValidationResult(
                step_id=step.step_id, passed=True, score=0.5,
                reason="Validator output unparseable — passed with low confidence."
            )
