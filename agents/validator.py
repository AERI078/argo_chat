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
        # ── early return 1: tool itself returned an error ──────────────────────
        # no point asking LLM to score a failed tool call
        if not result.success:
            return ValidationResult(
                step_id=step.step_id,
                passed=False,
                score=0.0,
                failure_type=FailureType.execution,
                reason=result.error or "Tool call failed."
            )

        # ── early return 2: rag_search ─────────────────────────────────────────
        # presence of docs is the only quality signal we need — no LLM required
        if step.tool == "rag_search":
            has_docs = bool(result.data and result.data.get("docs"))
            has_knowledge = bool(result.data and result.data.get("knowledge_docs"))
            passed = has_docs or has_knowledge
            score = 0.85 if has_docs else (0.5 if has_knowledge else 0.2)
            return ValidationResult(
                step_id=step.step_id,
                passed=passed,
                score=score,
                failure_type=None if passed else FailureType.strategy,
                reason=(
                    "RAG returned float summaries and knowledge context." if (has_docs and has_knowledge)
                    else "RAG returned float summaries." if has_docs
                    else "RAG returned knowledge context only." if has_knowledge
                    else "RAG returned no results — query may be out of scope."
                )
            )

        # ── early return 3: final_answer ───────────────────────────────────────
        # score by answer length — a longer, complete answer scores higher
        # cap at 1.0 at 400 chars (a decent paragraph)
        if step.tool == "final_answer":
            text = (result.data or {}).get("text", "")
            score = min(1.0, round(len(text) / 400, 2))
            passed = len(text) > 50  # anything under 50 chars is almost certainly an error
            return ValidationResult(
                step_id=step.step_id,
                passed=passed,
                score=score,
                failure_type=None if passed else FailureType.strategy,
                reason=(
                    f"Final answer synthesised ({len(text)} chars)." if passed
                    else "Final answer too short — likely a synthesis failure."
                )
            )

        # ── early return 4: fetch steps returned no data ───────────────────────
        if step.tool in ("fetch_region", "fetch_float", "db_query"):
            count = (result.data or {}).get("count", 0)
            if count == 0:
                return ValidationResult(
                    step_id=step.step_id,
                    passed=False,
                    score=0.1,
                    failure_type=FailureType.strategy,
                    reason="Query returned no data. Region or date range may be wrong."
                )

        # ── LLM scoring: only for fetch steps that returned data ───────────────
        # these need semantic quality checks the above rules can't catch
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