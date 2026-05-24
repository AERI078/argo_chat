# agents/executor.py — runs exactly one TaskStep using the bounded tool set
# single responsibility: call the right tool, return a StepResult

import json
from agents.llm_caller import LLMCaller
from agents.parser import parse_action
from agents.prompt import executor_prompt, synthesizer_prompt
from agents.contracts import TaskStep, StepResult
from agents import tools as T
from rag.pipeline import RAGPipeline


class ExecutorAgent:
    def __init__(self, llm: LLMCaller, rag: RAGPipeline):
        self.llm = llm
        self.rag = rag

    def execute(self, step: TaskStep, context: str) -> StepResult:
        # final_answer step is handled differently — needs LLM to synthesise
        if step.tool == "final_answer":
            return self._synthesise(step, context)

        # for all other steps, ask LLM to produce the Action call
        messages = executor_prompt(step.model_dump(), context)
        raw = self.llm.call(messages, temperature=0.0)
        action = parse_action(raw)

        if not action.success:
            return StepResult(step_id=step.step_id, tool=step.tool,
                              success=False, error=action.error)

        result = self._dispatch(action.tool, action.params)
        return StepResult(
            step_id=step.step_id,
            tool=action.tool,
            success=result.success,
            data=result.data,
            error=result.error
        )

    def _synthesise(self, step: TaskStep, context: str) -> StepResult:
        """For the final_answer step, synthesise a natural language response from all context."""
        messages = synthesizer_prompt(step.params.get("query", ""), context)
        answer = self.llm.call(messages, temperature=0.3)
        return StepResult(
            step_id=step.step_id,
            tool="final_answer",
            success=True,
            data={"text": answer, "chart_spec": step.params.get("chart_spec")}
        )

    def _dispatch(self, tool_name: str, params: dict) -> T.ToolResult:
        if tool_name == "rag_search":
            return T.rag_search(params.get("query", ""), self.rag)
        elif tool_name == "fetch_region":
            return T.fetch_region(
                params["lat_min"], params["lat_max"],
                params["lon_min"], params["lon_max"],
                params["date_start"], params["date_end"]
            )
        elif tool_name == "fetch_float":
            return T.fetch_float(params.get("float_id", ""))
        elif tool_name == "db_query":
            return T.db_query(params.get("sql", ""))
        elif tool_name == "generate_chart":
            return T.generate_chart(params.get("rows", []), params.get("chart_type", ""))
        else:
            return T.ToolResult(tool=tool_name, success=False, data=None,
                                error=f"Unknown tool: {tool_name}")
