# test_pipeline.py — runs at project root: python test_pipeline.py
# tests each layer independently so you know exactly where a failure is
# does NOT require live Argo data for most tests — uses mocks where possible

import sys
import json
import traceback
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

PASS = "  PASS"
FAIL = "  FAIL"
SKIP = "  SKIP"


def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def run(label: str, fn):
    try:
        fn()
        print(f"{PASS}  {label}")
        return True
    except Exception as e:
        print(f"{FAIL}  {label}")
        print(f"       {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


# ─────────────────────────────────────────────
# TEST 1 — CONFIG
# ─────────────────────────────────────────────
section("1. Config")

def test_config_imports():
    from config import (GROQ_API_KEY, LLM_MODEL, DATABASE_URL,
                        EMBED_MODEL, TOP_K, FAISS_INDEX_PATH, FAISS_DOCS_PATH,
                        MAX_PLAN_STEPS, MAX_REPLAN_ATTEMPTS, FAILURE_THRESHOLDS,
                        DEFAULT_LAT_RANGE, DEFAULT_LON_RANGE, DEFAULT_PRESSURE_RANGE)
    assert LLM_MODEL == "llama-3.1-70b-versatile"
    assert isinstance(FAILURE_THRESHOLDS, dict)
    assert "strategy" in FAILURE_THRESHOLDS

def test_config_env_vars():
    from config import GROQ_API_KEY, DATABASE_URL
    # warn if not set but don't fail — dev may not have .env yet
    if not GROQ_API_KEY:
        print(f"  WARN  GROQ_API_KEY not set in .env")
    if not DATABASE_URL:
        print(f"  WARN  DATABASE_URL not set in .env")

run("config imports and constants", test_config_imports)
run("env vars present", test_config_env_vars)


# ─────────────────────────────────────────────
# TEST 2 — LOGGER
# ─────────────────────────────────────────────
section("2. Logger")

def test_logger_basic():
    from agents.logger import PipelineLogger
    log = PipelineLogger(query="test query")
    log.start("planner")
    log.success("planner", {"steps": 3})
    log.start("executor", step_id=1, tool="rag_search")
    log.failure("executor", error="timeout", failure_type="execution", step_id=1)
    log.replan(reason="strategy failure", attempt=1)
    log.finish(success=True, answer_length=200)
    trace = log.trace()
    assert trace["query"] == "test query"
    assert len(trace["events"]) == 6
    assert trace["events"][0]["stage"] == "planner"
    assert trace["events"][0]["status"] == "started"

def test_logger_trace_structure():
    from agents.logger import PipelineLogger
    log = PipelineLogger(query="salinity query")
    log.start("executor", step_id=1, tool="fetch_region")
    log.success("executor", step_id=1, data={"count": 50})
    trace = log.trace()
    assert "run_id" in trace
    assert "events" in trace

run("logger stages and trace", test_logger_basic)
run("logger trace structure", test_logger_trace_structure)


# ─────────────────────────────────────────────
# TEST 3 — CONTRACTS
# ─────────────────────────────────────────────
section("3. Contracts (Pydantic)")

def test_contracts_task_plan():
    from agents.contracts import TaskPlan, TaskStep
    step = TaskStep(step_id=1, description="search", tool="rag_search",
                    params={"query": "salinity"}, depends_on=[])
    plan = TaskPlan(query="test", steps=[step], strategy_rationale="test rationale")
    assert plan.steps[0].tool == "rag_search"

def test_contracts_validation_result():
    from agents.contracts import ValidationResult, FailureType
    v = ValidationResult(step_id=1, passed=False, score=0.1,
                         failure_type=FailureType.strategy, reason="bad approach")
    assert v.failure_type == FailureType.strategy

def test_contracts_orchestrator_response():
    from agents.contracts import OrchestratorResponse
    r = OrchestratorResponse(query="q", answer="a", success=True)
    assert r.chart_spec is None
    assert r.step_results == []

run("TaskPlan and TaskStep", test_contracts_task_plan)
run("ValidationResult with FailureType", test_contracts_validation_result)
run("OrchestratorResponse defaults", test_contracts_orchestrator_response)


# ─────────────────────────────────────────────
# TEST 4 — PARSER
# ─────────────────────────────────────────────
section("4. Parser")

def test_parser_valid():
    from agents.parser import parse_action
    raw = 'Thought: I need to search\nAction: {"tool": "rag_search", "params": {"query": "salinity"}}'
    result = parse_action(raw)
    assert result.success
    assert result.tool == "rag_search"
    assert result.params["query"] == "salinity"

def test_parser_no_action():
    from agents.parser import parse_action
    result = parse_action("I will now think about this problem...")
    assert not result.success
    assert "No Action block" in result.error

def test_parser_bad_json():
    from agents.parser import parse_action
    result = parse_action('Action: {"tool": "rag_search", "params": {bad json}}')
    assert not result.success
    assert "JSON parse error" in result.error

def test_parser_multiline_json():
    from agents.parser import parse_action
    raw = 'Thought: fetching\nAction: {\n  "tool": "fetch_region",\n  "params": {"lat_min": 10}\n}'
    result = parse_action(raw)
    assert result.success
    assert result.tool == "fetch_region"

run("valid action parse", test_parser_valid)
run("missing action block", test_parser_no_action)
run("malformed JSON", test_parser_bad_json)
run("multiline JSON action", test_parser_multiline_json)


# ─────────────────────────────────────────────
# TEST 5 — RAG COMPONENTS (mocked embeddings)
# ─────────────────────────────────────────────
section("5. RAG Components")

def test_embedder_shape():
    from rag.embedder import Embedder
    e = Embedder()
    vecs = e.embed(["Arabian Sea salinity", "Bay of Bengal temperature"])
    assert vecs.shape == (2, 384)
    assert vecs.dtype == np.float32

def test_embedder_single():
    from rag.embedder import Embedder
    e = Embedder()
    v = e.embed_one("test query")
    assert v.shape == (384,)

def test_vector_store_add_search():
    from rag.vector_store import VectorStore
    store = VectorStore(dim=384)
    docs = ["Float 1234 near Arabian Sea", "Float 5678 in Bay of Bengal"]
    vecs = np.random.rand(2, 384).astype("float32")
    store.add(vecs, docs)
    assert not store.is_empty
    query_vec = np.random.rand(384).astype("float32")
    results = store.search(query_vec, k=2)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)

def test_summarizer():
    from rag.summarizer import summarize_profiles
    df = pd.DataFrame({
        "float_id": ["1234", "1234"],
        "lat": [15.0, 15.0],
        "lon": [65.0, 65.0],
        "date": [pd.Timestamp("2023-03-01").date(), pd.Timestamp("2023-03-01").date()],
        "pressure_dbar": [10.0, 50.0],
        "temperature_c": [28.5, 26.1],
        "salinity_psu": [36.2, 36.5]
    })
    summaries = summarize_profiles(df)
    assert len(summaries) == 1  # grouped by float+date
    assert "1234" in summaries[0]
    assert "Arabian Sea" in summaries[0] or "Indian Ocean" in summaries[0]

run("embedder output shape", test_embedder_shape)
run("embedder single query", test_embedder_single)
run("vector store add and search", test_vector_store_add_search)
run("summarizer groups by float+date", test_summarizer)


# ─────────────────────────────────────────────
# TEST 6 — TOOLS (mocked external calls)
# ─────────────────────────────────────────────
section("6. Tools")

def test_tool_rag_search_mock():
    from agents.tools import rag_search
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = MagicMock(docs=["Float 123 in Arabian Sea"])
    result = rag_search("salinity", mock_rag)
    assert result.success
    assert "docs" in result.data
    assert len(result.data["docs"]) == 1

def test_tool_rag_search_error():
    from agents.tools import rag_search
    mock_rag = MagicMock()
    mock_rag.retrieve.side_effect = Exception("FAISS index not loaded")
    result = rag_search("salinity", mock_rag)
    assert not result.success
    assert "FAISS index not loaded" in result.error

def test_tool_db_query_blocks_mutation():
    from agents.tools import db_query
    result = db_query("DELETE FROM argo_profiles")
    assert not result.success
    assert "Only SELECT" in result.error

def test_tool_generate_chart_depth_profile():
    from agents.tools import generate_chart
    rows = [
        {"pressure_dbar": 10.0, "temperature_c": 28.5, "salinity_psu": 36.2, "float_id": "123"},
        {"pressure_dbar": 50.0, "temperature_c": 26.1, "salinity_psu": 36.5, "float_id": "123"},
    ]
    result = generate_chart(rows, "depth_profile")
    assert result.success
    assert result.data["type"] == "depth_profile"
    assert len(result.data["data"]["pressure"]) == 2

def test_tool_generate_chart_trajectory():
    from agents.tools import generate_chart
    rows = [{"lat": 15.0, "lon": 65.0, "float_id": "123", "date": "2023-03-01"}]
    result = generate_chart(rows, "trajectory")
    assert result.success
    assert result.data["type"] == "trajectory"

def test_tool_generate_chart_bad_type():
    from agents.tools import generate_chart
    result = generate_chart([], "heatmap")
    assert not result.success
    assert "Unknown chart_type" in result.error

run("rag_search with mock pipeline", test_tool_rag_search_mock)
run("rag_search handles error", test_tool_rag_search_error)
run("db_query blocks non-SELECT", test_tool_db_query_blocks_mutation)
run("generate_chart depth_profile", test_tool_generate_chart_depth_profile)
run("generate_chart trajectory", test_tool_generate_chart_trajectory)
run("generate_chart unknown type", test_tool_generate_chart_bad_type)


# ─────────────────────────────────────────────
# TEST 7 — PLAN EVALUATOR
# ─────────────────────────────────────────────
section("7. Plan Evaluator")

def _make_plan():
    from agents.contracts import TaskPlan, TaskStep
    return TaskPlan(
        query="test",
        steps=[
            TaskStep(step_id=1, description="s1", tool="rag_search", params={}, depends_on=[]),
            TaskStep(step_id=2, description="s2", tool="fetch_region", params={}, depends_on=[1]),
            TaskStep(step_id=3, description="s3", tool="final_answer", params={}, depends_on=[2]),
        ],
        strategy_rationale="test"
    )

def test_evaluator_coherent():
    from agents.plan_evaluator import PlanEvaluator
    from agents.contracts import ValidationResult, PlanStatus
    ev = PlanEvaluator()
    plan = _make_plan()
    v = ValidationResult(step_id=1, passed=True, score=0.9, reason="good")
    result = ev.evaluate(plan, v, completed_ids=[1])
    assert result.status == PlanStatus.coherent

def test_evaluator_strategy_failure_triggers_replan():
    from agents.plan_evaluator import PlanEvaluator
    from agents.contracts import ValidationResult, FailureType, PlanStatus
    ev = PlanEvaluator()
    plan = _make_plan()
    v = ValidationResult(step_id=1, passed=False, score=0.1,
                         failure_type=FailureType.strategy, reason="wrong tool")
    result = ev.evaluate(plan, v, completed_ids=[])
    assert result.status == PlanStatus.replan

def test_evaluator_execution_failure_allows_retries():
    from agents.plan_evaluator import PlanEvaluator
    from agents.contracts import ValidationResult, FailureType, PlanStatus
    ev = PlanEvaluator()
    plan = _make_plan()
    v = ValidationResult(step_id=1, passed=False, score=0.0,
                         failure_type=FailureType.execution, reason="timeout")
    # first execution failure — should still be coherent (threshold is 3)
    result = ev.evaluate(plan, v, completed_ids=[])
    assert result.status == PlanStatus.coherent

def test_evaluator_unmet_dependency():
    from agents.plan_evaluator import PlanEvaluator
    from agents.contracts import ValidationResult, PlanStatus
    ev = PlanEvaluator()
    plan = _make_plan()
    v = ValidationResult(step_id=2, passed=True, score=0.9, reason="ok")
    # step 2 depends on step 1, but step 1 not in completed_ids
    result = ev.evaluate(plan, v, completed_ids=[])
    assert result.status == PlanStatus.replan

run("coherent plan stays coherent", test_evaluator_coherent)
run("strategy failure triggers replan", test_evaluator_strategy_failure_triggers_replan)
run("execution failure allows retries", test_evaluator_execution_failure_allows_retries)
run("unmet dependency triggers replan", test_evaluator_unmet_dependency)


# ─────────────────────────────────────────────
# TEST 8 — PLANNER (mocked LLM)
# ─────────────────────────────────────────────
section("8. Planner Agent")

def test_planner_valid_output():
    from agents.planner import PlannerAgent
    mock_llm = MagicMock()
    mock_llm.call.return_value = json.dumps({
        "steps": [
            {"step_id": 1, "description": "search", "tool": "rag_search",
             "params": {"query": "salinity"}, "depends_on": []},
            {"step_id": 2, "description": "answer", "tool": "final_answer",
             "params": {"text": "", "chart_spec": None}, "depends_on": [1]}
        ],
        "strategy_rationale": "simple query needs only RAG"
    })
    planner = PlannerAgent(mock_llm)
    plan = planner.plan("what is salinity?")
    assert len(plan.steps) == 2
    assert plan.steps[0].tool == "rag_search"
    assert plan.steps[-1].tool == "final_answer"

def test_planner_fallback_on_bad_json():
    from agents.planner import PlannerAgent
    mock_llm = MagicMock()
    mock_llm.call.return_value = "I will now create a plan for you..."
    planner = PlannerAgent(mock_llm)
    plan = planner.plan("show me temperature profiles")
    # fallback plan should still be usable
    assert len(plan.steps) >= 2
    assert plan.steps[-1].tool == "final_answer"

def test_planner_respects_max_steps():
    from agents.planner import PlannerAgent
    from config import MAX_PLAN_STEPS
    mock_llm = MagicMock()
    # return more steps than allowed
    steps = [{"step_id": i, "description": f"step {i}", "tool": "rag_search",
              "params": {}, "depends_on": []} for i in range(1, 10)]
    mock_llm.call.return_value = json.dumps({"steps": steps, "strategy_rationale": "test"})
    planner = PlannerAgent(mock_llm)
    plan = planner.plan("complex query")
    assert len(plan.steps) <= MAX_PLAN_STEPS

run("valid LLM output produces correct plan", test_planner_valid_output)
run("bad LLM output falls back gracefully", test_planner_fallback_on_bad_json)
run("max steps cap is enforced", test_planner_respects_max_steps)


# ─────────────────────────────────────────────
# TEST 9 — VALIDATOR (mocked LLM)
# ─────────────────────────────────────────────
section("9. Validator Agent")

def test_validator_tool_error_is_execution_failure():
    from agents.validator import ValidatorAgent
    from agents.contracts import TaskStep, StepResult, FailureType
    mock_llm = MagicMock()
    v = ValidatorAgent(mock_llm)
    step = TaskStep(step_id=1, description="fetch", tool="fetch_region", params={}, depends_on=[])
    result = StepResult(step_id=1, tool="fetch_region", success=False, error="connection refused")
    validation = v.validate(step, result)
    assert not validation.passed
    assert validation.failure_type == FailureType.execution
    mock_llm.call.assert_not_called()  # should not waste LLM call on obvious tool error

def test_validator_empty_fetch_is_strategy_failure():
    from agents.validator import ValidatorAgent
    from agents.contracts import TaskStep, StepResult, FailureType
    mock_llm = MagicMock()
    v = ValidatorAgent(mock_llm)
    step = TaskStep(step_id=1, description="fetch", tool="fetch_region", params={}, depends_on=[])
    result = StepResult(step_id=1, tool="fetch_region", success=True, data={"rows": [], "count": 0})
    validation = v.validate(step, result)
    assert not validation.passed
    assert validation.failure_type == FailureType.strategy

def test_validator_good_result_passes():
    from agents.validator import ValidatorAgent
    from agents.contracts import TaskStep, StepResult
    mock_llm = MagicMock()
    mock_llm.call.return_value = json.dumps({
        "passed": True, "score": 0.9, "failure_type": None, "reason": "good data"
    })
    v = ValidatorAgent(mock_llm)
    step = TaskStep(step_id=1, description="search", tool="rag_search", params={}, depends_on=[])
    result = StepResult(step_id=1, tool="rag_search", success=True,
                        data={"docs": ["Float 123 in Arabian Sea"]})
    validation = v.validate(step, result)
    assert validation.passed
    assert validation.score == 0.9

run("tool error → execution failure (no LLM call)", test_validator_tool_error_is_execution_failure)
run("empty fetch → strategy failure", test_validator_empty_fetch_is_strategy_failure)
run("good result passes validation", test_validator_good_result_passes)


# ─────────────────────────────────────────────
# TEST 10 — FULL ORCHESTRATOR (fully mocked)
# ─────────────────────────────────────────────
section("10. Orchestrator (end-to-end mock)")

def test_orchestrator_happy_path():
    from agents.orchestrator import Orchestrator
    from agents.contracts import (TaskPlan, TaskStep, StepResult,
                                   ValidationResult, OrchestratorResponse)

    # mock planner returns a simple 2-step plan
    mock_planner = MagicMock()
    mock_planner.plan.return_value = TaskPlan(
        query="what is the salinity?",
        steps=[
            TaskStep(step_id=1, description="search", tool="rag_search",
                     params={"query": "salinity"}, depends_on=[]),
            TaskStep(step_id=2, description="answer", tool="final_answer",
                     params={"text": "", "chart_spec": None}, depends_on=[1]),
        ],
        strategy_rationale="simple"
    )
    # mock executor returns success for both steps
    mock_executor = MagicMock()
    mock_executor.execute.side_effect = [
        StepResult(step_id=1, tool="rag_search", success=True,
                   data={"docs": ["Float 123 Arabian Sea salinity 36.2 PSU"]}),
        StepResult(step_id=2, tool="final_answer", success=True,
                   data={"text": "Salinity in the Arabian Sea averages 36.2 PSU.", "chart_spec": None}),
    ]
    # mock validator always passes
    mock_validator = MagicMock()
    mock_validator.validate.return_value = ValidationResult(
        step_id=1, passed=True, score=0.9, reason="good"
    )
    mock_replan = MagicMock()

    orch = Orchestrator(mock_planner, mock_executor, mock_validator, mock_replan)
    response, trace = orch.run("what is the salinity?")

    assert response.success
    assert "36.2" in response.answer
    assert "run_id" in trace
    assert len(trace["events"]) > 0

def test_orchestrator_replan_on_strategy_failure():
    from agents.orchestrator import Orchestrator
    from agents.contracts import (TaskPlan, TaskStep, StepResult,
                                   ValidationResult, FailureType)

    def make_plan():
        return TaskPlan(
            query="show temperature",
            steps=[
                TaskStep(step_id=1, description="fetch", tool="fetch_region",
                         params={"lat_min":0,"lat_max":10,"lon_min":60,"lon_max":80,
                                 "date_start":"2023-01","date_end":"2023-03"},
                         depends_on=[]),
                TaskStep(step_id=2, description="answer", tool="final_answer",
                         params={}, depends_on=[1]),
            ],
            strategy_rationale="test"
        )

    mock_planner = MagicMock()
    mock_planner.plan.return_value = make_plan()

    call_count = {"n": 0}
    def executor_side_effect(step, context, query=""):
        call_count["n"] += 1
        if step.tool == "fetch_region" and call_count["n"] == 1:
            return StepResult(step_id=1, tool="fetch_region", success=True,
                              data={"rows": [], "count": 0})  # empty = strategy failure
        return StepResult(step_id=step.step_id, tool="final_answer", success=True,
                          data={"text": "Here is the temperature data.", "chart_spec": None})

    mock_executor = MagicMock()
    mock_executor.execute.side_effect = executor_side_effect

    mock_validator = MagicMock()
    mock_validator.validate.side_effect = lambda step, result: ValidationResult(
        step_id=step.step_id,
        passed=result.data.get("count", 1) > 0 if result.success and result.data else result.success,
        score=0.9 if result.success else 0.0,
        failure_type=FailureType.strategy if (result.success and result.data and result.data.get("count", 1) == 0) else None,
        reason="empty" if (result.success and result.data and result.data.get("count", 1) == 0) else "ok"
    )

    mock_replan = MagicMock()
    mock_replan.replan.return_value = TaskPlan(
        query="show temperature",
        steps=[
            TaskStep(step_id=1, description="answer directly", tool="final_answer",
                     params={}, depends_on=[]),
        ],
        strategy_rationale="replanned"
    )

    orch = Orchestrator(mock_planner, mock_executor, mock_validator, mock_replan)
    response, trace = orch.run("show temperature")
    assert mock_replan.replan.called

run("happy path returns answer + trace", test_orchestrator_happy_path)
run("strategy failure triggers replan", test_orchestrator_replan_on_strategy_failure)


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print("  Done. Add --verbose flag for full tracebacks on failures.")
print(f"{'='*55}\n")