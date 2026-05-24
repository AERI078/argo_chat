# agents/contracts.py — typed data contracts for every agent handoff
# Pydantic enforces shape at runtime — no silent data corruption between agents

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class FailureType(str, Enum):
    execution   = "execution"    # tool call failed (network, timeout, bad response)
    dependency  = "dependency"   # step needs data that wasn't fetched yet
    strategy    = "strategy"     # the chosen approach is wrong for this query
    invalidation = "invalidation" # external state changed, plan is now incoherent


class TaskStep(BaseModel):
    step_id: int
    description: str                          # plain English — what this step does
    tool: str                                 # which tool the executor should call
    params: dict                              # tool parameters
    depends_on: list[int] = []               # step_ids this step needs to complete first


class TaskPlan(BaseModel):
    query: str
    steps: list[TaskStep]
    strategy_rationale: str                   # why the planner chose this approach


class StepResult(BaseModel):
    step_id: int
    tool: str
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class ValidationResult(BaseModel):
    step_id: int
    passed: bool
    score: float                              # 0.0 to 1.0
    failure_type: Optional[FailureType] = None
    reason: str


class PlanStatus(str, Enum):
    coherent    = "coherent"     # plan is still valid, continue executing
    replan      = "replan"       # plan needs to be regenerated
    unrecoverable = "unrecoverable"  # too many failures, give up


class PlanEvaluation(BaseModel):
    status: PlanStatus
    reason: str
    failure_counts: dict[str, int]            # tracks failures per FailureType


class OrchestratorResponse(BaseModel):
    query: str
    answer: str
    chart_spec: Optional[dict] = None
    plan: Optional[TaskPlan] = None
    step_results: list[StepResult] = []
    success: bool