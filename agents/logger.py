# logger.py — structured pipeline logger for FloatChat
# tracks every stage of the multi-agent pipeline in one place
# lives at project root so every module can import it cleanly

import logging
import json
import time
from datetime import datetime, timezone


class PipelineLogger:
    """
    Structured logger that tracks the full pipeline as a sequence of events.
    Each event has a stage, status, and optional data payload.
    
    Usage:
        log = PipelineLogger(query="show me salinity in Arabian Sea")
        log.start("planner")
        log.success("planner", {"steps": 4})
        log.start("executor", step_id=1, tool="rag_search")
        log.failure("executor", step_id=1, error="timeout")
        log.finish()  # prints full trace summary
    """

    def __init__(self, query: str):
        self.query = query
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.events: list[dict] = []
        self._timers: dict[str, float] = {}

        # standard Python logger — outputs to console, picked up by Render/Streamlit logs
        self._log = logging.getLogger("floatchat")
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self._log.addHandler(handler)
            self._log.setLevel(logging.INFO)

        self._log.info(f"[{self.run_id}] NEW QUERY: {query}")

    def start(self, stage: str, **kwargs):
        """Call when a pipeline stage begins."""
        self._timers[stage] = time.time()
        event = {"stage": stage, "status": "started", "ts": _now(), **kwargs}
        self.events.append(event)
        self._log.info(f"[{self.run_id}] START  {stage}" + (f" | {kwargs}" if kwargs else ""))

    def success(self, stage: str, data: dict = None, **kwargs):
        """Call when a stage completes successfully."""
        elapsed = self._elapsed(stage)
        event = {"stage": stage, "status": "success", "ts": _now(),
                 "elapsed_ms": elapsed, **(data or {}), **kwargs}
        self.events.append(event)
        self._log.info(f"[{self.run_id}] OK     {stage} ({elapsed}ms)" +
                       (f" | {_truncate(data)}" if data else ""))

    def failure(self, stage: str, error: str, failure_type: str = None, **kwargs):
        """Call when a stage fails."""
        elapsed = self._elapsed(stage)
        event = {"stage": stage, "status": "failure", "ts": _now(),
                 "elapsed_ms": elapsed, "error": error,
                 "failure_type": failure_type, **kwargs}
        self.events.append(event)
        self._log.warning(f"[{self.run_id}] FAIL   {stage} ({elapsed}ms) | {error}" +
                          (f" [{failure_type}]" if failure_type else ""))

    def replan(self, reason: str, attempt: int):
        """Call when the replan engine is triggered."""
        event = {"stage": "replan_engine", "status": "replan",
                 "ts": _now(), "reason": reason, "attempt": attempt}
        self.events.append(event)
        self._log.warning(f"[{self.run_id}] REPLAN attempt={attempt} | {reason}")

    def finish(self, success: bool, answer_length: int = 0):
        """Call at the end of the full pipeline run."""
        total = sum(e.get("elapsed_ms", 0) for e in self.events)
        event = {"stage": "orchestrator", "status": "done", "ts": _now(),
                 "success": success, "total_ms": total,
                 "steps_run": len([e for e in self.events if e["stage"] == "executor"]),
                 "answer_length": answer_length}
        self.events.append(event)
        status_str = "SUCCESS" if success else "FAILED"
        self._log.info(
            f"[{self.run_id}] {status_str} | total={total}ms | "
            f"steps={event['steps_run']} | answer_len={answer_length}"
        )

    def trace(self) -> dict:
        """Returns the full structured trace — used by API to expose /trace endpoint."""
        return {"run_id": self.run_id, "query": self.query, "events": self.events}

    def _elapsed(self, stage: str) -> int:
        start = self._timers.pop(stage, time.time())
        return int((time.time() - start) * 1000)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(data: dict, max_len: int = 120) -> str:
    s = json.dumps(data)
    return s[:max_len] + "..." if len(s) > max_len else s