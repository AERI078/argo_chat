# agents/parser.py — extracts Action JSON from raw LLM output
# uses brace-balancing instead of regex — regex breaks on nested JSON objects

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedAction:
    tool: str
    params: dict
    success: bool
    error: Optional[str] = None


def parse_action(llm_output: str) -> ParsedAction:
    json_str = _extract_action_json(llm_output)
    if json_str is None:
        return ParsedAction(tool="", params={}, success=False,
                            error="No Action block found in LLM output.")
    try:
        action = json.loads(json_str)
        return ParsedAction(tool=action.get("tool", ""), params=action.get("params", {}), success=True)
    except json.JSONDecodeError as e:
        return ParsedAction(tool="", params={}, success=False, error=f"JSON parse error: {e}")


def _extract_action_json(text: str) -> Optional[str]:
    """
    Finds 'Action:' then walks forward counting braces until the JSON object closes.
    Handles nested objects correctly — regex with .*? stops at the first } it finds.
    """
    action_pos = text.find("Action:")
    if action_pos == -1:
        return None
    brace_start = text.find("{", action_pos)
    if brace_start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]
    return None