# agents/tools.py — all tools the agent can call, plus their schemas
# each tool takes inputs, does one thing, returns a ToolResult

from dataclasses import dataclass
from typing import Optional, Any
from data_pipeline.fetcher import fetch_profiles_by_region, fetch_float_by_id
from data_pipeline.db import cache_profiles, query_profiles
from rag.pipeline import RAGPipeline


@dataclass
class ToolResult:
    tool: str
    success: bool
    data: Any
    error: Optional[str] = None


def rag_search(query: str, rag: RAGPipeline) -> ToolResult:
    """Semantic search over float summaries. Use for conceptual or exploratory questions."""
    try:
        result = rag.retrieve(query)
        return ToolResult(tool="rag_search", success=True, data={"docs": result.docs})
    except Exception as e:
        return ToolResult(tool="rag_search", success=False, data=None, error=str(e))


def fetch_region(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                 date_start: str, date_end: str) -> ToolResult:
    """
    Fetches live Argo profiles for a region and date range from argopy.
    Use when the user specifies a location and time — this gets real data.
    Also caches results to Supabase so repeat queries are instant.
    """
    try:
        df = fetch_profiles_by_region(lat_min, lat_max, lon_min, lon_max, date_start, date_end)
        cache_profiles(df)
        rows = df.to_dict(orient="records")
        # convert date objects to strings for JSON serialisation
        for r in rows:
            r["date"] = str(r["date"])
        return ToolResult(tool="fetch_region", success=True, data={"rows": rows, "count": len(rows)})
    except Exception as e:
        return ToolResult(tool="fetch_region", success=False, data=None, error=str(e))


def fetch_float(float_id: str) -> ToolResult:
    """Fetches full profile history for a specific Argo float by ID."""
    try:
        df = fetch_float_by_id(float_id)
        rows = df.to_dict(orient="records")
        for r in rows:
            r["date"] = str(r["date"])
        return ToolResult(tool="fetch_float", success=True, data={"rows": rows, "count": len(rows)})
    except Exception as e:
        return ToolResult(tool="fetch_float", success=False, data=None, error=str(e))


def db_query(sql: str) -> ToolResult:
    """
    Runs a SELECT query against cached profiles in Supabase.
    Use after fetch_region has already populated the cache for this region.
    Only SELECT allowed — no mutations.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return ToolResult(tool="db_query", success=False, data=None,
                          error="Only SELECT queries are allowed.")
    try:
        rows = query_profiles(sql)
        for r in rows:
            if "date" in r:
                r["date"] = str(r["date"])
        return ToolResult(tool="db_query", success=True, data={"rows": rows, "count": len(rows)})
    except Exception as e:
        return ToolResult(tool="db_query", success=False, data=None, error=str(e))


def generate_chart(rows: list[dict], chart_type: str) -> ToolResult:
    """
    Converts data rows into a Plotly chart spec dict.
    The frontend receives this and renders it directly with st.plotly_chart().
    Supported: depth_profile | trajectory | time_series
    """
    try:
        if chart_type == "depth_profile":
            spec = {
                "type": "depth_profile",
                "data": {
                    "pressure": [r["pressure_dbar"] for r in rows],
                    "temperature": [r["temperature_c"] for r in rows],
                    "salinity": [r["salinity_psu"] for r in rows],
                    "float_id": [r["float_id"] for r in rows],
                }
            }
        elif chart_type == "trajectory":
            spec = {
                "type": "trajectory",
                "data": {
                    "lat": [r["lat"] for r in rows],
                    "lon": [r["lon"] for r in rows],
                    "float_id": [r["float_id"] for r in rows],
                    "date": [r["date"] for r in rows],
                }
            }
        elif chart_type == "time_series":
            spec = {
                "type": "time_series",
                "data": {
                    "date": [r["date"] for r in rows],
                    "temperature": [r["temperature_c"] for r in rows],
                    "salinity": [r["salinity_psu"] for r in rows],
                }
            }
        else:
            return ToolResult(tool="generate_chart", success=False, data=None,
                              error=f"Unknown chart_type: {chart_type}. Use depth_profile, trajectory, or time_series.")
        return ToolResult(tool="generate_chart", success=True, data=spec)
    except Exception as e:
        return ToolResult(tool="generate_chart", success=False, data=None, error=str(e))


# tool schemas — agent reads these to know what tools exist and how to call them
TOOL_SCHEMAS = {
    "rag_search": {
        "description": "Semantic search over Argo float summaries in FAISS. Use for conceptual or exploratory questions.",
        "params": {"query": "string"}
    },
    "fetch_region": {
        "description": "Fetch live Argo profiles for a region and time range from argopy. Use when user specifies a location and dates.",
        "params": {
            "lat_min": "float", "lat_max": "float",
            "lon_min": "float", "lon_max": "float",
            "date_start": "string (YYYY-MM)", "date_end": "string (YYYY-MM)"
        }
    },
    "fetch_float": {
        "description": "Fetch full profile history for a specific Argo float by its ID.",
        "params": {"float_id": "string"}
    },
    "db_query": {
        "description": "Run a SELECT query on cached profiles in Supabase. Use after fetch_region has populated the cache.",
        "params": {"sql": "string — SELECT query against argo_profiles table"}
    },
    "generate_chart": {
        "description": "Generate a Plotly chart spec from data rows.",
        "params": {
            "rows": "list of dicts from fetch_region or db_query",
            "chart_type": "depth_profile | trajectory | time_series"
        }
    },
    "final_answer": {
        "description": "Return the final response to the user. Always end with this.",
        "params": {
            "text": "string — plain language answer",
            "chart_spec": "optional — chart spec dict from generate_chart, or null"
        }
    }
}