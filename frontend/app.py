# frontend/app.py — Streamlit chat UI

import streamlit as st
import httpx
import plotly.graph_objects as go
import os
import threading

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ArgoChat", page_icon="🌊", layout="wide")


# ── helpers ───────────────────────────────────────────────────────────────────

def get_backend_status() -> dict:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable", "orchestrator_ready": False}


def send_message(message: str) -> dict:
    try:
        r = httpx.post(f"{BACKEND_URL}/chat", json={"message": message}, timeout=120)
        if r.status_code == 503:
            return {
                "answer": "⏳ The system is still initialising. Please wait 60 seconds and try again.",
                "success": False
            }
        return r.json()
    except httpx.TimeoutException:
        return {"answer": "Request timed out. Try a more specific query.", "success": False}
    except Exception as e:
        return {"answer": f"Could not reach backend: {e}", "success": False}


def send_message_with_status(message: str) -> dict:
    """
    Runs send_message in a background thread while showing rotating
    stage indicators so the user sees progress instead of a frozen spinner.
    """
    stages = [
        "🗺️ Planning query approach...",
        "🔍 Searching float summaries...",
        "🌊 Retrieving ocean context...",
        "🧠 Synthesising answer...",
    ]

    status_placeholder = st.empty()
    result_container = {}
    done_flag = threading.Event()

    def fetch():
        result_container["result"] = send_message(message)
        done_flag.set()

    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()

    i = 0
    while not done_flag.is_set():
        status_placeholder.info(stages[i % len(stages)])
        done_flag.wait(timeout=4)
        i += 1

    status_placeholder.empty()
    return result_container.get("result", {"answer": "No response received.", "success": False})


# def render_confidence(confidence: float):
#     """Color-coded confidence badge based on average validation score."""
#     if confidence >= 0.7:
#         st.success(f"🎯 High confidence: {confidence:.0%}", icon="✅")
#     elif confidence >= 0.5:
#         st.info(f"📊 Medium confidence: {confidence:.0%}", icon="ℹ️")
#     else:
#         st.warning(f"⚠️ Low confidence: {confidence:.0%}", icon="⚠️")
#         st.caption("Some steps had difficulty. Results may be incomplete.")


def render_chart(chart_spec: dict):
    chart_type = chart_spec.get("type")
    data = chart_spec.get("data", {})

    if chart_type == "depth_profile":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.get("temperature", []), y=data.get("pressure", []),
            mode="lines+markers", name="Temperature (°C)", line=dict(color="#e74c3c")
        ))
        fig.add_trace(go.Scatter(
            x=data.get("salinity", []), y=data.get("pressure", []),
            mode="lines+markers", name="Salinity (PSU)",
            line=dict(color="#3498db"), xaxis="x2"
        ))
        fig.update_layout(
            title="Depth Profile",
            yaxis=dict(title="Pressure (dbar)", autorange="reversed"),
            xaxis=dict(title="Temperature (°C)"),
            xaxis2=dict(title="Salinity (PSU)", overlaying="x", side="top"),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "trajectory":
        fig = go.Figure(go.Scattergeo(
            lat=data.get("lat", []), lon=data.get("lon", []),
            mode="lines+markers", text=data.get("date", []),
            marker=dict(size=6, color="#2ecc71"),
            line=dict(width=1, color="#2ecc71")
        ))
        fig.update_layout(
            title="Float Trajectory",
            geo=dict(
                showland=True, landcolor="lightgray",
                showocean=True, oceancolor="#d6eaf8",
                projection_type="natural earth",
                center=dict(lat=15, lon=70),
                projection_scale=3
            ),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "time_series":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.get("date", []), y=data.get("temperature", []),
            mode="lines+markers", name="Temperature (°C)", line=dict(color="#e74c3c")
        ))
        fig.add_trace(go.Scatter(
            x=data.get("date", []), y=data.get("salinity", []),
            mode="lines+markers", name="Salinity (PSU)",
            line=dict(color="#3498db"), yaxis="y2"
        ))
        fig.update_layout(
            title="Time Series",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Salinity (PSU)", overlaying="y", side="right"),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def render_trace(trace: dict):
    with st.expander("🔍 How I answered this", expanded=False):
        for event in trace.get("events", []):
            stage = event.get("stage", "")
            status = event.get("status", "")
            icon = {"success": "✅", "failure": "❌", "replan": "🔄",
                    "started": "⏳", "done": "🏁"}.get(status, "•")
            elapsed = f" ({event['elapsed_ms']}ms)" if "elapsed_ms" in event else ""
            detail = ""
            if stage == "planner" and status == "success":
                detail = f" — {event.get('steps', '')} steps planned"
            elif stage == "executor":
                detail = f" — {event.get('tool', '')} step {event.get('step_id', '')}"
            elif stage == "validator":
                score = event.get('score', '')
                detail = f" — score {score}" if status == "success" else f" — {event.get('error', '')}"
            elif stage == "replan":
                detail = f" — {event.get('reason', '')}"
            st.markdown(f"{icon} **{stage}**{detail}{elapsed}")


def _render_assistant_message(result: dict):
    """
    Renders answer, confidence, chart, and trace for one assistant response.
    Handles both live results (key: 'answer') and session-stored messages (key: 'content').
    """
    answer = result.get("answer") or result.get("content", "")
    # st.markdown(answer)
    # if result.get("confidence") is not None:
    #     render_confidence(result["confidence"])
    # if result.get("chart_spec"):
    #     render_chart(result["chart_spec"])
    # if result.get("trace"):
    #     render_trace(result["trace"])


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌊 ArgoChat")
    st.caption("AI-powered Argo ocean data explorer")
    st.divider()

    status = get_backend_status()
    if status.get("orchestrator_ready"):
        st.success("Ready", icon="🟢")
    elif status.get("status") == "ok":
        st.warning("Initialising — building ocean data index...", icon="🟡")
        st.caption("This takes ~60 seconds on first start.")
    else:
        st.error("Backend offline", icon="🔴")
        st.caption(f"Expecting backend at {BACKEND_URL}")

    st.divider()
    st.markdown("**Try these queries**")
    examples = [
        "What is a thermocline and why does it matter?",
        "Why is the Arabian Sea saltier than the Bay of Bengal?",
        "What's happening in the Arabian Sea?",
        "Show temperature profiles near lat 15, lon 65 in June 2023",
        "What are Argo floats and how do they work?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── main ──────────────────────────────────────────────────────────────────────

st.title("ArgoChat 🌊")
st.caption("Ask anything about Argo ocean float data — in plain English or technical terms.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# render full chat history — always runs on every Streamlit rerender
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant_message(msg)
        else:
            st.markdown(msg["content"])


def _process_and_display(query: str):
    """
    Shared logic for both sidebar button clicks and typed input.
    Appends user message, fetches response, renders it, appends assistant message.
    Storing all fields in session state ensures re-renders show full history.
    """
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = send_message_with_status(query)
        _render_assistant_message(result)

    # store ALL fields so _render_assistant_message works correctly on re-render
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),   # stored as "content" for history
        "chart_spec": result.get("chart_spec"),
        "trace": result.get("trace"),
        # "confidence": result.get("confidence"),
    })


# sidebar example button click
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    _process_and_display(query)

# typed chat input
if query := st.chat_input("Ask about ocean data..."):
    _process_and_display(query)