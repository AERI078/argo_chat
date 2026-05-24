# frontend/app.py — Streamlit chat UI
# talks to the FastAPI backend at BACKEND_URL
# run with: streamlit run frontend/app.py

import streamlit as st
import httpx
import plotly.graph_objects as go
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FloatChat",
    page_icon="🌊",
    layout="wide"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def check_backend() -> bool:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        return r.json().get("orchestrator_ready", False)
    except Exception:
        return False


def send_message(message: str) -> dict:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/chat",
            json={"message": message},
            timeout=120  # agent pipeline can take up to 2 min on first fetch
        )
        return r.json()
    except httpx.TimeoutException:
        return {"answer": "Request timed out. The agent is still processing — try a simpler query.", "success": False}
    except Exception as e:
        return {"answer": f"Could not reach backend: {e}", "success": False}


def render_chart(chart_spec: dict):
    """Renders a Plotly chart from the spec dict returned by the agent."""
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
            lat=data.get("lat", []),
            lon=data.get("lon", []),
            mode="lines+markers",
            text=data.get("date", []),
            marker=dict(size=6, color="#2ecc71"),
            line=dict(width=1, color="#2ecc71")
        ))
        fig.update_layout(
            title="Float Trajectory",
            geo=dict(
                showland=True, landcolor="lightgray",
                showocean=True, oceancolor="#d6eaf8",
                projection_type="natural earth",
                center=dict(lat=15, lon=70),  # Indian Ocean
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
    """Shows the agent's reasoning steps in an expander."""
    with st.expander("🔍 How I answered this", expanded=False):
        for event in trace.get("events", []):
            stage = event.get("stage", "")
            status = event.get("status", "")
            icon = {"success": "✅", "failure": "❌", "replan": "🔄", "started": "⏳", "done": "🏁"}.get(status, "•")
            elapsed = f" ({event['elapsed_ms']}ms)" if "elapsed_ms" in event else ""
            detail = ""
            if stage == "planner" and status == "success":
                detail = f" — {event.get('steps', '')} steps planned"
            elif stage == "executor":
                detail = f" — {event.get('tool', '')} step {event.get('step_id', '')}"
            elif stage == "validator":
                detail = f" — score {event.get('score', '')}" if status == "success" else f" — {event.get('error', '')}"
            elif stage == "replan":
                detail = f" — {event.get('reason', '')}"
            st.markdown(f"{icon} **{stage}**{detail}{elapsed}")


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌊 FloatChat")
    st.caption("AI-powered Argo ocean data explorer")
    st.divider()

    # backend status indicator
    if check_backend():
        st.success("Backend connected", icon="🟢")
    else:
        st.error("Backend offline", icon="🔴")
        st.caption(f"Expecting backend at {BACKEND_URL}")

    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "What is salinity in the Arabian Sea?",
        "Show temperature profiles near lat 15, lon 65 in mid-2023",
        "Compare BGC parameters in the Indian Ocean",
        "What are Argo floats and how do they work?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── main chat area ─────────────────────────────────────────────────────────────

st.title("FloatChat 🌊")
st.caption("Ask anything about Argo ocean float data — in plain English or technical terms.")

# initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart_spec"):
            render_chart(msg["chart_spec"])
        if msg.get("trace"):
            render_trace(msg["trace"])

# handle sidebar example button clicks
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = send_message(query)
        st.markdown(result.get("answer", ""))
        if result.get("chart_spec"):
            render_chart(result["chart_spec"])
        if result.get("trace"):
            render_trace(result["trace"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "chart_spec": result.get("chart_spec"),
        "trace": result.get("trace")
    })
    st.rerun()

# handle typed input
if query := st.chat_input("Ask about ocean data..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = send_message(query)
        st.markdown(result.get("answer", ""))
        if result.get("chart_spec"):
            render_chart(result["chart_spec"])
        if result.get("trace"):
            render_trace(result["trace"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", ""),
        "chart_spec": result.get("chart_spec"),
        "trace": result.get("trace")
    })