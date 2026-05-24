# api/main.py — FastAPI app
# Port binds immediately on startup.
# FAISS index build runs in a background thread so Render doesn't timeout.

from fastapi import FastAPI
import threading
from agents.factory import build_orchestrator
from api.routes import chat, health

app = FastAPI(
    title="FloatChat API",
    description="AI-powered conversational interface for Argo ocean float data",
    version="1.0.0"
)

app.include_router(chat.router)
app.include_router(health.router)

# shared state — None until background init finishes
app.state.orchestrator = None
app.state.ready = False


def _init_orchestrator():
    """Runs in background thread — builds FAISS index and wires all agents."""
    try:
        print("Background init: loading FAISS index and wiring agents...")
        app.state.orchestrator = build_orchestrator()
        app.state.ready = True
        print("Background init: Ready.")
    except Exception as e:
        print(f"Background init FAILED: {e}")
        app.state.ready = False


@app.on_event("startup")
async def startup():
    # bind port immediately, then start heavy work in background
    thread = threading.Thread(target=_init_orchestrator, daemon=True)
    thread.start()