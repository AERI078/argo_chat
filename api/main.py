# api/main.py — FastAPI app entry point
# build_orchestrator() runs once at startup, shared across all requests

from fastapi import FastAPI
from contextlib import asynccontextmanager
from agents.factory import build_orchestrator
from api.routes import chat, health

orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    print("Starting up — loading FAISS index and wiring agents...")
    orchestrator = build_orchestrator()
    print("Ready.")
    yield

app = FastAPI(
    title="FloatChat API",
    description="AI-powered conversational interface for Argo ocean float data",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(chat.router)
app.include_router(health.router)