# api/routes/chat.py — POST /chat
# receives a user message, runs the orchestrator, returns answer + chart + trace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import api.main as state

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    chart_spec: Optional[dict] = None
    trace: Optional[dict] = None
    success: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not state.orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not ready yet.")

    response, trace = state.orchestrator.run(request.message)

    return ChatResponse(
        answer=response.answer,
        chart_spec=response.chart_spec,
        trace=trace,
        success=response.success
    )