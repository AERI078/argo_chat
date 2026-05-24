# api/routes/chat.py — POST /chat

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    chart_spec: Optional[dict] = None
    trace: Optional[dict] = None
    success: bool
    confidence: Optional[float] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    if not request.app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="System is still initialising — FAISS index is being built. Try again in 60 seconds."
        )

    response, trace = request.app.state.orchestrator.run(body.message)

    return ChatResponse(
        answer=response.answer,
        chart_spec=response.chart_spec,
        trace=trace,
        success=response.success,
        confidence=response.confidence
    )