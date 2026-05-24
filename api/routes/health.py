# api/routes/health.py — GET /health
# Render uses this to check if the service is up.
# Returns 200 immediately even during init — port is bound, service is alive.
# Frontend checks orchestrator_ready to know if chat is available yet.

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "orchestrator_ready": request.app.state.ready
    }