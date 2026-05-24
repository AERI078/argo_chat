# api/routes/health.py — GET /health
# used by Render to confirm the service is up, and by the frontend to check connectivity

from fastapi import APIRouter
import api.main as state

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "orchestrator_ready": state.orchestrator is not None
    }