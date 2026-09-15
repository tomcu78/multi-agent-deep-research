"""FastAPI routes for research orchestration, status polling, and WebSocket / SSE streaming."""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import json

from src.api.schemas import ResearchRequest, JobStatusResponse
from src.graph.workflow import ResearchWorkflowRunner
from src.llm import get_llm

router = APIRouter(prefix="/api", tags=["Research API"])

# In-memory storage for active research jobs
active_jobs: Dict[str, Dict[str, Any]] = {}
active_connections: Dict[str, list[WebSocket]] = {}


async def _run_job_background(job_id: str, request: ResearchRequest):
    job = active_jobs[job_id]
    job["status"] = "running"

    def _event_handler(event: Dict[str, Any]):
        job["events"].append(event)
        # Broadcast to active WebSocket clients
        if job_id in active_connections:
            for ws in active_connections[job_id]:
                asyncio.create_task(ws.send_text(json.dumps({"type": "event", "data": event})))

    llm = get_llm(
        provider=request.llm_provider,
        model_name=request.model_name,
        api_key=request.api_key
    )
    runner = ResearchWorkflowRunner(llm=llm, event_callback=_event_handler)

    try:
        result = await runner.run(
            query=request.query,
            clarifications=request.clarifications,
            max_iterations=request.max_iterations
        )
        job["status"] = result.get("status", "completed")
        job["plan"] = result.get("plan")
        job["critic_reviews"] = result.get("critic_reviews", [])
        job["final_report"] = result.get("final_report")
        job["current_iteration"] = result.get("current_iteration", 1)

        if job_id in active_connections:
            for ws in active_connections[job_id]:
                report_dict = job["final_report"].model_dump() if job["final_report"] else None
                asyncio.create_task(ws.send_text(json.dumps({
                    "type": "completed",
                    "data": {
                        "status": "completed",
                        "report": report_dict
                    }
                })))

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        if job_id in active_connections:
            for ws in active_connections[job_id]:
                asyncio.create_task(ws.send_text(json.dumps({
                    "type": "error",
                    "data": {"error": str(e)}
                })))


@router.post("/research", response_model=JobStatusResponse)
async def create_research_job(req: ResearchRequest):
    """Start an asynchronous deep research multi-agent workflow."""
    job_id = str(uuid.uuid4())[:8]
    job_record = {
        "job_id": job_id,
        "status": "queued",
        "query": req.query,
        "created_at": datetime.now().isoformat(),
        "current_iteration": 0,
        "plan": None,
        "critic_reviews": [],
        "final_report": None,
        "events": [],
        "error": None
    }
    active_jobs[job_id] = job_record

    # Dispatch non-blocking background task
    asyncio.create_task(_run_job_background(job_id, req))

    return JobStatusResponse(**job_record)


@router.get("/research/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Query current execution state and final synthesized report."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Research job not found")
    return JobStatusResponse(**active_jobs[job_id])


@router.get("/research/{job_id}/events")
async def stream_job_events_sse(job_id: str):
    """Server-Sent Events (SSE) endpoint to stream live multi-agent updates."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Research job not found")

    async def event_generator():
        last_index = 0
        while True:
            job = active_jobs.get(job_id)
            if not job:
                break

            events = job.get("events", [])
            while last_index < len(events):
                yield f"data: {json.dumps(events[last_index])}\n\n"
                last_index += 1

            if job.get("status") in ["completed", "failed"]:
                yield f"data: {json.dumps({'type': 'final_status', 'status': job.get('status')})}\n\n"
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.websocket("/ws/{job_id}")
async def websocket_job_stream(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for bi-directional and ultra-low latency event streaming."""
    await websocket.accept()
    if job_id not in active_connections:
        active_connections[job_id] = []
    active_connections[job_id].append(websocket)

    job = active_jobs.get(job_id)
    if job:
        for ev in job.get("events", []):
            await websocket.send_text(json.dumps({"type": "event", "data": ev}))
        if job.get("status") == "completed" and job.get("final_report"):
            await websocket.send_text(json.dumps({
                "type": "completed",
                "data": {"report": job["final_report"].model_dump()}
            }))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if job_id in active_connections:
            active_connections[job_id].remove(websocket)


@router.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
