"""Integration tests for FastAPI endpoints and health probe."""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_create_research_job():
    payload = {
        "query": "Autonomous Multi-Agent Architectures",
        "clarifications": "Focus on reflection loops",
        "llm_provider": "mock",
        "max_iterations": 2
    }
    response = client.post("/api/research", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ["queued", "running", "completed"]
    assert data["query"] == payload["query"]

    # Check job retrieval
    job_id = data["job_id"]
    status_resp = client.get(f"/api/research/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id
