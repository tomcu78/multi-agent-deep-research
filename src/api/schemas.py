"""FastAPI Pydantic Schemas for research requests and responses."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.models.report import ResearchReport
from src.models.plan import ResearchPlan
from src.models.critic import CriticEvaluation


class ResearchRequest(BaseModel):
    query: str = Field(..., description="The main research prompt or topic")
    clarifications: Optional[str] = Field(None, description="Optional constraints or focus areas")
    max_iterations: Optional[int] = Field(None, description="Override for maximum reflection iterations")
    llm_provider: Optional[str] = Field(None, description="LLM provider: openai, anthropic, google, mock")
    model_name: Optional[str] = Field(None, description="LLM model name")
    api_key: Optional[str] = Field(None, description="Optional API key provided at request time")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    query: str
    created_at: str
    current_iteration: int
    plan: Optional[ResearchPlan] = None
    critic_reviews: List[CriticEvaluation] = Field(default_factory=list)
    final_report: Optional[ResearchReport] = None
    events: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
