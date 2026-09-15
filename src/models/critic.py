"""Data models for reflection, critique, and fact-checking."""
from typing import List, Literal
from pydantic import BaseModel, Field


class CriticEvaluation(BaseModel):
    """Evaluation result produced by the Critic / Fact-Checker Agent."""
    completeness_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall percentage score of query coverage and depth (0-100)"
    )
    hallucination_risk: Literal["low", "medium", "high"] = Field(
        description="Assessed risk of ungrounded or contradictory claims"
    )
    identified_gaps: List[str] = Field(
        default_factory=list,
        description="Specific unanswered questions or missing perspectives"
    )
    contradictions_found: List[str] = Field(
        default_factory=list,
        description="Discrepancies found between different scraped sources"
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Targeted follow-up search queries if revision is requested"
    )
    verdict: Literal["APPROVE", "REVISE"] = Field(
        description="Whether to proceed to Writer (APPROVE) or perform corrective research (REVISE)"
    )
    reasoning: str = Field(
        description="Detailed analytical critique justifying the verdict"
    )
