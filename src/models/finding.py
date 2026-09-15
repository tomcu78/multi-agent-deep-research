"""Data models for extracted evidence, web search results, and citations."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Reference to a verified web or document source."""
    id: int = Field(description="Unique index for the source (e.g. 1 for [1])")
    title: str = Field(description="Title of the webpage or document")
    url: str = Field(description="Full URL of the source")
    domain: str = Field(description="Extracted domain name (e.g. arxiv.org, github.com)")
    snippet: str = Field(description="Relevant snippet or excerpt")
    reliability_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Estimated reliability score (0.0 to 1.0)"
    )


class FactExtraction(BaseModel):
    """An individual verifiable fact or data point extracted from sources."""
    claim: str = Field(description="Direct factual statement or finding")
    supporting_quote: str = Field(description="Verbatim quote from the source supporting the claim")
    source_id: int = Field(description="The source ID referencing SourceCitation.id")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence score")


class ResearchFinding(BaseModel):
    """Aggregated findings for a specific research subtask."""
    subtask_id: str = Field(description="ID of the subtask this finding resolves")
    query: str = Field(description="Search query or angle explored")
    facts: List[FactExtraction] = Field(default_factory=list, description="Extracted factual claims")
    sources: List[SourceCitation] = Field(default_factory=list, description="Sources discovered and scraped")
    summary: str = Field(description="Synthesized summary of the subtask's evidence")
    has_insufficient_data: bool = Field(
        default=False,
        description="True if public web sources yielded no verifiable data on this subtask"
    )
