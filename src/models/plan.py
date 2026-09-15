"""Data models for research plans and subtasks."""
from typing import List
from pydantic import BaseModel, Field


class ResearchSubtask(BaseModel):
    """An independent research subtask intended for parallel execution."""
    id: str = Field(description="Unique identifier for the subtask (e.g. subtask_1)")
    title: str = Field(description="Clear title of the subtopic")
    description: str = Field(description="Detailed objective and requirements of this subtask")
    target_search_queries: List[str] = Field(
        description="List of precise, keyword-rich search queries to run"
    )
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Key aspects, metrics, or technical nuances to specifically check"
    )
    priority: int = Field(default=1, description="Priority level (1 is highest)")


class ResearchPlan(BaseModel):
    """Structured decomposition of a complex research request."""
    main_query: str = Field(description="Original user request or research goal")
    research_objective: str = Field(description="Formal statement of the research objective")
    scope: str = Field(description="In-scope and out-of-scope boundaries")
    subtasks: List[ResearchSubtask] = Field(
        description="List of discrete subtasks to be parallelized across researcher sub-agents"
    )
    evaluation_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria the Critic should verify before approving final synthesis"
    )
