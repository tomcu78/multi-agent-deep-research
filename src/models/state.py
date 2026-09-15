"""Centralized typed state definitions for the LangGraph multi-agent research workflow."""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import operator
from src.models.plan import ResearchPlan, ResearchSubtask
from src.models.finding import ResearchFinding, SourceCitation
from src.models.critic import CriticEvaluation
from src.models.report import ResearchReport


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer to merge dictionary states."""
    res = dict(a)
    res.update(b)
    return res


def merge_citations(existing: List[SourceCitation], incoming: List[SourceCitation]) -> List[SourceCitation]:
    """Reducer to deduplicate sources by URL and maintain consistent numeric IDs."""
    seen_urls = {item.url: item for item in existing}
    current_max_id = max([item.id for item in existing], default=0)

    for item in incoming:
        if item.url not in seen_urls:
            current_max_id += 1
            new_item = item.model_copy(update={"id": current_max_id})
            seen_urls[item.url] = new_item
    return list(seen_urls.values())


class ResearchState(TypedDict):
    """Centralized state machine record passed across Planner, Researcher, Critic, and Writer nodes."""
    query: str
    clarifications: Optional[str]
    plan: Optional[ResearchPlan]
    subtasks: List[ResearchSubtask]
    findings: Annotated[List[ResearchFinding], operator.add]
    citations: Annotated[List[SourceCitation], merge_citations]
    critic_reviews: Annotated[List[CriticEvaluation], operator.add]
    current_iteration: int
    max_iterations: int
    final_report: Optional[ResearchReport]
    activity_logs: Annotated[List[Dict[str, Any]], operator.add]
    status: str
