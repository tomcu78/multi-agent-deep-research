"""Unit tests for centralized state management, Pydantic models, and reducers."""
import pytest
from src.models.finding import SourceCitation, FactExtraction, ResearchFinding
from src.models.plan import ResearchPlan, ResearchSubtask
from src.models.critic import CriticEvaluation
from src.models.report import ResearchReport, ReportSection
from src.models.state import merge_citations


def test_merge_citations_deduplication():
    src1 = SourceCitation(id=1, title="Article 1", url="https://example.com/1", domain="example.com", snippet="Test 1")
    src2 = SourceCitation(id=2, title="Article 2", url="https://example.com/2", domain="example.com", snippet="Test 2")
    src3_dup = SourceCitation(id=1, title="Article 1 Dup", url="https://example.com/1", domain="example.com", snippet="Test 1")
    src4_new = SourceCitation(id=99, title="Article 3", url="https://example.com/3", domain="example.com", snippet="Test 3")

    existing = [src1, src2]
    incoming = [src3_dup, src4_new]

    merged = merge_citations(existing, incoming)

    # Should have 3 unique URLs
    assert len(merged) == 3
    urls = [s.url for s in merged]
    assert "https://example.com/1" in urls
    assert "https://example.com/2" in urls
    assert "https://example.com/3" in urls
    new_src = next(s for s in merged if s.url == "https://example.com/3")
    assert new_src.id == 3


def test_research_plan_validation():
    subtask = ResearchSubtask(
        id="subtask_1",
        title="Architecture",
        description="Study agent architectures",
        target_search_queries=["multi-agent state machine"]
    )
    plan = ResearchPlan(
        main_query="Agentic AI",
        research_objective="Comprehensive investigation into Agentic AI",
        scope="Technical architectures and state graphs",
        subtasks=[subtask]
    )
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].id == "subtask_1"


def test_report_markdown_assembly():
    src = SourceCitation(id=1, title="Paper A", url="https://arxiv.org/123", domain="arxiv.org", snippet="Summary", reliability_score=0.95)
    report = ResearchReport(
        title="Test Deep Research",
        executive_summary="Summary of findings.",
        methodology="Conducted with multi-agent pipeline.",
        sections=[
            ReportSection(
                title="State Management",
                content_markdown="LangGraph provides state graph management [1].",
                key_takeaways=[],
                citations_used=[1]
            )
        ],
        critical_analysis_and_limitations="Some models exhibit latency overhead.",
        bibliography=[src]
    )

    markdown = report.assemble_markdown()
    assert "# Test Deep Research" in markdown
    assert "## Synthèse générale" in markdown
    assert "## 1. State Management" in markdown
    assert "[1]" in markdown
    assert "https://arxiv.org/123" in markdown
