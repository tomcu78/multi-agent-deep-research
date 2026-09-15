"""Unit tests for the Critic Agent and reflection loop mechanisms."""
import pytest
from src.agents.critic import CriticAgent
from src.models.plan import ResearchPlan, ResearchSubtask
from src.models.finding import ResearchFinding, SourceCitation, FactExtraction
from src.llm import MockChatModel


@pytest.mark.asyncio
async def test_critic_approval():
    plan = ResearchPlan(
        main_query="Test query",
        research_objective="Verify critic evaluation logic",
        scope="Testing",
        subtasks=[ResearchSubtask(id="st_1", title="Task 1", description="Desc", target_search_queries=["q1"])]
    )
    findings = [
        ResearchFinding(
            subtask_id="st_1",
            query="Task 1",
            facts=[FactExtraction(claim="Fact 1", supporting_quote="Quote 1", source_id=1, confidence=0.95)],
            sources=[SourceCitation(id=1, title="Src 1", url="https://example.com/1", domain="example.com", snippet="Snippet 1")],
            summary="Complete evidence summary."
        )
    ]
    citations = [SourceCitation(id=1, title="Src 1", url="https://example.com/1", domain="example.com", snippet="Snippet 1")]

    critic = CriticAgent(llm=MockChatModel())
    evaluation = await critic.evaluate(plan=plan, findings=findings, citations=citations, current_iteration=1, max_iterations=3)

    assert evaluation.verdict in ["APPROVE", "REVISE"]
    assert evaluation.completeness_score >= 0.0
    assert evaluation.hallucination_risk in ["low", "medium", "high"]


@pytest.mark.asyncio
async def test_critic_max_iterations_override():
    plan = ResearchPlan(
        main_query="Test query",
        research_objective="Verify loop boundary",
        scope="Testing",
        subtasks=[]
    )
    critic = CriticAgent(llm=MockChatModel())
    # Force max iteration reached
    evaluation = await critic.evaluate(plan=plan, findings=[], citations=[], current_iteration=3, max_iterations=3)
    assert evaluation.verdict == "APPROVE"
