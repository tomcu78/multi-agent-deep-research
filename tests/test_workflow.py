"""Integration tests for the LangGraph multi-agent research workflow."""
import pytest
from src.graph.workflow import ResearchWorkflowRunner
from src.llm import MockChatModel


@pytest.mark.asyncio
async def test_workflow_end_to_end():
    events = []

    def on_event(ev):
        events.append(ev)

    runner = ResearchWorkflowRunner(llm=MockChatModel(), event_callback=on_event)

    result = await runner.run(
        query="What are the key advancements in agentic workflows and test-time compute in 2025/2026?",
        clarifications="Focus on state machines and reflection loops",
        max_iterations=2
    )

    # 1. State machine completion assertions
    assert result["status"] == "completed"
    assert result["plan"] is not None
    assert len(result["plan"].subtasks) > 0

    # 2. Findings & Sources assertions
    assert len(result["findings"]) > 0
    assert len(result["citations"]) > 0

    # 3. Critic assertions
    assert len(result["critic_reviews"]) > 0
    assert result["critic_reviews"][-1].verdict == "APPROVE"

    # 4. Final Report & Citations assertions
    report = result["final_report"]
    assert report is not None
    assert report.title != ""
    assert len(report.sections) > 0
    assert len(report.bibliography) > 0
    assert len(report.full_markdown) > 0

    # 5. Event streaming telemetry assertions
    assert len(events) >= 5
    agent_names = [e["agent"] for e in events]
    assert any("Planner" in a for a in agent_names)
    assert any("Researcher" in a for a in agent_names)
    assert any("Critic" in a for a in agent_names)
    assert any("Writer" in a for a in agent_names)


@pytest.mark.asyncio
async def test_workflow_async_event_stream():
    runner = ResearchWorkflowRunner(llm=MockChatModel())
    streamed = []

    async for event in runner.stream_events(query="Test stream query"):
        streamed.append(event)

    assert len(streamed) > 0
    last_event = streamed[-1]
    assert last_event["type"] == "workflow_finished"
