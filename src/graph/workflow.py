"""Workflow execution runner with async streaming and lifecycle management."""
import asyncio
from typing import Optional, Callable, Dict, Any, AsyncGenerator
from src.graph.state_graph import create_research_graph
from src.models.state import ResearchState
from src.models.report import ResearchReport
from src.config import settings


class ResearchWorkflowRunner:
    """Manages the lifecycle of a Deep Research job."""
    def __init__(
        self,
        llm: Optional[Any] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.llm = llm
        self.event_callback = event_callback
        self.graph = create_research_graph(llm=llm, event_callback=event_callback)

    async def run(
        self,
        query: str,
        clarifications: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> ResearchState:
        """Run the research workflow from start to finish."""
        initial_state: ResearchState = {
            "query": query,
            "clarifications": clarifications,
            "plan": None,
            "subtasks": [],
            "findings": [],
            "citations": [],
            "critic_reviews": [],
            "current_iteration": 0,
            "max_iterations": max_iterations or settings.MAX_RESEARCH_ITERATIONS,
            "final_report": None,
            "activity_logs": [],
            "status": "initialized"
        }

        final_state = await self.graph.ainvoke(initial_state)
        return final_state

    async def stream_events(
        self,
        query: str,
        clarifications: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream real-time agent updates and state transitions using an async queue."""
        event_queue = asyncio.Queue()

        def _queue_callback(event: Dict[str, Any]):
            event_queue.put_nowait(event)

        runner = ResearchWorkflowRunner(llm=self.llm, event_callback=_queue_callback)

        async def _execute():
            try:
                res = await runner.run(query=query, clarifications=clarifications)
                await event_queue.put({"type": "workflow_finished", "payload": {"status": res.get("status"), "report": res.get("final_report")}})
            except Exception as e:
                await event_queue.put({"type": "workflow_error", "payload": {"error": str(e)}})
            finally:
                await event_queue.put(None)  # Sentinel to terminate generator

        task = asyncio.create_task(_execute())

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event

        await task
