"""LangGraph StateGraph definition for Multi-Agent Deep Research."""
import asyncio
from typing import Dict, Any, List, Literal, Optional, Callable
from langgraph.graph import StateGraph, START, END
from src.models.state import ResearchState
from src.models.finding import SourceCitation, ResearchFinding
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.critic import CriticAgent
from src.agents.writer import WriterAgent
from src.config import settings


def create_research_graph(
    llm: Optional[Any] = None,
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
):
    """Construct and compile the LangGraph multi-agent research workflow."""
    planner = PlannerAgent(llm=llm, event_callback=event_callback)
    critic = CriticAgent(llm=llm, event_callback=event_callback)
    writer = WriterAgent(llm=llm, event_callback=event_callback)

    # 1. Planner Node
    async def planner_node(state: ResearchState) -> Dict[str, Any]:
        latest_critique = state["critic_reviews"][-1] if state.get("critic_reviews") else None
        plan = await planner.plan(
            query=state["query"],
            clarifications=state.get("clarifications"),
            critic_review=latest_critique
        )
        return {
            "plan": plan,
            "subtasks": plan.subtasks,
            "status": "planned",
            "activity_logs": [
                {
                    "step": "planning",
                    "message": f"Generated research plan with {len(plan.subtasks)} subtasks.",
                    "details": {"subtasks": [s.title for s in plan.subtasks]}
                }
            ]
        }

    # 2. Parallel Research Node (Fan-Out / Map-Reduce)
    async def parallel_research_node(state: ResearchState) -> Dict[str, Any]:
        subtasks = state.get("subtasks", [])
        if not subtasks and state.get("plan"):
            subtasks = state["plan"].subtasks

        existing_citations_count = len(state.get("citations", []))

        # Launch researcher agents concurrently for all subtasks
        async def run_subtask(subtask):
            agent = ResearcherAgent(
                subtask_id=subtask.id,
                llm=llm,
                event_callback=event_callback
            )
            return await agent.execute_subtask(
                subtask=subtask,
                main_query=state.get("query"),
                existing_source_count=existing_citations_count
            )

        tasks = [run_subtask(st) for st in subtasks]
        new_findings: List[ResearchFinding] = await asyncio.gather(*tasks)

        # Collect all sources discovered
        all_new_sources: List[SourceCitation] = []
        for f in new_findings:
            all_new_sources.extend(f.sources)

        return {
            "findings": new_findings,
            "citations": all_new_sources,
            "status": "researched",
            "activity_logs": [
                {
                    "step": "research",
                    "message": f"Collected {len(new_findings)} findings across {len(subtasks)} subtasks.",
                    "details": {"new_sources_count": len(all_new_sources)}
                }
            ]
        }

    # 3. Critic Node
    async def critic_node(state: ResearchState) -> Dict[str, Any]:
        current_iter = state.get("current_iteration", 0) + 1
        evaluation = await critic.evaluate(
            plan=state["plan"],
            findings=state.get("findings", []),
            citations=state.get("citations", []),
            current_iteration=current_iter,
            max_iterations=state.get("max_iterations", settings.MAX_RESEARCH_ITERATIONS)
        )
        return {
            "critic_reviews": [evaluation],
            "current_iteration": current_iter,
            "status": "evaluated",
            "activity_logs": [
                {
                    "step": "critic",
                    "message": f"Completed review (Iteration {current_iter}): Verdict {evaluation.verdict} (Score: {evaluation.completeness_score}%)",
                    "details": {"verdict": evaluation.verdict, "score": evaluation.completeness_score}
                }
            ]
        }

    # 4. Conditional Edge Router (Reflection Loop)
    def route_critic_decision(state: ResearchState) -> Literal["replan", "synthesize"]:
        latest_critique = state["critic_reviews"][-1]
        current_iter = state.get("current_iteration", 1)
        max_iter = state.get("max_iterations", settings.MAX_RESEARCH_ITERATIONS)

        if latest_critique.verdict == "REVISE" and current_iter < max_iter:
            return "replan"
        return "synthesize"

    # 5. Writer Node
    async def writer_node(state: ResearchState) -> Dict[str, Any]:
        latest_critique = state["critic_reviews"][-1] if state.get("critic_reviews") else None
        report = await writer.write_report(
            query=state["query"],
            plan=state["plan"],
            findings=state.get("findings", []),
            citations=state.get("citations", []),
            critic_review=latest_critique
        )
        return {
            "final_report": report,
            "status": "completed",
            "activity_logs": [
                {
                    "step": "writer",
                    "message": f"Successfully compiled final report '{report.title}' with {len(report.sections)} sections.",
                    "details": {"sections": len(report.sections), "citations": len(report.bibliography)}
                }
            ]
        }

    # Construct the graph
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("parallel_researcher", parallel_research_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    # Define edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "parallel_researcher")
    workflow.add_edge("parallel_researcher", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_critic_decision,
        {
            "replan": "planner",
            "synthesize": "writer"
        }
    )

    workflow.add_edge("writer", END)

    return workflow.compile()
