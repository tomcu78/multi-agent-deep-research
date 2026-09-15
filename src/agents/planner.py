"""Planner Agent: Formulates structured research strategies and task decompositions."""
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseAgent
from src.models.plan import ResearchPlan, ResearchSubtask
from src.models.critic import CriticEvaluation
from src.config import settings

PLANNER_SYSTEM_PROMPT = """You are an elite Research Strategist and Planner Agent.
Your goal is to deconstruct a complex user query or research prompt into a structured, highly focused research plan.
You must split the objective into {max_subtasks} distinct, non-overlapping subtasks that can be executed in parallel by researcher agents.

Requirements:
1. Formulate a precise, formal research objective and defined scope.
2. For each subtask, generate specific, keyword-dense search queries designed to find empirical data, official documentation, academic papers, and benchmark metrics.
3. Define key focus areas and concrete evaluation criteria to ensure exhaustiveness and factual accuracy.
"""

REPLAN_SYSTEM_PROMPT = """You are an elite Research Strategist revising an existing research plan based on feedback from the Critic / Fact-Checker.
The Critic has identified specific gaps, missing angles, or unverified claims.
Generate a targeted set of corrective subtasks and search queries to fill these specific gaps without re-doing existing valid work.
"""


class PlannerAgent(BaseAgent):
    def __init__(self, llm: Optional[Any] = None, event_callback: Optional[Any] = None):
        super().__init__(
            name="Planner",
            role="Research Strategist",
            system_prompt=PLANNER_SYSTEM_PROMPT.format(max_subtasks=settings.MAX_SUBTASKS),
            llm=llm,
            event_callback=event_callback
        )

    async def plan(
        self,
        query: str,
        clarifications: Optional[str] = None,
        critic_review: Optional[CriticEvaluation] = None
    ) -> ResearchPlan:
        """Create or adapt a structured research plan."""
        is_replan = critic_review is not None and critic_review.verdict == "REVISE"

        if is_replan:
            self.emit_event(
                "planning_revision_started",
                "Revising research strategy based on Critic feedback...",
                {"gaps": critic_review.identified_gaps, "follow_ups": critic_review.follow_up_queries}
            )
            prompt = (
                f"Original Query: {query}\n\n"
                f"Critic Feedback Reasoning:\n{critic_review.reasoning}\n\n"
                f"Identified Gaps:\n" + "\n".join(f"- {g}" for g in critic_review.identified_gaps) + "\n\n"
                f"Suggested Follow-up Queries:\n" + "\n".join(f"- {q}" for q in critic_review.follow_up_queries) + "\n\n"
                "Please generate a targeted corrective ResearchPlan addressing these specific gaps."
            )
            sys_msg = REPLAN_SYSTEM_PROMPT
        else:
            self.emit_event(
                "planning_started",
                f"Deconstructing research prompt: '{query[:80]}...' into parallel subtasks.",
                {"query": query}
            )
            prompt = f"User Research Request: {query}\n"
            if clarifications:
                prompt += f"User Clarifications & Constraints: {clarifications}\n"
            sys_msg = self.system_prompt

        structured_llm = self.llm.with_structured_output(ResearchPlan)
        messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(content=prompt)
        ]

        try:
            plan_result: ResearchPlan = await structured_llm.ainvoke(messages)
        except Exception as e:
            self.emit_event("planning_fallback", f"Structured output parsing error ({e}), using default structured decomposition.")
            # Fallback plan
            plan_result = ResearchPlan(
                main_query=query,
                research_objective=f"Investigate core dimensions and verifiable facts regarding {query}",
                scope="Comprehensive state-of-the-art overview, architecture, benchmarks, and trade-offs.",
                subtasks=[
                    ResearchSubtask(
                        id="subtask_1",
                        title="Foundations, Architectures & State of the Art",
                        description=f"Analyze foundational concepts and current state of the art for {query}",
                        target_search_queries=[f"{query} architecture state of the art", f"{query} overview benchmark"],
                        focus_areas=["Core architecture", "Key paradigms"]
                    ),
                    ResearchSubtask(
                        id="subtask_2",
                        title="Empirical Benchmarks, Case Studies & Practical Implementations",
                        description=f"Find benchmark results, real-world case studies, and engineering implementations for {query}",
                        target_search_queries=[f"{query} benchmark results case studies", f"{query} production guide"],
                        focus_areas=["Benchmarks", "Production metrics", "Real-world trade-offs"]
                    )
                ],
                evaluation_criteria=["Factual grounding", "Clear citations", "Actionable depth"]
            )

        self.emit_event(
            "planning_completed",
            f"Formulated research plan with {len(plan_result.subtasks)} parallel subtasks.",
            {
                "objective": plan_result.research_objective,
                "subtasks": [s.title for s in plan_result.subtasks]
            }
        )
        return plan_result
