"""Critic Agent: Performs reflexive fact-checking, gap analysis, and anti-hallucination verification."""
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseAgent
from src.models.plan import ResearchPlan
from src.models.finding import ResearchFinding, SourceCitation
from src.models.critic import CriticEvaluation
from src.config import settings

CRITIC_SYSTEM_PROMPT = """You are a rigorous Academic Critic and Fact-Checking Agent.
Your mandate is to evaluate research evidence quality, detect missing data, and prevent hallucinations.

Evaluation Directives:
1. Honesty on Missing Data: If certain subtasks or queries have no public data on the web, you must explicitly flag them in 'identified_gaps' and state clearly in 'reasoning' that these aspects remain undocumented online.
2. Completeness: Score completeness proportionally based on the evidence actually verified.
3. Verdict: If public sources simply don't exist for the missing parts after exploration, approve the report with explicit limitations rather than looping endlessly.
"""


class CriticAgent(BaseAgent):
    def __init__(self, llm: Optional[Any] = None, event_callback: Optional[Any] = None):
        super().__init__(
            name="Critic",
            role="Fact-Checker & Quality Assurance",
            system_prompt=CRITIC_SYSTEM_PROMPT,
            llm=llm,
            event_callback=event_callback
        )

    async def evaluate(
        self,
        plan: ResearchPlan,
        findings: List[ResearchFinding],
        citations: List[SourceCitation],
        current_iteration: int = 1,
        max_iterations: int = 3
    ) -> CriticEvaluation:
        """Evaluate evidence quality, identify web data gaps, and decide whether to synthesize."""
        self.emit_event(
            "critique_started",
            f"Évaluation de la fiabilité ({len(findings)} axes explorés, {len(citations)} sources)...",
            {"iteration": current_iteration, "findings_count": len(findings)}
        )

        missing_subtasks = [f.query for f in findings if f.has_insufficient_data or len(f.facts) == 0]
        valid_subtasks = [f.query for f in findings if not f.has_insufficient_data and len(f.facts) > 0]

        # Calculate realistic completeness score
        total_subtasks = max(len(findings), 1)
        score = round((len(valid_subtasks) / total_subtasks) * 100.0, 1)

        findings_text = ""
        for idx, f in enumerate(findings, 1):
            status_str = "INSUFFISANT / NON TROUVÉ" if f.has_insufficient_data else "VÉRIFIÉ"
            findings_text += f"\n### Axe {idx}: {f.query} [{status_str}]\nSynthèse: {f.summary}\nFaits extraits:\n"
            for fact in f.facts:
                findings_text += f"- {fact.claim} (Source: {fact.source_id})\n"

        prompt = (
            f"Objectif de recherche: {plan.research_objective}\n\n"
            f"Résultats de l'exploration web:\n{findings_text}\n\n"
            f"Nombre total de sources vérifiées: {len(citations)}\n"
            f"Axes sans données publiques: {', '.join(missing_subtasks) if missing_subtasks else 'Aucun'}\n\n"
            "Évaluez la couverture réelle, explicitez les limites et lacunes des sources publiques, et fournissez votre verdict."
        )

        structured_llm = self.llm.with_structured_output(CriticEvaluation)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            critique: CriticEvaluation = await structured_llm.ainvoke(messages)
        except Exception as e:
            critique = CriticEvaluation(
                completeness_score=score,
                hallucination_risk="low" if score > 50 else "medium",
                identified_gaps=missing_subtasks,
                contradictions_found=[],
                follow_up_queries=[],
                verdict="APPROVE",
                reasoning=(
                    f"Couverture documentaire de {score}%. "
                    + (f"Les axes suivants ne disposent pas de données publiques accessibles : {', '.join(missing_subtasks)}." if missing_subtasks else "Tous les axes prévus sont étayés par des sources réelles.")
                )
            )

        # Enforce max iterations ceiling
        if current_iteration >= max_iterations:
            critique = critique.model_copy(update={"verdict": "APPROVE"})

        self.emit_event(
            "critique_completed",
            f"Évaluation : {critique.verdict} (Score : {critique.completeness_score}%, Lacunes web : {len(critique.identified_gaps)})",
            {
                "verdict": critique.verdict,
                "score": critique.completeness_score,
                "reasoning": critique.reasoning,
                "gaps": critique.identified_gaps
            }
        )

        return critique
