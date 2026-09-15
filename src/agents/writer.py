"""Writer / Synthesizer Agent: Compiles verified research into publication-grade reports with exact citations and transparent gap reporting."""
import re
from typing import List, Optional, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseAgent
from src.models.plan import ResearchPlan
from src.models.finding import ResearchFinding, SourceCitation
from src.models.report import ResearchReport, ReportSection
from src.models.critic import CriticEvaluation

WRITER_SYSTEM_PROMPT = """You are an elite Deep Research Synthesizer.
Your task is to write a comprehensive, rigorous, and highly honest research report in French based exclusively on the provided verified evidence.

Core Directives:
1. Strict Citation Protocol: Whenever stating a factual claim or finding, insert inline numeric citations matching the provided source IDs (e.g., [1], [2]).
2. Transparent Gap Reporting: If an axis has no verified facts (has_insufficient_data), explicitly state that public online sources did not yield reliable documented data for this specific subtopic. Never invent or hallucinate facts to fill empty sections.
3. Completely Unknown Entities: If NO verified sources or facts exist for the subject at all, state in the executive summary that the entity has no documented public presence on the web.
4. Limitations Section: Explicitly mention which aspects could not be verified on the public web.
"""


class WriterAgent(BaseAgent):
    def __init__(self, llm: Optional[Any] = None, event_callback: Optional[Any] = None):
        super().__init__(
            name="Writer",
            role="Research Synthesizer",
            system_prompt=WRITER_SYSTEM_PROMPT,
            llm=llm,
            event_callback=event_callback
        )

    async def write_report(
        self,
        query: str,
        plan: ResearchPlan,
        findings: List[ResearchFinding],
        citations: List[SourceCitation],
        critic_review: Optional[CriticEvaluation] = None
    ) -> ResearchReport:
        """Synthesize verified evidence into an exhaustive structured report with transparent gap reporting."""
        self.emit_event(
            "synthesis_started",
            f"Rédaction de la synthèse pour '{query[:60]}...' ({len(findings)} axes, {len(citations)} sources).",
            {"citations_count": len(citations)}
        )

        all_empty = all(f.has_insufficient_data or len(f.facts) == 0 for f in findings) or len(citations) == 0

        findings_context = ""
        missing_axes = []
        for idx, f in enumerate(findings, 1):
            if f.has_insufficient_data or len(f.facts) == 0:
                missing_axes.append(f.query)
                findings_context += f"### Axe {idx}: {f.query} [NON TROUVÉ / DONNÉES INSUFFISANTES]\nStatut: Aucune source publique n'a fourni de données exploitables sur cet axe.\n\n"
            else:
                findings_context += f"### Axe {idx}: {f.query} [VÉRIFIÉ]\nSynthèse de l'axe: {f.summary}\nFaits extraits:\n"
                for fact in f.facts:
                    findings_context += f"- {fact.claim} (Source ID: [{fact.source_id}])\n"
                findings_context += "\n"

        citations_list_text = "\n".join(
            f"[{c.id}] {c.title} — {c.url}"
            for c in citations
        )

        prompt = (
            f"Sujet de recherche: {query}\n\n"
            f"Objectif: {plan.research_objective}\n\n"
            f"Sources disponibles pour citations:\n{citations_list_text if citations else 'Aucune source publique trouvée'}\n\n"
            f"Données collectées et état des axes:\n{findings_context}\n\n"
            "Rédigez un rapport complet (ResearchReport) en français avec citations [X] pour les faits prouvés, et en mentionnant honnêtement les axes non trouvés."
        )

        structured_llm = self.llm.with_structured_output(ResearchReport)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            report: ResearchReport = await structured_llm.ainvoke(messages)
        except Exception as e:
            self.emit_event("synthesis_fallback", f"Synthesis fallback: {e}")
            sections = []
            for idx, f in enumerate(findings, 1):
                if f.has_insufficient_data or len(f.facts) == 0:
                    content = "Information non documentée publiquement : Les sources en ligne ouvertes ne contiennent pas de données exploitables sur cet aspect précis au moment de la recherche."
                    used_cits = []
                else:
                    used_cits = [fact.source_id for fact in f.facts if fact.source_id <= len(citations)]
                    cit_str = f" [{used_cits[0]}]" if used_cits else ""
                    content = f"{f.summary}{cit_str}"

                sections.append(
                    ReportSection(
                        title=f.query,
                        content_markdown=content,
                        key_takeaways=[],
                        citations_used=used_cits
                    )
                )

            if all_empty:
                exec_summary = f"Aucune information publique n'a été trouvée pour « **{query}** ». L'exploration web approfondie sur les sources ouvertes n'a identifié aucun document fiable ni aucune donnée publique vérifiable correspondant exactement à cette entité ou personne."
                limitations_text = f"L'entité « {query} » n'est référencée dans aucune source publique indexée. Aucune donnée n'a été extrapolée afin de prévenir toute confusion d'homonymie ou hallucination."
            else:
                exec_summary = f"Ce rapport présente une analyse vérifiée sur **{query}**. Basé sur l'exploration de sources indépendantes, il détaille les faits documentés et signale les zones non couvertes par les données publiques en ligne [1]."
                limitations_text = "Les données présentées correspondent aux sources publiques accessibles au moment de la consultation."
                if missing_axes:
                    limitations_text += f"\n\n**Limites identifiées :** Les axes suivants n'ont pas pu être documentés à partir du web ouvert : {', '.join(missing_axes)}."

            report = ResearchReport(
                title=f"Rapport de recherche : {query}",
                subtitle=f"Synthèse factuelle et état des connaissances sur {query}",
                executive_summary=exec_summary,
                methodology="La recherche a été conduite de façon autonome : cadrage, collecte web, vérification critique et synthèse.",
                sections=sections,
                critical_analysis_and_limitations=limitations_text,
                bibliography=citations if not all_empty else []
            )

        # STRICTLY enforce the clean citations list from state and negative report honesty
        if all_empty:
            report.subtitle = "Recherche infructueuse — Aucune donnée publique vérifiée"
            report.executive_summary = f"Aucune information publique n'a été trouvée pour « **{query}** ». L'exploration web approfondie sur les sources ouvertes n'a identifié aucun document fiable ni aucune donnée publique vérifiable correspondant exactement à cette entité ou ce sujet."
            report.critical_analysis_and_limitations = f"L'entité « {query} » n'est référencée dans aucune source publique indexée. Aucune donnée n'a été extrapolée afin de prévenir toute confusion d'homonymie ou hallucination."
            report.bibliography = []
            for sec in report.sections:
                sec.citations_used = []
                if not sec.content_markdown or "Information non documentée" not in sec.content_markdown:
                    sec.content_markdown = f"Information non documentée publiquement : Les sources en ligne ouvertes ne contiennent aucune donnée vérifiable sur « {query} » pour cet axe."
        else:
            clean_bibliography = []
            seen_urls = set()
            for idx, src in enumerate(citations, 1):
                if src.url not in seen_urls:
                    seen_urls.add(src.url)
                    clean_src = src.model_copy(update={
                        "id": len(clean_bibliography) + 1,
                        "title": re.sub(r"[\n\r]+", " ", src.title).strip()[:120]
                    })
                    clean_bibliography.append(clean_src)
            report.bibliography = clean_bibliography

        report.assemble_markdown()

        self.emit_event(
            "synthesis_completed",
            f"Rapport généré : '{report.title}' ({len(report.sections)} sections, {len(report.bibliography)} sources).",
            {"title": report.title, "sections_count": len(report.sections), "markdown_length": len(report.full_markdown)}
        )

        return report
