"""Researcher Sub-Agent: Performs parallelized web exploration, scraping, and fact extraction with strict entity alignment."""
import asyncio
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.base import BaseAgent
from src.models.plan import ResearchSubtask
from src.models.finding import ResearchFinding, SourceCitation, FactExtraction
from src.tools.search import search_web, SearchResult, extract_entity_tokens
from src.tools.scraper import scrape_multiple_urls, ScrapedDocument
from src.config import settings

RESEARCHER_SYSTEM_PROMPT = """You are an expert Research Sub-Agent and Evidence Extractor.
Your job is to rigorously extract factual claims, benchmark statistics, dates, and direct quotations from raw scraped web content to address your assigned subtask.

Rules:
1. Strict Entity Alignment: Only extract facts that explicitly pertain to the requested subject. If a document discusses a different entity, phonetically similar name, or unrelated person, discard it completely.
2. Honest Negative Reporting: If the provided source documents do NOT contain verifiable data on the subtask, you MUST return an empty facts list and set has_insufficient_data=True. Never invent or hallucinate facts.
3. For each extracted fact, provide a short direct quote from the source that confirms it.
"""


class ExtractionOutput(BaseModel):
    """Pydantic model for structured extraction from scraped documents."""
    facts: List[FactExtraction] = Field(default_factory=list, description="List of verified factual claims with supporting quotes")
    summary: str = Field(description="Concise synthesis of all evidence collected, or statement of missing data")
    has_insufficient_data: bool = Field(default=False, description="True if online sources do not provide sufficient verifiable data")


def document_contains_query_keywords(text: str, title: str, query: str, main_query: Optional[str] = None) -> bool:
    """Verify that scraped document actually contains the core entity tokens of the research subject."""
    target = main_query if main_query and main_query.strip() else query
    tokens = extract_entity_tokens(target)
    if not tokens:
        return True
    combined = f"{title.lower()} {text.lower()}"
    if len(tokens) == 1:
        return tokens[0] in combined
    if len(tokens) == 2:
        exact_phrase = f"{tokens[0]} {tokens[1]}"
        if exact_phrase in combined:
            return True
        return tokens[0] in combined and tokens[1] in combined
    matched_count = sum(1 for t in tokens if t in combined)
    return matched_count >= min(2, len(tokens))


class ResearcherAgent(BaseAgent):
    def __init__(
        self,
        subtask_id: str,
        llm: Optional[Any] = None,
        event_callback: Optional[Any] = None
    ):
        super().__init__(
            name=f"Researcher-{subtask_id}",
            role="Explorer & Fact Extractor",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            llm=llm,
            event_callback=event_callback
        )
        self.subtask_id = subtask_id

    async def execute_subtask(
        self,
        subtask: ResearchSubtask,
        main_query: Optional[str] = None,
        existing_source_count: int = 0
    ) -> ResearchFinding:
        """Execute web searches, scrape content, and extract structured findings."""
        self.emit_event(
            "research_started",
            f"Exploration du sous-sujet : '{subtask.title}'",
            {"queries": subtask.target_search_queries}
        )

        # 1. Execute search queries concurrently with main_subject constraint
        search_tasks = [search_web(q, main_subject=main_query) for q in subtask.target_search_queries]
        search_results_lists = await asyncio.gather(*search_tasks, return_exceptions=False)

        seen_urls = set()
        deduped_results: List[SearchResult] = []
        for r_list in search_results_lists:
            for item in r_list:
                if item.url and item.url.startswith("http") and item.url not in seen_urls:
                    seen_urls.add(item.url)
                    deduped_results.append(item)

        # If zero search results found, report insufficient data immediately
        if not deduped_results:
            self.emit_event(
                "subtask_insufficient_data",
                f"Aucune source publique trouvée pour '{subtask.title}'.",
                {"queries": subtask.target_search_queries}
            )
            return ResearchFinding(
                subtask_id=subtask.id,
                query=subtask.title,
                facts=[],
                sources=[],
                summary="Données publiques non documentées : les requêtes web n'ont renvoyé aucun résultat accessible sur cet aspect.",
                has_insufficient_data=True
            )

        deduped_results = deduped_results[:4]

        # 2. Scrape target URLs
        target_urls = [r.url for r in deduped_results]
        scraped_docs_map = await scrape_multiple_urls(target_urls)

        # 3. Create clean SourceCitation records with strict keyword validation
        sources: List[SourceCitation] = []
        current_id = existing_source_count + 1
        for res in deduped_results:
            scraped_doc = scraped_docs_map.get(res.url)
            text_snippet = scraped_doc.text[:280] if scraped_doc and scraped_doc.text else res.snippet
            title = scraped_doc.title if scraped_doc and scraped_doc.title and "Unavailable" not in scraped_doc.title else res.title

            # Strict relevance check on scraped body
            if scraped_doc and scraped_doc.text:
                if not document_contains_query_keywords(scraped_doc.text, title, subtask.title, main_query=main_query):
                    continue

            clean_title = re.sub(r"\s+", " ", title).strip()[:100]

            sources.append(
                SourceCitation(
                    id=current_id,
                    title=clean_title or res.domain,
                    url=res.url,
                    domain=res.domain,
                    snippet=text_snippet.strip()[:250],
                    reliability_score=0.92 if any(k in res.domain for k in ["wikipedia", "gouv", "edu", "nature", "science", "academie", "revue", "inserm", "ncbi"]) else 0.85
                )
            )
            current_id += 1

        # If after scraping all documents failed keyword validation
        if not sources:
            return ResearchFinding(
                subtask_id=subtask.id,
                query=subtask.title,
                facts=[],
                sources=[],
                summary="Données publiques non documentées : les pages trouvées ne correspondent pas à l'entité recherchée.",
                has_insufficient_data=True
            )

        # 4. Extract structured facts using LLM
        sources_context = ""
        for src in sources:
            scraped = scraped_docs_map.get(src.url)
            body = scraped.text if scraped and scraped.text else src.snippet
            if body and len(body.strip()) > 30:
                sources_context += f"--- Source [{src.id}]: {src.title} ({src.url}) ---\n{body[:1200]}\n\n"

        if not sources_context.strip():
            return ResearchFinding(
                subtask_id=subtask.id,
                query=subtask.title,
                facts=[],
                sources=[],
                summary="Contenu web inaccessible : les pages identifiées n'ont pas permis d'extraire de texte exploitable.",
                has_insufficient_data=True
            )

        prompt = (
            f"Subtask: {subtask.title}\n"
            f"Subject: {main_query or subtask.title}\n"
            f"Description: {subtask.description}\n"
            f"Focus Areas: {', '.join(subtask.focus_areas)}\n\n"
            f"Scraped Source Documents:\n{sources_context}\n\n"
            "Extract verified facts from these documents. If the documents do NOT contain facts addressing the subtask, set has_insufficient_data=True and state so in the summary."
        )

        structured_llm = self.llm.with_structured_output(ExtractionOutput)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        try:
            extraction: ExtractionOutput = await structured_llm.ainvoke(messages)
            facts = extraction.facts
            summary = extraction.summary
            insufficient = extraction.has_insufficient_data or len(facts) == 0
        except Exception as e:
            facts = []
            summary = "Données insuffisantes dans les sources publiques consultées."
            insufficient = True

        finding = ResearchFinding(
            subtask_id=subtask.id,
            query=subtask.title,
            facts=facts if not insufficient else [],
            sources=sources if not insufficient else [],
            summary=summary,
            has_insufficient_data=insufficient
        )

        self.emit_event(
            "subtask_completed",
            f"Analyse terminée pour '{subtask.title}' ({len(facts)} faits extraits, Manque de données: {insufficient}).",
            {"facts_count": len(facts), "has_insufficient_data": insufficient}
        )

        return finding
