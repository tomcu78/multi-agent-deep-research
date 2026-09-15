"""Search tool integrations supporting DDGS with strict entity grounding and multi-backend fallbacks."""
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from src.config import settings

logger = logging.getLogger(__name__)


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str, domain: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.domain = domain or urlparse(url).netloc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain
        }


STOP_WORDS = {
    "de", "la", "le", "un", "une", "du", "des", "en", "et", "ou", "sur", "pour", "dans", "par",
    "the", "and", "for", "with", "from", "about", "into", "over", "after",
    "biographie", "biography", "parcours", "profil", "profile", "publications", "articles",
    "citations", "actualite", "actualites", "news", "architecture", "overview", "guide",
    "theses", "thèses", "recherche", "recherches", "research", "linkedin", "benchmark", "benchmarks",
    "definition", "définition", "impact", "application", "applications", "state", "art",
    "fondements", "origines", "fondateur", "science", "donnees", "données", "faits", "information",
    "informations", "carriere", "carrière", "postes", "projets", "distinctions", "collaborations",
    "limites", "perspectives", "mecanismes", "mécanismes", "fonctionnement", "enjeux", "cas", "usage",
    "production", "results", "analysis", "etude", "étude"
}


def extract_entity_tokens(query: str) -> List[str]:
    """Extract distinctive entity tokens from a search query or subject string."""
    raw_tokens = re.findall(r"[a-zA-Z0-9\u00C0-\u024F]+", query.lower())
    tokens = [t for t in raw_tokens if len(t) > 2 and t not in STOP_WORDS]
    if not tokens:
        # Fallback to non-short tokens if all were filtered
        tokens = [t for t in raw_tokens if len(t) > 2]
    return tokens


def is_relevant_result(result: SearchResult, query: str, main_subject: Optional[str] = None) -> bool:
    """Strict entity alignment: verify that search results actually contain the query's core entity terms."""
    target = main_subject if main_subject and main_subject.strip() else query
    tokens = extract_entity_tokens(target)

    title_lower = result.title.lower()
    snippet_lower = result.snippet.lower()
    url_lower = result.url.lower()
    combined = f"{title_lower} {snippet_lower} {url_lower}"

    # Discard disambiguation and homonym pages
    if "homonymie" in title_lower or "disambiguation" in title_lower or "(prénom)" in title_lower or "(given name)" in title_lower:
        return False

    if not tokens:
        return True

    # Single token query (e.g. 'caca', 'langgraph')
    if len(tokens) == 1:
        tok = tokens[0]
        # Check if the exact token appears in the content
        return tok in combined

    # Multi-token query / Proper Name (e.g. 'Inys Maani', 'Karim Dorgham', 'Quantum Computing')
    if len(tokens) == 2:
        # Require BOTH core tokens to be present, or the full exact phrase
        exact_phrase = f"{tokens[0]} {tokens[1]}"
        if exact_phrase in combined:
            return True
        return tokens[0] in combined and tokens[1] in combined

    # 3+ tokens: require at least 2 distinctive tokens
    matched_count = sum(1 for t in tokens if t in combined)
    return matched_count >= min(2, len(tokens))


import httpx

async def search_ddgs(query: str, max_results: int = 6, main_subject: Optional[str] = None) -> List[SearchResult]:
    """Execute search using DDGS with auto backend and fast timeout."""
    clean_q = re.sub(r'["\']', '', query).strip()

    def _run():
        results = []
        try:
            from ddgs import DDGS
            with DDGS(timeout=4.0) as ddgs:
                try:
                    raw_results = list(ddgs.text(clean_q, max_results=max_results))
                    if raw_results:
                        for r in raw_results:
                            title = r.get("title", "")
                            url = r.get("href", "")
                            body = r.get("body", "")
                            domain = urlparse(url).netloc
                            res = SearchResult(title=title, url=url, snippet=body, domain=domain)
                            if is_relevant_result(res, clean_q, main_subject=main_subject):
                                results.append(res)
                except Exception as be_err:
                    logger.debug(f"DDGS error: {be_err}")
        except Exception as e:
            logger.warning(f"DDGS search error for query '{clean_q}': {e}")
        return results

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run), timeout=5.0)
    except (asyncio.TimeoutError, Exception):
        return []


async def search_wikipedia(query: str, max_results: int = 4, main_subject: Optional[str] = None) -> List[SearchResult]:
    """Direct search on Wikimedia / Wikipedia API for high-authority structured knowledge."""
    clean_q = re.sub(r'["\']', '', query).strip()
    headers = {"User-Agent": "DeepResearchMultiAgent/1.0 (https://cottutom.fr)"}
    results = []
    
    async with httpx.AsyncClient(timeout=4.0, headers=headers) as client:
        for lang in ["fr", "en"]:
            try:
                url = f"https://{lang}.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": clean_q,
                    "format": "json",
                    "srlimit": max_results
                }
                r = await client.get(url, params=params)
                if r.status_code == 200:
                    data = r.json()
                    for item in data.get("query", {}).get("search", []):
                        title = item.get("title", "")
                        raw_snippet = item.get("snippet", "")
                        snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()
                        page_url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        domain = f"{lang}.wikipedia.org"
                        res_obj = SearchResult(title=title, url=page_url, snippet=snippet, domain=domain)
                        if is_relevant_result(res_obj, clean_q, main_subject=main_subject):
                            results.append(res_obj)
            except Exception as e:
                logger.debug(f"Wikipedia search failed for {lang}: {e}")
                
    return results


async def search_tavily(query: str, max_results: int = 6, main_subject: Optional[str] = None) -> List[SearchResult]:
    """Execute search query using Tavily API if key is available."""
    if not settings.TAVILY_API_KEY:
        return []

    def _run_tavily():
        results = []
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            res = client.search(query=query, max_results=max_results)
            for r in res.get("results", []):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
                domain = urlparse(url).netloc
                res_obj = SearchResult(title=title, url=url, snippet=content, domain=domain)
                if is_relevant_result(res_obj, query, main_subject=main_subject):
                    results.append(res_obj)
        except Exception as e:
            logger.warning(f"Tavily search error for query '{query}': {e}")
        return results

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run_tavily), timeout=8.5)
    except asyncio.TimeoutError:
        return []


async def search_web(query: str, max_results: Optional[int] = None, main_subject: Optional[str] = None) -> List[SearchResult]:
    """Unified search: Tavily -> DDGS -> Wikipedia with strict entity relevance checking."""
    limit = max_results or settings.MAX_SEARCH_RESULTS_PER_QUERY

    if settings.TAVILY_API_KEY:
        tavily_results = await search_tavily(query, max_results=limit, main_subject=main_subject)
        if tavily_results:
            return tavily_results

    ddgs_results = await search_ddgs(query, max_results=limit, main_subject=main_subject)
    if ddgs_results:
        return ddgs_results

    wiki_results = await search_wikipedia(query, max_results=limit, main_subject=main_subject)
    if wiki_results:
        return wiki_results

    return []


def generate_fallback_search_results(query: str, max_results: int = 3) -> List[SearchResult]:
    return []
