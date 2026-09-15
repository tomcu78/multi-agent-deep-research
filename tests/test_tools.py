"""Unit tests for web search tools and scrapers."""
import pytest
from src.tools.search import search_web, generate_fallback_search_results
from src.tools.scraper import scrape_url, ScrapedDocument


@pytest.mark.asyncio
async def test_search_web_fallback():
    query = "LangGraph multi-agent deep research"
    results = await search_web(query, max_results=3)
    assert len(results) > 0
    assert results[0].title != ""
    assert results[0].url.startswith("http")
    assert results[0].domain != ""


@pytest.mark.asyncio
async def test_scraper_fallback_graceful():
    url = "https://non-existent-domain-for-test-xyz-987.org/article"
    doc = await scrape_url(url)
    assert isinstance(doc, ScrapedDocument)
    assert doc.url == url
    assert doc.title != ""
