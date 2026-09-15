"""Export search and scraping tool interfaces."""
from src.tools.search import search_web, SearchResult
from src.tools.scraper import scrape_url, scrape_multiple_urls, ScrapedDocument

__all__ = [
    "search_web",
    "SearchResult",
    "scrape_url",
    "scrape_multiple_urls",
    "ScrapedDocument",
]
