"""Asynchronous web scraping and content extraction tool with advanced boilerplate filtering."""
import asyncio
import logging
import re
import httpx
import trafilatura
from bs4 import BeautifulSoup
from typing import Optional, Dict
from src.config import settings

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
}

BOILERPLATE_PATTERNS = [
    r"aller directement à.*",
    r"skip to (?:content|search|navigation|main).*",
    r"please enable javascript.*",
    r"veuillez activer javascript.*",
    r"bienvenue sur (?:idref|le site).*",
    r"politique (?:de confidentialité|des cookies).*",
    r"tous droits réservés.*",
    r"all rights reserved.*",
    r"sign in|log in|se connecter|créer un compte",
    r"accept all cookies|accepter tous les cookies",
    r"par souci de protection des données.*",
    r"menu de navigation|navigation menu",
    r"barre de recherche|search bar",
    r"_waf_|cloudflare|challenge-platform|__cf_bm",
    r"\{\s*\"_waf",
]


def is_gibberish_or_token(text: str) -> bool:
    """Detect if a text contains base64/token/WAF payloads or random hashes."""
    if "{" in text and "}" in text and ":" in text:
        return True
    if "_waf_" in text or "waf" in text.lower():
        return True
    # Count non-space sequence length
    words = text.split()
    if any(len(w) > 35 and not w.startswith("http") for w in words):
        return True
    return False


def sanitize_text(text: str) -> str:
    """Strip unprintable chars, WAF payloads and web boilerplate noise."""
    if not text:
        return ""
    
    # 1. Strip binary / control chars
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd\xad]", " ", text)
    cleaned = re.sub(r"[^\x20-\x7E\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F\n\r\t.,;:!?'\"()\[\]/\\%-]", " ", cleaned)
    
    # 2. Filter lines with boilerplate navigation phrases or WAF tokens
    lines = []
    for line in cleaned.split("\n"):
        line_str = line.strip()
        if len(line_str) < 15:
            continue
        if is_gibberish_or_token(line_str):
            continue
        is_junk = any(re.search(pat, line_str, re.IGNORECASE) for pat in BOILERPLATE_PATTERNS)
        if not is_junk:
            if line_str.count("|") > 3 and len(line_str) < 60:
                continue
            lines.append(line_str)

    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
    return cleaned_text.strip()


class ScrapedDocument:
    def __init__(self, url: str, title: str, text: str, success: bool = True, error: Optional[str] = None):
        self.url = url
        self.title = title
        self.text = text
        self.success = success
        self.error = error


async def scrape_url(url: str, max_length: int = 4000) -> ScrapedDocument:
    """Fetch URL and extract clean, readable text without boilerplate."""
    if not url.startswith("http://") and not url.startswith("https://"):
        return ScrapedDocument(
            url=url,
            title="Local Reference",
            text="",
            success=True
        )

    # Skip raw PDF or binary media files
    if any(url.lower().endswith(ext) for ext in [".pdf", ".png", ".jpg", ".jpeg", ".zip", ".tar", ".gz"]):
        return ScrapedDocument(
            url=url,
            title=url.split("/")[-1],
            text="",
            success=False,
            error="Skipped binary document"
        )

    try:
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            timeout=settings.SCRAPE_TIMEOUT_SECONDS,
            follow_redirects=True,
            verify=False
        ) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return ScrapedDocument(
                    url=url,
                    title="Unavailable Source",
                    text="",
                    success=False,
                    error=f"HTTP status {response.status_code}"
                )

            c_type = response.headers.get("content-type", "").lower()
            if "pdf" in c_type or "octet-stream" in c_type:
                return ScrapedDocument(
                    url=url,
                    title=url.split("/")[-1],
                    text="",
                    success=False,
                    error="Skipped non-HTML content"
                )

            html_content = response.text
            
            # Check for WAF / Captcha page
            if "challenge-running" in html_content or "_waf_" in html_content:
                return ScrapedDocument(
                    url=url,
                    title="Protected Page",
                    text="",
                    success=False,
                    error="WAF Challenge"
                )

            # 1. Try Trafilatura
            extracted = trafilatura.extract(
                html_content,
                include_links=False,
                include_images=False,
                output_format="txt"
            )

            # 2. Fallback to BeautifulSoup
            if not extracted:
                soup = BeautifulSoup(html_content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                    tag.extract()
                extracted = " ".join(soup.stripped_strings)

            soup_title = BeautifulSoup(html_content, "html.parser").title
            title = soup_title.string.strip() if soup_title and soup_title.string else url

            clean_text = sanitize_text(extracted or "")
            clean_title = sanitize_text(title)

            if len(clean_text) > max_length:
                clean_text = clean_text[:max_length]

            return ScrapedDocument(
                url=url,
                title=clean_title,
                text=clean_text,
                success=bool(clean_text)
            )

    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ScrapedDocument(
            url=url,
            title="Document Web",
            text="",
            success=False,
            error=str(e)
        )


async def scrape_multiple_urls(urls: list[str], max_length: int = 4000) -> Dict[str, ScrapedDocument]:
    """Scrape multiple URLs concurrently."""
    tasks = [scrape_url(url, max_length=max_length) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return {doc.url: doc for doc in results}
