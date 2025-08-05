"""
Robust scrape(url) helper:

1. Try newspaper3k (fastest).
2. If it 4xx/5xx's, fall back to requests + basic <p> extraction.
"""

import requests, validators, bs4
from newspaper import Article
from newspaper.article import ArticleException

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def _fallback(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = bs4.BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string if soup.title else ""
    body  = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    return f"{title}\n{body}"

def scrape(url: str) -> str:
    if not validators.url(url):
        raise ValueError("Invalid URL")
    art = Article(url)
    try:
        art.download(); art.parse()
        return f"{art.title}\n{art.text}"
    except ArticleException:
        return _fallback(url)
