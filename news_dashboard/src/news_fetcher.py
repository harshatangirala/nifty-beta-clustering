"""NewsAPI fetcher with rate limiting and SQLite caching."""

import time
from datetime import datetime, timedelta, date
from typing import Optional
import requests

from .database import NewsDatabase


class NewsAPIFetcher:
    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str, db: NewsDatabase, cache_expiry_hours: int = 2):
        self.api_key = api_key
        self.db = db
        self.cache_expiry_hours = cache_expiry_hours
        self._last_call = 0.0
        self._min_interval = 0.5  # 500 ms between calls

    # ── Internal HTTP ────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        params["apiKey"] = self.api_key
        try:
            resp = requests.get(
                f"{self.BASE_URL}/{endpoint}", params=params, timeout=15
            )
            self._last_call = time.time()
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "message": resp.text, "articles": []}
        except Exception as exc:
            return {"status": "error", "message": str(exc), "articles": []}

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch_everything(
        self,
        query: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20,
    ) -> list[dict]:
        params: dict = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
        }
        if from_date:
            params["from"] = from_date.isoformat() if isinstance(from_date, (date, datetime)) else from_date
        if to_date:
            params["to"] = to_date.isoformat() if isinstance(to_date, (date, datetime)) else to_date

        data = self._get("everything", params)
        return data.get("articles", [])

    def fetch_top_headlines(
        self,
        country: Optional[str] = None,
        query: Optional[str] = None,
        category: Optional[str] = "business",
        page_size: int = 20,
    ) -> list[dict]:
        params: dict = {"pageSize": min(page_size, 100)}
        if country:
            params["country"] = country
        if query:
            params["q"] = query
        if category:
            params["category"] = category

        data = self._get("top-headlines", params)
        return data.get("articles", [])

    # ── High-level helpers ───────────────────────────────────────────────────

    def fetch_for_continent(
        self,
        continent_name: str,
        continent_cfg: dict,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        max_per_query: int = 20,
    ) -> list[dict]:
        """Fetch articles for an entire continent using configured queries."""
        all_raw: list[dict] = []

        for query in continent_cfg.get("global_queries", []):
            raw = self.fetch_everything(
                query=query,
                from_date=from_date,
                to_date=to_date,
                page_size=max_per_query,
            )
            all_raw.extend(raw)

        return self._normalize_and_cache(
            all_raw, continent_name, continent_cfg, from_date, to_date
        )

    def fetch_for_country(
        self,
        country_name: str,
        country_cfg: dict,
        continent_name: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        max_per_query: int = 20,
    ) -> list[dict]:
        """Fetch articles for a specific country."""
        code = country_cfg.get("code", "")
        all_raw: list[dict] = []

        # Top headlines for the country (business)
        if code:
            raw = self.fetch_top_headlines(
                country=code,
                category="business",
                page_size=max_per_query,
            )
            all_raw.extend(raw)

        # Keyword-based search
        primary_kw = country_cfg.get("keywords", [country_name])[:3]
        for kw in primary_kw[:2]:
            raw = self.fetch_everything(
                query=f"{kw} stock market economy",
                from_date=from_date,
                to_date=to_date,
                page_size=max_per_query,
            )
            all_raw.extend(raw)

        return self._normalize_and_cache(
            all_raw, continent_name, {}, from_date, to_date, country_name
        )

    # ── Normalisation ────────────────────────────────────────────────────────

    def _normalize_and_cache(
        self,
        raw_articles: list[dict],
        continent_name: str,
        continent_cfg: dict,
        from_date,
        to_date,
        forced_country: Optional[str] = None,
    ) -> list[dict]:
        seen_urls: set = set()
        normalized: list[dict] = []

        for art in raw_articles:
            url = art.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # Skip removed / [Removed] articles
            if art.get("title") in (None, "[Removed]"):
                continue

            country = forced_country or self._detect_country(art, continent_cfg)

            norm = {
                "url":          url,
                "title":        art.get("title") or "",
                "description":  art.get("description") or "",
                "content":      (art.get("content") or "")[:1000],
                "source_name":  (art.get("source") or {}).get("name", ""),
                "author":       art.get("author") or "",
                "published_at": art.get("publishedAt") or "",
                "country":      country,
                "continent":    continent_name,
                "query_used":   "",
            }

            # Cache
            if not self.db.is_cached(url, self.cache_expiry_hours):
                self.db.upsert_article(norm)

            normalized.append(norm)

        return normalized

    def _detect_country(self, article: dict, continent_cfg: dict) -> str:
        text = " ".join([
            article.get("title", "") or "",
            article.get("description", "") or "",
        ]).lower()

        best_country = ""
        best_count = 0
        for cname, ccfg in continent_cfg.get("countries", {}).items():
            count = sum(
                1 for kw in ccfg.get("keywords", []) if kw.lower() in text
            )
            if count > best_count:
                best_count = count
                best_country = cname

        return best_country or "Unknown"
