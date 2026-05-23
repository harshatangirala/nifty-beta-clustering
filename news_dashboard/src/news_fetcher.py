"""NewsAPI fetcher with rate limiting, SQLite caching, and visible error reporting."""

import time
import logging
from datetime import date, datetime
from typing import Optional

import requests

from .database import NewsDatabase

logger = logging.getLogger(__name__)


class NewsAPIFetcher:
    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str, db: NewsDatabase, cache_expiry_hours: int = 2):
        self.api_key  = api_key
        self.db       = db
        self.cache_expiry_hours = cache_expiry_hours
        self._last_call  = 0.0
        self._min_interval = 0.5  # 500 ms between requests
        self.last_error: Optional[str] = None
        self.last_status: Optional[int] = None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> dict:
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        params["apiKey"] = self.api_key
        self.last_error  = None

        try:
            resp = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params, timeout=15,
            )
            self._last_call  = time.time()
            self.last_status = resp.status_code

            data = resp.json()

            if resp.status_code == 200:
                return data

            # Surface the error clearly
            msg = data.get("message", resp.text[:200])
            self.last_error = f"[{resp.status_code}] {msg}"
            logger.warning("NewsAPI error %s: %s", endpoint, self.last_error)
            return {"status": "error", "articles": [], "message": self.last_error}

        except Exception as exc:
            self.last_error  = str(exc)
            self.last_status = 0
            logger.warning("NewsAPI request failed: %s", exc)
            return {"status": "error", "articles": [], "message": str(exc)}

    # ── Public endpoints ──────────────────────────────────────────────────────

    def fetch_everything(
        self,
        query: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20,
    ) -> tuple[list[dict], Optional[str]]:
        """Returns (articles, error_msg).  error_msg is None on success."""
        params: dict = {
            "q":        query,
            "language": language,
            "sortBy":   sort_by,
            "pageSize": min(page_size, 100),
        }
        if from_date:
            params["from"] = from_date.isoformat() if hasattr(from_date, "isoformat") else str(from_date)
        if to_date:
            params["to"]   = to_date.isoformat() if hasattr(to_date, "isoformat") else str(to_date)

        data = self._get("everything", params)
        return data.get("articles", []), self.last_error

    def fetch_top_headlines(
        self,
        country: Optional[str] = None,
        query: Optional[str] = None,
        category: str = "business",
        page_size: int = 20,
    ) -> tuple[list[dict], Optional[str]]:
        params: dict = {"pageSize": min(page_size, 100)}
        if country:
            params["country"]  = country
        if query:
            params["q"]        = query
        if category:
            params["category"] = category

        data = self._get("top-headlines", params)
        return data.get("articles", []), self.last_error

    # ── High-level ────────────────────────────────────────────────────────────

    def fetch_for_continent(
        self,
        continent_name: str,
        continent_cfg: dict,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        max_per_query: int = 20,
    ) -> tuple[list[dict], list[str]]:
        """Returns (articles, errors)."""
        all_raw: list[dict] = []
        errors:  list[str]  = []

        for query in continent_cfg.get("global_queries", []):
            articles, err = self.fetch_everything(
                query=query,
                from_date=from_date,
                to_date=to_date,
                page_size=max_per_query,
            )
            all_raw.extend(articles)
            if err:
                errors.append(f"Query '{query[:40]}': {err}")
                break  # stop on hard error (e.g. 401 bad key, 429 rate limit)

        normalized = self._normalize(all_raw, continent_name, continent_cfg)
        return normalized, errors

    def fetch_for_country(
        self,
        country_name: str,
        country_cfg: dict,
        continent_name: str,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        max_per_query: int = 20,
    ) -> tuple[list[dict], list[str]]:
        code   = country_cfg.get("code", "")
        all_raw: list[dict] = []
        errors: list[str]   = []

        # Top headlines
        if code:
            articles, err = self.fetch_top_headlines(
                country=code, category="business", page_size=max_per_query
            )
            all_raw.extend(articles)
            if err:
                errors.append(f"Headlines {code}: {err}")

        # Keyword search (first two keywords only to save quota)
        for kw in country_cfg.get("keywords", [country_name])[:2]:
            articles, err = self.fetch_everything(
                query=f"{kw} stock market economy",
                from_date=from_date,
                to_date=to_date,
                page_size=max_per_query,
            )
            all_raw.extend(articles)
            if err:
                errors.append(f"Search '{kw}': {err}")
                break

        dummy_cfg = {"countries": {country_name: country_cfg}}
        normalized = self._normalize(all_raw, continent_name, dummy_cfg, forced_country=country_name)
        return normalized, errors

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalize(
        self,
        raw: list[dict],
        continent: str,
        cont_cfg: dict,
        forced_country: Optional[str] = None,
    ) -> list[dict]:
        seen: set = set()
        out:  list[dict] = []
        for art in raw:
            url = art.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            title = art.get("title") or ""
            if title in ("[Removed]", ""):
                continue

            country = forced_country or self._detect_country(art, cont_cfg)
            norm = {
                "url":          url,
                "title":        title,
                "description":  (art.get("description") or "")[:500],
                "content":      (art.get("content")     or "")[:1000],
                "source_name":  (art.get("source") or {}).get("name", ""),
                "author":       art.get("author") or "",
                "published_at": art.get("publishedAt") or "",
                "country":      country,
                "continent":    continent,
                "query_used":   "",
            }
            if not self.db.is_cached(url, self.cache_expiry_hours):
                self.db.upsert_article(norm)
            out.append(norm)
        return out

    @staticmethod
    def _detect_country(article: dict, cont_cfg: dict) -> str:
        text = " ".join([
            article.get("title",       "") or "",
            article.get("description", "") or "",
        ]).lower()
        best, best_n = "", 0
        for cname, ccfg in cont_cfg.get("countries", {}).items():
            n = sum(1 for kw in ccfg.get("keywords", []) if kw.lower() in text)
            if n > best_n:
                best_n, best = n, cname
        return best or "Unknown"
