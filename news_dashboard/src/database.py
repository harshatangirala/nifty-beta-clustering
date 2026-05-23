"""SQLite caching layer for news articles, sentiments, and stock mentions."""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Optional
import pandas as pd


class NewsDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS articles (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    url          TEXT    UNIQUE NOT NULL,
                    title        TEXT,
                    description  TEXT,
                    content      TEXT,
                    source_name  TEXT,
                    author       TEXT,
                    published_at TEXT,
                    country      TEXT,
                    continent    TEXT,
                    query_used   TEXT,
                    fetched_at   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS sentiments (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id      INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                    model           TEXT NOT NULL,
                    sentiment_label TEXT,
                    positive_score  REAL,
                    negative_score  REAL,
                    neutral_score   REAL,
                    compound_score  REAL,
                    analyzed_at     TEXT DEFAULT (datetime('now')),
                    UNIQUE(article_id, model)
                );

                CREATE TABLE IF NOT EXISTS stock_mentions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id      INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                    symbol          TEXT,
                    company_name    TEXT,
                    market          TEXT,
                    industry        TEXT,
                    relevance_score REAL,
                    mention_count   INTEGER DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_articles_continent
                    ON articles(continent);
                CREATE INDEX IF NOT EXISTS idx_articles_country
                    ON articles(country);
                CREATE INDEX IF NOT EXISTS idx_articles_published
                    ON articles(published_at);
                CREATE INDEX IF NOT EXISTS idx_stock_symbol
                    ON stock_mentions(symbol);
            """)

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert_article(self, article: dict) -> int:
        """Insert or ignore duplicate URL; return row id."""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO articles
                   (url, title, description, content, source_name, author,
                    published_at, country, continent, query_used)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    article.get("url", ""),
                    article.get("title", ""),
                    article.get("description", ""),
                    article.get("content", ""),
                    article.get("source_name", ""),
                    article.get("author", ""),
                    article.get("published_at", ""),
                    article.get("country", ""),
                    article.get("continent", ""),
                    article.get("query_used", ""),
                ),
            )
            row_id = cur.lastrowid
            if row_id == 0:
                row = conn.execute(
                    "SELECT id FROM articles WHERE url=?", (article["url"],)
                ).fetchone()
                row_id = row["id"] if row else 0
            return row_id

    def upsert_sentiment(self, article_id: int, model: str, result: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sentiments
                   (article_id, model, sentiment_label, positive_score,
                    negative_score, neutral_score, compound_score)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    article_id, model,
                    result.get("label", "neutral"),
                    result.get("positive", 0.0),
                    result.get("negative", 0.0),
                    result.get("neutral", 1.0),
                    result.get("compound", 0.0),
                ),
            )

    def upsert_stock_mentions(self, article_id: int, mentions: list[dict]):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM stock_mentions WHERE article_id=?", (article_id,)
            )
            for m in mentions:
                conn.execute(
                    """INSERT INTO stock_mentions
                       (article_id, symbol, company_name, market, industry,
                        relevance_score, mention_count)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        article_id,
                        m["symbol"], m["company_name"], m["market"],
                        m.get("industry", ""), m.get("relevance_score", 1.0),
                        m.get("mention_count", 1),
                    ),
                )

    # ── Read ─────────────────────────────────────────────────────────────────

    def is_cached(self, url: str, expiry_hours: int = 2) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT fetched_at FROM articles WHERE url=?""", (url,)
            ).fetchone()
            if not row:
                return False
            fetched = datetime.fromisoformat(row["fetched_at"].replace("Z", "+00:00"))
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            return age_hours < expiry_hours

    def get_articles(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        continent: Optional[str] = None,
        country: Optional[str] = None,
        sentiment: Optional[str] = None,
        model: str = "finbert",
        limit: int = 500,
    ) -> pd.DataFrame:
        query = """
            SELECT
                a.id, a.url, a.title, a.description, a.source_name,
                a.published_at, a.country, a.continent,
                s.sentiment_label, s.positive_score, s.negative_score,
                s.neutral_score, s.compound_score
            FROM articles a
            LEFT JOIN sentiments s ON a.id = s.article_id AND s.model = ?
            WHERE 1=1
        """
        params: list = [model]

        if from_date:
            query += " AND a.published_at >= ?"
            params.append(from_date)
        if to_date:
            query += " AND a.published_at <= ?"
            params.append(to_date + "T23:59:59")
        if continent:
            query += " AND a.continent = ?"
            params.append(continent)
        if country:
            query += " AND a.country = ?"
            params.append(country)
        if sentiment:
            query += " AND s.sentiment_label = ?"
            params.append(sentiment)

        query += " ORDER BY a.published_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_stock_impact(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        market: Optional[str] = None,
        model: str = "finbert",
    ) -> pd.DataFrame:
        query = """
            SELECT
                sm.symbol, sm.company_name, sm.market, sm.industry,
                sm.relevance_score, sm.mention_count,
                s.sentiment_label, s.positive_score, s.negative_score,
                s.neutral_score, s.compound_score,
                a.published_at, a.title, a.url, a.country, a.continent
            FROM stock_mentions sm
            JOIN articles a ON sm.article_id = a.id
            LEFT JOIN sentiments s ON a.id = s.article_id AND s.model = ?
            WHERE 1=1
        """
        params: list = [model]

        if from_date:
            query += " AND a.published_at >= ?"
            params.append(from_date)
        if to_date:
            query += " AND a.published_at <= ?"
            params.append(to_date + "T23:59:59")
        if market:
            query += " AND sm.market = ?"
            params.append(market)

        query += " ORDER BY a.published_at DESC"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_sentiment_timeseries(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        continent: Optional[str] = None,
        model: str = "finbert",
    ) -> pd.DataFrame:
        query = """
            SELECT
                date(a.published_at) AS pub_date,
                a.continent,
                s.sentiment_label,
                COUNT(*) AS article_count,
                AVG(s.positive_score)  AS avg_positive,
                AVG(s.negative_score)  AS avg_negative,
                AVG(s.neutral_score)   AS avg_neutral,
                AVG(s.compound_score)  AS avg_compound
            FROM articles a
            JOIN sentiments s ON a.id = s.article_id AND s.model = ?
            WHERE 1=1
        """
        params: list = [model]
        if from_date:
            query += " AND a.published_at >= ?"
            params.append(from_date)
        if to_date:
            query += " AND a.published_at <= ?"
            params.append(to_date + "T23:59:59")
        if continent:
            query += " AND a.continent = ?"
            params.append(continent)

        query += " GROUP BY pub_date, a.continent, s.sentiment_label ORDER BY pub_date"

        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            analyzed = conn.execute(
                "SELECT COUNT(DISTINCT article_id) FROM sentiments"
            ).fetchone()[0]
            mapped = conn.execute(
                "SELECT COUNT(DISTINCT article_id) FROM stock_mentions"
            ).fetchone()[0]
        return {"total_articles": total, "analyzed": analyzed, "stock_mapped": mapped}
