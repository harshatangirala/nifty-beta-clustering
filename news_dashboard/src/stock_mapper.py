"""
Map news articles to stocks using keyword matching.

Strategy (scored additively):
  +3  ticker symbol found (word-boundary match)
  +2  full company name found
  +1  each keyword found
  ×2  title match vs body match
"""

from __future__ import annotations
import re
from typing import Optional
import pandas as pd


def _wb(term: str) -> re.Pattern:
    """Word-boundary regex for a term."""
    escaped = re.escape(term)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


class StockMapper:

    def __init__(self, nse_df: pd.DataFrame, sp500_df: pd.DataFrame):
        self._stocks: list[dict] = []
        self._build_index(nse_df, sp500_df)

    def _build_index(self, nse_df: pd.DataFrame, sp500_df: pd.DataFrame):
        # NSE stocks
        for _, row in nse_df.iterrows():
            symbol   = str(row.get("Symbol", "")).strip()
            name     = str(row.get("Company Name", "")).strip()
            industry = str(row.get("Industry", "")).strip()
            index_nm = str(row.get("Index Name", "")).strip()
            if not symbol:
                continue

            short_name = name.replace(" Ltd.", "").replace(" Limited", "").replace(" Ltd", "").strip()

            keywords = [name, short_name, symbol]
            # Add significant words from company name (length > 4)
            for word in name.split():
                clean = word.strip(".,()&")
                if len(clean) > 4 and clean not in {"India", "Indian", "Limited", "Industries"}:
                    keywords.append(clean)

            self._stocks.append({
                "symbol":   symbol,
                "name":     name,
                "industry": industry,
                "index":    index_nm,
                "market":   "NSE",
                "patterns": self._compile_patterns(keywords),
                "ticker_re": _wb(symbol),
                "name_re":   _wb(re.escape(short_name)),
            })

        # S&P 500 stocks
        for _, row in sp500_df.iterrows():
            symbol   = str(row.get("Symbol", "")).strip()
            name     = str(row.get("Company Name", "")).strip()
            industry = str(row.get("Industry", "")).strip()
            kws      = row.get("keywords", []) or []
            if not symbol:
                continue

            short_name = name.split("(")[0].strip()

            all_kws = list(kws) + [name, short_name, symbol]
            self._stocks.append({
                "symbol":   symbol,
                "name":     name,
                "industry": industry,
                "index":    "S&P 500",
                "market":   "S&P 500",
                "patterns": self._compile_patterns(all_kws),
                "ticker_re": _wb(symbol),
                "name_re":   _wb(re.escape(short_name)),
            })

    @staticmethod
    def _compile_patterns(keywords: list[str]) -> list[re.Pattern]:
        seen: set = set()
        patterns = []
        for kw in keywords:
            kw = kw.strip()
            low = kw.lower()
            if low in seen or len(kw) < 3:
                continue
            seen.add(low)
            patterns.append(_wb(kw))
        return patterns

    # ── Public ───────────────────────────────────────────────────────────────

    def find_related_stocks(
        self,
        title: str,
        body: str,
        market: Optional[str] = None,
        top_n: int = 10,
        min_score: float = 1.0,
    ) -> list[dict]:
        title_text = (title or "").strip()
        body_text  = (body  or "").strip()
        combined   = title_text + " " + body_text

        matches: list[dict] = []

        for stock in self._stocks:
            if market and stock["market"] != market:
                continue

            score = 0.0
            count = 0

            # Ticker match (high weight)
            ticker_in_title = len(stock["ticker_re"].findall(title_text))
            ticker_in_body  = len(stock["ticker_re"].findall(body_text))
            if ticker_in_title:
                score += ticker_in_title * 6
                count += ticker_in_title
            if ticker_in_body:
                score += ticker_in_body * 3
                count += ticker_in_body

            # Full / short name match
            name_in_title = len(stock["name_re"].findall(title_text))
            name_in_body  = len(stock["name_re"].findall(body_text))
            if name_in_title:
                score += name_in_title * 4
                count += name_in_title
            if name_in_body:
                score += name_in_body * 2
                count += name_in_body

            # Keyword patterns
            for pat in stock["patterns"]:
                hits_title = len(pat.findall(title_text))
                hits_body  = len(pat.findall(body_text))
                if hits_title:
                    score += hits_title * 2
                    count += hits_title
                if hits_body:
                    score += hits_body * 1
                    count += hits_body

            if score >= min_score:
                matches.append({
                    "symbol":          stock["symbol"],
                    "company_name":    stock["name"],
                    "market":          stock["market"],
                    "industry":        stock["industry"],
                    "relevance_score": round(score, 2),
                    "mention_count":   count,
                })

        matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        return matches[:top_n]
