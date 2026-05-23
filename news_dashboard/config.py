"""Central configuration for the Global Financial News Sentiment Dashboard."""

import os

# ── API ─────────────────────────────────────────────────────────────────────
def _load_api_key() -> str:
    # 1) Streamlit secrets (cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "NEWS_API_KEY" in st.secrets:
            return st.secrets["NEWS_API_KEY"]
    except Exception:
        pass
    # 2) Environment variable
    return os.environ.get("NEWS_API_KEY", "92899cc630874540b41418cc2b3f63e5")

NEWS_API_KEY = _load_api_key()

def _load_hf_token() -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass
    return os.environ.get("HF_TOKEN", "")

HF_TOKEN = _load_hf_token()

# ── Model ────────────────────────────────────────────────────────────────────
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
FINBERT_MAX_TOKENS = 512

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "news_sentiment.db")
_csv_local  = os.path.join(BASE_DIR, "master_stock_universe.csv")
_csv_parent = os.path.join(os.path.dirname(BASE_DIR), "master_stock_universe.csv")
NSE_STOCKS_CSV = _csv_local if os.path.exists(_csv_local) else _csv_parent

# ── Cache ─────────────────────────────────────────────────────────────────
CACHE_EXPIRY_HOURS = 2
MAX_ARTICLES_PER_QUERY = 20
DEFAULT_LOOKBACK_DAYS = 7

# ── Geography ────────────────────────────────────────────────────────────────
CONTINENTS = {
    "Asia": {
        "icon": "🌏",
        "color": "#FF6B6B",
        "countries": {
            "India": {
                "code": "in", "iso3": "IND", "lat": 20.59, "lon": 78.96,
                "keywords": ["India", "Indian", "NSE", "BSE", "Nifty", "Sensex",
                             "SEBI", "RBI", "Mumbai", "Delhi", "Bengaluru", "rupee"],
            },
            "China": {
                "code": "cn", "iso3": "CHN", "lat": 35.86, "lon": 104.19,
                "keywords": ["China", "Chinese", "Shanghai", "Shenzhen", "CSI",
                             "PBOC", "Beijing", "Yuan", "RMB", "Hang Seng"],
            },
            "Japan": {
                "code": "jp", "iso3": "JPN", "lat": 36.20, "lon": 138.25,
                "keywords": ["Japan", "Japanese", "Nikkei", "Topix", "Bank of Japan",
                             "BOJ", "Tokyo", "Yen", "JPY"],
            },
            "South Korea": {
                "code": "kr", "iso3": "KOR", "lat": 35.91, "lon": 127.77,
                "keywords": ["South Korea", "Korean", "KOSPI", "Seoul",
                             "Bank of Korea", "Won", "KRW", "Samsung"],
            },
            "Singapore": {
                "code": "sg", "iso3": "SGP", "lat": 1.35, "lon": 103.82,
                "keywords": ["Singapore", "SGX", "MAS", "Straits Times", "SGD"],
            },
            "Hong Kong": {
                "code": "hk", "iso3": "HKG", "lat": 22.32, "lon": 114.17,
                "keywords": ["Hong Kong", "Hang Seng", "HKEX", "HKD", "HKMA"],
            },
            "Taiwan": {
                "code": "tw", "iso3": "TWN", "lat": 23.70, "lon": 120.96,
                "keywords": ["Taiwan", "Taiwanese", "TSMC", "TAIEX", "TWD", "Taipei"],
            },
            "Indonesia": {
                "code": "id", "iso3": "IDN", "lat": -0.79, "lon": 113.92,
                "keywords": ["Indonesia", "Indonesian", "IDX", "Jakarta", "Rupiah"],
            },
            "Malaysia": {
                "code": "my", "iso3": "MYS", "lat": 4.21, "lon": 101.98,
                "keywords": ["Malaysia", "Malaysian", "Bursa", "KLCI", "Ringgit"],
            },
            "Thailand": {
                "code": "th", "iso3": "THA", "lat": 15.87, "lon": 100.99,
                "keywords": ["Thailand", "Thai", "SET", "Bangkok", "Baht"],
            },
            "Vietnam": {
                "code": "vn", "iso3": "VNM", "lat": 14.06, "lon": 108.28,
                "keywords": ["Vietnam", "Vietnamese", "VNIndex", "Ho Chi Minh", "Dong"],
            },
        },
        "global_queries": [
            "Asia stock market economy",
            "Asian markets earnings",
            "India NSE Nifty economy",
            "China economy markets PBOC",
            "Japan Nikkei economy BOJ",
            "ASEAN emerging markets",
        ],
    },
    "Europe": {
        "icon": "🌍",
        "color": "#4ECDC4",
        "countries": {
            "United Kingdom": {
                "code": "gb", "iso3": "GBR", "lat": 55.38, "lon": -3.44,
                "keywords": ["United Kingdom", "UK", "Britain", "British", "FTSE",
                             "London", "Bank of England", "BOE", "Pound", "GBP", "Sterling"],
            },
            "Germany": {
                "code": "de", "iso3": "DEU", "lat": 51.17, "lon": 10.45,
                "keywords": ["Germany", "German", "DAX", "Frankfurt", "Deutsche",
                             "Bundesbank", "Euro", "EUR", "Berlin"],
            },
            "France": {
                "code": "fr", "iso3": "FRA", "lat": 46.23, "lon": 2.21,
                "keywords": ["France", "French", "CAC 40", "Paris", "Banque de France", "EUR"],
            },
            "Italy": {
                "code": "it", "iso3": "ITA", "lat": 41.87, "lon": 12.57,
                "keywords": ["Italy", "Italian", "MIB", "Milan", "Borsa Italiana", "EUR"],
            },
            "Spain": {
                "code": "es", "iso3": "ESP", "lat": 40.46, "lon": -3.75,
                "keywords": ["Spain", "Spanish", "IBEX", "Madrid", "Banco de España"],
            },
            "Netherlands": {
                "code": "nl", "iso3": "NLD", "lat": 52.13, "lon": 5.29,
                "keywords": ["Netherlands", "Dutch", "AEX", "Amsterdam", "EUR"],
            },
            "Switzerland": {
                "code": "ch", "iso3": "CHE", "lat": 46.82, "lon": 8.23,
                "keywords": ["Switzerland", "Swiss", "SMI", "Zurich", "SNB", "Franc", "CHF"],
            },
            "Sweden": {
                "code": "se", "iso3": "SWE", "lat": 60.13, "lon": 18.64,
                "keywords": ["Sweden", "Swedish", "OMX", "Stockholm", "Riksbank", "Krona"],
            },
        },
        "global_queries": [
            "European economy ECB",
            "Eurozone monetary policy",
            "FTSE DAX CAC European markets",
            "European Central Bank rates",
            "EU economy trade",
            "European stocks earnings",
        ],
    },
    "North America": {
        "icon": "🌎",
        "color": "#45B7D1",
        "countries": {
            "United States": {
                "code": "us", "iso3": "USA", "lat": 37.09, "lon": -95.71,
                "keywords": ["United States", "US", "USA", "American", "Federal Reserve",
                             "Fed", "S&P 500", "Nasdaq", "Dow Jones", "Wall Street",
                             "NYSE", "SEC", "Dollar", "USD", "New York"],
            },
            "Canada": {
                "code": "ca", "iso3": "CAN", "lat": 56.13, "lon": -106.35,
                "keywords": ["Canada", "Canadian", "TSX", "Toronto Stock Exchange",
                             "Bank of Canada", "CAD", "Ottawa", "Toronto"],
            },
            "Mexico": {
                "code": "mx", "iso3": "MEX", "lat": 23.63, "lon": -102.55,
                "keywords": ["Mexico", "Mexican", "IPC", "BMV", "Banxico", "Peso", "MXN"],
            },
        },
        "global_queries": [
            "S&P 500 stock market",
            "Federal Reserve interest rates inflation",
            "US economy GDP earnings",
            "Wall Street Nasdaq technology",
            "Dow Jones market outlook",
            "US Federal Reserve monetary policy",
        ],
    },
}

# ── UI ───────────────────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral":  "#95A5A6",
}
SENTIMENT_EMOJIS = {
    "positive": "📈",
    "negative": "📉",
    "neutral":  "➡️",
}

PAGE_TITLE = "Global Financial News Sentiment Dashboard"
PAGE_ICON  = "📊"
