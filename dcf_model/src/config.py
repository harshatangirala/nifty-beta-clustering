"""
DCF Model Configuration
Nifty 50 stock universe, market timing constants, and DCF defaults.
"""

import pytz

IST = pytz.timezone("Asia/Kolkata")

# NSE market hours (IST)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MIN = 30

# Cache staleness thresholds
PRICE_STALE_SECONDS = 300          # 5 min during live hours
PRICE_STALE_AFTERHOURS_DAYS = 1   # 1 day after close
FINANCIALS_STALE_DAYS = 90        # 3 months

# India macro defaults (update periodically)
INDIA_RISK_FREE_RATE = 0.0700      # 10-Year G-Sec ~7%
INDIA_EQUITY_RISK_PREMIUM = 0.0700 # Damodaran ERP for India ~7%

DCF_DEFAULTS = {
    "risk_free_rate": INDIA_RISK_FREE_RATE,
    "equity_risk_premium": INDIA_EQUITY_RISK_PREMIUM,
    "terminal_growth_rate": 0.04,   # 4% perpetuity growth (nominal GDP)
    "projection_years": 5,
    "growth_stage1_years": 3,       # years 1-3: high growth
    "wacc_min": 0.06,
    "wacc_max": 0.18,
    "growth_rate_max": 0.50,        # flag if >50% CAGR
    "perpetuity_growth_max": 0.07,  # flag if terminal g > 7%
}

# Nifty 50 constituents → yfinance symbol (append .NS)
NIFTY50_STOCKS: dict[str, str] = {
    "ADANIENT":    "Adani Enterprises",
    "ADANIPORTS":  "Adani Ports",
    "APOLLOHOSP":  "Apollo Hospitals",
    "ASIANPAINT":  "Asian Paints",
    "AXISBANK":    "Axis Bank",
    "BAJAJ-AUTO":  "Bajaj Auto",
    "BAJAJFINSV":  "Bajaj Finserv",
    "BAJFINANCE":  "Bajaj Finance",
    "BHARTIARTL":  "Bharti Airtel",
    "BPCL":        "BPCL",
    "BRITANNIA":   "Britannia",
    "CIPLA":       "Cipla",
    "COALINDIA":   "Coal India",
    "DIVISLAB":    "Divi's Labs",
    "DRREDDY":     "Dr. Reddy's",
    "EICHERMOT":   "Eicher Motors",
    "GRASIM":      "Grasim Industries",
    "HCLTECH":     "HCL Technologies",
    "HDFCBANK":    "HDFC Bank",
    "HDFCLIFE":    "HDFC Life Insurance",
    "HEROMOTOCO":  "Hero MotoCorp",
    "HINDALCO":    "Hindalco",
    "HINDUNILVR":  "Hindustan Unilever",
    "ICICIBANK":   "ICICI Bank",
    "INDUSINDBK":  "IndusInd Bank",
    "INFY":        "Infosys",
    "ITC":         "ITC",
    "JSWSTEEL":    "JSW Steel",
    "KOTAKBANK":   "Kotak Mahindra Bank",
    "LT":          "Larsen & Toubro",
    "M&M":         "Mahindra & Mahindra",
    "MARUTI":      "Maruti Suzuki",
    "NESTLEIND":   "Nestle India",
    "NTPC":        "NTPC",
    "ONGC":        "ONGC",
    "POWERGRID":   "Power Grid Corp",
    "RELIANCE":    "Reliance Industries",
    "SBILIFE":     "SBI Life Insurance",
    "SBIN":        "State Bank of India",
    "SHREECEM":    "Shree Cement",
    "SUNPHARMA":   "Sun Pharma",
    "TATACONSUM":  "Tata Consumer",
    "TATAMOTORS":  "Tata Motors",
    "TATASTEEL":   "Tata Steel",
    "TCS":         "TCS",
    "TECHM":       "Tech Mahindra",
    "TITAN":       "Titan Company",
    "ULTRACEMCO":  "UltraTech Cement",
    "WIPRO":       "Wipro",
    "ZOMATO":      "Zomato",
}

# Quarterly earnings announcement months (NSE pattern)
EARNINGS_MONTHS = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct

# INR crore conversion factor
CR = 1e7  # 1 crore = 10 million
