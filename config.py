# risk_engine/config.py
# Central configuration — edit this file to change the portfolio or date range.

import os

# --- Portfolio ---
TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "SPY"]

PORTFOLIO_WEIGHTS = {
    "AAPL": 0.25,
    "JPM":  0.20,
    "XOM":  0.20,
    "JNJ":  0.15,
    "SPY":  0.20,
}

PORTFOLIO_VALUE = 1_000_000  # USD notional

# --- Data ---
START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "market_data.db")

# --- Risk parameters ---
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
VAR_WINDOW = 252          # 1-year rolling window (trading days)
HOLDING_PERIOD = 1        # days (Basel standard for daily VaR)
MONTE_CARLO_SIMS = 10_000 # number of MC paths
EWMA_LAMBDA      = 0.94   # RiskMetrics decay factor for EWMA volatility

# --- Options (Module 7: Greeks) ---
OPTION_TICKER         = "AAPL"   # underlying for the hypothetical option
OPTION_RISK_FREE_RATE = 0.05     # annualised, continuously compounded
OPTION_EXPIRY_DAYS    = 30       # calendar days to expiry
OPTION_CONTRACTS      = 10       # position size (1 contract = 100 shares)
