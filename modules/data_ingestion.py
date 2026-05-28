# risk_engine/modules/data_ingestion.py
"""
Module 1 — Data Ingestion
=========================
Fetches historical adjusted close prices and stores them in a local SQLite
database. Also computes and stores daily log returns.

Data source: yfinance (Yahoo Finance).
  - In environments without internet access, a synthetic data generator
    is provided as a drop-in fallback (see generate_synthetic_prices).

Two tables created / updated:
  prices   : adjusted close price per ticker per date
  returns  : daily log return per ticker per date

Usage (from project root):
    python -m risk_engine.modules.data_ingestion
"""

import sqlite3
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from risk_engine.config import TICKERS, START_DATE, END_DATE, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Custom SQLite aggregate: sample standard deviation ──────────────────────

class _StdDev:
    """Sample standard deviation as a SQLite aggregate function."""
    def __init__(self):
        self.vals = []
    def step(self, v):
        if v is not None:
            self.vals.append(v)
    def finalize(self):
        n = len(self.vals)
        if n < 2:
            return 0.0
        mean = sum(self.vals) / n
        return math.sqrt(sum((x - mean) ** 2 for x in self.vals) / (n - 1))


# ── Database helpers ─────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite DB, registering custom aggregates."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.create_aggregate("STDEV", 1, _StdDev)
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            date    TEXT NOT NULL,
            ticker  TEXT NOT NULL,
            close   REAL NOT NULL,
            volume  REAL,
            PRIMARY KEY (date, ticker)
        );
        CREATE TABLE IF NOT EXISTS returns (
            date    TEXT NOT NULL,
            ticker  TEXT NOT NULL,
            log_ret REAL NOT NULL,
            PRIMARY KEY (date, ticker)
        );
        CREATE INDEX IF NOT EXISTS idx_prices_ticker  ON prices  (ticker);
        CREATE INDEX IF NOT EXISTS idx_prices_date    ON prices  (date);
        CREATE INDEX IF NOT EXISTS idx_returns_ticker ON returns (ticker);
        CREATE INDEX IF NOT EXISTS idx_returns_date   ON returns (date);
    """)
    conn.commit()
    log.info("Tables ready.")


# ── Synthetic price generator ─────────────────────────────────────────────────

# Calibrated parameters approximate real equity behaviour (annualised).
SYNTHETIC_PARAMS = {
    #          mu      sigma   S0
    "AAPL": (0.22,   0.28,   150.0),
    "JPM":  (0.14,   0.24,    95.0),
    "XOM":  (0.10,   0.26,    70.0),
    "JNJ":  (0.09,   0.18,   130.0),
    "SPY":  (0.12,   0.18,   270.0),
}

# Typical cross-sector correlation matrix (order = SYNTHETIC_PARAMS keys)
_CORR = np.array([
    #AAPL  JPM   XOM   JNJ   SPY
    [1.00, 0.45, 0.25, 0.30, 0.65],
    [0.45, 1.00, 0.40, 0.30, 0.70],
    [0.25, 0.40, 1.00, 0.20, 0.55],
    [0.30, 0.30, 0.20, 1.00, 0.50],
    [0.65, 0.70, 0.55, 0.50, 1.00],
])
_ALL = list(SYNTHETIC_PARAMS.keys())


def generate_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate correlated GBM price paths.

    Uses Geometric Brownian Motion with Cholesky-decomposed correlated shocks:
        ln(S_t / S_{t-1}) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_corr
    where Z_corr ~ Cholesky(Corr) @ N(0,1).

    This is the same process that underlies Black-Scholes and many VaR models,
    making the generated data ideal for demonstrating risk methodology.
    """
    rng = np.random.default_rng(seed)
    trading_days = pd.bdate_range(start=start, end=end)
    n = len(trading_days)
    dt = 1 / 252

    mus    = np.array([SYNTHETIC_PARAMS[t][0] for t in tickers])
    sigmas = np.array([SYNTHETIC_PARAMS[t][1] for t in tickers])
    S0s    = np.array([SYNTHETIC_PARAMS[t][2] for t in tickers])

    idx      = [_ALL.index(t) for t in tickers]
    corr_sub = _CORR[np.ix_(idx, idx)]
    L        = np.linalg.cholesky(corr_sub)          # Cholesky factor

    Z      = rng.standard_normal((n, len(tickers)))
    Z_corr = Z @ L.T

    drift      = (mus - 0.5 * sigmas ** 2) * dt
    shock      = sigmas * np.sqrt(dt) * Z_corr
    log_returns = drift + shock                       # shape (n, k)

    prices = S0s * np.exp(np.cumsum(log_returns, axis=0))

    base_vol = np.array([50e6, 20e6, 25e6, 15e6, 80e6])
    volumes  = rng.lognormal(
        mean=np.log(base_vol[idx]),
        sigma=0.3,
        size=(n, len(tickers)),
    )

    rows = []
    for i, date in enumerate(trading_days):
        for j, ticker in enumerate(tickers):
            rows.append({
                "date":   date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "close":  round(float(prices[i, j]), 4),
                "volume": round(float(volumes[i, j]), 0),
            })

    df = pd.DataFrame(rows)
    log.info(
        "Generated %d synthetic price records (%d tickers × %d days).",
        len(df), len(tickers), n,
    )
    return df


# ── Fetch prices (live or synthetic) ─────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Try yfinance; fall back to synthetic generator if unavailable."""
    try:
        import yfinance as yf
        log.info("Attempting yfinance download…")
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("yfinance returned an empty DataFrame.")

        close  = raw["Close"].ffill(limit=2).dropna(how="all")
        volume = raw["Volume"]

        close_long  = close.reset_index().melt(
            id_vars="Date", var_name="ticker", value_name="close")
        volume_long = volume.reset_index().melt(
            id_vars="Date", var_name="ticker", value_name="volume")

        df = close_long.merge(volume_long, on=["Date", "ticker"])
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.dropna(subset=["close"], inplace=True)
        log.info("yfinance: fetched %d records.", len(df))
        return df[["date", "ticker", "close", "volume"]]

    except Exception as exc:
        log.warning("yfinance unavailable (%s). Using synthetic data.", exc)
        return generate_synthetic_prices(tickers, start, end)


# ── Compute log returns ───────────────────────────────────────────────────────

def compute_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log returns: r_t = ln(P_t / P_{t-1}).

    Preferred in risk modelling because they are:
      - Time-additive (multi-period return = sum of daily log returns)
      - More normally distributed than simple returns
      - Consistent with GBM / Black-Scholes assumptions
    """
    pivot = (
        prices_df
        .pivot(index="date", columns="ticker", values="close")
        .sort_index()
    )
    log_ret = np.log(pivot / pivot.shift(1)).dropna(how="all")

    long = (
        log_ret.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="log_ret")
        .dropna(subset=["log_ret"])
    )
    log.info("Computed %d return records.", len(long))
    return long[["date", "ticker", "log_ret"]]


# ── Write to DB ───────────────────────────────────────────────────────────────

def upsert_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    conn.execute(f"DELETE FROM {table}")
    df.to_sql(table, conn, if_exists="append", index=False,
              method="multi", chunksize=1000)
    conn.commit()
    log.info("%s: %d rows saved.", table, len(df))


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(conn: sqlite3.Connection) -> None:
    log.info("─── Validation ─────────────────────────────────")

    for tbl in ("prices", "returns"):
        (n,) = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
        log.info("  %-12s rows: %d", tbl, n)

    rows = conn.execute("""
        SELECT ticker, MIN(date), MAX(date), COUNT(*)
        FROM prices GROUP BY ticker ORDER BY ticker
    """).fetchall()
    log.info("  Ticker coverage:")
    for ticker, first, last, n in rows:
        log.info("    %-6s  %s → %s  (%d trading days)", ticker, first, last, n)

    rows = conn.execute("""
        SELECT ticker,
               ROUND(AVG(log_ret)*100, 4)   AS mean_pct,
               ROUND(STDEV(log_ret)*100, 4) AS std_pct,
               ROUND(MIN(log_ret)*100, 2)   AS min_pct,
               ROUND(MAX(log_ret)*100, 2)   AS max_pct
        FROM returns GROUP BY ticker ORDER BY ticker
    """).fetchall()
    log.info("  Return stats (daily log %%):  mean / std / min / max")
    for ticker, mean, std, mn, mx in rows:
        log.info(
            "    %-6s  %+.4f / %.4f / %+.2f / %+.2f",
            ticker, mean, std, mn, mx,
        )
    log.info("────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(tickers=None, start=START_DATE, end=END_DATE) -> sqlite3.Connection:
    """Full ingestion pipeline. Returns an open DB connection for downstream use."""
    if tickers is None:
        tickers = TICKERS
    conn       = get_connection()
    create_tables(conn)
    prices_df  = fetch_prices(tickers, start, end)
    returns_df = compute_log_returns(prices_df)
    upsert_df(conn, prices_df,  "prices")
    upsert_df(conn, returns_df, "returns")
    run_validation(conn)
    log.info("Module 1 complete.  DB: %s", DB_PATH)
    return conn


if __name__ == "__main__":
    run()
