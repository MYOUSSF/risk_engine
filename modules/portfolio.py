# risk_engine/modules/portfolio.py
"""
Module 2 — Portfolio Construction
==================================
Defines the portfolio positions, computes daily P&L at both the position
and portfolio level, and stores results in the SQLite database.

Three new tables are created:
  positions        : static position book (shares held per ticker)
  position_pnl     : daily mark-to-market P&L per position
  portfolio_returns : daily portfolio-level log return and P&L

Key concepts demonstrated:
  - Mark-to-market valuation
  - Dollar P&L vs percentage return
  - Portfolio-level aggregation via weighted returns
  - Risk decomposition: which positions contribute most to portfolio vol?

Usage:
    python -m risk_engine.modules.portfolio
"""

import sqlite3
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from risk_engine.config import (
    TICKERS, PORTFOLIO_WEIGHTS, PORTFOLIO_VALUE,
    START_DATE, END_DATE, DB_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Custom SQLite aggregate ───────────────────────────────────────────────────

class _StdDev:
    def __init__(self): self.vals = []
    def step(self, v):
        if v is not None: self.vals.append(v)
    def finalize(self):
        n = len(self.vals)
        if n < 2: return 0.0
        m = sum(self.vals) / n
        return math.sqrt(sum((x - m) ** 2 for x in self.vals) / (n - 1))


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.create_aggregate("STDEV", 1, _StdDev)
    return conn


# ── Table setup ───────────────────────────────────────────────────────────────

def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            ticker          TEXT PRIMARY KEY,
            shares          REAL NOT NULL,
            entry_price     REAL NOT NULL,
            market_value    REAL NOT NULL,   -- shares × entry_price
            weight          REAL NOT NULL    -- target portfolio weight
        );

        CREATE TABLE IF NOT EXISTS position_pnl (
            date            TEXT NOT NULL,
            ticker          TEXT NOT NULL,
            close           REAL NOT NULL,
            market_value    REAL NOT NULL,   -- shares × close
            daily_pnl       REAL NOT NULL,   -- MtM change vs previous day
            cum_pnl         REAL NOT NULL,   -- cumulative P&L since inception
            daily_ret       REAL NOT NULL,   -- log return for this position
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS portfolio_returns (
            date            TEXT PRIMARY KEY,
            portfolio_ret   REAL NOT NULL,   -- weighted sum of position log returns
            portfolio_pnl   REAL NOT NULL,   -- dollar P&L for the day
            cum_pnl         REAL NOT NULL,   -- cumulative portfolio P&L
            portfolio_value REAL NOT NULL    -- total MtM value
        );

        CREATE INDEX IF NOT EXISTS idx_pospnl_ticker ON position_pnl (ticker);
        CREATE INDEX IF NOT EXISTS idx_pospnl_date   ON position_pnl (date);
    """)
    conn.commit()
    log.info("Portfolio tables ready.")


# ── Position construction ─────────────────────────────────────────────────────

def build_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Construct the position book.

    Each ticker is allocated a dollar amount = PORTFOLIO_VALUE × weight.
    Shares held = dollar allocation / entry price (first available price).

    This mirrors how a portfolio manager would size a new book on day 1.
    """
    # Use first date's prices as entry prices
    first_date = conn.execute(
        "SELECT MIN(date) FROM prices"
    ).fetchone()[0]

    prices = pd.read_sql(
        "SELECT ticker, close AS entry_price FROM prices WHERE date = ?",
        conn, params=(first_date,),
    )

    rows = []
    for _, row in prices.iterrows():
        ticker = row["ticker"]
        if ticker not in PORTFOLIO_WEIGHTS:
            continue
        weight       = PORTFOLIO_WEIGHTS[ticker]
        dollar_alloc = PORTFOLIO_VALUE * weight
        shares       = dollar_alloc / row["entry_price"]
        rows.append({
            "ticker":       ticker,
            "shares":       round(shares, 4),
            "entry_price":  round(row["entry_price"], 4),
            "market_value": round(dollar_alloc, 2),
            "weight":       weight,
        })

    positions = pd.DataFrame(rows)
    log.info("Built %d positions (entry date: %s, notional: $%.0f)",
             len(positions), first_date, PORTFOLIO_VALUE)
    for _, r in positions.iterrows():
        log.info("  %-6s  %.2f shares @ $%.2f  = $%.0f  (%.0f%%)",
                 r.ticker, r.shares, r.entry_price,
                 r.market_value, r.weight * 100)
    return positions


# ── Daily P&L calculation ─────────────────────────────────────────────────────

def compute_position_pnl(
    conn: sqlite3.Connection,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mark-to-market daily P&L for each position.

    daily_pnl  = shares × (close_t − close_{t-1})
    cum_pnl    = shares × (close_t − entry_price)
    daily_ret  = ln(close_t / close_{t-1})   [from returns table]
    """
    prices = pd.read_sql(
        """
        SELECT p.date, p.ticker, p.close, r.log_ret
        FROM prices p
        JOIN returns r ON p.date = r.date AND p.ticker = r.ticker
        ORDER BY p.ticker, p.date
        """,
        conn,
    )

    pos_map = positions.set_index("ticker")[["shares", "entry_price"]]
    results = []

    for ticker, grp in prices.groupby("ticker"):
        if ticker not in pos_map.index:
            continue
        shares      = pos_map.loc[ticker, "shares"]
        entry_price = pos_map.loc[ticker, "entry_price"]

        grp = grp.sort_values("date").copy()
        grp["prev_close"]   = grp["close"].shift(1)
        grp["daily_pnl"]    = shares * (grp["close"] - grp["prev_close"])
        grp["cum_pnl"]      = shares * (grp["close"] - entry_price)
        grp["market_value"] = shares * grp["close"]
        grp["daily_ret"]    = grp["log_ret"]

        # First row has no prev_close — set daily_pnl = 0
        grp["daily_pnl"] = grp["daily_pnl"].fillna(0.0)

        results.append(grp[["date", "ticker", "close", "market_value",
                             "daily_pnl", "cum_pnl", "daily_ret"]])

    pnl_df = pd.concat(results).sort_values(["date", "ticker"])
    log.info("Computed position P&L: %d records.", len(pnl_df))
    return pnl_df


def compute_portfolio_returns(
    position_pnl: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate position-level P&L to portfolio level.

    portfolio_ret = sum(weight_i × daily_ret_i)
      — uses static entry weights (consistent with a buy-and-hold book)

    portfolio_pnl = sum(daily_pnl_i)
    portfolio_value = sum(market_value_i)
    """
    weights = positions.set_index("ticker")["weight"].to_dict()

    # Weighted return per day
    position_pnl = position_pnl.copy()
    position_pnl["weighted_ret"] = position_pnl.apply(
        lambda r: r["daily_ret"] * weights.get(r["ticker"], 0), axis=1
    )

    daily = (
        position_pnl
        .groupby("date")
        .agg(
            portfolio_ret   = ("weighted_ret",   "sum"),
            portfolio_pnl   = ("daily_pnl",      "sum"),
            portfolio_value = ("market_value",    "sum"),
        )
        .reset_index()
        .sort_values("date")
    )

    daily["cum_pnl"] = daily["portfolio_pnl"].cumsum()
    log.info("Computed portfolio returns: %d trading days.", len(daily))
    return daily[["date", "portfolio_ret", "portfolio_pnl",
                  "cum_pnl", "portfolio_value"]]


# ── Write to DB ───────────────────────────────────────────────────────────────

def save_positions(conn: sqlite3.Connection, positions: pd.DataFrame) -> None:
    conn.execute("DELETE FROM positions")
    positions.to_sql("positions", conn, if_exists="append", index=False)
    conn.commit()
    log.info("positions: saved %d rows.", len(positions))


def save_pnl(conn: sqlite3.Connection, df: pd.DataFrame, table: str) -> None:
    conn.execute(f"DELETE FROM {table}")
    df.to_sql(table, conn, if_exists="append", index=False,
              method="multi", chunksize=1000)
    conn.commit()
    log.info("%s: saved %d rows.", table, len(df))


# ── Risk decomposition ────────────────────────────────────────────────────────

def risk_decomposition(conn: sqlite3.Connection) -> None:
    """
    Print a risk attribution table:
      - Annualised volatility per position
      - Marginal contribution to portfolio volatility
      - % of total portfolio risk

    This mirrors the kind of output a middle office team would review daily.
    """
    log.info("─── Risk decomposition ──────────────────────────")

    ret_wide = pd.read_sql(
        """
        SELECT date, ticker, daily_ret
        FROM position_pnl
        ORDER BY date
        """,
        conn,
    ).pivot(index="date", columns="ticker", values="daily_ret").dropna()

    positions = pd.read_sql("SELECT ticker, weight FROM positions", conn)
    weights   = positions.set_index("ticker")["weight"].reindex(ret_wide.columns).values

    # Covariance matrix (daily), annualised
    cov_daily  = ret_wide.cov().values
    cov_annual = cov_daily * 252

    # Portfolio variance and volatility
    port_var = weights @ cov_annual @ weights
    port_vol = math.sqrt(port_var)

    # Marginal contribution to risk: (Cov × w)_i / port_vol
    mcr = (cov_annual @ weights) / port_vol

    log.info("  %-6s  %8s  %10s  %6s", "Ticker", "Ann.Vol", "MCR", "% Risk")
    log.info("  " + "-" * 40)
    total_pct = 0.0
    for i, ticker in enumerate(ret_wide.columns):
        ann_vol    = math.sqrt(cov_annual[i, i]) * 100
        contribution = weights[i] * mcr[i]
        pct_risk   = (contribution / port_vol) * 100
        total_pct += pct_risk
        log.info("  %-6s  %7.2f%%  %9.4f  %5.1f%%",
                 ticker, ann_vol, contribution, pct_risk)

    log.info("  " + "-" * 40)
    log.info("  %-6s  %7.2f%%  %9s  %5.1f%%",
             "PORT", port_vol * 100, "", total_pct)
    log.info("──────────────────────────────────────────────")


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(conn: sqlite3.Connection) -> None:
    log.info("─── Portfolio validation ────────────────────────")

    (n_days,) = conn.execute(
        "SELECT COUNT(*) FROM portfolio_returns"
    ).fetchone()
    log.info("  Trading days in portfolio: %d", n_days)

    row = conn.execute("""
        SELECT
            MIN(date)                             AS start_date,
            MAX(date)                             AS end_date,
            ROUND(MIN(portfolio_value), 0)        AS min_value,
            ROUND(MAX(portfolio_value), 0)        AS max_value,
            ROUND(SUM(portfolio_pnl), 0)          AS total_pnl,
            ROUND(AVG(portfolio_ret)*100, 4)      AS mean_ret_pct,
            ROUND(STDEV(portfolio_ret)*100, 4)    AS std_ret_pct
        FROM portfolio_returns
    """).fetchone()

    labels = ["Start", "End", "Min value ($)", "Max value ($)",
              "Total P&L ($)", "Mean daily ret (%)", "Daily vol (%)"]
    for label, val in zip(labels, row):
        log.info("  %-22s  %s", label, val)

    # Best / worst days
    best = conn.execute("""
        SELECT date, ROUND(portfolio_pnl, 0), ROUND(portfolio_ret*100, 3)
        FROM portfolio_returns ORDER BY portfolio_pnl DESC LIMIT 1
    """).fetchone()
    worst = conn.execute("""
        SELECT date, ROUND(portfolio_pnl, 0), ROUND(portfolio_ret*100, 3)
        FROM portfolio_returns ORDER BY portfolio_pnl ASC LIMIT 1
    """).fetchone()

    log.info("  Best day   %s  $%.0f  (%.3f%%)", best[0], best[1], best[2])
    log.info("  Worst day  %s  $%.0f  (%.3f%%)", worst[0], worst[1], worst[2])
    log.info("────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> tuple[sqlite3.Connection, pd.DataFrame]:
    """
    Full Module 2 pipeline.
    Returns (conn, positions) for use by downstream modules.
    """
    conn = get_connection()
    create_tables(conn)

    positions    = build_positions(conn)
    position_pnl = compute_position_pnl(conn, positions)
    port_returns = compute_portfolio_returns(position_pnl, positions)

    save_positions(conn, positions)
    save_pnl(conn, position_pnl,  "position_pnl")
    save_pnl(conn, port_returns,  "portfolio_returns")

    run_validation(conn)
    risk_decomposition(conn)

    log.info("Module 2 complete.")
    return conn, positions


if __name__ == "__main__":
    run()
