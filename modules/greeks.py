# risk_engine/modules/greeks.py
"""
Module 7 — Black-Scholes Greeks
================================
Prices a hypothetical at-the-money (ATM) European call option on AAPL using
the Black-Scholes model and computes the five standard option Greeks.

Inputs (pulled from DB + config):
  S  : latest AAPL closing price from the `prices` table
  K  : strike = S (ATM)
  T  : OPTION_EXPIRY_DAYS / 365  (calendar-day convention)
  r  : OPTION_RISK_FREE_RATE (annualised, continuously compounded)
  σ  : AAPL annualised historical vol from last 252 log returns

Greeks computed (all per-share of the underlying):
  Delta Δ — ∂C/∂S    directional sensitivity
  Gamma Γ — ∂²C/∂S²  convexity; rate of change of delta
  Vega  ν — ∂C/∂σ    vol sensitivity    (reported per 1% change in σ)
  Theta Θ — ∂C/∂t    time decay         (reported per calendar day)
  Rho   ρ — ∂C/∂r    rate sensitivity   (reported per 1% change in r)

Two tables are written to the DB:
  option_snapshot  — one-row ATM snapshot for the latest date
  option_profile   — greeks across a 0.6K–1.4K spot range (for the dashboard
                     delta S-curve and option-price chart)

Usage:
    python -m risk_engine.modules.greeks
"""

import math
import logging
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from risk_engine.config import (
    DB_PATH,
    OPTION_TICKER,
    OPTION_RISK_FREE_RATE,
    OPTION_EXPIRY_DAYS,
    OPTION_CONTRACTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS option_snapshot (
            computed_at    TEXT PRIMARY KEY,
            ticker         TEXT    NOT NULL,
            spot_price     REAL    NOT NULL,   -- S: latest close
            strike         REAL    NOT NULL,   -- K: ATM = S
            t_years        REAL    NOT NULL,   -- T: years to expiry
            risk_free_rate REAL    NOT NULL,   -- r: annualised continuous
            implied_vol    REAL    NOT NULL,   -- σ: 252-day historical vol
            option_price   REAL    NOT NULL,   -- Black-Scholes call price
            delta          REAL    NOT NULL,   -- N(d1)
            gamma          REAL    NOT NULL,   -- φ(d1) / (S σ √T)
            vega           REAL    NOT NULL,   -- per 1% change in σ
            theta          REAL    NOT NULL,   -- per calendar day
            rho            REAL    NOT NULL,   -- per 1% change in r
            n_contracts    INTEGER NOT NULL    -- position size
        );

        CREATE TABLE IF NOT EXISTS option_profile (
            spot_price   REAL PRIMARY KEY,
            option_price REAL NOT NULL,
            delta        REAL NOT NULL,
            gamma        REAL NOT NULL,
            vega         REAL NOT NULL,
            theta        REAL NOT NULL,
            rho          REAL NOT NULL
        );
    """)
    conn.commit()
    log.info("Greeks tables ready.")


# ── Black-Scholes core ────────────────────────────────────────────────────────

def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Black-Scholes d1 and d2 intermediates."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float
) -> dict:
    """
    Compute Black-Scholes call price and all five Greeks.

    Vega  is scaled to per 1% change in σ  (raw ÷ 100).
    Rho   is scaled to per 1% change in r  (raw ÷ 100).
    Theta is per calendar day              (raw ÷ 365).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    phi_d1 = stats.norm.pdf(d1)
    N_d1   = stats.norm.cdf(d1)
    N_d2   = stats.norm.cdf(d2)
    disc   = math.exp(-r * T)

    price = S * N_d1 - K * disc * N_d2
    delta = N_d1
    gamma = phi_d1 / (S * sigma * math.sqrt(T))
    vega  = S * phi_d1 * math.sqrt(T) / 100
    theta = (-(S * phi_d1 * sigma) / (2 * math.sqrt(T))
             - r * K * disc * N_d2) / 365
    rho   = K * T * disc * N_d2 / 100

    return {
        "option_price": price,
        "delta": delta,
        "gamma": gamma,
        "vega":  vega,
        "theta": theta,
        "rho":   rho,
    }


# ── Market inputs from DB ─────────────────────────────────────────────────────

def fetch_market_inputs(
    conn: sqlite3.Connection,
) -> tuple[float, float, str]:
    """
    Pull the latest AAPL spot price and its 252-day annualised vol from the DB.

    Returns: (spot_price, ann_vol, latest_date)
    """
    row = conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
        (OPTION_TICKER,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"No price data for {OPTION_TICKER}. Run data_ingestion first."
        )
    latest_date, spot = row

    ret_rows = conn.execute(
        "SELECT log_ret FROM returns WHERE ticker = ? ORDER BY date DESC LIMIT 252",
        (OPTION_TICKER,),
    ).fetchall()
    if not ret_rows:
        raise RuntimeError(f"No return data for {OPTION_TICKER}.")

    daily_rets = np.array([r[0] for r in ret_rows], dtype=float)
    ann_vol = daily_rets.std(ddof=1) * math.sqrt(252)

    log.info("%-6s  spot=$%.2f  252-day ann. vol=%.2f%%  date=%s",
             OPTION_TICKER, spot, ann_vol * 100, latest_date)
    return spot, ann_vol, latest_date


# ── Profile computation ───────────────────────────────────────────────────────

def compute_profile(
    K: float, T: float, r: float, sigma: float, n_points: int = 200
) -> pd.DataFrame:
    """
    Compute Greeks at 200 spot prices spanning 0.6K to 1.4K.
    The strike K, expiry T, rate r, and vol sigma are held fixed —
    only spot varies. Used for the delta S-curve and option-price charts.
    """
    spots = np.linspace(0.60 * K, 1.40 * K, n_points)
    rows = [{"spot_price": float(S), **bs_greeks(float(S), K, T, r, sigma)}
            for S in spots]
    return pd.DataFrame(rows)


# ── Save + validate ───────────────────────────────────────────────────────────

def save_results(
    conn: sqlite3.Connection,
    snapshot: dict,
    profile_df: pd.DataFrame,
) -> None:
    conn.execute("DELETE FROM option_snapshot")
    conn.execute("DELETE FROM option_profile")

    cols = (
        "computed_at", "ticker", "spot_price", "strike", "t_years",
        "risk_free_rate", "implied_vol", "option_price",
        "delta", "gamma", "vega", "theta", "rho", "n_contracts",
    )
    conn.execute(
        f"INSERT INTO option_snapshot VALUES ({', '.join(':' + c for c in cols)})",
        snapshot,
    )
    profile_df.to_sql("option_profile", conn, if_exists="append",
                      index=False, method="multi")
    conn.commit()
    log.info("option_snapshot written  |  option_profile: %d rows", len(profile_df))


def run_validation(conn: sqlite3.Connection) -> None:
    row  = conn.execute("SELECT * FROM option_snapshot").fetchone()
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM option_snapshot LIMIT 0"
    ).description]
    s = dict(zip(cols, row))

    pos_value = s["option_price"] * 100 * s["n_contracts"]

    log.info("─── Black-Scholes Greeks  (%s ATM Call) ─────────────", s["ticker"])
    log.info("  Spot $%-8.2f │ Strike $%-8.2f │ T = %d days │ σ = %.2f%% │ r = %.0f%%",
             s["spot_price"], s["strike"],
             round(s["t_years"] * 365),
             s["implied_vol"] * 100,
             s["risk_free_rate"] * 100)
    log.info("  Call price: $%.4f / share  ·  Position value: $%.2f  (%d × 100 shares)",
             s["option_price"], pos_value, s["n_contracts"])
    log.info("")
    log.info("  %-8s  %-6s  %10s  %s", "Greek", "Symbol", "Value", "Interpretation (per share)")
    log.info("  %s", "-" * 70)
    log.info("  %-8s  %-6s  %+10.4f  $%+.4f per $1 move in spot",
             "Delta", "Δ", s["delta"], s["delta"])
    log.info("  %-8s  %-6s  %+10.6f  Δ changes by %.6f per $1 move in spot",
             "Gamma", "Γ", s["gamma"], s["gamma"])
    log.info("  %-8s  %-6s  %+10.4f  $%+.4f per 1%% rise in vol",
             "Vega", "ν", s["vega"], s["vega"])
    log.info("  %-8s  %-6s  %+10.4f  $%+.4f per calendar day (time decay)",
             "Theta", "Θ", s["theta"], s["theta"])
    log.info("  %-8s  %-6s  %+10.4f  $%+.4f per 1%% rise in risk-free rate",
             "Rho", "ρ", s["rho"], s["rho"])
    log.info("  %s", "-" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> sqlite3.Connection:
    conn = get_connection()
    create_tables(conn)

    spot, ann_vol, latest_date = fetch_market_inputs(conn)

    K = spot                          # ATM: strike equals spot
    T = OPTION_EXPIRY_DAYS / 365      # calendar-day convention
    r = OPTION_RISK_FREE_RATE

    greeks = bs_greeks(spot, K, T, r, ann_vol)

    snapshot = {
        "computed_at":    latest_date,
        "ticker":         OPTION_TICKER,
        "spot_price":     spot,
        "strike":         K,
        "t_years":        T,
        "risk_free_rate": r,
        "implied_vol":    ann_vol,
        "n_contracts":    OPTION_CONTRACTS,
        **greeks,
    }

    profile_df = compute_profile(K, T, r, ann_vol)
    save_results(conn, snapshot, profile_df)
    run_validation(conn)

    log.info("Module 7 complete.")
    return conn


if __name__ == "__main__":
    run()
