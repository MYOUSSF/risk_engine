# risk_engine/modules/var_engine.py
"""
Module 3 — Value at Risk Engine
================================
Computes daily rolling VaR using three standard methodologies:

  1. Historical Simulation  — non-parametric; uses empirical return distribution
  2. Parametric (Variance-Covariance) — assumes normally distributed returns
  3. Monte Carlo Simulation  — simulates future paths via correlated GBM

For each method, VaR is computed at both 95% and 99% confidence levels.
Conditional VaR (CVaR / Expected Shortfall) is also computed — the average
loss in the worst (1-confidence)% of days, which Basel III requires alongside VaR.

Results are stored in the `var_results` table and a daily summary is written
to `var_summary` for easy querying.

Usage:
    python -m risk_engine.modules.var_engine
"""

import sqlite3
import logging
import math
import os
import sys
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from risk_engine.config import (
    DB_PATH, VAR_CONFIDENCE_LEVELS, VAR_WINDOW,
    MONTE_CARLO_SIMS, PORTFOLIO_VALUE, EWMA_LAMBDA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

Method = Literal["historical", "parametric", "monte_carlo"]


# ── DB helpers ────────────────────────────────────────────────────────────────

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


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS var_results (
            date        TEXT    NOT NULL,
            method      TEXT    NOT NULL,   -- historical / parametric / monte_carlo
            confidence  REAL    NOT NULL,   -- 0.95 or 0.99
            var_pct     REAL    NOT NULL,   -- VaR as % of portfolio value
            var_dollar  REAL    NOT NULL,   -- VaR in USD
            cvar_pct    REAL    NOT NULL,   -- CVaR (Expected Shortfall) as %
            cvar_dollar REAL    NOT NULL,   -- CVaR in USD
            PRIMARY KEY (date, method, confidence)
        );

        CREATE TABLE IF NOT EXISTS var_summary (
            date            TEXT PRIMARY KEY,
            port_value      REAL NOT NULL,
            hist_var95      REAL NOT NULL,
            hist_var99      REAL NOT NULL,
            param_var95     REAL NOT NULL,
            param_var99     REAL NOT NULL,
            mc_var95        REAL NOT NULL,
            mc_var99        REAL NOT NULL,
            hist_cvar95     REAL NOT NULL,
            hist_cvar99     REAL NOT NULL,
            ewma_var95      REAL,
            ewma_var99      REAL
        );

        CREATE INDEX IF NOT EXISTS idx_var_date   ON var_results (date);
        CREATE INDEX IF NOT EXISTS idx_var_method ON var_results (method);
    """)
    conn.commit()
    # Migration: add EWMA columns to existing databases that pre-date this method.
    for col in ("ewma_var95", "ewma_var99"):
        try:
            conn.execute(f"ALTER TABLE var_summary ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    log.info("VaR tables ready.")


# ── Core VaR calculators ──────────────────────────────────────────────────────

def historical_var(
    returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
) -> dict:
    """
    Historical Simulation VaR.

    Sort the last `window` daily returns and read off the (1-confidence)
    percentile. No distributional assumption — uses the empirical distribution.

    Pros: captures fat tails, skewness, and non-linearity.
    Cons: fully backward-looking; slow to adapt to regime changes.
    """
    sorted_rets = np.sort(returns)          # ascending: worst losses first
    cutoff_idx  = int(len(sorted_rets) * (1 - confidence))
    cutoff_idx  = max(cutoff_idx, 1)        # ensure at least one observation

    var_pct  = abs(sorted_rets[cutoff_idx - 1])
    # CVaR = mean of losses beyond VaR threshold (Expected Shortfall)
    cvar_pct = abs(np.mean(sorted_rets[:cutoff_idx]))

    return {
        "var_pct":    var_pct,
        "var_dollar": var_pct * portfolio_value,
        "cvar_pct":   cvar_pct,
        "cvar_dollar": cvar_pct * portfolio_value,
    }


def parametric_var(
    returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
) -> dict:
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns are normally distributed:
        VaR = μ - z_α × σ

    where z_α is the standard normal quantile at confidence level α.

    Pros: simple, fast, analytically tractable.
    Cons: underestimates tail risk when returns are fat-tailed (leptokurtic).
    The 2008 crisis showed this limitation clearly.
    """
    mu    = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z     = stats.norm.ppf(1 - confidence)   # negative (left tail)

    var_pct  = abs(mu + z * sigma)
    # For normal distribution, CVaR has a closed form:
    #   CVaR = -μ + σ × φ(z_α) / (1 - α)
    cvar_pct = abs(-mu + sigma * stats.norm.pdf(z) / (1 - confidence))

    return {
        "var_pct":     var_pct,
        "var_dollar":  var_pct * portfolio_value,
        "cvar_pct":    cvar_pct,
        "cvar_dollar": cvar_pct * portfolio_value,
    }


def monte_carlo_var(
    returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
    n_sims: int,
    rng: np.random.Generator,
) -> dict:
    """
    Monte Carlo VaR.

    Fits a normal distribution to the historical returns in the window,
    then draws n_sims one-day return scenarios. VaR and CVaR are read
    from the simulated distribution's empirical percentiles.

    Pros: flexible — can incorporate non-normal distributions, fat tails,
          or multi-asset correlations (extended version).
    Cons: computationally expensive; results vary slightly across runs
          (use a fixed seed for reproducibility).

    Note: this single-asset MC version uses the portfolio's univariate
    return series. The multi-asset version in Module 5 (stress testing)
    uses the full covariance matrix.
    """
    mu    = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    simulated = rng.normal(loc=mu, scale=sigma, size=n_sims)
    simulated.sort()

    cutoff  = int(n_sims * (1 - confidence))
    cutoff  = max(cutoff, 1)

    var_pct  = abs(simulated[cutoff - 1])
    cvar_pct = abs(np.mean(simulated[:cutoff]))

    return {
        "var_pct":     var_pct,
        "var_dollar":  var_pct * portfolio_value,
        "cvar_pct":    cvar_pct,
        "cvar_dollar": cvar_pct * portfolio_value,
    }


def ewma_var(
    returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
    lambda_: float = EWMA_LAMBDA,
) -> dict:
    """
    EWMA (Exponentially Weighted Moving Average) VaR — RiskMetrics methodology.

    Computes volatility using exponentially decaying weights on squared returns:
        σ²_t = Σ w_i · r²_i   where w_i ∝ λ^(n−1−i), normalised to Σw_i = 1

    With λ = 0.94 (J.P. Morgan RiskMetrics standard), recent observations receive
    far more weight than older ones, so the model adapts quickly after a volatility
    spike — the key advantage over the equal-weight historical and parametric methods.

    Zero-mean assumption (RiskMetrics standard): daily return mean is treated as 0
    for the variance calculation, which is conservative for short horizons.

    CVaR uses the same closed-form as parametric VaR (normal distribution assumed):
        CVaR = σ_ewma · φ(z_α) / (1 − α)

    Pros: reacts faster to volatility regime changes than equal-weight methods.
    Cons: still assumes normally distributed returns; no fat-tail correction.
    """
    n = len(returns)
    # returns[0] = oldest, returns[-1] = most recent.
    # Assign weight λ^(n−1−i) so the most recent observation gets λ^0 = 1 (pre-norm).
    weights = lambda_ ** np.arange(n - 1, -1, -1)
    weights /= weights.sum()                          # normalise for finite window

    variance = np.dot(weights, returns ** 2)          # zero-mean EWMA variance
    sigma    = np.sqrt(variance)

    z        = stats.norm.ppf(1 - confidence)         # negative left-tail quantile
    var_pct  = abs(z * sigma)
    cvar_pct = sigma * stats.norm.pdf(z) / (1 - confidence)

    return {
        "var_pct":     var_pct,
        "var_dollar":  var_pct * portfolio_value,
        "cvar_pct":    cvar_pct,
        "cvar_dollar": cvar_pct * portfolio_value,
    }


# ── Rolling VaR engine ────────────────────────────────────────────────────────

def compute_rolling_var(
    conn: sqlite3.Connection,
    window: int = VAR_WINDOW,
    confidence_levels: list = VAR_CONFIDENCE_LEVELS,
    n_sims: int = MONTE_CARLO_SIMS,
    ewma_lambda: float = EWMA_LAMBDA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling VaR for every date that has a full `window` of history.

    For each date t:
      - Use returns from [t-window, t-1] as the estimation window
      - Report VaR as the 1-day loss not exceeded at the given confidence level

    This rolling approach is what risk systems use in practice — the model
    is continuously re-estimated as new data arrives.
    """
    port_df = pd.read_sql(
        "SELECT date, portfolio_ret, portfolio_value FROM portfolio_returns ORDER BY date",
        conn,
    )
    returns = port_df["portfolio_ret"].values
    dates   = port_df["date"].values
    values  = port_df["portfolio_value"].values

    rng = np.random.default_rng(42)          # fixed seed → reproducible MC

    var_rows     = []
    summary_rows = []

    n = len(returns)
    log.info("Computing rolling VaR over %d dates (window=%d)…", n - window, window)

    for t in range(window, n):
        date            = dates[t]
        portfolio_value = values[t]
        window_returns  = returns[t - window : t]   # last `window` days

        summary = {"date": date, "port_value": portfolio_value}

        for conf in confidence_levels:
            conf_label = int(conf * 100)   # 95 or 99

            # ── Historical ──────────────────────────────────────────────────
            h = historical_var(window_returns, conf, portfolio_value)
            var_rows.append({
                "date": date, "method": "historical",
                "confidence": conf, **h,
            })
            summary[f"hist_var{conf_label}"]  = h["var_dollar"]
            summary[f"hist_cvar{conf_label}"] = h["cvar_dollar"]

            # ── Parametric ──────────────────────────────────────────────────
            p = parametric_var(window_returns, conf, portfolio_value)
            var_rows.append({
                "date": date, "method": "parametric",
                "confidence": conf, **p,
            })
            summary[f"param_var{conf_label}"] = p["var_dollar"]

            # ── Monte Carlo ─────────────────────────────────────────────────
            mc = monte_carlo_var(window_returns, conf, portfolio_value, n_sims, rng)
            var_rows.append({
                "date": date, "method": "monte_carlo",
                "confidence": conf, **mc,
            })
            summary[f"mc_var{conf_label}"] = mc["var_dollar"]

            # ── EWMA ─────────────────────────────────────────────────────────
            ew = ewma_var(window_returns, conf, portfolio_value, ewma_lambda)
            var_rows.append({
                "date": date, "method": "ewma",
                "confidence": conf, **ew,
            })
            summary[f"ewma_var{conf_label}"] = ew["var_dollar"]

        summary_rows.append(summary)

    var_df     = pd.DataFrame(var_rows)
    summary_df = pd.DataFrame(summary_rows)

    log.info("VaR computed: %d records across %d dates.", len(var_df), len(summary_rows))
    return var_df, summary_df


# ── Write + validate ──────────────────────────────────────────────────────────

def save_results(
    conn: sqlite3.Connection,
    var_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    for tbl, df in [("var_results", var_df), ("var_summary", summary_df)]:
        conn.execute(f"DELETE FROM {tbl}")
        df.to_sql(tbl, conn, if_exists="append", index=False,
                  method="multi", chunksize=2000)
    conn.commit()
    log.info("var_results: %d rows  |  var_summary: %d rows",
             len(var_df), len(summary_df))


def run_validation(conn: sqlite3.Connection) -> None:
    log.info("─── VaR validation ──────────────────────────────")

    # Latest date — snapshot of all three methods side by side
    latest = conn.execute(
        "SELECT MAX(date) FROM var_summary"
    ).fetchone()[0]

    row = conn.execute(
        "SELECT * FROM var_summary WHERE date = ?", (latest,)
    ).fetchone()
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM var_summary LIMIT 0"
    ).description]
    snap = dict(zip(cols, row))

    log.info("  Latest date: %s  |  Portfolio value: $%.0f", latest, snap["port_value"])
    log.info("  %-14s  %10s  %10s", "Method", "VaR 95%", "VaR 99%")
    log.info("  " + "-" * 38)
    for label, k95, k99 in [
        ("Historical",   "hist_var95",  "hist_var99"),
        ("Parametric",   "param_var95", "param_var99"),
        ("Monte Carlo",  "mc_var95",    "mc_var99"),
        ("EWMA",         "ewma_var95",  "ewma_var99"),
    ]:
        log.info("  %-14s  $%8.0f  $%8.0f", label, snap[k95], snap[k99])

    log.info("  " + "-" * 38)

    # Average VaR over the full history
    avgs = conn.execute("""
        SELECT method,
               ROUND(AVG(var_dollar), 0) AS avg_var95,
               ROUND(AVG(cvar_dollar), 0) AS avg_cvar95
        FROM var_results
        WHERE confidence = 0.95
        GROUP BY method ORDER BY method
    """).fetchall()
    log.info("  Average 95%% VaR and CVaR over history:")
    for method, avg_var, avg_cvar in avgs:
        log.info("    %-14s  VaR=$%7.0f  CVaR=$%7.0f", method, avg_var, avg_cvar)

    # VaR as % of portfolio value
    row = conn.execute("""
        SELECT ROUND(AVG(var_pct)*100, 3)
        FROM var_results
        WHERE method='historical' AND confidence=0.99
    """).fetchone()[0]
    log.info("  Avg 99%% Historical VaR: %.3f%% of portfolio", row)
    log.info("────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> sqlite3.Connection:
    conn = get_connection()
    create_tables(conn)

    var_df, summary_df = compute_rolling_var(conn)
    save_results(conn, var_df, summary_df)
    run_validation(conn)

    log.info("Module 3 complete.")
    return conn


if __name__ == "__main__":
    run()
