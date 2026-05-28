# risk_engine/modules/backtesting.py
"""
Module 4 — VaR Backtesting
============================
Tests whether the VaR model is well-calibrated by comparing predicted VaR
against realised daily P&L over rolling 250-day windows.

Implements the Basel II / III backtesting framework:

  1. Exception counting  — count days where actual loss > VaR
  2. Traffic light test  — Basel zones based on exception count in 250 days
       Green  : 0–4 exceptions   → model acceptable
       Amber  : 5–9 exceptions   → supervisory concern, capital add-on possible
       Red    : 10+ exceptions   → model presumed inadequate
  3. Kupiec POF test     — likelihood ratio test: are exceptions occurring at
                           the right frequency? (tests unconditional coverage)
  4. Christoffersen test — are exceptions independent? Clustered exceptions
                           (e.g. all during one crash) suggest model is too slow
                           to adapt. (tests conditional coverage)

All results are stored in `backtest_results` and `backtest_summary`.

Usage:
    python -m risk_engine.modules.backtesting
"""

import sqlite3
import logging
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from risk_engine.config import DB_PATH, VAR_CONFIDENCE_LEVELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACKTEST_WINDOW = 250   # Basel standard: 1 trading year


# ── Basel traffic light thresholds (250-day window, 99% VaR) ─────────────────
# Source: Basel Committee on Banking Supervision (BCBS) 1996, updated 2019
TRAFFIC_LIGHT = {
    "green":  (0, 4),    # 0–4 exceptions: model OK
    "amber":  (5, 9),    # 5–9 exceptions: under scrutiny
    "red":    (10, 999), # 10+: model rejected
}


def _traffic_light_zone(n_exceptions: int) -> str:
    for zone, (lo, hi) in TRAFFIC_LIGHT.items():
        if lo <= n_exceptions <= hi:
            return zone
    return "red"


# ── Statistical tests ─────────────────────────────────────────────────────────

def kupiec_pof_test(n_exceptions: int, n_obs: int, confidence: float) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    H0: p_actual = p_expected   (model is correctly calibrated)
    H1: p_actual ≠ p_expected

    Test statistic follows chi-squared(1) under H0.

    A p-value > 0.05 means we cannot reject H0 — the model is well-calibrated.
    A p-value < 0.05 means the exception frequency is statistically too high
    (or too low), suggesting the model is mis-calibrated.
    """
    p_expected = 1 - confidence
    p_actual   = n_exceptions / n_obs if n_obs > 0 else 0

    if n_exceptions == 0:
        # Log-likelihood undefined at 0 exceptions; return borderline pass
        return {"statistic": 0.0, "p_value": 1.0, "reject_h0": False,
                "p_expected": p_expected, "p_actual": 0.0}

    if n_exceptions == n_obs:
        return {"statistic": np.inf, "p_value": 0.0, "reject_h0": True,
                "p_expected": p_expected, "p_actual": 1.0}

    # Log-likelihood ratio statistic
    lr = -2 * (
        n_exceptions * math.log(p_expected)
        + (n_obs - n_exceptions) * math.log(1 - p_expected)
        - n_exceptions * math.log(p_actual)
        - (n_obs - n_exceptions) * math.log(1 - p_actual)
    )

    p_value   = 1 - stats.chi2.cdf(lr, df=1)
    reject_h0 = p_value < 0.05

    return {
        "statistic": round(lr, 4),
        "p_value":   round(p_value, 4),
        "reject_h0": reject_h0,
        "p_expected": round(p_expected, 4),
        "p_actual":   round(p_actual, 4),
    }


def christoffersen_test(exceptions: np.ndarray) -> dict:
    """
    Christoffersen (1998) Interval Forecast test.

    Tests whether exceptions are independently distributed (no clustering).
    A model that only fails during crises produces clustered exceptions —
    it's not truly capturing daily risk.

    H0: exceptions are independent (no clustering)
    H1: exceptions are serially dependent

    Computes transition probabilities:
      pi01 = P(exception today | no exception yesterday)
      pi11 = P(exception today | exception yesterday)

    Under H0: pi01 = pi11 (today's exception doesn't depend on yesterday's).
    """
    n = len(exceptions)
    if n < 2:
        return {"statistic": np.nan, "p_value": np.nan,
                "reject_h0": False, "pi01": np.nan, "pi11": np.nan}

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = exceptions[i-1], exceptions[i]
        if   prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Avoid log(0)
    eps = 1e-10
    pi01c, pi11c, pic = max(pi01, eps), max(pi11, eps), max(pi, eps)
    pi01c = min(pi01c, 1-eps); pi11c = min(pi11c, 1-eps); pic = min(pic, 1-eps)

    ll_restricted = (
        (n00 + n10) * math.log(1 - pic)
        + (n01 + n11) * math.log(pic)
    )
    ll_unrestricted = (
        n00 * math.log(1 - pi01c) + n01 * math.log(pi01c)
        + n10 * math.log(1 - pi11c) + n11 * math.log(pi11c)
    )

    lr      = -2 * (ll_restricted - ll_unrestricted)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "statistic": round(lr, 4),
        "p_value":   round(p_value, 4),
        "reject_h0": p_value < 0.05,
        "pi01":      round(pi01, 4),
        "pi11":      round(pi11, 4),
    }


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
        CREATE TABLE IF NOT EXISTS backtest_results (
            date            TEXT    NOT NULL,
            method          TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            var_dollar      REAL    NOT NULL,
            actual_pnl      REAL    NOT NULL,
            is_exception    INTEGER NOT NULL,  -- 1 if actual loss > VaR
            exception_label TEXT    NOT NULL,  -- 'breach' or 'ok'
            PRIMARY KEY (date, method, confidence)
        );

        CREATE TABLE IF NOT EXISTS backtest_summary (
            window_end      TEXT    NOT NULL,
            method          TEXT    NOT NULL,
            confidence      REAL    NOT NULL,
            n_obs           INTEGER NOT NULL,
            n_exceptions    INTEGER NOT NULL,
            exception_rate  REAL    NOT NULL,
            zone            TEXT    NOT NULL,  -- green / amber / red
            kupiec_stat     REAL,
            kupiec_pval     REAL,
            kupiec_reject   INTEGER,
            christo_stat    REAL,
            christo_pval    REAL,
            christo_reject  INTEGER,
            pi01            REAL,              -- P(breach | no breach yesterday)
            pi11            REAL,              -- P(breach | breach yesterday)
            PRIMARY KEY (window_end, method, confidence)
        );

        CREATE INDEX IF NOT EXISTS idx_bt_date   ON backtest_results (date);
        CREATE INDEX IF NOT EXISTS idx_bt_method ON backtest_results (method);
    """)
    conn.commit()
    log.info("Backtesting tables ready.")


# ── Core backtesting logic ────────────────────────────────────────────────────

def run_backtests(
    conn: sqlite3.Connection,
    window: int = BACKTEST_WINDOW,
    confidence_levels: list = VAR_CONFIDENCE_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each VaR method and confidence level:
      1. Flag each day as an exception if -actual_pnl > var_dollar
      2. Over the most recent `window` days, compute:
         - Exception count and rate
         - Basel traffic light zone
         - Kupiec POF test result
         - Christoffersen independence test result
    """
    # Join VaR estimates with realised P&L
    df = pd.read_sql("""
        SELECT
            v.date,
            p.portfolio_pnl  AS actual_pnl,
            v.hist_var95, v.hist_var99,
            v.param_var95, v.param_var99,
            v.mc_var95, v.mc_var99
        FROM var_summary v
        JOIN portfolio_returns p ON v.date = p.date
        ORDER BY v.date
    """, conn)

    method_map = {
        "historical":  {0.95: "hist_var95",  0.99: "hist_var99"},
        "parametric":  {0.95: "param_var95", 0.99: "param_var99"},
        "monte_carlo": {0.95: "mc_var95",    0.99: "mc_var99"},
    }

    result_rows  = []
    summary_rows = []

    for method, conf_cols in method_map.items():
        for conf, col in conf_cols.items():
            for _, row in df.iterrows():
                loss         = -row["actual_pnl"]     # positive = loss
                var_dollar   = row[col]
                is_exception = int(loss > var_dollar)
                result_rows.append({
                    "date":            row["date"],
                    "method":          method,
                    "confidence":      conf,
                    "var_dollar":      round(var_dollar, 2),
                    "actual_pnl":      round(row["actual_pnl"], 2),
                    "is_exception":    is_exception,
                    "exception_label": "breach" if is_exception else "ok",
                })

            # Rolling window summary (use last `window` observations)
            series  = df[[col, "actual_pnl", "date"]].copy()
            series["is_exception"] = (-series["actual_pnl"] > series[col]).astype(int)

            if len(series) >= window:
                window_df    = series.iloc[-window:]
                window_end   = window_df["date"].iloc[-1]
                exc_arr      = window_df["is_exception"].values
                n_exceptions = int(exc_arr.sum())
                n_obs        = len(window_df)
                exc_rate     = n_exceptions / n_obs

                kupiec  = kupiec_pof_test(n_exceptions, n_obs, conf)
                christo = christoffersen_test(exc_arr)
                zone    = _traffic_light_zone(n_exceptions)

                summary_rows.append({
                    "window_end":     window_end,
                    "method":         method,
                    "confidence":     conf,
                    "n_obs":          n_obs,
                    "n_exceptions":   n_exceptions,
                    "exception_rate": round(exc_rate, 4),
                    "zone":           zone,
                    "kupiec_stat":    kupiec["statistic"],
                    "kupiec_pval":    kupiec["p_value"],
                    "kupiec_reject":  int(kupiec["reject_h0"]),
                    "christo_stat":   christo["statistic"],
                    "christo_pval":   christo["p_value"],
                    "christo_reject": int(christo["reject_h0"]),
                    "pi01":           christo["pi01"],
                    "pi11":           christo["pi11"],
                })

    results_df = pd.DataFrame(result_rows)
    summary_df = pd.DataFrame(summary_rows)
    log.info("Backtesting complete: %d daily records, %d window summaries.",
             len(results_df), len(summary_df))
    return results_df, summary_df


# ── Write to DB ───────────────────────────────────────────────────────────────

def save_results(conn, results_df, summary_df):
    for tbl, df in [("backtest_results", results_df),
                    ("backtest_summary", summary_df)]:
        conn.execute(f"DELETE FROM {tbl}")
        df.to_sql(tbl, conn, if_exists="append", index=False,
                  method="multi", chunksize=2000)
    conn.commit()
    log.info("backtest_results: %d rows  |  backtest_summary: %d rows",
             len(results_df), len(summary_df))


# ── Validation & reporting ────────────────────────────────────────────────────

def run_validation(conn: sqlite3.Connection) -> None:
    log.info("─── Backtesting results ─────────────────────────")
    log.info("  Basel 250-day window  |  99%% VaR")
    log.info("  %-14s  %6s  %5s  %6s  %-8s  %8s  %8s",
             "Method", "Excep.", "Rate", "Zone",
             "Kupiec p", "Christo p", "Clustered?")
    log.info("  " + "-" * 72)

    rows = conn.execute("""
        SELECT method, n_exceptions, exception_rate, zone,
               kupiec_pval, christo_pval, christo_reject, pi11
        FROM backtest_summary
        WHERE confidence = 0.99
        ORDER BY method
    """).fetchall()

    for method, n_exc, exc_rate, zone, kp, cp, cr, pi11 in rows:
        clustered = "YES" if cr else "no"
        log.info("  %-14s  %6d  %4.2f%%  %-8s  %8.4f  %9.4f  %9s",
                 method, n_exc, exc_rate * 100, zone.upper(),
                 kp if kp is not None else float("nan"),
                 cp if cp is not None else float("nan"),
                 clustered)

    log.info("  " + "-" * 72)
    log.info("  Note: p-value > 0.05 = model NOT rejected by that test")
    log.info("")
    log.info("  Basel zones (250-day, 99%% VaR):")
    log.info("    GREEN  0–4 exceptions  → model acceptable")
    log.info("    AMBER  5–9 exceptions  → supervisory concern")
    log.info("    RED    10+ exceptions  → model rejected")
    log.info("")

    # Show actual breach dates (99% Historical)
    breaches = conn.execute("""
        SELECT date, var_dollar, actual_pnl
        FROM backtest_results
        WHERE method='historical' AND confidence=0.99 AND is_exception=1
        ORDER BY date
    """).fetchall()
    log.info("  Historical 99%% VaR breach dates (%d total):", len(breaches))
    for date, var_d, pnl in breaches:
        loss = -pnl
        log.info("    %s  VaR=$%7.0f  Actual loss=$%7.0f  Excess=$%6.0f",
                 date, var_d, loss, loss - var_d)
    log.info("────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> sqlite3.Connection:
    conn = get_connection()
    create_tables(conn)
    results_df, summary_df = run_backtests(conn)
    save_results(conn, results_df, summary_df)
    run_validation(conn)
    log.info("Module 4 complete.")
    return conn


if __name__ == "__main__":
    run()
