# risk_engine/modules/stress_testing.py
"""
Module 5 — Stress Testing
==========================
Tests portfolio resilience under extreme but plausible scenarios.

VaR measures "normal" risk. Stress testing asks: what happens in a crisis?

Three scenario types are implemented:

  1. Historical scenarios   — replay actual market crashes on the current book
       - 2008 Global Financial Crisis (Sep–Nov 2008 peak drawdown window)
       - 2020 COVID crash (Feb–Mar 2020)
       - 2022 Rate shock (Jan–Jun 2022, fastest Fed hiking cycle in 40 years)

  2. Hypothetical scenarios — user-defined factor shocks
       - Equity crash:  broad equity -20%, credit widening, vol spike
       - Oil shock:     crude -40%, energy sector -30%, broader market -5%
       - Rate shock:    +200bps parallel shift, financials hit, duration losses
       - Tech selloff:  growth/tech -35%, defensives outperform

  3. Reverse stress test    — work backwards: what market move would wipe out
       a given % of the portfolio? (regulatory requirement under Basel III)

Results stored in `stress_scenarios` and `stress_results`.

Usage:
    python -m risk_engine.modules.stress_testing
"""

import sqlite3
import logging
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from risk_engine.config import DB_PATH, TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Scenario definitions ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    name:        str
    description: str
    scenario_type: str          # 'historical' | 'hypothetical' | 'reverse'
    shocks: dict                # {ticker: shock_return}  (e.g. -0.20 = -20%)
    holding_days: int = 10      # multi-day horizon (Basel uses 10-day for stress)
    notes: str = ""


# Calibrated to actual correlations from the data
SCENARIOS = [

    # ── Historical ──────────────────────────────────────────────────────────

    Scenario(
        name="2008 GFC peak",
        description="Global Financial Crisis — Sep 29 to Nov 20 2008 (worst 38 days)",
        scenario_type="historical",
        holding_days=38,
        shocks={
            # Approximate peak-to-trough returns for each sector in that window
            "AAPL": -0.52,   # tech sold off heavily
            "JPM":  -0.60,   # financials epicentre
            "XOM":  -0.40,   # energy fell with oil
            "JNJ":  -0.18,   # defensives outperformed
            "SPY":  -0.38,   # S&P 500 total drawdown in window
        },
        notes="Peak drawdown period. JPM required TARP capital injection.",
    ),

    Scenario(
        name="2020 COVID crash",
        description="COVID-19 market crash — Feb 19 to Mar 23 2020 (33 days)",
        scenario_type="historical",
        holding_days=33,
        shocks={
            "AAPL": -0.32,
            "JPM":  -0.45,   # financials hit by rate cut fears
            "XOM":  -0.55,   # oil demand collapse + Saudi-Russia price war
            "JNJ":  -0.16,   # healthcare relative outperformer
            "SPY":  -0.34,
        },
        notes="Fastest 30% decline in S&P history. XOM hit by simultaneous demand and supply shock.",
    ),

    Scenario(
        name="2022 rate shock",
        description="Fed hiking cycle — Jan to Jun 2022 (+300bps in 6 months)",
        scenario_type="historical",
        holding_days=126,
        shocks={
            "AAPL": -0.28,   # high-duration growth stock
            "JPM":  -0.22,   # NIM benefit partly offset by recession fears
            "XOM":  +0.38,   # energy outperformed sharply (Ukraine war + supply)
            "JNJ":  -0.05,   # defensive, low duration
            "SPY":  -0.21,
        },
        notes="First time since 1994 that bonds and equities fell simultaneously. XOM was top performer.",
    ),

    # ── Hypothetical ────────────────────────────────────────────────────────

    Scenario(
        name="Equity crash",
        description="Sudden broad equity selloff — global risk-off, VIX >50",
        scenario_type="hypothetical",
        holding_days=5,
        shocks={
            "AAPL": -0.25,
            "JPM":  -0.30,
            "XOM":  -0.22,
            "JNJ":  -0.12,   # defensive, lower beta
            "SPY":  -0.20,
        },
        notes="Calibrated to 2-sigma weekly drawdown based on observed vols.",
    ),

    Scenario(
        name="Oil price collapse",
        description="Oil -40% (demand shock / OPEC+ breakdown), energy sector -30%",
        scenario_type="hypothetical",
        holding_days=10,
        shocks={
            "AAPL": -0.05,   # minimal direct exposure
            "JPM":  -0.08,   # credit exposure to energy sector
            "XOM":  -0.30,   # direct commodity exposure
            "JNJ":  -0.03,   # near-zero sensitivity
            "SPY":  -0.07,   # energy is ~4% of S&P 500
        },
        notes="Scenario based on Apr 2020 oil price collapse (WTI went negative).",
    ),

    Scenario(
        name="Rate spike +200bps",
        description="Sudden +200bps parallel shift in yield curve (taper tantrum style)",
        scenario_type="hypothetical",
        holding_days=5,
        shocks={
            "AAPL": -0.15,   # high P/E = high duration, hurt by rate rise
            "JPM":  +0.05,   # NIM expansion outweighs mark-to-market bond losses
            "XOM":  -0.04,   # modest sensitivity
            "JNJ":  -0.08,   # moderate duration sensitivity
            "SPY":  -0.10,
        },
        notes="Based on 2013 taper tantrum. Financials can benefit from steeper curve.",
    ),

    Scenario(
        name="Tech selloff",
        description="Growth/tech de-rating — P/E compression, rotation to value",
        scenario_type="hypothetical",
        holding_days=10,
        shocks={
            "AAPL": -0.35,   # highest P/E in book
            "JPM":  +0.03,   # value stock, benefits from rotation
            "XOM":  +0.05,   # value / commodity, benefits from rotation
            "JNJ":  +0.02,   # defensive, slight outperform
            "SPY":  -0.12,   # tech is ~30% of S&P
        },
        notes="Calibrated to 2022 Nasdaq drawdown pattern. Rotation into value/cyclicals.",
    ),

    # ── Reverse stress test ─────────────────────────────────────────────────

    Scenario(
        name="Reverse: -20% portfolio",
        description="What uniform market shock wipes out 20% of portfolio?",
        scenario_type="reverse",
        holding_days=1,
        shocks={t: -0.20 for t in TICKERS},   # uniform shock — solved analytically
        notes="Regulatory reverse stress test. Identifies the break-even market move.",
    ),
]


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
        CREATE TABLE IF NOT EXISTS stress_scenarios (
            scenario_name   TEXT PRIMARY KEY,
            description     TEXT NOT NULL,
            scenario_type   TEXT NOT NULL,
            holding_days    INTEGER NOT NULL,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS stress_results (
            scenario_name       TEXT NOT NULL,
            ticker              TEXT NOT NULL,
            shock_pct           REAL NOT NULL,   -- applied shock as decimal
            position_value      REAL NOT NULL,   -- current market value
            stressed_value      REAL NOT NULL,   -- value after shock
            position_pnl        REAL NOT NULL,   -- dollar P&L for this position
            pct_contribution    REAL,            -- % of total scenario P&L
            PRIMARY KEY (scenario_name, ticker)
        );

        CREATE TABLE IF NOT EXISTS stress_summary (
            scenario_name       TEXT PRIMARY KEY,
            scenario_type       TEXT NOT NULL,
            holding_days        INTEGER NOT NULL,
            portfolio_value     REAL NOT NULL,
            stressed_value      REAL NOT NULL,
            total_pnl           REAL NOT NULL,
            total_pnl_pct       REAL NOT NULL,   -- as % of portfolio
            worst_position      TEXT NOT NULL,
            worst_position_pnl  REAL NOT NULL
        );
    """)
    conn.commit()
    log.info("Stress testing tables ready.")


# ── Scenario engine ───────────────────────────────────────────────────────────

def apply_scenario(
    scenario: Scenario,
    positions: pd.DataFrame,
    current_prices: pd.DataFrame,
) -> dict:
    """
    Apply a shock scenario to the current position book.

    For each position:
        stressed_value = shares × close × (1 + shock)
        position_pnl   = shares × close × shock

    Portfolio P&L = sum of position P&Ls
    """
    pos = positions.set_index("ticker")
    px  = current_prices.set_index("ticker")["close"]

    results   = []
    total_pnl = 0.0
    port_val  = 0.0

    for ticker in pos.index:
        if ticker not in scenario.shocks:
            continue
        shares    = pos.loc[ticker, "shares"]
        price     = px.loc[ticker]
        mkt_val   = shares * price
        shock     = scenario.shocks[ticker]
        pos_pnl   = mkt_val * shock
        stressed  = mkt_val * (1 + shock)

        total_pnl += pos_pnl
        port_val  += mkt_val

        results.append({
            "scenario_name":   scenario.name,
            "ticker":          ticker,
            "shock_pct":       round(shock * 100, 2),
            "position_value":  round(mkt_val, 2),
            "stressed_value":  round(stressed, 2),
            "position_pnl":    round(pos_pnl, 2),
            "pct_contribution": None,  # filled after totals known
        })

    # Fill % contribution
    for r in results:
        r["pct_contribution"] = round(
            r["position_pnl"] / total_pnl * 100 if total_pnl != 0 else 0, 2
        )

    worst = min(results, key=lambda r: r["position_pnl"])

    summary = {
        "scenario_name":      scenario.name,
        "scenario_type":      scenario.scenario_type,
        "holding_days":       scenario.holding_days,
        "portfolio_value":    round(port_val, 2),
        "stressed_value":     round(port_val + total_pnl, 2),
        "total_pnl":          round(total_pnl, 2),
        "total_pnl_pct":      round(total_pnl / port_val * 100, 2) if port_val else 0,
        "worst_position":     worst["ticker"],
        "worst_position_pnl": worst["position_pnl"],
    }

    return {"results": results, "summary": summary}


def reverse_stress_test(
    positions: pd.DataFrame,
    current_prices: pd.DataFrame,
    target_loss_pct: float = 0.20,
) -> float:
    """
    Reverse stress test: find the uniform shock x such that
    portfolio loss = target_loss_pct × portfolio_value.

    For a uniform shock x:
        loss = sum(shares_i × price_i × x) = x × portfolio_value
        => x = target_loss_pct   (trivially, for uniform shock)

    More usefully: reports the required shock per-ticker
    given the actual portfolio weights and current prices,
    and identifies at what index-level move this is equivalent.
    """
    pos = positions.set_index("ticker")
    px  = current_prices.set_index("ticker")["close"]

    port_val  = sum(pos.loc[t, "shares"] * px.loc[t] for t in pos.index)
    target    = target_loss_pct * port_val

    # For a value-weighted portfolio, uniform shock is simply target_loss_pct
    uniform_shock = -target_loss_pct

    log.info("  Reverse stress test (target loss: %.0f%%):", target_loss_pct * 100)
    log.info("    Portfolio value: $%.0f", port_val)
    log.info("    Target loss:     $%.0f", target)
    log.info("    Required uniform shock: %.1f%%", uniform_shock * 100)

    # More interesting: what shock to SPY (proxy index) achieves same loss
    # given the portfolio's beta to SPY?
    spy_weight = pos.loc["SPY", "weight"] if "SPY" in pos.index else 0.20
    avg_beta   = 0.95  # approximate beta of this portfolio to SPY
    spy_shock  = uniform_shock / avg_beta
    log.info("    Equivalent SPY move (beta=%.2f): %.1f%%", avg_beta, spy_shock * 100)

    return uniform_shock


# ── Run all scenarios ─────────────────────────────────────────────────────────

def run_all_scenarios(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = pd.read_sql("SELECT * FROM positions", conn)

    latest_date = conn.execute(
        "SELECT MAX(date) FROM prices"
    ).fetchone()[0]
    current_prices = pd.read_sql(
        "SELECT ticker, close FROM prices WHERE date = ?",
        conn, params=(latest_date,),
    )

    log.info("Running stress scenarios on portfolio as of %s", latest_date)

    all_results  = []
    all_summaries = []

    for scenario in SCENARIOS:
        out = apply_scenario(scenario, positions, current_prices)
        all_results.extend(out["results"])
        all_summaries.append(out["summary"])

        s = out["summary"]
        log.info("  %-28s  P&L: $%+10.0f  (%+.1f%%)  Worst: %s",
                 scenario.name, s["total_pnl"], s["total_pnl_pct"],
                 s["worst_position"])

    # Reverse stress test (supplementary)
    log.info("  " + "-" * 60)
    reverse_stress_test(positions, current_prices, target_loss_pct=0.20)

    results_df  = pd.DataFrame(all_results)
    summary_df  = pd.DataFrame(all_summaries)
    return results_df, summary_df


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(
    conn: sqlite3.Connection,
    scenarios: list,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    # Save scenario metadata
    scenario_rows = [{
        "scenario_name": s.name,
        "description":   s.description,
        "scenario_type": s.scenario_type,
        "holding_days":  s.holding_days,
        "notes":         s.notes,
    } for s in scenarios]

    for tbl in ("stress_scenarios", "stress_results", "stress_summary"):
        conn.execute(f"DELETE FROM {tbl}")

    pd.DataFrame(scenario_rows).to_sql(
        "stress_scenarios", conn, if_exists="append", index=False)
    results_df.to_sql(
        "stress_results", conn, if_exists="append", index=False,
        method="multi", chunksize=500)
    summary_df.to_sql(
        "stress_summary", conn, if_exists="append", index=False)

    conn.commit()
    log.info("Saved: %d scenario records, %d position results, %d summaries.",
             len(scenario_rows), len(results_df), len(summary_df))


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(conn: sqlite3.Connection) -> None:
    log.info("─── Stress test results ─────────────────────────")
    log.info("  %-28s  %-12s  %10s  %8s  %-8s",
             "Scenario", "Type", "P&L ($)", "P&L (%)", "Worst pos.")
    log.info("  " + "-" * 72)

    rows = conn.execute("""
        SELECT scenario_name, scenario_type, total_pnl,
               total_pnl_pct, worst_position, worst_position_pnl
        FROM stress_summary ORDER BY total_pnl ASC
    """).fetchall()

    for name, stype, pnl, pct, worst, worst_pnl in rows:
        log.info("  %-28s  %-12s  %+10.0f  %+7.1f%%  %s ($%+.0f)",
                 name, stype, pnl, pct, worst, worst_pnl)

    log.info("  " + "-" * 72)

    # Worst scenario detail
    worst_name = conn.execute(
        "SELECT scenario_name FROM stress_summary ORDER BY total_pnl ASC LIMIT 1"
    ).fetchone()[0]
    log.info("  Worst scenario: %s", worst_name)
    detail = conn.execute("""
        SELECT ticker, shock_pct, position_value, position_pnl, pct_contribution
        FROM stress_results WHERE scenario_name = ?
        ORDER BY position_pnl ASC
    """, (worst_name,)).fetchall()
    log.info("  %-6s  %8s  %12s  %12s  %10s",
             "Ticker", "Shock%", "Pos. Value", "P&L", "Contribution")
    for ticker, shock, pv, ppnl, contrib in detail:
        log.info("  %-6s  %+7.1f%%  $%10.0f  $%+10.0f  %+9.1f%%",
                 ticker, shock, pv, ppnl, contrib)
    log.info("────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> sqlite3.Connection:
    conn = get_connection()
    create_tables(conn)
    results_df, summary_df = run_all_scenarios(conn)
    save_results(conn, SCENARIOS, results_df, summary_df)
    run_validation(conn)
    log.info("Module 5 complete.")
    return conn


if __name__ == "__main__":
    run()
