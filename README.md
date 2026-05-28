# Market Risk Engine

A Python-based market risk analytics toolkit simulating a middle office workflow.
Covers data ingestion, VaR (three methods), Basel backtesting, stress testing,
and an interactive [Streamlit dashboard](https://market-risk-engine.streamlit.app).

---

## Quick start

```bash
# 1. Clone / unzip the project
cd risk_engine

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Run the full pipeline (modules must run in this order)
python -m risk_engine.modules.data_ingestion
python -m risk_engine.modules.portfolio
python -m risk_engine.modules.var_engine
python -m risk_engine.modules.backtesting
python -m risk_engine.modules.stress_testing

# 4. Launch the dashboard
streamlit run risk_engine/modules/dashboard.py
```

The dashboard opens at **http://localhost:8501** in your browser.

> **Note:** A pre-populated `market_data.db` is included, so you can skip
> steps 3 and go straight to step 4 if you just want to explore the dashboard.

---

## Project structure

```
risk_engine/
├── CLAUDE.md               # Context file for Claude Code (AI-assisted development)
├── README.md               # This file
├── config.py               # All parameters: tickers, dates, risk settings
├── requirements.txt
├── data/
│   └── market_data.db      # SQLite database (pre-populated, or auto-created on first run)
├── modules/
│   ├── data_ingestion.py   # Module 1 — fetch prices, compute returns, store in DB
│   ├── portfolio.py        # Module 2 — positions, mark-to-market P&L, risk decomposition
│   ├── var_engine.py       # Module 3 — Historical, Parametric, Monte Carlo VaR + CVaR
│   ├── backtesting.py      # Module 4 — Basel traffic-light, Kupiec & Christoffersen tests
│   ├── stress_testing.py   # Module 5 — historical scenarios, hypothetical shocks, reverse stress
│   └── dashboard.py        # Module 6 — Streamlit interactive dashboard
├── reports/                # Output directory for generated reports
└── tests/                  # Unit tests (pytest)
```

---

## Running individual modules

Each module is self-contained and can be run on its own. **Modules must be run
in order** — each one reads from tables written by the previous module.

```bash
# Module 1 — Data ingestion
# Fetches prices from Yahoo Finance (falls back to synthetic GBM data if offline)
# Writes: prices, returns
python -m risk_engine.modules.data_ingestion

# Module 2 — Portfolio construction
# Builds position book, computes daily mark-to-market P&L, risk decomposition
# Writes: positions, position_pnl, portfolio_returns
python -m risk_engine.modules.portfolio

# Module 3 — VaR engine
# Rolling 1-day VaR at 95% and 99% confidence using three methods
# Writes: var_results, var_summary
python -m risk_engine.modules.var_engine

# Module 4 — Backtesting
# Basel traffic-light test, Kupiec POF test, Christoffersen independence test
# Writes: backtest_results, backtest_summary
python -m risk_engine.modules.backtesting

# Module 5 — Stress testing
# 3 historical scenarios, 4 hypothetical shocks, 1 reverse stress test
# Writes: stress_scenarios, stress_results, stress_summary
python -m risk_engine.modules.stress_testing

# Module 6 — Dashboard
streamlit run risk_engine/modules/dashboard.py
```

Re-running any module is safe — it clears and recomputes its tables each time.

---

## Configuration

All parameters are in `config.py`. Edit this file to change the portfolio,
date range, or risk settings — never hardcode values in the modules.

```python
TICKERS               = ["AAPL", "JPM", "XOM", "JNJ", "SPY"]
PORTFOLIO_WEIGHTS     = {"AAPL": 0.25, "JPM": 0.20, "XOM": 0.20, "JNJ": 0.15, "SPY": 0.20}
PORTFOLIO_VALUE       = 1_000_000        # USD notional
START_DATE            = "2018-01-01"
END_DATE              = "2024-12-31"
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
VAR_WINDOW            = 252              # rolling window (trading days)
MONTE_CARLO_SIMS      = 10_000
```

---

## Dashboard pages

| Page | Contents |
|---|---|
| Portfolio overview | Positions, weights, cumulative P&L chart, risk decomposition table |
| VaR analysis | Rolling VaR (all 3 methods), return distribution vs normal, CVaR table |
| Backtesting | Basel traffic-light cards, breach timeline, Kupiec/Christoffersen results |
| Stress testing | Scenario waterfall chart, position breakdown, shock heatmap |
| Methodology | Model write-ups with formulas, assumptions, and limitations |

---

## Portfolio

| Ticker | Sector | Weight |
|---|---|---|
| AAPL | Technology | 25% |
| JPM | Financials | 20% |
| XOM | Energy | 20% |
| JNJ | Healthcare | 15% |
| SPY | Broad market (benchmark) | 20% |

**Initial notional:** $1,000,000 · **Period:** 2018–2024 · **Data:** Yahoo Finance (adjusted close)

---

## Methodology

### Value at Risk

Three methods are computed daily on a rolling 252-day window:

| Method | Assumption | Key limitation |
|---|---|---|
| Historical simulation | Non-parametric (empirical distribution) | Backward-looking, slow to adapt |
| Parametric (Variance-Covariance) | Returns ~ Normal | Underestimates fat tails |
| Monte Carlo | Returns ~ Normal (fitted, 10,000 paths) | Same as parametric for univariate |

CVaR (Expected Shortfall) is computed alongside VaR for all methods.

### Backtesting

- **Kupiec POF test** — likelihood ratio test for correct exception frequency
- **Christoffersen test** — independence test for exception clustering
- **Basel traffic light** (250-day window, 99% VaR):
  - 🟢 Green: 0–4 exceptions → model acceptable
  - 🟡 Amber: 5–9 exceptions → supervisory concern
  - 🔴 Red: 10+ exceptions → model rejected

### Stress testing

- **Historical:** 2008 GFC peak drawdown, 2020 COVID crash, 2022 rate shock
- **Hypothetical:** Equity crash, oil collapse, rate spike +200bps, tech selloff
- **Reverse:** What uniform shock produces a 20% portfolio loss?

---

## Requirements

```
Python        3.10+
yfinance      0.2.40+     # market data
pandas        2.0+
numpy         1.26+
scipy         1.13+       # statistical distributions
matplotlib    3.8+        # charts
streamlit     1.35+       # dashboard
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Skills demonstrated

- **Python** — pandas, NumPy, SciPy, matplotlib, Streamlit
- **SQL** — SQLite schema design, aggregation queries, indexing, custom aggregates
- **Market risk** — VaR, CVaR, Greeks (extension), stress testing, model validation
- **Financial mathematics** — GBM, Cholesky decomposition, Kupiec/Christoffersen LR tests
- **Software engineering** — modular architecture, idempotent pipelines, reproducible results
