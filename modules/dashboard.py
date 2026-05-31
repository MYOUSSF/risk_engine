# risk_engine/modules/dashboard.py
"""
Module 6 — Risk Dashboard (Streamlit)
======================================
Interactive risk reporting app. Pulls all computed results from the SQLite
database and presents them across five pages:

  1. Portfolio overview  — positions, weights, cumulative P&L, risk decomposition
  2. VaR analysis        — rolling VaR (all methods), return distribution, CVaR
  3. Backtesting         — Basel traffic light, breach timeline, statistical tests
  4. Stress testing      — scenario waterfall, position contributions, heatmap
  5. Methodology         — explanations of each model, assumptions, limitations

Usage (from project root):
    streamlit run risk_engine/modules/dashboard.py
"""

import math
import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# Resolve DB path relative to this file — works locally and on Streamlit Cloud
_HERE   = os.path.dirname(os.path.abspath(__file__))   # .../modules/
_ROOT   = os.path.dirname(_HERE)                        # .../risk_engine/
DB_PATH = os.path.join(_ROOT, "data", "market_data.db")

sys.path.insert(0, os.path.join(_ROOT, ".."))           # keep package imports working

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Risk Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 12px; color: #6c757d; margin: 0; }
    .metric-value { font-size: 24px; font-weight: 600; margin: 4px 0 0; }
    .zone-green { background:#d4edda; color:#155724;
                  padding:4px 12px; border-radius:20px; font-size:12px;
                  font-weight:600; display:inline-block; }
    .zone-amber { background:#fff3cd; color:#856404;
                  padding:4px 12px; border-radius:20px; font-size:12px;
                  font-weight:600; display:inline-block; }
    .zone-red   { background:#f8d7da; color:#721c24;
                  padding:4px 12px; border-radius:20px; font-size:12px;
                  font-weight:600; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ── DB connection ─────────────────────────────────────────────────────────────

class _StdDev:
    def __init__(self): self.vals = []
    def step(self, v):
        if v is not None: self.vals.append(v)
    def finalize(self):
        n = len(self.vals)
        if n < 2: return 0.0
        m = sum(self.vals) / n
        return math.sqrt(sum((x - m) ** 2 for x in self.vals) / (n - 1))


@st.cache_resource
def get_conn():
    if not os.path.exists(DB_PATH):
        st.error(
            f"Database not found at `{DB_PATH}`.\n\n"
            "Run the pipeline first:\n"
            "```\n"
            "python -m risk_engine.modules.data_ingestion\n"
            "python -m risk_engine.modules.portfolio\n"
            "python -m risk_engine.modules.var_engine\n"
            "python -m risk_engine.modules.backtesting\n"
            "python -m risk_engine.modules.stress_testing\n"
            "```"
        )
        st.stop()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.create_aggregate("STDEV", 1, _StdDev)
    return conn


@st.cache_data(ttl=300)
def query(_conn, sql, params=None):
    if params:
        return pd.read_sql(sql, _conn, params=params)
    return pd.read_sql(sql, _conn)


conn = get_conn()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Market Risk Engine")
    st.markdown("*Portfolio risk analytics toolkit*")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Portfolio Overview", "VaR Analysis",
         "Backtesting", "Stress Testing", "Greeks", "Methodology"],
        label_visibility="collapsed",
    )
    st.divider()

    # Key stats in sidebar
    latest_val = query(conn,
        "SELECT portfolio_value FROM portfolio_returns ORDER BY date DESC LIMIT 1"
    ).iloc[0, 0]
    latest_var = query(conn,
        "SELECT hist_var99 FROM var_summary ORDER BY date DESC LIMIT 1"
    ).iloc[0, 0]

    st.markdown(f"**Portfolio value**  \n`${latest_val:,.0f}`")
    st.markdown(f"**99% VaR (1-day)**  \n`${latest_var:,.0f}`")
    st.markdown(f"**VaR / NAV**  \n`{latest_var/latest_val*100:.2f}%`")
    st.divider()
    st.caption("Data: 2018–2024 · 5 tickers · $1M initial notional")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Portfolio Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.caption("Mark-to-market positions, daily P&L, and risk decomposition")

    # ── Top metrics ──────────────────────────────────────────────────────────
    port = query(conn, "SELECT * FROM portfolio_returns ORDER BY date")
    positions = query(conn, "SELECT * FROM positions ORDER BY ticker")

    total_pnl  = port["portfolio_pnl"].sum()
    cum_ret    = (port["portfolio_value"].iloc[-1] /
                  port["portfolio_value"].iloc[0] - 1) * 100
    daily_vol  = port["portfolio_ret"].std() * math.sqrt(252) * 100
    sharpe     = (port["portfolio_ret"].mean() * 252 /
                  (port["portfolio_ret"].std() * math.sqrt(252)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio value",    f"${latest_val:,.0f}")
    c2.metric("Total P&L",          f"${total_pnl:,.0f}")
    c3.metric("Ann. volatility",    f"{daily_vol:.1f}%")
    c4.metric("Sharpe ratio",       f"{sharpe:.2f}")

    st.divider()

    # ── Cumulative P&L chart ─────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Cumulative P&L")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.fill_between(range(len(port)), port["cum_pnl"],
                        where=port["cum_pnl"] >= 0,
                        color="#1a7a4a", alpha=0.3, label="Gain")
        ax.fill_between(range(len(port)), port["cum_pnl"],
                        where=port["cum_pnl"] < 0,
                        color="#c0392b", alpha=0.3, label="Loss")
        ax.plot(port["cum_pnl"].values, color="#1a7a4a", linewidth=1.2)
        ax.axhline(0, color="#999", linewidth=0.5, linestyle="--")
        ax.set_xticks(np.linspace(0, len(port)-1, 7).astype(int))
        ax.set_xticklabels(
            port["date"].iloc[np.linspace(0, len(port)-1, 7).astype(int)].str[:7],
            fontsize=9, rotation=20
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.set_ylabel("Cumulative P&L ($)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Positions")
        for _, row in positions.iterrows():
            mkt_val = query(conn,
                f"SELECT close FROM prices WHERE ticker='{row.ticker}' "
                "ORDER BY date DESC LIMIT 1"
            ).iloc[0, 0] * row["shares"]
            st.markdown(f"**{row.ticker}**  \n"
                        f"`{row.weight*100:.0f}%` · ${mkt_val:,.0f}")

    # ── Risk decomposition ───────────────────────────────────────────────────
    st.subheader("Risk decomposition")
    ret_wide = query(conn,
        "SELECT date, ticker, daily_ret FROM position_pnl ORDER BY date"
    ).pivot(index="date", columns="ticker", values="daily_ret").dropna()

    weights = positions.set_index("ticker")["weight"].reindex(ret_wide.columns).values
    cov_annual = ret_wide.cov().values * 252
    port_vol = math.sqrt(weights @ cov_annual @ weights)
    mcr = (cov_annual @ weights) / port_vol

    risk_rows = []
    for i, ticker in enumerate(ret_wide.columns):
        ann_vol = math.sqrt(cov_annual[i, i]) * 100
        contrib = weights[i] * mcr[i]
        pct_risk = contrib / port_vol * 100
        risk_rows.append({
            "Ticker": ticker,
            "Weight": f"{weights[i]*100:.0f}%",
            "Ann. vol": f"{ann_vol:.1f}%",
            "Risk contrib.": f"{pct_risk:.1f}%",
        })

    risk_df = pd.DataFrame(risk_rows)
    st.dataframe(risk_df, width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — VaR Analysis
# ════════════════════════════════════════════════════════════════════════════
elif page == "VaR Analysis":
    st.title("Value at Risk Analysis")
    st.caption("Rolling 1-day VaR at 95% and 99% confidence · Historical, Parametric, Monte Carlo")

    var_sum = query(conn, "SELECT * FROM var_summary ORDER BY date")
    port    = query(conn, "SELECT date, portfolio_ret FROM portfolio_returns ORDER BY date")

    # ── Latest snapshot ──────────────────────────────────────────────────────
    latest = var_sum.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hist VaR 95%",  f"${latest.hist_var95:,.0f}")
    c2.metric("Hist VaR 99%",  f"${latest.hist_var99:,.0f}")
    c3.metric("Param VaR 95%", f"${latest.param_var95:,.0f}")
    c4.metric("MC VaR 99%",    f"${latest.mc_var99:,.0f}")
    c5.metric("EWMA VaR 99%",  f"${latest.ewma_var99:,.0f}")

    st.divider()

    # ── Method selector ──────────────────────────────────────────────────────
    conf = st.radio(
        "Confidence level", ["95%", "99%"], horizontal=True
    )
    suffix = "95" if conf == "95%" else "99"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Rolling {conf} VaR — all methods")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(var_sum[f"hist_var{suffix}"].values,
                color="#185FA5", linewidth=1.3, label="Historical")
        ax.plot(var_sum[f"param_var{suffix}"].values,
                color="#3B6D11", linewidth=1.1, linestyle="--", label="Parametric")
        ax.plot(var_sum[f"mc_var{suffix}"].values,
                color="#854F0B", linewidth=1.0, linestyle=":", label="Monte Carlo")
        ax.plot(var_sum[f"ewma_var{suffix}"].values,
                color="#7B2D8B", linewidth=1.0, linestyle=(0, (3, 1, 1, 1)), label="EWMA")
        ticks = np.linspace(0, len(var_sum)-1, 6).astype(int)
        ax.set_xticks(ticks)
        ax.set_xticklabels(var_sum["date"].iloc[ticks].str[:7],
                           fontsize=8, rotation=20)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.set_ylabel("VaR ($)", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Return distribution vs normal")
        rets = port["portfolio_ret"].values * 100
        fig, ax = plt.subplots(figsize=(7, 3.5))
        n_bins = 50
        counts, bins, patches = ax.hist(rets, bins=n_bins,
                                         color="#185FA5", alpha=0.6,
                                         edgecolor="none")
        # Colour the left tail
        var99_pct = -latest.hist_var99 / latest.port_value * 100
        for patch, left in zip(patches, bins[:-1]):
            if left < var99_pct:
                patch.set_facecolor("#E24B4A")
                patch.set_alpha(0.7)
        # Normal overlay
        mu, sigma = rets.mean(), rets.std()
        x = np.linspace(rets.min(), rets.max(), 200)
        bw = (bins[1] - bins[0])
        ax.plot(x, len(rets) * bw *
                (1/(sigma*math.sqrt(2*math.pi))) *
                np.exp(-0.5*((x-mu)/sigma)**2),
                color="#E24B4A", linewidth=1.5, linestyle="--", label="Normal fit")
        ax.axvline(var99_pct, color="#E24B4A", linewidth=1,
                   linestyle="-", alpha=0.8, label=f"99% VaR ({var99_pct:.2f}%)")
        ax.set_xlabel("Daily return (%)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── CVaR comparison ──────────────────────────────────────────────────────
    st.subheader("Average VaR and CVaR over history")
    avg = query(conn, """
        SELECT method,
               ROUND(AVG(var_dollar),0)  AS avg_var_95,
               ROUND(AVG(cvar_dollar),0) AS avg_cvar_95
        FROM var_results WHERE confidence=0.95
        GROUP BY method ORDER BY method
    """)
    avg.columns = ["Method", "Avg VaR 95% ($)", "Avg CVaR 95% ($)"]
    avg["Method"] = (avg["Method"].str.replace("_", " ").str.title()
                     .str.replace("Ewma", "EWMA"))
    st.dataframe(avg, width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Backtesting
# ════════════════════════════════════════════════════════════════════════════
elif page == "Backtesting":
    st.title("VaR Backtesting")
    st.caption("Basel II/III traffic light · Kupiec POF test · Christoffersen independence test")

    bt_sum = query(conn,
        "SELECT * FROM backtest_summary WHERE confidence=0.99 ORDER BY method")

    # ── Traffic light cards ──────────────────────────────────────────────────
    st.subheader("Basel traffic light — 250-day window, 99% VaR")
    cols = st.columns(3)
    for col, (_, row) in zip(cols, bt_sum.iterrows()):
        zone_class = f"zone-{row.zone}"
        col.markdown(
            f"<div class='metric-card'>"
            f"<span class='{zone_class}'>{row.zone.upper()}</span><br><br>"
            f"<strong>{row.method.replace('_',' ').title()}</strong><br>"
            f"<span style='font-size:13px;color:#555'>"
            f"{row.n_exceptions} exceptions · {row.exception_rate*100:.1f}% rate</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Breach timeline ──────────────────────────────────────────────────────
    st.subheader("Historical 99% VaR — breach timeline")
    bt_daily = query(conn, """
        SELECT b.date, b.var_dollar, b.actual_pnl, b.is_exception
        FROM backtest_results b
        WHERE b.method='historical' AND b.confidence=0.99
        ORDER BY b.date
    """)
    bt_daily["actual_loss"] = -bt_daily["actual_pnl"]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(bt_daily["var_dollar"].values,
            color="#185FA5", linewidth=1.3, label="99% VaR limit", zorder=2)
    ax.fill_between(range(len(bt_daily)),
                    bt_daily["actual_loss"].values,
                    where=bt_daily["actual_loss"] > 0,
                    color="#888", alpha=0.25, label="Daily loss")
    ax.fill_between(range(len(bt_daily)),
                    bt_daily["actual_loss"].values,
                    where=bt_daily["actual_loss"] <= 0,
                    color="#1a7a4a", alpha=0.2, label="Daily gain")

    breaches = bt_daily[bt_daily["is_exception"] == 1]
    breach_idx = breaches.index.tolist()
    # Reindex relative to bt_daily positional index
    breach_pos = [bt_daily.index.get_loc(i) for i in breaches.index]
    ax.scatter(breach_pos, breaches["actual_loss"].values,
               color="#E24B4A", zorder=5, s=50, label="VaR breach", marker="o")

    ticks = np.linspace(0, len(bt_daily)-1, 8).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels(bt_daily["date"].iloc[ticks].str[:7],
                       fontsize=8, rotation=20)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.set_ylabel("P&L ($)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── Statistical tests table ───────────────────────────────────────────────
    st.subheader("Statistical test results")
    tests = query(conn, """
        SELECT method, n_exceptions, exception_rate,
               kupiec_pval, kupiec_reject,
               christo_pval, christo_reject, pi11
        FROM backtest_summary WHERE confidence=0.99 ORDER BY method
    """)
    tests["Method"] = tests["method"].str.replace("_"," ").str.title()
    tests["Exceptions"] = tests["n_exceptions"]
    tests["Rate"] = (tests["exception_rate"] * 100).map("{:.2f}%".format)
    tests["Kupiec p-val"] = tests["kupiec_pval"].map("{:.4f}".format)
    tests["Kupiec"] = tests["kupiec_reject"].map(
        {0: "✅ PASS", 1: "❌ FAIL"})
    tests["Christoffersen p"] = tests["christo_pval"].map("{:.4f}".format)
    tests["Independence"] = tests["christo_reject"].map(
        {0: "✅ Not clustered", 1: "❌ Clustered"})
    tests["π11 (cluster prob)"] = tests["pi11"].map("{:.4f}".format)

    display_cols = ["Method","Exceptions","Rate","Kupiec p-val","Kupiec",
                    "Christoffersen p","Independence","π11 (cluster prob)"]
    st.dataframe(tests[display_cols], width='stretch', hide_index=True)

    # ── All breach dates ──────────────────────────────────────────────────────
    with st.expander("View all 99% VaR breach dates"):
        breaches_detail = query(conn, """
            SELECT date, ROUND(var_dollar,0) AS var_limit,
                   ROUND(-actual_pnl,0) AS actual_loss,
                   ROUND(-actual_pnl - var_dollar,0) AS excess
            FROM backtest_results
            WHERE method='historical' AND confidence=0.99 AND is_exception=1
            ORDER BY date
        """)
        breaches_detail.columns = ["Date","VaR Limit ($)","Actual Loss ($)","Excess ($)"]
        st.dataframe(breaches_detail, width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Stress Testing
# ══════════════════════════════════════
elif page == "Stress Testing":
    st.title("Stress Testing")
    st.caption("Historical replays · Hypothetical shocks · Reverse stress test")

    summary = query(conn,
        "SELECT * FROM stress_summary ORDER BY total_pnl ASC")

    # ── Scenario waterfall ───────────────────────────────────────────────────
    st.subheader("Portfolio P&L under stress scenarios")
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = []
    for _, row in summary.iterrows():
        if row.scenario_type == "historical":
            colors.append("#A32D2D")
        elif row.scenario_type == "reverse":
            colors.append("#854F0B")
        else:
            colors.append("#185FA5")

    bars = ax.barh(summary["scenario_name"], summary["total_pnl"],
                   color=colors, height=0.6)
    ax.axvline(0, color="#999", linewidth=0.8)
    for bar, val in zip(bars, summary["total_pnl"]):
        ax.text(val - 8000, bar.get_y() + bar.get_height()/2,
                f"${val/1000:.0f}k",
                va="center", ha="right", fontsize=9, color="white", fontweight="bold")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#A32D2D", label="Historical"),
        mpatches.Patch(color="#185FA5", label="Hypothetical"),
        mpatches.Patch(color="#854F0B", label="Reverse"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Scenario selector for detail ─────────────────────────────────────────
    st.subheader("Position breakdown")
    selected = st.selectbox("Select scenario", summary["scenario_name"].tolist())

    col1, col2 = st.columns([1, 2])

    with col1:
        row = summary[summary["scenario_name"] == selected].iloc[0]
        st.metric("Portfolio P&L",
                  f"${row.total_pnl:,.0f}",
                  delta=f"{row.total_pnl_pct:.1f}%")
        st.metric("Stressed value", f"${row.stressed_value:,.0f}")
        st.metric("Holding period", f"{row.holding_days} days")
        st.markdown(f"**Worst position:** `{row.worst_position}`  \n"
                    f"P&L: `${row.worst_position_pnl:,.0f}`")

    with col2:
        detail = query(conn, """
            SELECT ticker, shock_pct, position_value,
                   position_pnl, pct_contribution
            FROM stress_results WHERE scenario_name = ?
            ORDER BY position_pnl ASC
        """, params=(selected,))

        fig, ax = plt.subplots(figsize=(6, 3))
        bar_colors = ["#E24B4A" if v < 0 else "#1a7a4a"
                      for v in detail["position_pnl"]]
        ax.barh(detail["ticker"], detail["position_pnl"],
                color=bar_colors, height=0.5)
        ax.axvline(0, color="#999", linewidth=0.8)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.tick_params(labelsize=9)
        ax.set_title("Position P&L", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── Shock heatmap ─────────────────────────────────────────────────────────
    st.subheader("Shock heatmap — all scenarios × all tickers")
    heatmap_data = query(conn, """
        SELECT scenario_name, ticker, shock_pct
        FROM stress_results ORDER BY scenario_name, ticker
    """).pivot(index="scenario_name", columns="ticker", values="shock_pct")

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto",
                   vmin=-60, vmax=40)
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, fontsize=10)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=9)
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            ax.text(j, i, f"{val:+.0f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > 25 else "black")
    plt.colorbar(im, ax=ax, label="Shock (%)", fraction=0.03)
    ax.set_title("Applied shocks by scenario and ticker (%)", fontsize=11, pad=12)
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Greeks
# ════════════════════════════════════════════════════════════════════════════
elif page == "Greeks":
    st.title("Black-Scholes Greeks")
    st.caption(
        "Hypothetical ATM call on AAPL · 30-day expiry · 5% risk-free rate · "
        "historical implied vol from last 252 trading days"
    )

    # Graceful fallback if the module has not been run yet
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='option_snapshot'"
    ).fetchone()
    if not has_table:
        st.warning(
            "Option data not found. "
            "Run `python -m risk_engine.modules.greeks` first, then refresh."
        )
        st.stop()

    snap    = query(conn, "SELECT * FROM option_snapshot").iloc[0]
    profile = query(conn, "SELECT * FROM option_profile ORDER BY spot_price")

    S       = snap["spot_price"]
    K       = snap["strike"]
    expiry  = round(snap["t_years"] * 365)
    sigma   = snap["implied_vol"]
    r       = snap["risk_free_rate"]
    price   = snap["option_price"]
    n_cont  = int(snap["n_contracts"])
    pos_val = price * 100 * n_cont

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AAPL Spot",      f"${S:,.2f}")
    c2.metric("Strike (ATM)",   f"${K:,.2f}")
    c3.metric("Expiry",         f"{expiry} days")
    c4.metric("Implied vol",    f"{sigma*100:.1f}%")
    c5.metric("Call price",     f"${price:.4f}/sh",
              help=f"Position value: ${pos_val:,.2f}  ({n_cont} contracts × 100 shares)")

    st.divider()

    # ── Greeks table + Delta S-curve ─────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Option Greeks")
        st.caption("Per share of AAPL underlying")

        greeks_rows = [
            {
                "Greek":  "Delta  Δ",
                "Value":  f"{snap['delta']:+.4f}",
                "Sensitivity": f"${snap['delta']:.4f} per $1 spot move",
            },
            {
                "Greek":  "Gamma  Γ",
                "Value":  f"{snap['gamma']:+.6f}",
                "Sensitivity": f"Δ shifts {snap['gamma']:.6f} per $1 spot move",
            },
            {
                "Greek":  "Vega   ν",
                "Value":  f"{snap['vega']:+.4f}",
                "Sensitivity": f"${snap['vega']:.4f} per 1% rise in vol",
            },
            {
                "Greek":  "Theta  Θ",
                "Value":  f"{snap['theta']:+.4f}",
                "Sensitivity": f"${snap['theta']:.4f} per calendar day",
            },
            {
                "Greek":  "Rho    ρ",
                "Value":  f"{snap['rho']:+.4f}",
                "Sensitivity": f"${snap['rho']:.4f} per 1% rise in rate",
            },
        ]
        st.dataframe(
            pd.DataFrame(greeks_rows),
            width='stretch',
            hide_index=True,
        )

        st.divider()
        st.markdown(f"**Position size:** {n_cont} contracts (×100 shares)")
        st.markdown(f"**Total position value:** `${pos_val:,.2f}`")
        st.markdown(
            f"**Dollar delta:** `${snap['delta']*100*n_cont:,.0f}` "
            f"(P&L per $1 move in AAPL)"
        )

    with col2:
        st.subheader("Delta profile — sensitivity across spot prices")
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(profile["spot_price"], profile["delta"],
                color="#185FA5", linewidth=2, label="Delta Δ")
        ax.fill_between(profile["spot_price"], profile["delta"],
                        alpha=0.12, color="#185FA5")

        # ATM marker
        ax.axvline(S, color="#E24B4A", linewidth=1.2, linestyle="--",
                   label=f"ATM spot = ${S:.2f}")
        ax.axhline(0.5, color="#888", linewidth=0.7, linestyle=":", alpha=0.7)
        ax.scatter([S], [snap["delta"]], color="#E24B4A", zorder=5, s=60)
        ax.annotate(
            f"  Δ = {snap['delta']:.4f}",
            xy=(S, snap["delta"]),
            fontsize=9, color="#E24B4A",
        )

        ax.set_xlabel("Spot price ($)", fontsize=9)
        ax.set_ylabel("Delta", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    # ── Option price vs spot ──────────────────────────────────────────────────
    st.subheader("Option price vs spot — Black-Scholes vs intrinsic value")
    fig, ax = plt.subplots(figsize=(12, 3.5))

    spots   = profile["spot_price"].values
    intrinsic = np.maximum(spots - K, 0)

    ax.plot(spots, profile["option_price"].values,
            color="#185FA5", linewidth=2, label="Black-Scholes price")
    ax.plot(spots, intrinsic,
            color="#E24B4A", linewidth=1.2, linestyle="--", label="Intrinsic value")
    ax.fill_between(
        spots,
        profile["option_price"].values,
        intrinsic,
        alpha=0.15,
        color="#1a7a4a",
        label="Time value",
    )

    ax.axvline(S, color="#888", linewidth=1, linestyle=":", alpha=0.8)
    ax.scatter([S], [price], color="#185FA5", zorder=5, s=60)
    ax.annotate(
        f"  ${price:.4f}",
        xy=(S, price),
        fontsize=9, color="#185FA5",
    )

    ax.set_xlabel("Spot price ($)", fontsize=9)
    ax.set_ylabel("Option price ($)", fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # ── Methodology note ──────────────────────────────────────────────────────
    with st.expander("📐 Black-Scholes model assumptions"):
        st.markdown(f"""
**Call price formula:**

`C = S·N(d₁) − K·e^(−rT)·N(d₂)`

where `d₁ = [ln(S/K) + (r + ½σ²)T] / (σ√T)` and `d₂ = d₁ − σ√T`

**Parameters used:**  S = ${S:.2f}  ·  K = ${K:.2f}  ·  T = {expiry}/365  ·  r = {r*100:.0f}%  ·  σ = {sigma*100:.1f}%

**Greeks conventions in this module:**
- Vega is per 1% absolute change in σ (raw vega ÷ 100)
- Theta is per calendar day (raw theta ÷ 365)
- Rho is per 1% absolute change in r (raw rho ÷ 100)

**Key assumptions and limitations:**
- Continuous trading, no dividends, constant vol and rate until expiry
- Lognormal returns — does not capture vol skew or term structure
- Historical vol used as a proxy for implied vol; in practice these diverge
- European exercise only (no early exercise premium as in American options)
        """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Methodology
# ════════════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.title("Methodology")
    st.caption("Model descriptions, assumptions, and known limitations")

    with st.expander("📐 Historical Simulation VaR", expanded=True):
        st.markdown("""
**How it works:**
Rank the last 252 daily log returns from worst to best. The VaR at confidence
level α is the return at the (1−α) percentile.

**Formula:**  `VaR_α = −Q_{1−α}(r_1, ..., r_T)`

**Pros:**
- Non-parametric — no distributional assumption
- Automatically captures fat tails, skewness, and non-linearity
- Simple to explain to regulators

**Cons:**
- Fully backward-looking — slow to adapt after regime changes
- Requires sufficient history (≥252 days) before first estimate
- Gives equal weight to all observations regardless of age

**Our implementation:** 252-day rolling window, computed daily.
        """)

    with st.expander("📊 Parametric (Variance-Covariance) VaR"):
        st.markdown("""
**How it works:**
Assumes daily returns are normally distributed. VaR is derived analytically
from the mean and standard deviation of the return window.

**Formula:**  `VaR_α = −(μ − z_α · σ)`

where z_α is the standard normal quantile (−1.645 at 95%, −2.326 at 99%).

**CVaR (closed form):**  `CVaR_α = −μ + σ · φ(z_α) / (1−α)`

**Pros:** Fast, analytically tractable, easy to decompose into components.

**Cons:**
- Underestimates tail risk when returns have excess kurtosis (fat tails)
- The 2008 GFC exposed this limitation across the industry

**Our implementation:** Rolling 252-day window, μ and σ re-estimated daily.
        """)

    with st.expander("🎲 Monte Carlo VaR"):
        st.markdown("""
**How it works:**
Fits a normal distribution to the rolling return window and draws 10,000
one-day return scenarios. VaR and CVaR are read from the simulated distribution's
empirical percentiles.

**Formula:**  `r_sim ~ N(μ_window, σ_window)` × 10,000 draws

**Pros:**
- Flexible — can incorporate non-normal distributions (fat-tailed, skewed)
- Naturally extends to multi-asset simulation with Cholesky correlation
- Results converge to parametric for normal inputs (sanity check)

**Cons:** Computationally expensive; slightly noisy across runs (mitigated with fixed seed).

**Our implementation:** 10,000 simulations per day, fixed seed (42), rolling 252-day window.
        """)

    with st.expander("✅ Basel Backtesting Framework"):
        st.markdown("""
**Kupiec POF (Proportion of Failures) test**
Tests whether exceptions occur at the right frequency.

- H₀: actual exception rate = expected rate (1% for 99% VaR)
- Test statistic: likelihood ratio, χ²(1) distribution
- p-value > 0.05 → cannot reject H₀ → model calibration acceptable

**Christoffersen Independence test**
Tests whether exceptions cluster in time.

- H₀: today's exception is independent of yesterday's
- Transition probabilities π₀₁ and π₁₁ are estimated
- If π₁₁ >> π₀₁, exceptions cluster → model fails to adapt

**Basel traffic light (250-day window, 99% VaR):**

| Exceptions | Zone | Regulatory action |
|---|---|---|
| 0–4 | 🟢 Green | Model acceptable |
| 5–9 | 🟡 Amber | Supervisory concern |
| 10+ | 🔴 Red | Model presumed inadequate |
        """)

    with st.expander("⚡ Stress Testing"):
        st.markdown("""
**Historical scenarios** replay actual market crashes on the current position book.
Shocks are calibrated to observed peak-to-trough drawdowns for each sector.

**Hypothetical scenarios** apply user-defined factor shocks derived from:
- Historical volatility per ticker (sigma calibration)
- Cross-sector correlations from the data
- Specific macro drivers (rate sensitivity, commodity exposure, beta)

**Reverse stress test** works backwards from an outcome:
*"What uniform market shock would produce a 20% portfolio loss?"*
This is a Basel III regulatory requirement for systematically important institutions.

**Limitation:** Static shocks assume constant positions and ignore:
- Liquidity risk (inability to exit positions)
- Correlation breakdown during crises (correlations approach 1)
- Second-order effects (margin calls, forced deleveraging)
        """)