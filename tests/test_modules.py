# risk_engine/tests/test_modules.py
"""
Unit tests for core computation functions across all modules.
All tests use synthetic numpy/pandas inputs — no database required.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Add the project root (parent of risk_engine/) so the package is importable.
# Mirrors the sys.path.insert pattern used in every module file.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from risk_engine.modules.var_engine import historical_var, parametric_var, monte_carlo_var
from risk_engine.modules.backtesting import kupiec_pof_test, christoffersen_test
from risk_engine.modules.stress_testing import apply_scenario, Scenario
from risk_engine.modules.data_ingestion import compute_log_returns


# ── Shared fixtures / helpers ─────────────────────────────────────────────────

PORTFOLIO_VALUE = 1_000_000

# Ten-element returns array: a clean symmetric spread with slight positive drift
RETURNS_10 = np.array([-0.04, -0.03, -0.02, -0.01, 0.00,
                         0.01,  0.02,  0.03,  0.04,  0.05])


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _positions(*args) -> pd.DataFrame:
    """Build a positions DataFrame: _positions(("AAPL", 100, 0.5), ...)."""
    return pd.DataFrame(
        [{"ticker": t, "shares": s, "weight": w} for t, s, w in args]
    )


def _prices(**kwargs) -> pd.DataFrame:
    """Build a current_prices DataFrame: _prices(AAPL=200.0, JPM=100.0)."""
    return pd.DataFrame([{"ticker": t, "close": p} for t, p in kwargs.items()])


def _scenario(shocks: dict, **kw) -> Scenario:
    defaults = dict(
        name="test", description="unit test", scenario_type="hypothetical",
        holding_days=1,
    )
    defaults.update(kw)
    return Scenario(**defaults, shocks=shocks)


# ════════════════════════════════════════════════════════════════════════════
# historical_var
# ════════════════════════════════════════════════════════════════════════════

class TestHistoricalVaR:

    def test_output_keys(self):
        result = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert set(result) == {"var_pct", "var_dollar", "cvar_pct", "cvar_dollar"}

    def test_var_dollar_equals_pct_times_value(self):
        result = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert result["var_dollar"] == pytest.approx(result["var_pct"] * PORTFOLIO_VALUE)

    def test_cvar_dollar_equals_pct_times_value(self):
        result = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert result["cvar_dollar"] == pytest.approx(result["cvar_pct"] * PORTFOLIO_VALUE)

    def test_cvar_geq_var(self):
        result = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert result["cvar_pct"] >= result["var_pct"]

    def test_var_positive(self):
        result = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert result["var_pct"] > 0

    def test_99pct_var_geq_95pct_var(self):
        r95 = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        r99 = historical_var(RETURNS_10, 0.99, PORTFOLIO_VALUE)
        assert r99["var_pct"] >= r95["var_pct"]

    def test_known_value_90pct_confidence(self):
        # sorted RETURNS_10: [-0.04, -0.03, ..., 0.05]
        # cutoff = int(10 * 0.10) = 1 → var = abs(sorted[0]) = 0.04
        # CVaR = mean([-0.04]) = 0.04
        result = historical_var(RETURNS_10, 0.90, PORTFOLIO_VALUE)
        assert result["var_pct"] == pytest.approx(0.04)
        assert result["cvar_pct"] == pytest.approx(0.04)

    def test_known_value_80pct_confidence(self):
        # int(10 * (1 - 0.80)) = int(1.9999...) = 1 due to floating point,
        # so cutoff = max(1, 1) = 1 → var = abs(sorted[0]) = 0.04
        result = historical_var(RETURNS_10, 0.80, PORTFOLIO_VALUE)
        assert result["var_pct"] == pytest.approx(0.04)
        assert result["cvar_pct"] == pytest.approx(0.04)

    def test_single_observation(self):
        # cutoff forced to max(0, 1) = 1 → var = abs(sole element)
        result = historical_var(np.array([-0.05]), 0.95, PORTFOLIO_VALUE)
        assert result["var_pct"] == pytest.approx(0.05)
        assert result["cvar_pct"] == pytest.approx(0.05)

    def test_scales_with_portfolio_value(self):
        r1 = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        r2 = historical_var(RETURNS_10, 0.95, PORTFOLIO_VALUE * 5)
        assert r2["var_dollar"] == pytest.approx(r1["var_dollar"] * 5)
        # percentage unchanged
        assert r2["var_pct"] == pytest.approx(r1["var_pct"])

    def test_all_positive_returns_no_crash(self):
        # var still computed (smallest return in tail); function must not crash
        result = historical_var(np.array([0.01, 0.02, 0.03, 0.04, 0.05]), 0.95,
                                PORTFOLIO_VALUE)
        assert result["var_pct"] >= 0


# ════════════════════════════════════════════════════════════════════════════
# parametric_var
# ════════════════════════════════════════════════════════════════════════════

class TestParametricVaR:

    def test_output_keys(self):
        result = parametric_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert set(result) == {"var_pct", "var_dollar", "cvar_pct", "cvar_dollar"}

    def test_var_dollar_equals_pct_times_value(self):
        result = parametric_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert result["var_dollar"] == pytest.approx(result["var_pct"] * PORTFOLIO_VALUE)

    def test_cvar_geq_var(self):
        result = parametric_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        assert result["cvar_pct"] >= result["var_pct"]

    def test_99pct_var_greater_than_95pct(self):
        r95 = parametric_var(RETURNS_10, 0.95, PORTFOLIO_VALUE)
        r99 = parametric_var(RETURNS_10, 0.99, PORTFOLIO_VALUE)
        assert r99["var_pct"] > r95["var_pct"]

    def test_formula_applied_correctly(self):
        # Verify the function implements exactly: VaR = |µ + z·σ| and
        # CVaR = |-µ + σ·φ(z)/(1-α)|, so we can catch any formula inversion.
        conf   = 0.95
        result = parametric_var(RETURNS_10, conf, PORTFOLIO_VALUE)
        mu     = RETURNS_10.mean()
        sigma  = RETURNS_10.std(ddof=1)
        z      = stats.norm.ppf(1 - conf)
        assert result["var_pct"]  == pytest.approx(abs(mu + z * sigma))
        assert result["cvar_pct"] == pytest.approx(
            abs(-mu + sigma * stats.norm.pdf(z) / (1 - conf))
        )

    def test_higher_volatility_raises_var(self):
        low_vol  = np.full(100, 0.001)
        high_vol = np.linspace(-0.05, 0.05, 100)
        r_low  = parametric_var(low_vol,  0.95, PORTFOLIO_VALUE)
        r_high = parametric_var(high_vol, 0.95, PORTFOLIO_VALUE)
        assert r_high["var_pct"] > r_low["var_pct"]

    def test_zero_variance_var_equals_abs_mean(self):
        # All identical returns → sigma=0, VaR = |µ + z·0| = |µ|
        constant = 0.01
        result   = parametric_var(np.full(20, constant), 0.95, PORTFOLIO_VALUE)
        assert result["var_pct"] == pytest.approx(abs(constant))

    def test_symmetric_returns_cvar_formula(self):
        # Zero-mean symmetric returns: CVaR = σ·φ(z)/(1-α)
        symmetric = np.linspace(-0.03, 0.03, 200)     # mean ≈ 0
        conf      = 0.95
        result    = parametric_var(symmetric, conf, PORTFOLIO_VALUE)
        sigma     = symmetric.std(ddof=1)
        z         = stats.norm.ppf(1 - conf)
        expected_cvar = sigma * stats.norm.pdf(z) / (1 - conf)
        assert result["cvar_pct"] == pytest.approx(expected_cvar, rel=1e-4)


# ════════════════════════════════════════════════════════════════════════════
# monte_carlo_var
# ════════════════════════════════════════════════════════════════════════════

class TestMonteCarloVaR:

    def test_output_keys(self):
        result = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng())
        assert set(result) == {"var_pct", "var_dollar", "cvar_pct", "cvar_dollar"}

    def test_var_dollar_equals_pct_times_value(self):
        result = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng())
        assert result["var_dollar"] == pytest.approx(result["var_pct"] * PORTFOLIO_VALUE)

    def test_cvar_geq_var(self):
        result = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng())
        assert result["cvar_pct"] >= result["var_pct"]

    def test_var_positive(self):
        result = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng())
        assert result["var_pct"] > 0

    def test_99pct_var_geq_95pct_var(self):
        # Same seed → same 10 000 draws; 99th percentile loss ≥ 95th percentile loss
        r95 = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng(42))
        r99 = monte_carlo_var(RETURNS_10, 0.99, PORTFOLIO_VALUE, 10_000, _rng(42))
        assert r99["var_pct"] >= r95["var_pct"]

    def test_reproducible_with_same_seed(self):
        r1 = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng(42))
        r2 = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng(42))
        assert r1["var_pct"] == r2["var_pct"]
        assert r1["cvar_pct"] == r2["cvar_pct"]

    def test_different_seeds_produce_different_results(self):
        r1 = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng(1))
        r2 = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 10_000, _rng(2))
        assert r1["var_pct"] != r2["var_pct"]

    def test_converges_to_parametric(self):
        # With 100 000 sims drawn from the same N(µ,σ), the MC estimate should
        # be within 5% of the closed-form parametric VaR.
        rng_data     = np.random.default_rng(0)
        large_returns = rng_data.normal(loc=0.0, scale=0.01, size=500)
        mc    = monte_carlo_var(large_returns, 0.95, PORTFOLIO_VALUE, 100_000, _rng(99))
        param = parametric_var(large_returns, 0.95, PORTFOLIO_VALUE)
        assert mc["var_pct"] == pytest.approx(param["var_pct"], rel=0.05)

    def test_single_simulation_no_crash(self):
        # n_sims=1: cutoff forced to max(1,1)=1 → reads simulated[0]
        result = monte_carlo_var(RETURNS_10, 0.95, PORTFOLIO_VALUE, 1, _rng())
        assert result["var_pct"] >= 0
        assert "var_dollar" in result


# ════════════════════════════════════════════════════════════════════════════
# kupiec_pof_test
# ════════════════════════════════════════════════════════════════════════════

class TestKupiecPOFTest:

    def test_output_keys(self):
        result = kupiec_pof_test(5, 250, 0.99)
        assert {"statistic", "p_value", "reject_h0", "p_expected", "p_actual"} <= set(result)

    def test_zero_exceptions_not_rejected(self):
        result = kupiec_pof_test(0, 250, 0.99)
        assert result["statistic"] == 0.0
        assert result["p_value"]   == 1.0
        assert not result["reject_h0"]
        assert result["p_actual"]  == 0.0

    def test_all_exceptions_rejected(self):
        result = kupiec_pof_test(250, 250, 0.99)
        assert result["reject_h0"]
        assert result["p_value"]   == 0.0

    def test_perfectly_calibrated_not_rejected(self):
        # n_exceptions / n_obs = 1 - confidence exactly → LR = 0 → p = 1.0
        result = kupiec_pof_test(5, 500, 0.99)   # 5/500 = 0.01 = 1 - 0.99
        assert result["p_value"]   == pytest.approx(1.0)
        assert not result["reject_h0"]

    def test_over_exception_rate_rejected(self):
        # 50 exceptions in 250 obs → 20% rate vs 1% expected
        result = kupiec_pof_test(50, 250, 0.99)
        assert result["reject_h0"]
        assert result["p_value"]   < 0.001

    def test_extreme_under_exception_rejected(self):
        # 1 exception in 1000 obs at 95% conf → far too few (expected ~50)
        result = kupiec_pof_test(1, 1000, 0.95)
        assert result["reject_h0"]

    def test_p_expected_matches_confidence(self):
        assert kupiec_pof_test(5, 250, 0.99)["p_expected"] == pytest.approx(0.01)
        assert kupiec_pof_test(5, 250, 0.95)["p_expected"] == pytest.approx(0.05)

    def test_p_actual_computed_correctly(self):
        result = kupiec_pof_test(10, 200, 0.99)
        assert result["p_actual"] == pytest.approx(10 / 200)

    def test_statistic_nonnegative_across_inputs(self):
        for n_exc in [0, 1, 5, 10, 25]:
            assert kupiec_pof_test(n_exc, 250, 0.99)["statistic"] >= 0

    def test_p_value_in_unit_interval(self):
        for n_exc in [0, 2, 5, 10, 25, 50]:
            pv = kupiec_pof_test(n_exc, 250, 0.99)["p_value"]
            assert 0.0 <= pv <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# christoffersen_test
# ════════════════════════════════════════════════════════════════════════════

class TestChristoffersenTest:

    def test_single_observation_returns_nan(self):
        result = christoffersen_test(np.array([1], dtype=int))
        assert math.isnan(result["statistic"])
        assert math.isnan(result["p_value"])
        assert not result["reject_h0"]

    def test_output_keys(self):
        result = christoffersen_test(np.zeros(20, dtype=int))
        assert {"statistic", "p_value", "reject_h0", "pi01", "pi11"} <= set(result)

    def test_no_exceptions_not_rejected(self):
        # All zeros: pi01 = pi11 = 0, no clustering possible
        result = christoffersen_test(np.zeros(250, dtype=int))
        assert not result["reject_h0"]
        assert result["pi01"] == pytest.approx(0.0)
        assert result["pi11"] == pytest.approx(0.0)

    def test_all_exceptions_not_rejected(self):
        # All ones: trivially "serial", but the test has nothing to compare against
        # (n01=n10=n00=0), LR collapses to 0 → not rejected
        result = christoffersen_test(np.ones(50, dtype=int))
        assert not result["reject_h0"]
        assert result["pi11"] == pytest.approx(1.0)

    def test_clustered_exceptions_rejected(self):
        # Two large blocks of exceptions: pi11 >> pi01 → strong clustering
        # [0]*20 + [1]*20 + [0]*20 + [1]*20
        # n00=38, n01=2, n10=1, n11=38 → pi01≈0.05, pi11≈0.97
        exceptions = np.array([0]*20 + [1]*20 + [0]*20 + [1]*20, dtype=int)
        result = christoffersen_test(exceptions)
        assert result["reject_h0"]
        assert result["pi11"] > result["pi01"]

    def test_clustered_pi11_greater_than_pi01(self):
        exceptions = np.array([0]*20 + [1]*20 + [0]*20 + [1]*20, dtype=int)
        result = christoffersen_test(exceptions)
        # pi11 ≈ 38/39 ≈ 0.97,  pi01 = 2/40 = 0.05
        assert result["pi11"] == pytest.approx(38 / 39, rel=1e-4)
        assert result["pi01"] == pytest.approx(2 / 40,  rel=1e-4)

    def test_p_value_in_unit_interval(self):
        exceptions = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=int)
        result = christoffersen_test(exceptions)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_two_observations_no_crash(self):
        result = christoffersen_test(np.array([0, 1], dtype=int))
        assert "p_value" in result


# ════════════════════════════════════════════════════════════════════════════
# apply_scenario
# ════════════════════════════════════════════════════════════════════════════

class TestApplyScenario:
    # 100 AAPL @ $200 = $20 000,  50 JPM @ $100 = $5 000  → portfolio = $25 000
    _pos    = _positions(("AAPL", 100, 0.80), ("JPM", 50, 0.20))
    _prices = _prices(AAPL=200.0, JPM=100.0)

    def test_output_keys(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        assert set(out) == {"results", "summary"}

    def test_position_pnl_formula(self):
        # AAPL: 100 × $200 × −10% = −$2 000
        # JPM:  50  × $100 × −5%  = −$250
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        by = {r["ticker"]: r for r in out["results"]}
        assert by["AAPL"]["position_pnl"] == pytest.approx(-2_000.0)
        assert by["JPM"]["position_pnl"]  == pytest.approx(-250.0)

    def test_position_value_formula(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        by = {r["ticker"]: r for r in out["results"]}
        assert by["AAPL"]["position_value"] == pytest.approx(20_000.0)
        assert by["JPM"]["position_value"]  == pytest.approx(5_000.0)

    def test_stressed_value_per_position(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        for r in out["results"]:
            assert r["stressed_value"] == pytest.approx(
                r["position_value"] + r["position_pnl"]
            )

    def test_total_pnl_equals_sum_of_positions(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        pos_sum = sum(r["position_pnl"] for r in out["results"])
        assert out["summary"]["total_pnl"] == pytest.approx(pos_sum)

    def test_portfolio_stressed_value(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        s = out["summary"]
        assert s["stressed_value"] == pytest.approx(s["portfolio_value"] + s["total_pnl"])

    def test_pct_contribution_sums_to_100(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        total = sum(r["pct_contribution"] for r in out["results"])
        assert total == pytest.approx(100.0, abs=0.01)

    def test_total_pnl_pct_formula(self):
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        s   = out["summary"]
        expected_pct = s["total_pnl"] / s["portfolio_value"] * 100
        assert s["total_pnl_pct"] == pytest.approx(expected_pct, abs=0.01)

    def test_shock_pct_stored_as_percentage(self):
        # shock=-0.10 must be stored as -10.0, not -0.10
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        by = {r["ticker"]: r for r in out["results"]}
        assert by["AAPL"]["shock_pct"] == pytest.approx(-10.0)
        assert by["JPM"]["shock_pct"]  == pytest.approx(-5.0)

    def test_worst_position_identified_correctly(self):
        # AAPL loss −$2 000 > JPM loss −$250 → AAPL is worst
        out = apply_scenario(_scenario({"AAPL": -0.10, "JPM": -0.05}),
                             self._pos, self._prices)
        assert out["summary"]["worst_position"] == "AAPL"

    def test_zero_shock_gives_zero_pnl(self):
        out = apply_scenario(_scenario({"AAPL": 0.0, "JPM": 0.0}),
                             self._pos, self._prices)
        assert out["summary"]["total_pnl"] == pytest.approx(0.0)
        for r in out["results"]:
            assert r["position_pnl"] == pytest.approx(0.0)

    def test_positive_shock_gives_positive_pnl(self):
        out = apply_scenario(_scenario({"AAPL": +0.10, "JPM": +0.05}),
                             self._pos, self._prices)
        assert out["summary"]["total_pnl"] > 0

    def test_single_position(self):
        pos    = _positions(("AAPL", 100, 1.0))
        prices = _prices(AAPL=200.0)
        out    = apply_scenario(_scenario({"AAPL": -0.20}), pos, prices)
        # 100 × $200 × −20% = −$4 000
        assert out["summary"]["total_pnl"] == pytest.approx(-4_000.0)
        assert out["summary"]["worst_position"] == "AAPL"

    def test_ticker_absent_from_shocks_is_skipped(self):
        # JPM has no entry in shocks → skipped; only AAPL counted
        out = apply_scenario(_scenario({"AAPL": -0.10}), self._pos, self._prices)
        tickers = [r["ticker"] for r in out["results"]]
        assert "JPM" not in tickers
        assert out["summary"]["portfolio_value"] == pytest.approx(20_000.0)


# ════════════════════════════════════════════════════════════════════════════
# compute_log_returns
# ════════════════════════════════════════════════════════════════════════════

class TestComputeLogReturns:

    def _prices_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"date": "2024-01-02", "ticker": "AAPL", "close": 100.0},
            {"date": "2024-01-03", "ticker": "AAPL", "close": 110.0},
            {"date": "2024-01-04", "ticker": "AAPL", "close": 105.0},
            {"date": "2024-01-02", "ticker": "JPM",  "close":  50.0},
            {"date": "2024-01-03", "ticker": "JPM",  "close":  55.0},
            {"date": "2024-01-04", "ticker": "JPM",  "close":  52.5},
        ])

    def test_output_columns(self):
        df = compute_log_returns(self._prices_df())
        assert set(df.columns) == {"date", "ticker", "log_ret"}

    def test_row_count_drops_first_date(self):
        # 3 dates × 2 tickers → 2 return rows × 2 tickers = 4 total
        df = compute_log_returns(self._prices_df())
        assert len(df) == 4

    def test_aapl_log_returns_correct(self):
        df   = compute_log_returns(self._prices_df())
        aapl = df[df["ticker"] == "AAPL"].sort_values("date")["log_ret"].values
        assert aapl[0] == pytest.approx(math.log(110.0 / 100.0))
        assert aapl[1] == pytest.approx(math.log(105.0 / 110.0))

    def test_jpm_log_returns_correct(self):
        df  = compute_log_returns(self._prices_df())
        jpm = df[df["ticker"] == "JPM"].sort_values("date")["log_ret"].values
        assert jpm[0] == pytest.approx(math.log(55.0  / 50.0))
        assert jpm[1] == pytest.approx(math.log(52.5  / 55.0))

    def test_no_nan_in_output(self):
        df = compute_log_returns(self._prices_df())
        assert not df["log_ret"].isna().any()

    def test_single_ticker_two_dates(self):
        prices = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "SPY", "close": 400.0},
            {"date": "2024-01-03", "ticker": "SPY", "close": 404.0},
        ])
        df = compute_log_returns(prices)
        assert len(df) == 1
        assert df.iloc[0]["ticker"]  == "SPY"
        assert df.iloc[0]["log_ret"] == pytest.approx(math.log(404.0 / 400.0))

    def test_flat_prices_give_zero_return(self):
        prices = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "XOM", "close": 80.0},
            {"date": "2024-01-03", "ticker": "XOM", "close": 80.0},
        ])
        df = compute_log_returns(prices)
        assert df.iloc[0]["log_ret"] == pytest.approx(0.0)

    def test_log_return_not_simple_return(self):
        # +10% simple return: simple = 0.10, log = ln(1.1) ≈ 0.0953
        prices = pd.DataFrame([
            {"date": "2024-01-02", "ticker": "JNJ", "close": 100.0},
            {"date": "2024-01-03", "ticker": "JNJ", "close": 110.0},
        ])
        df = compute_log_returns(prices)
        log_ret = df.iloc[0]["log_ret"]
        assert log_ret == pytest.approx(math.log(1.1))
        assert log_ret != pytest.approx(0.10)   # not the simple return

    def test_output_sorted_by_date_within_ticker(self):
        df   = compute_log_returns(self._prices_df())
        aapl = df[df["ticker"] == "AAPL"]["date"].tolist()
        assert aapl == sorted(aapl)
