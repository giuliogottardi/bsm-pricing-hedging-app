import math
import numpy as np
import pytest

import core


def test_normal_cdf_against_erfc_grid():
    xs = np.linspace(-8, 8, 2001)
    ref = np.array([0.5 * math.erfc(-float(x) / math.sqrt(2)) for x in xs])
    err = np.max(np.abs(core.norm_cdf(xs) - ref))
    # A&S 26.2.17 approximation has a small, nonzero approximation error.
    assert err < 8e-8


@pytest.mark.parametrize("q", [0.0, 0.025])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
def test_put_call_parity(q, T):
    S, K, r, vol = 102.0, 100.0, 0.035, 0.24
    c = core.BSMCall(S, K, T, r, vol, q)
    p = core.BSMPut(S, K, T, r, vol, q)
    rhs = S * math.exp(-q*T) - K * math.exp(-r*T)
    assert abs((c-p)-rhs) < 2e-6


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_finite_difference_greeks_close_to_analytic(option_type):
    analytic = core.analytic_greeks(100, 100, 0.8, 0.03, 0.22, option_type, q=0.015)
    fd = core.compute_all_greeks(100, 100, 0.8, 0.03, 0.22, option_type, 1e-4, q=0.015)
    tolerances = {"Delta": 2e-5, "Gamma": 2e-5, "Theta": 2e-3, "Vega": 2e-4, "Rho": 1e-3}
    for greek, tol in tolerances.items():
        assert abs(float(fd[greek])-analytic[greek]) < tol, (greek, fd[greek], analytic[greek])


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_round_trip(option_type):
    S, K, T, r, q, vol = 100.0, 103.0, 0.7, 0.035, 0.01, 0.31
    price = core.bsm_price(S, K, T, r, vol, option_type, q)
    out = core.implied_vol_bisection(price, S, K, T, r, option_type, q=q, return_details=True)
    assert out["status"] == "converged"
    assert abs(out["iv"] - vol) < 2e-5
    assert abs(out["price_residual"]) < 1e-7


def test_iv_rejects_price_outside_bounds():
    out = core.implied_vol_bisection(150.0, 100.0, 100.0, 1.0, 0.03, "call", return_details=True)
    assert math.isnan(out["iv"])
    assert out["status"] == "outside_no_arbitrage_bounds"


def test_yield_curve_interpolation():
    assert core.risk_free_rate_for_maturity({0.25: 0.02, 1.0: 0.05}, 0.625) == pytest.approx(0.035)


def test_synthetic_chain_iv_surface_round_trip(monkeypatch):
    monkeypatch.setattr(core, "YFINANCE_AVAILABLE", False)
    curve = {0.25: 0.04, 1.0: 0.04, 2.0: 0.04, 5.0: 0.04, 10.0: 0.04, 30.0: 0.04}
    result = core.build_iv_surface("TEST_NO_NETWORK", 100.0, 0.04, 25, q=0.0, curve=curve)
    assert result["chain"]["source"].startswith("synthetic")
    assert len(result["implied_rows"]) > 20
    errors = [abs(row["iv"] - result["chain"]["true_vols"][(row["T"], row["K"], row["option_type"])])
              for row in result["implied_rows"]]
    assert max(errors) < 2e-5


def test_mc_reproducible_and_vectorized():
    args = (100.0, 100.0, 0.25, 0.03, 0.2, "call", 1e-4, "weekly", 3, 100)
    p1, _ = core.run_mc_batch(*args, seed=123, keep_paths=0)
    p2, _ = core.run_mc_batch(*args, seed=123, keep_paths=0)
    assert np.allclose(p1, p2)
    assert np.isfinite(p1).all()


def test_hedging_scenario_accepts_realized_vol_and_costs():
    out = core.run_delta_hedge_simulation(100, 100, 0.25, 0.03, 0.2, "call", 1e-4,
        "weekly", 3, rng=np.random.default_rng(10), realized_vol=0.3, real_drift=0.08,
        transaction_cost_bps=2.0)
    assert out["transaction_costs"] > 0
    assert out["realized_vol"] == pytest.approx(0.3)
    assert np.isfinite(out["hedging_pnl"])


def test_low_vega_quote_is_flagged_as_ill_conditioned():
    out = core.implied_vol_bisection(0.001, 100.0, 180.0, 0.05, 0.03, "call", return_details=True)
    assert out["status"] == "ill_conditioned_low_vega"


def test_surface_can_filter_selected_option_type(monkeypatch):
    monkeypatch.setattr(core, "YFINANCE_AVAILABLE", False)
    curve = {0.25: 0.04, 1.0: 0.04, 2.0: 0.04, 5.0: 0.04, 10.0: 0.04, 30.0: 0.04}
    result = core.build_iv_surface("TEST_NO_NETWORK", 100.0, 0.04, 20,
                                   q=0.0, curve=curve, option_type="put")
    assert result["implied_rows"]
    assert {row["option_type"] for row in result["implied_rows"]} == {"put"}
