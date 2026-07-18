"""
Core BSM pricing / Greeks / hedging / implied-vol logic.

Ported from Giulio's QFM notebook (Deliverable 2 - Proprietary Pricing Engine).
Pure computation only - no I/O, no plotting, no interactive input - so it can
be safely imported and cached by the Streamlit app.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

ArrayOrFloat = Union[float, np.ndarray]


# --------------------------------------------------------------------------
# 1. Market data
# --------------------------------------------------------------------------
def fetch_price_history(ticker: str, lookback_days: int, seed: int = 42):
    """Daily close prices for `ticker`; falls back to synthetic GBM if offline."""
    calendar_days_needed = int(lookback_days * 1.6) + 30
    source = "synthetic"

    if YFINANCE_AVAILABLE:
        try:
            raw = yf.download(ticker, period=f"{calendar_days_needed}d",
                               progress=False, auto_adjust=True)
            if raw is not None and len(raw) > 5:
                close = raw[["Close"]].dropna()
                close.columns = ["Close"]
                return close.tail(lookback_days + 1), "yfinance"
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    n = lookback_days + 1
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    n = len(dates)  # bdate_range can return a count that differs slightly from the request
    s0, mu, sigma_synth = 100.0, 0.08, 0.30
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma_synth ** 2) * dt, sigma_synth * np.sqrt(dt), size=n - 1)
    log_path = np.concatenate([[np.log(s0)], np.log(s0) + np.cumsum(shocks)])
    prices = np.exp(log_path)
    return pd.DataFrame({"Close": prices}, index=dates), source


def compute_volatility(price_history: pd.DataFrame, rolling_window: int = 21):
    log_returns = np.log(price_history["Close"] / price_history["Close"].shift(1)).dropna()
    daily_vol = log_returns.std()
    annualized_vol = float(daily_vol * np.sqrt(252))
    rolling_vol_annualized = log_returns.rolling(rolling_window).std() * np.sqrt(252)
    return log_returns, rolling_vol_annualized, annualized_vol


# --------------------------------------------------------------------------
# 1b. Live risk-free rate curve (Treasury yield proxies)
# --------------------------------------------------------------------------
# Maturity (years) -> Yahoo Finance ticker for the closest benchmark yield.
TREASURY_TICKERS = {
    0.25: "^IRX",   # 13-week T-bill discount rate
    5.0: "^FVX",    # 5-year Treasury yield
    10.0: "^TNX",   # 10-year Treasury yield
    30.0: "^TYX",   # 30-year Treasury yield
}

# Fallback curve (annualized decimal rates) used if live data is unavailable.
_FALLBACK_CURVE = {0.25: 0.045, 5.0: 0.040, 10.0: 0.042, 30.0: 0.043}


def fetch_risk_free_curve():
    """Fetch the latest Treasury yield curve points; falls back to a static curve.

    Returns
    -------
    tuple
        (curve: dict[float, float] maturity -> annualized decimal rate, source: str)
    """
    if YFINANCE_AVAILABLE:
        try:
            curve = {}
            for T, tkr in TREASURY_TICKERS.items():
                hist = yf.Ticker(tkr).history(period="5d")
                if hist is not None and len(hist) > 0:
                    curve[T] = float(hist["Close"].iloc[-1]) / 100.0
            if len(curve) >= 2:
                return curve, "yfinance"
        except Exception:
            pass
    return dict(_FALLBACK_CURVE), "fallback"


def risk_free_rate_for_maturity(curve: dict, T: float) -> float:
    """Linearly interpolate (flat-extrapolate at the ends) the yield curve at maturity T."""
    xs = np.array(sorted(curve.keys()))
    ys = np.array([curve[x] for x in xs])
    return float(np.interp(T, xs, ys))


# --------------------------------------------------------------------------
# 2. Custom Normal CDF (Abramowitz & Stegun 26.2.17) - no scipy
# --------------------------------------------------------------------------
def norm_cdf_scalar(x: float) -> float:
    p = 0.2316419
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    sign = 1.0
    z = x
    if z < 0:
        sign = -1.0
        z = -z
    k = 1.0 / (1.0 + p * z)
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
    cdf_positive = 1.0 - phi_z * poly
    return cdf_positive if sign > 0 else 1.0 - cdf_positive


def norm_pdf_scalar(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


norm_cdf = np.vectorize(norm_cdf_scalar)
norm_pdf = np.vectorize(norm_pdf_scalar)

NORMAL_CDF_REFERENCE = {
    0.0: 0.500000, 1.0: 0.841345, -1.0: 0.158655,
    1.96: 0.975002, -1.96: 0.024998, 2.33: 0.990097,
}


def validate_norm_cdf():
    """Returns (max_abs_error, rows) against textbook reference values."""
    rows = []
    max_error = 0.0
    for x_val, ref in NORMAL_CDF_REFERENCE.items():
        approx = norm_cdf_scalar(x_val)
        err = abs(approx - ref)
        max_error = max(max_error, err)
        rows.append((x_val, approx, ref, err))
    return max_error, rows


# --------------------------------------------------------------------------
# 3. BSM pricing engine
# --------------------------------------------------------------------------
def _d1_d2(S, K, T, r, vol):
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    T_safe = np.where(T > 1e-12, T, 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T_safe) / (vol * np.sqrt(T_safe))
    d2 = d1 - vol * np.sqrt(T_safe)
    return d1, d2


def BSMCall(S, K, T, r, vol):
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1, d2 = _d1_d2(S, K, T, r, vol)
    price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    intrinsic = np.maximum(S - K, 0.0)
    price = np.where(T > 1e-12, price, intrinsic)
    return float(price) if price.ndim == 0 else price


def BSMPut(S, K, T, r, vol):
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    d1, d2 = _d1_d2(S, K, T, r, vol)
    price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    intrinsic = np.maximum(K - S, 0.0)
    price = np.where(T > 1e-12, price, intrinsic)
    return float(price) if price.ndim == 0 else price


def bsm_price(S, K, T, r, vol, option_type):
    if option_type == "call":
        return BSMCall(S, K, T, r, vol)
    elif option_type == "put":
        return BSMPut(S, K, T, r, vol)
    raise ValueError(f"Unknown option_type: {option_type!r}")


# --------------------------------------------------------------------------
# 4. Numerical (finite-difference) Greeks
# --------------------------------------------------------------------------
def _rel_bump(x, step: float):
    x = np.asarray(x, dtype=float)
    bumped = np.maximum(np.abs(x) * step, step)
    return float(bumped) if bumped.ndim == 0 else bumped


def numerical_delta(S, K, T, r, vol, option_type, step):
    h = _rel_bump(S, step)
    up = bsm_price(S + h, K, T, r, vol, option_type)
    down = bsm_price(S - h, K, T, r, vol, option_type)
    return (up - down) / (2 * h)


def numerical_gamma(S, K, T, r, vol, option_type, step):
    h = _rel_bump(S, step)
    up = bsm_price(S + h, K, T, r, vol, option_type)
    mid = bsm_price(S, K, T, r, vol, option_type)
    down = bsm_price(S - h, K, T, r, vol, option_type)
    return (up - 2 * mid + down) / (h ** 2)


def numerical_vega(S, K, T, r, vol, option_type, step):
    h = _rel_bump(vol, step)
    up = bsm_price(S, K, T, r, vol + h, option_type)
    down = bsm_price(S, K, T, r, vol - h, option_type)
    return (up - down) / (2 * h)


def numerical_rho(S, K, T, r, vol, option_type, step):
    h = _rel_bump(r, step)
    up = bsm_price(S, K, T, r + h, vol, option_type)
    down = bsm_price(S, K, T, r - h, vol, option_type)
    return (up - down) / (2 * h)


def numerical_theta(S, K, T, r, vol, option_type, step):
    T = np.asarray(T, dtype=float)
    h = _rel_bump(T, step)
    h = np.where(T > 0, np.minimum(h, T * 0.5), h)
    price_now = bsm_price(S, K, T, r, vol, option_type)
    price_later = bsm_price(S, K, np.maximum(T - h, 1e-12), r, vol, option_type)
    return (price_later - price_now) / h


def compute_all_greeks(S, K, T, r, vol, option_type, step):
    return {
        "Delta": numerical_delta(S, K, T, r, vol, option_type, step),
        "Gamma": numerical_gamma(S, K, T, r, vol, option_type, step),
        "Theta": numerical_theta(S, K, T, r, vol, option_type, step),
        "Vega": numerical_vega(S, K, T, r, vol, option_type, step),
        "Rho": numerical_rho(S, K, T, r, vol, option_type, step),
    }


# --------------------------------------------------------------------------
# 5. Delta-hedging Monte Carlo
# --------------------------------------------------------------------------
def simulate_underlying_path(S0, mu, vol, T, n_steps, rng):
    dt = T / n_steps
    shocks = rng.normal((mu - 0.5 * vol ** 2) * dt, vol * np.sqrt(dt), size=n_steps)
    log_path = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(shocks)])
    return np.exp(log_path)


def rebalancing_step_size(rebalancing_frequency: str, custom_rebalancing_days: int) -> int:
    mapping = {"daily": 1, "weekly": 5, "monthly": 21}
    if rebalancing_frequency == "custom":
        return custom_rebalancing_days
    return mapping[rebalancing_frequency]


def run_delta_hedge_simulation(S0, K, T, r, vol, option_type, step,
                                rebalancing_frequency, custom_rebalancing_days,
                                n_daily_steps=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_daily_steps is None:
        n_daily_steps = max(int(round(T * 252)), 2)

    rebal_every = rebalancing_step_size(rebalancing_frequency, custom_rebalancing_days)
    price_path = simulate_underlying_path(S0, r, vol, T, n_daily_steps, rng)
    time_grid = np.linspace(0, T, n_daily_steps + 1)
    dt = T / n_daily_steps

    option_premium = bsm_price(S0, K, T, r, vol, option_type)
    delta_0 = numerical_delta(S0, K, T, r, vol, option_type, step)

    shares_held = delta_0
    cash = option_premium - shares_held * S0

    portfolio_values = np.empty(n_daily_steps + 1)
    delta_path = np.empty(n_daily_steps + 1)
    cash_path = np.empty(n_daily_steps + 1)
    delta_path[0] = delta_0
    cash_path[0] = cash
    portfolio_values[0] = -option_premium + shares_held * S0 + cash

    for i in range(1, n_daily_steps + 1):
        cash *= np.exp(r * dt)
        t_now = time_grid[i]
        T_remaining = max(T - t_now, 1e-12)
        S_now = price_path[i]
        option_value_now = bsm_price(S_now, K, T_remaining, r, vol, option_type)

        if i == n_daily_steps or (i % rebal_every == 0):
            new_delta = numerical_delta(S_now, K, T_remaining, r, vol, option_type, step) \
                if i < n_daily_steps else 0.0
            trade = new_delta - shares_held
            cash -= trade * S_now
            shares_held = new_delta

        delta_path[i] = shares_held
        cash_path[i] = cash
        portfolio_values[i] = -option_value_now + shares_held * S_now + cash

    hedging_pnl = portfolio_values[-1]
    return {
        "time_grid": time_grid, "price_path": price_path,
        "portfolio_values": portfolio_values, "cash_path": cash_path,
        "delta_path": delta_path, "hedging_pnl": hedging_pnl,
    }


def run_mc_batch(S0, K, T, r, vol, option_type, step, rebalancing_frequency,
                  custom_rebalancing_days, n_sims, seed=2024, keep_paths=30):
    mc_rng = np.random.default_rng(seed)
    pnls = np.empty(n_sims)
    kept_paths = []
    for i in range(n_sims):
        res = run_delta_hedge_simulation(
            S0, K, T, r, vol, option_type, step,
            rebalancing_frequency, custom_rebalancing_days, rng=mc_rng,
        )
        pnls[i] = res["hedging_pnl"]
        if i < keep_paths:
            kept_paths.append(res["portfolio_values"])
    return pnls, kept_paths


# --------------------------------------------------------------------------
# 6. Implied volatility (option chain + bisection inversion)
# --------------------------------------------------------------------------
def fetch_option_chain(ticker: str, spot: float, risk_free_rate: float,
                        max_expirations: int = 6, seed: int = 7):
    if YFINANCE_AVAILABLE:
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options[:max_expirations]
            if not expirations:
                raise ValueError("No expirations returned.")
            rows = []
            today = pd.Timestamp.today().normalize()
            for exp_str in expirations:
                T = (pd.Timestamp(exp_str) - today).days / 365.0
                if T <= 0.02:
                    continue
                calls = tk.option_chain(exp_str).calls
                calls = calls[(calls["strike"] > 0.7 * spot) & (calls["strike"] < 1.3 * spot)]
                for _, row in calls.iterrows():
                    bid, ask, last = row.get("bid", 0), row.get("ask", 0), row.get("lastPrice", 0)
                    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                    if mid and mid > 0.01:
                        rows.append((T, float(row["strike"]), float(mid)))
            if len(rows) < 10:
                raise ValueError("Too few usable quotes returned.")
            return {"rows": rows, "true_vols": None, "source": "yfinance"}
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    synth_maturities = [0.08, 0.25, 0.5, 1.0, 1.5, 2.0]
    synth_strikes = np.arange(round(0.6 * spot), round(1.4 * spot) + 1, max(1, round(spot * 0.04)))
    rows, true_vols = [], {}
    for T in synth_maturities:
        for K in synth_strikes:
            moneyness = np.log(K / spot)
            true_vol = 0.22 + 0.055 * moneyness ** 2 - 0.05 * moneyness - 0.03 * np.sqrt(T)
            true_vol = float(max(true_vol, 0.03))
            price = float(BSMCall(spot, K, T, risk_free_rate, true_vol))
            if price > 0.01:
                rows.append((T, float(K), price))
                true_vols[(T, float(K))] = true_vol
    return {"rows": rows, "true_vols": true_vols, "source": "synthetic"}


def implied_vol_bisection(market_price, S, K, T, r, option_type="call",
                           vol_lo=1e-4, vol_hi=5.0, tol=1e-6, max_iter=100):
    price_lo = bsm_price(S, K, T, r, vol_lo, option_type) - market_price
    price_hi = bsm_price(S, K, T, r, vol_hi, option_type) - market_price
    if price_lo * price_hi > 0:
        return np.nan
    vol_mid = 0.5 * (vol_lo + vol_hi)
    for _ in range(max_iter):
        vol_mid = 0.5 * (vol_lo + vol_hi)
        price_mid = bsm_price(S, K, T, r, vol_mid, option_type) - market_price
        if abs(price_mid) < tol:
            return vol_mid
        if price_lo * price_mid < 0:
            vol_hi = vol_mid
        else:
            vol_lo, price_lo = vol_mid, price_mid
    return vol_mid


def build_iv_surface(ticker, spot, risk_free_rate, n_price_points, max_expirations=6):
    chain = fetch_option_chain(ticker, spot, risk_free_rate, max_expirations)
    implied_rows = []
    for T, K, market_price in chain["rows"]:
        iv = implied_vol_bisection(market_price, spot, K, T, risk_free_rate, "call")
        if np.isfinite(iv):
            implied_rows.append((T, K, iv))

    by_maturity = defaultdict(list)
    for T, K, iv in implied_rows:
        by_maturity[T].append((K, iv))

    vol_T_axis = np.array(sorted(by_maturity.keys()))
    vol_K_axis = np.linspace(0.7 * spot, 1.3 * spot, n_price_points)
    vol_surface_grid = np.full((len(vol_T_axis), len(vol_K_axis)), np.nan)
    for i, T in enumerate(vol_T_axis):
        pts = sorted(by_maturity[T])
        strikes_i = np.array([p[0] for p in pts])
        vols_i = np.array([p[1] for p in pts])
        vol_surface_grid[i, :] = np.interp(vol_K_axis, strikes_i, vols_i)

    return {
        "chain": chain, "implied_rows": implied_rows,
        "T_axis": vol_T_axis, "K_axis": vol_K_axis,
        "grid": vol_surface_grid,
    }
