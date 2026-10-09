"""Black–Scholes–Merton pricing, numerical risk, IV inversion and hedging tools.

The core intentionally implements the Normal CDF without SciPy. Market data access
is isolated in fetch_* functions; all numerical routines are deterministic given
inputs/seeds. The model prices European options (not American exercise).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Union, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:  # offline / minimal test environment
    YFINANCE_AVAILABLE = False

ArrayOrFloat = Union[float, np.ndarray]

# ---------------------------------------------------------------------------
# Market data and yield curve
# ---------------------------------------------------------------------------
def fetch_price_history(ticker: str, lookback_days: int, seed: int = 42):
    """Return adjusted daily close data and source label; use reproducible GBM offline."""
    calendar_days_needed = int(lookback_days * 1.6) + 30
    if YFINANCE_AVAILABLE:
        try:
            raw = yf.download(ticker, period=f"{calendar_days_needed}d", progress=False,
                              auto_adjust=True, threads=False)
            if raw is not None and len(raw) > 5 and "Close" in raw:
                close = raw[["Close"]].dropna().tail(lookback_days + 1)
                if not close.empty:
                    close.columns = ["Close"]
                    return close, "yfinance"
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=lookback_days + 1)
    mu, vol, dt = 0.08, 0.30, 1 / 252
    shocks = rng.normal((mu - 0.5 * vol**2) * dt, vol * np.sqrt(dt), len(dates) - 1)
    prices = np.exp(np.r_[np.log(100.0), np.log(100.0) + np.cumsum(shocks)])
    return pd.DataFrame({"Close": prices}, index=dates), "synthetic"


def compute_volatility(price_history: pd.DataFrame, rolling_window: int = 21):
    log_returns = np.log(price_history["Close"] / price_history["Close"].shift(1)).dropna()
    annualized_vol = float(log_returns.std(ddof=1) * np.sqrt(252)) if len(log_returns) > 1 else 0.0
    rolling = log_returns.rolling(rolling_window).std() * np.sqrt(252)
    return log_returns, rolling, annualized_vol

TREASURY_TICKERS = {0.25: "^IRX", 2.0: "^UST2Y", 5.0: "^FVX", 10.0: "^TNX", 30.0: "^TYX"}
_FALLBACK_CURVE = {0.25: 0.045, 2.0: 0.041, 5.0: 0.040, 10.0: 0.042, 30.0: 0.043}


def _treasury_quote_to_continuous_rate(ticker: str, quoted_percent: float) -> float:
    """Approximate a continuously-compounded zero rate from Treasury yield proxies.

    Yahoo's ^IRX is a bank-discount yield proxy, not a zero-coupon continuously
    compounded rate. For consistency, treat quotes as annual bond-equivalent yields
    and convert with log(1+y). This is a documented approximation, not a bootstrapped curve.
    """
    y = float(quoted_percent) / 100.0
    if ticker == "^IRX":
        # ^IRX is quoted on a bank-discount basis; convert discount yield to a
        # simple annual investment yield using the standard 13-week approximation.
        d = np.clip(y, -0.99, 0.99)
        y = d / (1.0 - d * 91.0 / 360.0)
    return float(np.log1p(y))


def fetch_risk_free_curve():
    """Fetch proxy yields; returns approximate continuously compounded rates."""
    if YFINANCE_AVAILABLE:
        curve = {}
        for T, ticker in TREASURY_TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if hist is not None and not hist.empty:
                    curve[T] = _treasury_quote_to_continuous_rate(ticker, float(hist["Close"].iloc[-1]))
            except Exception:
                continue
        if len(curve) >= 2:
            return curve, "yfinance yield proxies (converted approximation)"
    return dict(_FALLBACK_CURVE), "fallback continuous-rate approximation"


def risk_free_rate_for_maturity(curve: dict, T: float) -> float:
    """Linearly interpolate zero-rate proxies; flat extrapolation outside curve nodes."""
    if not curve:
        raise ValueError("Yield curve cannot be empty")
    xs = np.array(sorted(float(x) for x in curve))
    ys = np.array([float(curve[x]) for x in xs])
    return float(np.interp(float(T), xs, ys))

# ---------------------------------------------------------------------------
# Custom Normal CDF/PDF (Abramowitz & Stegun 26.2.17)
# ---------------------------------------------------------------------------
def norm_cdf_scalar(x: float) -> float:
    z = abs(float(x))
    p = 0.2316419
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    k = 1.0 / (1.0 + p * z)
    poly = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    positive = 1.0 - math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * poly
    return positive if x >= 0 else 1.0 - positive


def norm_cdf(x: ArrayOrFloat):
    """Vectorized A&S CDF; returns float for scalar input and ndarray otherwise."""
    arr = np.asarray(x, dtype=float)
    z = np.abs(arr)
    p = 0.2316419
    k = 1.0 / (1.0 + p * z)
    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))))
    positive = 1.0 - np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * poly
    out = np.where(arr >= 0, positive, 1.0 - positive)
    return float(out) if out.ndim == 0 else out


def norm_pdf_scalar(x: float) -> float:
    return math.exp(-0.5 * float(x) ** 2) / math.sqrt(2.0 * math.pi)


def norm_pdf(x: ArrayOrFloat):
    arr = np.asarray(x, dtype=float)
    out = np.exp(-0.5 * arr**2) / np.sqrt(2.0 * np.pi)
    return float(out) if out.ndim == 0 else out

NORMAL_CDF_REFERENCE = {0.0: 0.500000, 1.0: 0.841345, -1.0: 0.158655,
                         1.96: 0.975002, -1.96: 0.024998, 2.33: 0.990097}


def validate_norm_cdf():
    rows, max_error = [], 0.0
    for x, ref in NORMAL_CDF_REFERENCE.items():
        approx = norm_cdf_scalar(x)
        err = abs(approx - ref)
        max_error = max(max_error, err)
        rows.append((x, approx, ref, err))
    return max_error, rows

# ---------------------------------------------------------------------------
# BSM pricing and analytical Greeks
# ---------------------------------------------------------------------------
def _validate_inputs(S, K, T, vol):
    if np.any(np.asarray(S) <= 0) or K <= 0:
        raise ValueError("Spot and strike must be strictly positive")
    if np.any(np.asarray(T) < 0):
        raise ValueError("Maturity cannot be negative")
    if np.any(np.asarray(vol) <= 0):
        raise ValueError("Volatility must be strictly positive")


def _d1_d2(S, K, T, r, vol, q=0.0):
    S, T = np.asarray(S, float), np.asarray(T, float)
    _validate_inputs(S, K, T, vol)
    ts = np.maximum(T, 1e-14)
    root = np.asarray(vol, float) * np.sqrt(ts)
    d1 = (np.log(S / K) + (np.asarray(r) - np.asarray(q) + 0.5 * np.asarray(vol)**2) * ts) / root
    return d1, d1 - root


def BSMCall(S, K, T, r, vol, q=0.0):
    S, T = np.asarray(S, float), np.asarray(T, float)
    d1, d2 = _d1_d2(S, K, T, r, vol, q)
    price = S * np.exp(-np.asarray(q) * T) * norm_cdf(d1) - K * np.exp(-np.asarray(r) * T) * norm_cdf(d2)
    price = np.where(T > 1e-12, price, np.maximum(S - K, 0.0))
    return float(price) if np.ndim(price) == 0 else price


def BSMPut(S, K, T, r, vol, q=0.0):
    S, T = np.asarray(S, float), np.asarray(T, float)
    d1, d2 = _d1_d2(S, K, T, r, vol, q)
    price = K * np.exp(-np.asarray(r) * T) * norm_cdf(-d2) - S * np.exp(-np.asarray(q) * T) * norm_cdf(-d1)
    price = np.where(T > 1e-12, price, np.maximum(K - S, 0.0))
    return float(price) if np.ndim(price) == 0 else price


def bsm_price(S, K, T, r, vol, option_type, q=0.0):
    if option_type == "call":
        return BSMCall(S, K, T, r, vol, q)
    if option_type == "put":
        return BSMPut(S, K, T, r, vol, q)
    raise ValueError(f"Unknown option_type: {option_type!r}")


def analytic_greeks(S, K, T, r, vol, option_type, q=0.0):
    """Closed-form BSM Greeks; Vega/Rho are per 1.00 change (not per 1%). Theta is per year."""
    if T <= 0:
        raise ValueError("Analytical Greeks require positive time to maturity")
    d1, d2 = _d1_d2(S, K, T, r, vol, q)
    d1, d2 = float(d1), float(d2)
    disc_q, disc_r = math.exp(-q*T), math.exp(-r*T)
    pdf = norm_pdf_scalar(d1)
    gamma = disc_q * pdf / (S * vol * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)
    if option_type == "call":
        delta = disc_q * norm_cdf_scalar(d1)
        theta = (-S*disc_q*pdf*vol/(2*math.sqrt(T)) - r*K*disc_r*norm_cdf_scalar(d2)
                 + q*S*disc_q*norm_cdf_scalar(d1))
        rho = K*T*disc_r*norm_cdf_scalar(d2)
    elif option_type == "put":
        delta = disc_q * (norm_cdf_scalar(d1)-1)
        theta = (-S*disc_q*pdf*vol/(2*math.sqrt(T)) + r*K*disc_r*norm_cdf_scalar(-d2)
                 - q*S*disc_q*norm_cdf_scalar(-d1))
        rho = -K*T*disc_r*norm_cdf_scalar(-d2)
    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")
    return {"Delta": float(delta), "Gamma": float(gamma), "Theta": float(theta),
            "Vega": float(vega), "Rho": float(rho)}

# ---------------------------------------------------------------------------
# Finite-difference Greeks and convergence validation
# ---------------------------------------------------------------------------
def _rel_bump(x, step):
    arr = np.asarray(x, float)
    h = np.maximum(np.abs(arr)*step, step)
    return float(h) if h.ndim == 0 else h


def numerical_delta(S, K, T, r, vol, option_type, step, q=0.0):
    h = _rel_bump(S, step)
    return (bsm_price(S+h,K,T,r,vol,option_type,q)-bsm_price(S-h,K,T,r,vol,option_type,q))/(2*h)


def numerical_gamma(S, K, T, r, vol, option_type, step, q=0.0):
    h = _rel_bump(S, step)
    return (bsm_price(S+h,K,T,r,vol,option_type,q)-2*bsm_price(S,K,T,r,vol,option_type,q)+bsm_price(S-h,K,T,r,vol,option_type,q))/(h*h)


def numerical_vega(S, K, T, r, vol, option_type, step, q=0.0):
    h = min(_rel_bump(vol, step), vol*0.49) if np.ndim(vol)==0 else np.minimum(_rel_bump(vol,step), np.asarray(vol)*0.49)
    return (bsm_price(S,K,T,r,np.asarray(vol)+h,option_type,q)-bsm_price(S,K,T,r,np.asarray(vol)-h,option_type,q))/(2*h)


def numerical_rho(S, K, T, r, vol, option_type, step, q=0.0):
    h = max(float(step), abs(float(r)) * float(step))
    return (bsm_price(S,K,T,r+h,vol,option_type,q)-bsm_price(S,K,T,r-h,vol,option_type,q))/(2*h)


def numerical_theta(S, K, T, r, vol, option_type, step, q=0.0):
    Tarr = np.asarray(T, float)
    h = np.minimum(_rel_bump(Tarr, step), np.maximum(Tarr*0.5, 1e-12))
    now = bsm_price(S,K,Tarr,r,vol,option_type,q)
    later = bsm_price(S,K,np.maximum(Tarr-h, 1e-12),r,vol,option_type,q)
    out = (later-now)/h
    return float(out) if np.ndim(out)==0 else out


def compute_all_greeks(S,K,T,r,vol,option_type,step,q=0.0):
    return {"Delta": numerical_delta(S,K,T,r,vol,option_type,step,q),
            "Gamma": numerical_gamma(S,K,T,r,vol,option_type,step,q),
            "Theta": numerical_theta(S,K,T,r,vol,option_type,step,q),
            "Vega": numerical_vega(S,K,T,r,vol,option_type,step,q),
            "Rho": numerical_rho(S,K,T,r,vol,option_type,step,q)}


def validate_greeks(S=100.0,K=100.0,T=1.0,r=0.03,vol=0.2,q=0.0,steps=(1e-2,1e-3,1e-4,1e-5)):
    """Return per-Greek absolute errors vs analytical formulas for multiple bump sizes."""
    rows=[]
    for typ in ("call","put"):
        ref=analytic_greeks(S,K,T,r,vol,typ,q)
        for h in steps:
            vals=compute_all_greeks(S,K,T,r,vol,typ,h,q)
            for name in ref:
                rows.append({"option_type":typ,"step":h,"Greek":name,"finite_difference":float(vals[name]),
                             "analytical":ref[name],"absolute_error":abs(float(vals[name])-ref[name])})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Discrete delta hedging
# ---------------------------------------------------------------------------
def simulate_underlying_path(S0, mu, vol, T, n_steps, rng):
    if S0 <= 0 or vol <= 0 or T <= 0 or n_steps < 1:
        raise ValueError("S0, vol, T and n_steps must be positive")
    dt=T/n_steps
    shocks=rng.normal((mu-0.5*vol**2)*dt,vol*np.sqrt(dt),size=n_steps)
    return np.exp(np.r_[np.log(S0),np.log(S0)+np.cumsum(shocks)])


def rebalancing_step_size(rebalancing_frequency, custom_rebalancing_days):
    mapping={"daily":1,"weekly":5,"monthly":21}
    if rebalancing_frequency=="custom":
        if custom_rebalancing_days < 1: raise ValueError("custom_rebalancing_days must be >= 1")
        return int(custom_rebalancing_days)
    if rebalancing_frequency not in mapping: raise ValueError(f"Unknown frequency: {rebalancing_frequency}")
    return mapping[rebalancing_frequency]


def run_delta_hedge_simulation(S0,K,T,r,vol,option_type,step,rebalancing_frequency,
                                custom_rebalancing_days,n_daily_steps=None,rng=None, q=0.0,
                                realized_vol=None, real_drift=None, transaction_cost_bps=0.0):
    """Short-one-option delta hedge. Returns terminal hedging P&L and diagnostics.

    `vol` is the pricing/implied volatility; `realized_vol` and `real_drift` drive paths.
    Transaction costs are charged on absolute stock turnover in basis points. Underlying
    dividends are approximated continuously at yield q. Default parameters reproduce the
    original risk-neutral/no-cost benchmark.
    """
    if rng is None: rng=np.random.default_rng()
    if n_daily_steps is None: n_daily_steps=max(int(round(T*252)),2)
    rebal_every=rebalancing_step_size(rebalancing_frequency,custom_rebalancing_days)
    path_vol=vol if realized_vol is None else float(realized_vol)
    mu=(r-q) if real_drift is None else float(real_drift)
    path=simulate_underlying_path(S0,mu,path_vol,T,n_daily_steps,rng)
    times=np.linspace(0,T,n_daily_steps+1); dt=T/n_daily_steps
    premium=bsm_price(S0,K,T,r,vol,option_type,q)
    delta=float(numerical_delta(S0,K,T,r,vol,option_type,step,q))
    shares=delta; cash=premium-shares*S0
    cost_rate=max(float(transaction_cost_bps),0.0)/10000.0
    initial_cost=cost_rate*abs(shares*S0); cash-=initial_cost
    portfolio=np.empty(n_daily_steps+1); deltas=np.empty(n_daily_steps+1); cash_values=np.empty(n_daily_steps+1)
    portfolio[0]=-premium+shares*S0+cash; deltas[0]=shares; cash_values[0]=cash
    total_cost=initial_cost
    delta_attr=gamma_attr=theta_attr=0.0
    for i in range(1,n_daily_steps+1):
        S_prev=float(path[i-1]); S_now=float(path[i]); dS=S_now-S_prev
        T_prev=max(T-times[i-1],1e-12)
        prev_greeks=analytic_greeks(S_prev,K,T_prev,r,vol,option_type,q)
        delta_attr+=(shares-prev_greeks["Delta"])*dS
        gamma_attr+=-0.5*prev_greeks["Gamma"]*(dS**2)
        theta_attr+=-prev_greeks["Theta"]*dt
        cash=cash*np.exp(r*dt)+shares*S_now*(np.exp(q*dt)-1.0)
        remaining=max(T-times[i],0.0)
        option_value=bsm_price(S_now,K,remaining,r,vol,option_type,q)
        if i==n_daily_steps or i%rebal_every==0:
            new_delta=0.0 if i==n_daily_steps else float(numerical_delta(S_now,K,remaining,r,vol,option_type,step,q))
            trade=new_delta-shares
            cost=cost_rate*abs(trade*S_now); total_cost+=cost
            cash-=trade*S_now+cost; shares=new_delta
        deltas[i]=shares; cash_values[i]=cash
        portfolio[i]=-option_value+shares*S_now+cash
    return {"time_grid":times,"price_path":path,"portfolio_values":portfolio,"cash_path":cash_values,
            "delta_path":deltas,"hedging_pnl":float(portfolio[-1]),"option_premium":float(premium),
            "pnl_over_premium":float(portfolio[-1]/premium) if premium else np.nan,
            "transaction_costs":float(total_cost),"realized_vol":path_vol,"real_drift":mu,
            "pnl_attribution":{"delta_mismatch_approx":float(delta_attr),"gamma_approx":float(gamma_attr),
                               "theta_approx":float(theta_attr),
                               "residual_financing_dividends_costs_higher_order":float(portfolio[-1]-delta_attr-gamma_attr-theta_attr)}}


def run_mc_batch(S0,K,T,r,vol,option_type,step,rebalancing_frequency,custom_rebalancing_days,
                 n_sims,seed=2024,keep_paths=30,q=0.0,realized_vol=None,real_drift=None,
                 transaction_cost_bps=0.0):
    """Vectorized-path Monte Carlo hedge P&L. The CDF/pricer support ndarray inputs."""
    if n_sims < 1: raise ValueError("n_sims must be positive")
    n_steps=max(int(round(T*252)),2); dt=T/n_steps
    rng=np.random.default_rng(seed)
    path_vol=vol if realized_vol is None else float(realized_vol)
    mu=(r-q) if real_drift is None else float(real_drift)
    z=rng.standard_normal((n_sims,n_steps))
    increments=(mu-0.5*path_vol**2)*dt+path_vol*np.sqrt(dt)*z
    paths=np.concatenate([np.full((n_sims,1),S0),S0*np.exp(np.cumsum(increments,axis=1))],axis=1)
    times=np.linspace(0,T,n_steps+1)
    premium=float(bsm_price(S0,K,T,r,vol,option_type,q))
    delta=np.asarray(numerical_delta(S0,K,T,r,vol,option_type,step,q),float).item()
    shares=np.full(n_sims,delta); cash=np.full(n_sims,premium-delta*S0)
    cost_rate=max(float(transaction_cost_bps),0.0)/10000.0
    costs=cost_rate*np.abs(shares*S0); cash-=costs
    every=rebalancing_step_size(rebalancing_frequency,custom_rebalancing_days)
    for i in range(1,n_steps+1):
        S=paths[:,i]
        cash=cash*np.exp(r*dt)+shares*S*(np.exp(q*dt)-1.0)
        remaining=max(T-times[i],0.0)
        if i==n_steps or i%every==0:
            if i==n_steps:
                new_delta=np.zeros(n_sims)
            else:
                new_delta=np.asarray(numerical_delta(S,K,remaining,r,vol,option_type,step,q))
            trade=new_delta-shares
            c=cost_rate*np.abs(trade*S); costs+=c
            cash-=trade*S+c; shares=new_delta
    terminal_option=np.maximum(paths[:,-1]-K,0.0) if option_type=="call" else np.maximum(K-paths[:,-1],0.0)
    pnls=-terminal_option+cash
    kept=[paths[i] for i in range(min(int(keep_paths),n_sims))]
    return pnls, kept


def compare_rebalancing_frequencies(S0,K,T,r,vol,option_type,step,custom_rebalancing_days,
                                   n_sims=1000,seed=777,q=0.0,realized_vol=None,real_drift=None,
                                   transaction_cost_bps=0.0):
    """Common-random-number comparison across frequencies to reduce sampling noise."""
    # Generate the same standard-normal paths in each scenario via identical seeds.
    result={}
    for freq in ("daily","weekly","monthly"):
        pnls,_=run_mc_batch(S0,K,T,r,vol,option_type,step,freq,custom_rebalancing_days,n_sims,
                            seed=seed,keep_paths=0,q=q,realized_vol=realized_vol,
                            real_drift=real_drift,transaction_cost_bps=transaction_cost_bps)
        result[freq]={"std":float(np.std(pnls,ddof=1)),"mean":float(np.mean(pnls)),"pnls":pnls}
    return result

# ---------------------------------------------------------------------------
# Implied volatility and surface construction
# ---------------------------------------------------------------------------
def _intrinsic_and_bounds(S,K,T,r,q,option_type):
    disc_s=S*np.exp(-q*T); disc_k=K*np.exp(-r*T)
    if option_type=="call": return max(disc_s-disc_k,0.0), max(disc_s,0.0)
    return max(disc_k-disc_s,0.0), disc_k


def implied_vol_bisection(market_price,S,K,T,r,option_type="call",vol_lo=1e-4,vol_hi=5.0,
                          tol=1e-9,max_iter=180,q=0.0,price_tol=1e-10,return_details=False,
                          min_vega=1.0):
    """Invert BSM price with bounds checks and volatility-width stopping criterion.

    Returns NaN for prices outside European BSM bounds or ill-conditioned low-Vega cases.
    Set return_details=True for a diagnostics dictionary.
    """
    def done(vol, status, residual=np.nan, iterations=0, vega=np.nan):
        d={"iv":float(vol) if np.isfinite(vol) else np.nan,"status":status,"price_residual":float(residual),
           "iterations":int(iterations),"vega":float(vega)}
        return d if return_details else d["iv"]
    if not all(np.isfinite(x) for x in (market_price,S,K,T,r,q)) or market_price < 0 or S<=0 or K<=0 or T<=0:
        return done(np.nan,"invalid_input")
    lower,upper=_intrinsic_and_bounds(S,K,T,r,q,option_type)
    if market_price < lower-price_tol or market_price > upper+price_tol:
        return done(np.nan,"outside_no_arbitrage_bounds")
    lo,hi=float(vol_lo),float(vol_hi)
    f_lo=bsm_price(S,K,T,r,lo,option_type,q)-market_price
    f_hi=bsm_price(S,K,T,r,hi,option_type,q)-market_price
    if f_lo*f_hi>0: return done(np.nan,"root_not_bracketed")
    mid=0.5*(lo+hi); residual=np.nan
    for i in range(1,max_iter+1):
        mid=0.5*(lo+hi); residual=float(bsm_price(S,K,T,r,mid,option_type,q)-market_price)
        if abs(residual)<=price_tol or (hi-lo)<=tol: break
        if f_lo*residual<=0: hi=mid; f_hi=residual
        else: lo=mid; f_lo=residual
    vega=float(analytic_greeks(S,K,T,r,mid,option_type,q)["Vega"])
    if vega < min_vega:
        return done(np.nan,"ill_conditioned_low_vega",residual,i,vega)
    status="converged" if abs(residual)<=max(price_tol,1e-6) else "volatility_tolerance_reached"
    return done(mid,status,residual,i,vega)


def fetch_option_chain(ticker,spot,risk_free_rate,max_expirations=8,seed=7,q=0.0,curve=None):
    """Fetch liquid-ish call/put mid quotes; explicitly label last-price fallback.

    American-listed equity options are treated as European BSM inputs here, so the
    resulting IVs are model-implied approximations, not exchange-standard IV marks.
    """
    if YFINANCE_AVAILABLE:
        try:
            tk=yf.Ticker(ticker)
            all_expirations=list(tk.options)
            if len(all_expirations) > max_expirations:
                # Sample across the available term structure instead of taking only
                # the nearest weekly expiries, which can create a misleadingly narrow surface.
                indices=np.linspace(0, len(all_expirations)-1, max_expirations).round().astype(int)
                expirations=[all_expirations[i] for i in sorted(set(indices))]
            else:
                expirations=all_expirations
            rows=[]; today=pd.Timestamp.today().normalize()
            for exp in expirations:
                T=(pd.Timestamp(exp)-today).days/365.0
                if T<=0.02: continue
                rT=risk_free_rate_for_maturity(curve,T) if curve else risk_free_rate
                chain=tk.option_chain(exp)
                for typ,frame in (("call",chain.calls),("put",chain.puts)):
                    if frame is None or frame.empty: continue
                    frame=frame[(frame["strike"]>=0.75*spot)&(frame["strike"]<=1.25*spot)]
                    for _,row in frame.iterrows():
                        bid=float(row.get("bid",0) or 0); ask=float(row.get("ask",0) or 0)
                        last=float(row.get("lastPrice",0) or 0)
                        if bid>0 and ask>=bid and ask>0:
                            mid=(bid+ask)/2; source="mid"
                            spread=(ask-bid)/mid if mid>0 else np.inf
                            if spread>0.75: continue
                        elif last>0:
                            mid=last; source="last_fallback"
                        else: continue
                        if mid<0.01: continue
                        rows.append({"T":T,"K":float(row["strike"]),"price":float(mid),"option_type":typ,
                                     "quote_source":source,"r":rT,"q":q,"bid":bid,"ask":ask})
            if len(rows)>=10:
                return {"rows":rows,"true_vols":None,"source":"yfinance quotes (European BSM approximation)"}
        except Exception:
            pass
    rng=np.random.default_rng(seed)
    maturities=[0.08,0.25,0.5,1.0,1.5,2.0]
    strikes=np.arange(round(0.7*spot),round(1.3*spot)+1,max(1,round(spot*0.025)))
    rows=[]; true_vols={}
    for T in maturities:
        rT=risk_free_rate_for_maturity(curve,T) if curve else risk_free_rate
        for K in strikes:
            m=np.log(K/spot)
            true_vol=float(np.clip(0.22+0.055*m*m-0.05*m-0.03*np.sqrt(T),0.03,2.0))
            for typ in ("call","put"):
                price=float(bsm_price(spot,K,T,rT,true_vol,typ,q))
                if price>0.01:
                    rows.append({"T":T,"K":float(K),"price":price,"option_type":typ,"quote_source":"synthetic",
                                 "r":rT,"q":q,"bid":np.nan,"ask":np.nan})
                    true_vols[(T,float(K),typ)]=true_vol
    return {"rows":rows,"true_vols":true_vols,"source":"synthetic model-generated quotes"}


def build_iv_surface(ticker,spot,risk_free_rate,n_price_points,max_expirations=8,q=0.0,curve=None,option_type=None):
    chain=fetch_option_chain(ticker,spot,risk_free_rate,max_expirations,q=q,curve=curve)
    implied_rows=[]; rejected=[]
    for row in chain["rows"]:
        if option_type is not None and row["option_type"] != option_type:
            continue
        diag=implied_vol_bisection(row["price"],spot,row["K"],row["T"],row["r"],row["option_type"],q=row["q"],return_details=True)
        if np.isfinite(diag["iv"]):
            implied_rows.append({**row,**diag})
        else:
            rejected.append({**row,**diag})
    by_maturity=defaultdict(list)
    for row in implied_rows: by_maturity[row["T"]].append((row["K"],row["iv"]))
    Ts=np.array(sorted(by_maturity.keys()),dtype=float)
    Ks=np.linspace(0.75*spot,1.25*spot,int(n_price_points))
    grid=np.full((len(Ts),len(Ks)),np.nan)
    for i,T in enumerate(Ts):
        # Calls and puts may both exist at the same strike. Aggregate duplicates
        # rather than arbitrarily keeping whichever quote appeared first.
        pts=sorted(by_maturity[T]); grouped=defaultdict(list)
        for strike, iv in pts: grouped[float(strike)].append(float(iv))
        unique=np.array(sorted(grouped), dtype=float)
        vols=np.array([float(np.median(grouped[k])) for k in unique])
        if len(unique)>=2:
            inside=(Ks>=unique.min()) & (Ks<=unique.max())
            grid[i,inside]=np.interp(Ks[inside],unique,vols)
        elif len(unique)==1:
            grid[i,np.argmin(abs(Ks-unique[0]))]=vols[0]
    return {"chain":chain,"implied_rows":implied_rows,"rejected_rows":rejected,
            "T_axis":Ts,"K_axis":Ks,"grid":grid}
