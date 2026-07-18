"""
BSM Pricing Engine & Dynamic Hedging Simulator - Streamlit app
Author: Giulio - LM-16 Financial Risk and Data Analysis, Sapienza Universita di Roma
Ported from a dependency-free BSM notebook (custom Normal CDF, numerical Greeks,
Monte Carlo delta hedging, bisection implied vol).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="BSM Pricing & Hedging Desk", page_icon="📈", layout="wide")

PRIMARY = "#7a1f2b"    # bordeaux
SECONDARY = "#1f4e7a"  # navy
ACCENT = "#c98a2b"      # gold


# --------------------------------------------------------------------------
# Cached data-fetching wrappers (avoid hammering yfinance on every rerun)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Scarico i dati di mercato...")
def cached_price_history(ticker, lookback_days):
    return core.fetch_price_history(ticker, lookback_days)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_risk_free_curve():
    return core.fetch_risk_free_curve()


@st.cache_data(ttl=3600, show_spinner="Costruisco la superficie di volatilita implicita...")
def cached_iv_surface(ticker, spot, r, n_points):
    return core.build_iv_surface(ticker, spot, r, n_points)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_mc_batch(S0, K, T, r, vol, option_type, step, rebal_freq, custom_rebal_days, n_sims):
    return core.run_mc_batch(S0, K, T, r, vol, option_type, step, rebal_freq, custom_rebal_days, n_sims)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_freq_comparison(S0, K, T, r, vol, option_type, step, custom_rebal_days, n_compare):
    freqs = ["daily", "weekly", "monthly"]
    compare_rng = np.random.default_rng(777)
    te_by_freq = {}
    for freq in freqs:
        pnl_batch = np.empty(n_compare)
        for i in range(n_compare):
            res = core.run_delta_hedge_simulation(
                S0, K, T, r, vol, option_type, step, freq, custom_rebal_days, rng=compare_rng,
            )
            pnl_batch[i] = res["hedging_pnl"]
        te_by_freq[freq] = float(np.std(pnl_batch))
    return te_by_freq


# --------------------------------------------------------------------------
# Sidebar - scenario parameters
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 Parametri scenario")
    ticker = st.text_input("Ticker (Yahoo Finance)", "NVDA")
    option_type = st.selectbox("Tipo opzione", ["call", "put"])
    strike = st.number_input("Strike K", min_value=0.01, value=150.0, step=1.0)
    maturity = st.slider("Maturity T (anni)", 0.05, 3.0, 0.5, step=0.05)

    st.divider()
    st.subheader("Volatilita")
    historical_window = st.slider("Finestra storica (giorni)", 60, 756, 252, step=21)
    vol_source = st.radio("Fonte volatilita", ["historical", "custom"], horizontal=True)
    custom_vol = None
    if vol_source == "custom":
        custom_vol = st.slider("Volatilita annualizzata custom", 0.01, 1.5, 0.35, step=0.01)

    # Live market data, fetched now so spot/rate toggles below can default to it.
    live_price_history, live_data_source = cached_price_history(ticker, historical_window)
    live_spot = float(live_price_history["Close"].iloc[-1])
    live_curve, curve_source = cached_risk_free_curve()
    live_rate = core.risk_free_rate_for_maturity(live_curve, maturity)

    st.divider()
    st.subheader("Spot price")
    spot_override = st.toggle("Override manuale", value=False, key="spot_override")
    if spot_override:
        spot_price = st.number_input("Spot price manuale", min_value=0.01,
                                      value=round(live_spot, 2), step=0.5)
        st.caption(f"Ultimo close reale: {live_spot:,.2f} ({live_data_source})")
    else:
        spot_price = live_spot
        st.metric("Spot (ultimo close)", f"{spot_price:,.2f}")
        st.caption(f"Fonte: {live_data_source}")

    st.subheader("Risk-free rate")
    rate_override = st.toggle("Override manuale", value=False, key="rate_override")
    if rate_override:
        risk_free_rate = st.slider("r manuale", -0.02, 0.10, round(live_rate, 4), step=0.0025,
                                    format="%.4f")
        st.caption(f"Tasso live interpolato su T={maturity:.2f}y: {live_rate:.3%} ({curve_source})")
    else:
        risk_free_rate = live_rate
        st.metric(f"r interpolato (T={maturity:.2f}y)", f"{risk_free_rate:.3%}")
        st.caption(f"Curva Treasury: {curve_source} — si aggiorna con la maturity scelta")

    st.divider()
    st.subheader("Hedging")
    rebal_freq = st.selectbox("Frequenza ribilanciamento", ["daily", "weekly", "monthly", "custom"], index=1)
    custom_rebal_days = 3
    if rebal_freq == "custom":
        custom_rebal_days = st.number_input("Intervallo custom (giorni)", min_value=1, value=3)
    n_sims = st.slider("N. simulazioni Monte Carlo", 50, 5000, 2000, step=50)

    st.divider()
    with st.expander("Impostazioni avanzate"):
        n_price_points = st.slider("Risoluzione griglia prezzo", 20, 100, 60, step=10)
        n_time_points = st.slider("Risoluzione griglia tempo", 20, 100, 60, step=10)
        fd_step = st.select_slider("Passo differenze finite", options=[1e-5, 1e-4, 1e-3, 1e-2],
                                    value=1e-4, format_func=lambda v: f"{v:.0e}")

# --------------------------------------------------------------------------
# Load market data + compute base quantities
# --------------------------------------------------------------------------
price_history, data_source = live_price_history, live_data_source
log_returns, rolling_vol, hist_vol = core.compute_volatility(price_history)
sigma = custom_vol if vol_source == "custom" else hist_vol

option_price = core.bsm_price(spot_price, strike, maturity, risk_free_rate, sigma, option_type)
greeks_at_spot = core.compute_all_greeks(spot_price, strike, maturity, risk_free_rate, sigma,
                                          option_type, fd_step)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Black-Scholes-Merton Pricing Engine & Dynamic Hedging Simulator")
st.caption(
    "Proprietary Options Desk Infrastructure — Dependency-Free Statistical Core · "
    "Giulio · LM-16 Financial Risk and Data Analysis, Sapienza Università di Roma"
)
if data_source != "yfinance":
    st.info("Dati live non disponibili in questo momento: sto usando un percorso GBM sintetico "
            "riproducibile come fallback (il motore di pricing sottostante è identico).", icon="ℹ️")

kpi_cols = st.columns(6)
kpi_cols[0].metric("Spot", f"{spot_price:,.2f}")
kpi_cols[1].metric(f"{option_type.capitalize()} price", f"{option_price:,.4f}")
kpi_cols[2].metric("Delta", f"{greeks_at_spot['Delta']:.4f}")
kpi_cols[3].metric("Gamma", f"{greeks_at_spot['Gamma']:.5f}")
kpi_cols[4].metric("Vega", f"{greeks_at_spot['Vega']:.4f}")
kpi_cols[5].metric("Sigma usato", f"{sigma:.2%}")

tab_pricing, tab_surfaces, tab_vol, tab_hedging, tab_about = st.tabs(
    ["💰 Pricing & Greeks", "🌐 Superfici 3D", "📊 Volatilita & IV surface",
     "🛡️ Hedging simulator", "ℹ️ Come funziona"]
)

PLOTLY_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=480)

# ============================================================ Pricing tab
with tab_pricing:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prezzo e Greche vs Spot")
        S_grid = np.linspace(0.4 * spot_price, 1.6 * spot_price, n_price_points)
        price_curve = core.bsm_price(S_grid, strike, maturity, risk_free_rate, sigma, option_type)
        intrinsic = (np.maximum(S_grid - strike, 0.0) if option_type == "call"
                     else np.maximum(strike - S_grid, 0.0))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_grid, y=price_curve, name=f"{option_type.capitalize()} price",
                                  line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=S_grid, y=intrinsic, name="Intrinsic value",
                                  line=dict(color=SECONDARY, dash="dash")))
        fig.add_vline(x=strike, line_dash="dot", line_color="gray", annotation_text="Strike")
        fig.add_vline(x=spot_price, line_dash="dot", line_color=ACCENT, annotation_text="Spot")
        fig.update_layout(xaxis_title="Underlying price S", yaxis_title="Price", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Greche numeriche (differenze finite)")
        greek_curves = {
            name: getattr(core, f"numerical_{name.lower()}")(
                S_grid, strike, maturity, risk_free_rate, sigma, option_type, fd_step)
            for name in ["Delta", "Gamma", "Vega"]
        }
        fig2 = go.Figure()
        colors = [PRIMARY, SECONDARY, ACCENT]
        for (name, curve), color in zip(greek_curves.items(), colors):
            fig2.add_trace(go.Scatter(x=S_grid, y=curve, name=name, line=dict(color=color)))
        fig2.add_vline(x=strike, line_dash="dot", line_color="gray")
        fig2.update_layout(xaxis_title="Underlying price S", yaxis_title="Value", **PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tutte le Greche puntuali")
    greek_cols = st.columns(5)
    for c, (name, value) in zip(greek_cols, greeks_at_spot.items()):
        c.metric(name, f"{value:.6f}")

    with st.expander("Validazione motore di pricing"):
        max_err, ref_rows = core.validate_norm_cdf()
        st.write(f"**Normal CDF custom (Abramowitz & Stegun 26.2.17)** — errore assoluto max "
                 f"vs valori tabulati: `{max_err:.2e}`")
        ref_df = pd.DataFrame(ref_rows, columns=["x", "N(x) custom", "Riferimento", "Errore assoluto"])
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

        c_test = core.BSMCall(100.0, 100.0, 1.0, 0.03, 0.20)
        p_test = core.BSMPut(100.0, 100.0, 1.0, 0.03, 0.20)
        parity_lhs = c_test - p_test
        parity_rhs = 100.0 - 100.0 * np.exp(-0.03 * 1.0)
        st.write(f"**Put-call parity check**: C-P = `{parity_lhs:.6f}`, "
                 f"S-Ke^(-rT) = `{parity_rhs:.6f}`, diff = `{abs(parity_lhs - parity_rhs):.2e}`")

# ============================================================ Surfaces tab
with tab_surfaces:
    st.subheader("Superfici (Prezzo x Time-to-maturity)")
    S_axis = np.linspace(0.5 * spot_price, 1.5 * spot_price, n_price_points)
    T_axis = np.linspace(maturity, 1e-3, n_time_points)
    S_mesh, T_mesh = np.meshgrid(S_axis, T_axis)

    price_surface = core.bsm_price(S_mesh, strike, T_mesh, risk_free_rate, sigma, option_type)
    surface_choice = st.selectbox("Superficie da esplorare",
                                   ["Price", "Delta", "Gamma", "Theta", "Vega", "Rho"])

    if surface_choice == "Price":
        Z = price_surface
        colorscale = "Magma"
    else:
        Z = getattr(core, f"numerical_{surface_choice.lower()}")(
            S_mesh, strike, T_mesh, risk_free_rate, sigma, option_type, fd_step)
        colorscale = "Viridis"

    fig3d = go.Figure(data=[go.Surface(x=S_axis, y=T_axis, z=Z, colorscale=colorscale)])
    fig3d.update_layout(
        scene=dict(xaxis_title="Underlying price S", yaxis_title="Time to maturity (years)",
                   zaxis_title=surface_choice),
        height=650, margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("Trascina per ruotare, scroll per zoom — stesso grafico che finisce nel report/slide, "
               "ma qui esplorabile live.")

# ============================================================ Volatility tab
with tab_vol:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volatilita storica")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                                      name="Rolling 21d annualized vol", line=dict(color=PRIMARY)))
        fig_vol.add_hline(y=hist_vol, line_dash="dash", line_color=SECONDARY,
                           annotation_text=f"Full-window vol ({hist_vol:.1%})")
        if vol_source == "custom":
            fig_vol.add_hline(y=custom_vol, line_dash="dot", line_color=ACCENT,
                               annotation_text=f"Sigma custom ({custom_vol:.1%})")
        fig_vol.update_layout(xaxis_title="Data", yaxis_title="Volatilita annualizzata", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.subheader("Prezzo storico")
        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(x=price_history.index, y=price_history["Close"],
                                     name="Close", line=dict(color=SECONDARY)))
        fig_px.update_layout(xaxis_title="Data", yaxis_title="Prezzo", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_px, use_container_width=True)

    st.divider()
    st.subheader("Superficie di volatilita implicita (option chain reale + bisezione)")
    st.caption("Recupera una catena di opzioni reale via yfinance (fallback su uno smile sintetico "
               "se non disponibile) e inverte il prezzo BSM per ricavare la IV, senza scipy.optimize.")
    if st.button("Costruisci IV surface", type="primary"):
        iv_data = cached_iv_surface(ticker, spot_price, risk_free_rate, n_price_points)
        chain_source = iv_data["chain"]["source"]
        st.write(f"Fonte dati: **{chain_source}** · "
                 f"{len(iv_data['implied_rows'])} quote invertite su "
                 f"{len(iv_data['T_axis'])} scadenze.")

        fig_iv = go.Figure(data=[go.Surface(
            x=iv_data["K_axis"], y=iv_data["T_axis"], z=iv_data["grid"], colorscale="Viridis")])
        fig_iv.update_layout(
            scene=dict(xaxis_title="Strike K", yaxis_title="Time to maturity (years)",
                       zaxis_title="Implied vol"),
            height=650, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_iv, use_container_width=True)

        if chain_source == "synthetic":
            true_vols = iv_data["chain"]["true_vols"]
            errs = [abs(iv - true_vols[(T, K)]) for T, K, iv in iv_data["implied_rows"]
                    if (T, K) in true_vols]
            if errs:
                st.write(f"Validazione (fallback sintetico): errore assoluto medio = "
                         f"`{np.mean(errs):.6f}`, massimo = `{np.max(errs):.6f}`.")

# ============================================================ Hedging tab
with tab_hedging:
    st.subheader("Simulazione di delta-hedging discreto")
    run_sim = st.button("Esegui simulazione Monte Carlo", type="primary")

    if run_sim:
        base_rng = np.random.default_rng(123)
        single_path = core.run_delta_hedge_simulation(
            spot_price, strike, maturity, risk_free_rate, sigma, option_type, fd_step,
            rebal_freq, custom_rebal_days, rng=base_rng,
        )
        with st.spinner(f"Eseguo {n_sims} simulazioni Monte Carlo..."):
            pnls, kept_paths = cached_mc_batch(
                spot_price, strike, maturity, risk_free_rate, sigma, option_type, fd_step,
                rebal_freq, custom_rebal_days, n_sims,
            )
        mean_pnl = float(np.mean(pnls))
        tracking_error = float(np.std(pnls))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean hedging P&L", f"{mean_pnl:.4f}")
        m2.metric("Tracking error (std)", f"{tracking_error:.4f}")
        m3.metric("Min P&L", f"{pnls.min():.4f}")
        m4.metric("Max P&L", f"{pnls.max():.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Un path illustrativo — prezzo, Delta, valore portafoglio**")
            fig_path = go.Figure()
            fig_path.add_trace(go.Scatter(x=single_path["time_grid"], y=single_path["price_path"],
                                           name="Underlying price", line=dict(color=SECONDARY)))
            fig_path.update_layout(xaxis_title="Tempo (anni)", yaxis_title="Prezzo", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_path, use_container_width=True)

            fig_delta = go.Figure()
            fig_delta.add_trace(go.Scatter(x=single_path["time_grid"], y=single_path["delta_path"],
                                            name="Delta", line=dict(color=PRIMARY)))
            fig_delta.update_layout(xaxis_title="Tempo (anni)", yaxis_title="Delta",
                                     height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_delta, use_container_width=True)

        with col2:
            st.markdown(f"**Distribuzione P&L su {n_sims} path Monte Carlo**")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=pnls, marker_color=PRIMARY, nbinsx=25, opacity=0.85))
            fig_hist.add_vline(x=mean_pnl, line_color=SECONDARY, line_dash="dash",
                                annotation_text=f"Mean={mean_pnl:.4f}")
            fig_hist.update_layout(xaxis_title="Hedging P&L", yaxis_title="Frequenza", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("**Tracking error per frequenza di ribilanciamento**")
            n_compare = min(n_sims, 100)
            with st.spinner("Confronto le frequenze di ribilanciamento..."):
                te_by_freq = cached_freq_comparison(
                    spot_price, strike, maturity, risk_free_rate, sigma, option_type, fd_step,
                    custom_rebal_days, n_compare,
                )
            fig_freq = go.Figure(data=[go.Bar(
                x=list(te_by_freq.keys()), y=list(te_by_freq.values()),
                marker_color=[PRIMARY, SECONDARY, ACCENT])])
            fig_freq.update_layout(xaxis_title="Frequenza", yaxis_title="Tracking error (std)",
                                    height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("Imposta i parametri di hedging nella sidebar e premi il bottone per lanciare "
                 f"{n_sims} simulazioni Monte Carlo di delta-hedging discreto.")

# ============================================================ About tab
with tab_about:
    st.markdown(f"""
### Cosa fa questa app

Motore di pricing Black-Scholes-Merton **senza dipendenze statistiche esterne**:

- **Normal CDF** implementata da zero con l'approssimazione polinomiale
  Abramowitz & Stegun (26.2.17) — non `scipy.stats.norm.cdf`.
- **Greeks** calcolate via **differenze finite** sul pricer, non in forma chiusa —
  cosi si validerebbe un pricer contro payoff esotici privi di formula analitica.
- **Delta hedging** simulato con ribilanciamento discreto (non continuo), per
  quantificare l'hedging error che la teoria continua nasconde.
- **Volatilita implicita** ricavata per bisezione da una catena di opzioni reale
  (yfinance) o da uno smile sintetico di fallback — nessun `scipy.optimize`.

Dati mancanti o rete non disponibile → l'app ricade automaticamente su un percorso
GBM sintetico riproducibile, cosi resta sempre utilizzabile end-to-end.

**Autore:** Giulio Donato Gottardi · LM-16 Financial Risk and Data Analysis, Sapienza Universita di Roma
Progetto originariamente sviluppato come deliverable per il corso di Quantitative
Financial Modelling (Prof. Sergio Bianchi).
""")
