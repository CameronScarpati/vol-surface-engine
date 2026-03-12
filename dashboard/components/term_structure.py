"""
Term structure panel and mispricing table.

1. ATM implied volatility vs expiry (the volatility term structure).
2. SVI parameter evolution across expiries.
3. Top mispricings ranked by |residual|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.components.helpers import compute_chain_fitted_iv
from src.svi_fitter import svi_total_variance


def render_term_structure(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
) -> None:
    """Render the ATM term structure and SVI parameter evolution."""
    st.subheader("Volatility Term Structure")

    if slice_params.empty:
        st.warning("No fitted slices available.")
        return

    sorted_sp = slice_params.sort_values("T")
    dte = sorted_sp["T"] * 365.25

    # ATM IV (k = 0) from each fitted slice
    atm_ivs = []
    for _, row in sorted_sp.iterrows():
        w = svi_total_variance(0.0, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        w_val = float(np.squeeze(w))
        atm_ivs.append(float(np.sqrt(max(w_val, 0.0) / row["T"])) if row["T"] > 0 else np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dte,
        y=atm_ivs,
        mode="lines+markers",
        name="ATM IV (SVI fit, k=0)",
        line=dict(width=2.5, color="#1f77b4"),
        marker=dict(size=7),
    ))

    # Overlay market ATM IV where available
    market_atm = []
    market_dte = []
    for T in sorted_sp["T"]:
        slice_data = chain[np.isclose(chain["T"], T, atol=1e-6)]
        if slice_data.empty:
            continue
        # Closest to ATM
        atm_row = slice_data.iloc[(slice_data["strike"] - slice_data["S"]).abs().argsort()[:3]]
        mean_iv = atm_row["iv"].dropna().mean()
        if np.isfinite(mean_iv):
            market_atm.append(mean_iv)
            market_dte.append(T * 365.25)

    if market_atm:
        fig.add_trace(go.Scatter(
            x=market_dte,
            y=market_atm,
            mode="markers",
            name="Market ATM IV",
            marker=dict(size=8, symbol="diamond", color="#ff7f0e"),
        ))

    fig.update_layout(
        xaxis_title="Days to Expiry",
        yaxis_title="ATM Implied Volatility",
        height=380,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # SVI parameter evolution
    st.subheader("SVI Parameter Evolution")

    param_names = [("a", "Level (a)"), ("b", "Angle (b)"), ("rho", "Skew (ρ)"),
                   ("m", "Translation (m)"), ("sigma", "Curvature (σ)")]

    fig2 = make_subplots(
        rows=3, cols=2,
        subplot_titles=[name for _, name in param_names] + [""],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
    for (col_name, label), (r, c) in zip(param_names, positions, strict=True):
        fig2.add_trace(
            go.Scatter(
                x=dte,
                y=sorted_sp[col_name],
                mode="lines+markers",
                name=label,
                marker=dict(size=5),
                showlegend=False,
            ),
            row=r, col=c,
        )
        fig2.update_xaxes(title_text="DTE" if r == 3 else "", row=r, col=c)
        fig2.update_yaxes(title_text=label, row=r, col=c)

    fig2.update_layout(height=650, margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig2, use_container_width=True)


def render_mispricing_table(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
    top_n: int = 10,
) -> None:
    """Top N mispricings ranked by |residual|."""
    st.subheader(f"Top {top_n} Mispricings")

    df = compute_chain_fitted_iv(chain, slice_params)
    if df.empty:
        st.warning("No data for mispricing analysis.")
        return
    df["abs_residual"] = df["residual"].abs()
    df["direction"] = np.where(df["residual"] > 0, "CHEAP (under-fitted)", "RICH (over-fitted)")
    df["DTE"] = (df["T"] * 365.25).round().astype(int)
    df["bid_ask_spread"] = df["ask"] - df["bid"]

    top = (
        df.dropna(subset=["residual"])
        .nlargest(top_n, "abs_residual")
        [["strike", "DTE", "option_type", "iv", "fitted_iv", "residual",
          "direction", "bid", "ask", "bid_ask_spread"]]
        .reset_index(drop=True)
    )

    # Format for display
    fmt = {
        "iv": "{:.4f}",
        "fitted_iv": "{:.4f}",
        "residual": "{:+.4f}",
        "bid": "{:.2f}",
        "ask": "{:.2f}",
        "bid_ask_spread": "{:.2f}",
    }

    st.dataframe(
        top.style.format(fmt),
        use_container_width=True,
        hide_index=True,
    )

    # Summary
    sigma_r = df["residual"].std()
    n_sig = (df["abs_residual"] > 2 * sigma_r).sum()
    st.caption(
        f"Residual σ = {sigma_r:.4f} | "
        f"{n_sig} options exceed 2σ significance threshold"
    )
