"""
Per-expiry smile slice plot.

Shows market-observed IV as scatter points overlaid with the smooth SVI
fit curve, plus confidence bands derived from the bid-ask IV range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.iv_engine import implied_volatility
from src.svi_fitter import svi_total_variance


def render_smile_slices(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the smile-slice panel in Streamlit."""
    st.subheader("Volatility Smile by Expiry")

    available_T = sorted(slice_params["T"].unique())
    if not available_T:
        st.warning("No fitted slices available.")
        return

    # Build human-readable labels
    dte_labels = [f"{T * 365.25:.0f}d (T={T:.4f})" for T in available_T]
    selected_label = st.selectbox("Select expiry", dte_labels, key="smile_expiry")
    selected_idx = dte_labels.index(selected_label)
    T_sel = available_T[selected_idx]

    # Market data for this slice
    tol = 1e-6
    slice_data = chain[np.isclose(chain["T"], T_sel, atol=tol)].copy()
    slice_data = slice_data.dropna(subset=["iv"]).sort_values("strike")

    sp_row = slice_params[np.isclose(slice_params["T"], T_sel, atol=tol)]
    if sp_row.empty:
        st.warning("No SVI params for this expiry.")
        return
    sp = sp_row.iloc[0]

    F = spot * np.exp((risk_free - div_yield) * T_sel)

    # Smooth SVI curve over a fine strike grid
    k_min = np.log(slice_data["strike"].min() / F) - 0.05
    k_max = np.log(slice_data["strike"].max() / F) + 0.05
    k_fine = np.linspace(k_min, k_max, 200)
    strikes_fine = F * np.exp(k_fine)
    w_fine = svi_total_variance(k_fine, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
    iv_fine = np.sqrt(np.maximum(w_fine, 0.0) / T_sel)

    fig = go.Figure()

    # Confidence band from bid/ask IV
    if "bid" in slice_data.columns and "ask" in slice_data.columns:
        iv_bid = []
        iv_ask = []
        valid_strikes = []
        for _, row in slice_data.iterrows():
            bid_iv = implied_volatility(
                row["bid"], row["S"], row["strike"], row["T"],
                row["r"], row["q"], row["option_type"],
            )
            ask_iv = implied_volatility(
                row["ask"], row["S"], row["strike"], row["T"],
                row["r"], row["q"], row["option_type"],
            )
            if np.isfinite(bid_iv) and np.isfinite(ask_iv):
                iv_bid.append(bid_iv)
                iv_ask.append(ask_iv)
                valid_strikes.append(row["strike"])

        if valid_strikes:
            fig.add_trace(go.Scatter(
                x=valid_strikes + valid_strikes[::-1],
                y=iv_ask + iv_bid[::-1],
                fill="toself",
                fillcolor="rgba(135, 206, 250, 0.25)",
                line=dict(width=0),
                name="Bid-Ask IV Band",
                hoverinfo="skip",
            ))

    # SVI fit curve
    fig.add_trace(go.Scatter(
        x=strikes_fine,
        y=iv_fine,
        mode="lines",
        name="SVI Fit",
        line=dict(color="#1f77b4", width=2.5),
    ))

    # Market IV scatter
    fig.add_trace(go.Scatter(
        x=slice_data["strike"],
        y=slice_data["iv"],
        mode="markers",
        name="Market IV",
        marker=dict(
            size=7,
            color=slice_data["iv"],
            colorscale="Viridis",
            showscale=False,
            line=dict(width=0.5, color="black"),
        ),
        hovertemplate=(
            "Strike: %{x:.1f}<br>"
            "Market IV: %{y:.4f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis_title="Strike",
        yaxis_title="Implied Volatility",
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Fit quality metrics
    cols = st.columns(3)
    cols[0].metric("R²", f"{sp['r_squared']:.6f}")
    cols[1].metric("RMSE (variance)", f"{sp['rmse']:.2e}")
    cols[2].metric("Max |error|", f"{sp['max_abs_error']:.2e}")
