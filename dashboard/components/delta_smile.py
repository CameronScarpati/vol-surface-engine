"""
Delta-space smile — implied volatility plotted against Black-Scholes delta.

Practitioners quote volatility in delta-space (e.g. "25-delta put vol"), not
strike-space.  This view normalises across expiries and makes skew/convexity
directly comparable, which is how desks actually manage risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from src.svi_fitter import svi_total_variance


def _strike_to_delta(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Convert strikes to Black-Scholes call delta."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return np.exp(-q * T) * norm.cdf(d1)


def render_delta_smile(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the delta-space volatility smile in Streamlit."""
    st.subheader("Volatility Smile in Delta-Space")

    if slice_params.empty:
        st.warning("No fitted slices available.")
        return

    sorted_sp = slice_params.sort_values("T")

    fig = go.Figure()

    for _, row in sorted_sp.iterrows():
        T = row["T"]
        dte = round(T * 365.25)
        F = spot * np.exp((risk_free - div_yield) * T)

        # Fine strike grid
        k_fine = np.linspace(-0.4, 0.4, 200)
        strikes = F * np.exp(k_fine)

        w = svi_total_variance(k_fine, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        iv = np.sqrt(np.maximum(w, 1e-10) / T)

        # Convert to delta
        delta = _strike_to_delta(spot, strikes, T, risk_free, div_yield, iv)

        # Filter to reasonable delta range [0.05, 0.95]
        mask = (delta >= 0.05) & (delta <= 0.95)
        if mask.sum() < 5:
            continue

        fig.add_trace(go.Scatter(
            x=delta[mask],
            y=iv[mask],
            mode="lines",
            name=f"{dte}d",
            line=dict(width=2),
            hovertemplate=(
                "Delta: %{x:.2f}<br>"
                "IV: %{y:.2%}<extra></extra>"
            ),
        ))

    # Mark standard delta pillars
    for d_val, label in [(0.25, "25Δ Put"), (0.50, "ATM"), (0.75, "25Δ Call")]:
        fig.add_vline(
            x=d_val, line_dash="dot", line_color="gray", line_width=0.8,
            annotation_text=label,
            annotation_position="top",
            annotation_font_size=9,
        )

    fig.update_layout(
        xaxis_title="Call Delta",
        yaxis_title="Implied Volatility",
        yaxis_tickformat=".1%",
        xaxis=dict(range=[0.05, 0.95], dtick=0.1),
        height=450,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font=dict(size=10)),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Skew metrics table
    st.markdown("**Skew & Convexity Metrics**")
    skew_rows = []
    for _, row in sorted_sp.iterrows():
        T = row["T"]
        dte = round(T * 365.25)
        F = spot * np.exp((risk_free - div_yield) * T)

        # Compute IV at standard delta pillars via SVI
        # 25D put ~ k ≈ -0.15 to -0.25, 25D call ~ k ≈ 0.10 to 0.20, ATM ~ k=0
        w_atm = float(np.squeeze(svi_total_variance(0.0, row["a"], row["b"], row["rho"], row["m"], row["sigma"])))
        iv_atm = np.sqrt(max(w_atm, 0.0) / T)

        # Approximate 25-delta strikes via SVI
        k_25p = -0.20  # rough 25Δ put moneyness
        k_25c = 0.15   # rough 25Δ call moneyness

        w_25p = float(np.squeeze(svi_total_variance(k_25p, row["a"], row["b"], row["rho"], row["m"], row["sigma"])))
        w_25c = float(np.squeeze(svi_total_variance(k_25c, row["a"], row["b"], row["rho"], row["m"], row["sigma"])))
        iv_25p = np.sqrt(max(w_25p, 0.0) / T)
        iv_25c = np.sqrt(max(w_25c, 0.0) / T)

        # Risk reversal (skew) and butterfly (convexity)
        rr_25 = iv_25c - iv_25p
        bf_25 = (iv_25c + iv_25p) / 2 - iv_atm

        skew_rows.append({
            "DTE": dte,
            "ATM IV": f"{iv_atm:.2%}",
            "25Δ RR": f"{rr_25:+.2%}",
            "25Δ BF": f"{bf_25:+.2%}",
            "25Δ Put IV": f"{iv_25p:.2%}",
            "25Δ Call IV": f"{iv_25c:.2%}",
        })

    if skew_rows:
        st.dataframe(
            pd.DataFrame(skew_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "**25Δ RR** = 25Δ Call IV − 25Δ Put IV (skew direction). "
            "**25Δ BF** = (25Δ Call + 25Δ Put)/2 − ATM IV (smile convexity/curvature)."
        )
