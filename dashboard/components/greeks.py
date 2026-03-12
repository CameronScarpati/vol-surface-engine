"""
Greeks panel — Delta, Gamma, Vega surfaces from the fitted SVI surface.

The primary purpose of constructing an arbitrage-free volatility surface is
to price and hedge derivatives.  This panel computes Black-Scholes Greeks
using the fitted IV at each (strike, T) point, giving traders the
sensitivity profile across the entire surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from src.svi_fitter import svi_total_variance


def _bs_greeks(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    sigma: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute Black-Scholes Greeks (call convention) for arrays of strikes/vols.

    Returns dict with keys: delta, gamma, vega, theta.
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    delta = np.exp(-q * T) * norm.cdf(d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * np.exp(-q * T) * sqrt_T * norm.pdf(d1) / 100  # per 1% vol move
    theta = (
        -(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * sqrt_T)
        - r * K * np.exp(-r * T) * norm.cdf(d2)
        + q * S * np.exp(-q * T) * norm.cdf(d1)
    ) / 365.25  # per calendar day

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def render_greeks(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the Greeks analysis panel in Streamlit."""
    st.subheader("Greeks Surface")

    if slice_params.empty:
        st.warning("No fitted slices available.")
        return

    greek_choice = st.radio(
        "Select Greek",
        ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)"],
        horizontal=True,
        key="greek_choice",
    )

    greek_key = {
        "Delta (Δ)": "delta",
        "Gamma (Γ)": "gamma",
        "Vega (ν)": "vega",
        "Theta (Θ)": "theta",
    }[greek_choice]

    sorted_sp = slice_params.sort_values("T")
    T_vals = sorted_sp["T"].values

    # Build strike grid
    strike_min = chain["strike"].min()
    strike_max = chain["strike"].max()
    n_strikes = 80
    strikes = np.linspace(strike_min, strike_max, n_strikes)

    # Compute Greeks on the grid
    greek_grid = np.full((len(T_vals), n_strikes), np.nan)

    for i, (_, row) in enumerate(sorted_sp.iterrows()):
        T = row["T"]
        F = spot * np.exp((risk_free - div_yield) * T)
        k = np.log(strikes / F)
        w = svi_total_variance(k, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        iv = np.sqrt(np.maximum(w, 1e-10) / T)

        greeks = _bs_greeks(spot, strikes, T, risk_free, div_yield, iv)
        greek_grid[i, :] = greeks[greek_key]

    # Colorscale and formatting
    colorscale_map = {
        "delta": "Viridis",
        "gamma": "Hot",
        "vega": "Cividis",
        "theta": "RdBu_r",
    }
    fmt_map = {
        "delta": ".3f",
        "gamma": ".6f",
        "vega": ".2f",
        "theta": ".3f",
    }
    unit_map = {
        "delta": "Δ",
        "gamma": "Γ",
        "vega": "ν (per 1% vol)",
        "theta": "Θ (per day)",
    }

    fig = go.Figure(
        data=[
            go.Surface(
                x=np.tile(strikes, (len(T_vals), 1)),
                y=np.tile(T_vals[:, None] * 365.25, (1, n_strikes)),
                z=greek_grid,
                colorscale=colorscale_map[greek_key],
                colorbar=dict(title=unit_map[greek_key], tickformat=fmt_map[greek_key]),
                hovertemplate=(
                    "Strike: %{x:.1f}<br>"
                    "DTE: %{y:.0f}<br>"
                    f"{unit_map[greek_key]}: %{{z:{fmt_map[greek_key]}}}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title=unit_map[greek_key],
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Per-slice greek profile
    st.markdown("**Cross-sectional view** — Greeks vs strike for each expiry slice:")

    fig2 = go.Figure()
    for i, (_, row) in enumerate(sorted_sp.iterrows()):
        dte = round(row["T"] * 365.25)
        fig2.add_trace(go.Scatter(
            x=strikes,
            y=greek_grid[i, :],
            mode="lines",
            name=f"{dte}d",
            line=dict(width=1.5),
        ))

    fig2.update_layout(
        xaxis_title="Strike",
        yaxis_title=unit_map[greek_key],
        height=380,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font=dict(size=10)),
    )

    # Add ATM reference line
    fig2.add_vline(
        x=spot, line_dash="dash", line_color="gray", line_width=1,
        annotation_text="ATM",
    )

    st.plotly_chart(fig2, use_container_width=True)
