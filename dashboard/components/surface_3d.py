"""
3D Volatility Surface visualisation.

Renders an interactive Plotly 3D surface of implied volatility over the
(strike, time-to-expiry) grid, with a toggle between market IV and
SVI-fitted IV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.svi_fitter import svi_total_variance


def _build_surface_grid(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
    n_strike: int = 80,
    n_expiry: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build regular grids for the market-IV and fitted-IV surfaces.

    Returns
    -------
    strikes_grid, T_grid : 2-D meshgrid arrays
    market_iv_grid, fitted_iv_grid : 2-D arrays (NaN where no data)
    """
    strikes = np.linspace(
        chain["strike"].min(),
        chain["strike"].max(),
        n_strike,
    )
    T_vals = np.sort(slice_params["T"].unique())

    strikes_grid = np.tile(strikes, (len(T_vals), 1))
    T_grid = np.tile(T_vals[:, None], (1, n_strike))

    fitted_iv_grid = np.full_like(strikes_grid, np.nan)
    market_iv_grid = np.full_like(strikes_grid, np.nan)

    for i, T in enumerate(T_vals):
        sp_row = slice_params[np.isclose(slice_params["T"], T, atol=1e-6)]
        if sp_row.empty:
            continue
        sp = sp_row.iloc[0]

        F = spot * np.exp((risk_free - div_yield) * T)
        k = np.log(strikes / F)
        w = svi_total_variance(k, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
        fitted_iv_grid[i, :] = np.sqrt(np.maximum(w, 0.0) / T)

        # Scatter market points onto nearest grid columns
        slice_data = chain[np.isclose(chain["T"], T, atol=1e-6)]
        for _, row in slice_data.iterrows():
            if np.isnan(row["iv"]):
                continue
            j = int(np.argmin(np.abs(strikes - row["strike"])))
            market_iv_grid[i, j] = row["iv"]

    return strikes_grid, T_grid, market_iv_grid, fitted_iv_grid


def render_surface_3d(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the 3D volatility surface in Streamlit."""
    st.subheader("3D Implied Volatility Surface")

    view_mode = st.radio(
        "Surface view",
        ["SVI-fitted", "Market IV", "Residual (Market − Fitted)"],
        horizontal=True,
        key="surface_view",
    )

    strikes_grid, T_grid, mkt_iv, fit_iv = _build_surface_grid(
        chain, slice_params, spot, risk_free, div_yield,
    )

    residuals = mkt_iv - fit_iv

    if view_mode == "SVI-fitted":
        z = fit_iv
        colorscale = "Viridis"
        cbar_title = "IV"
    elif view_mode == "Market IV":
        z = mkt_iv
        colorscale = "Viridis"
        cbar_title = "IV"
    else:
        z = residuals
        colorscale = "RdBu_r"
        cbar_title = "Residual"

    fig = go.Figure(
        data=[
            go.Surface(
                x=strikes_grid,
                y=T_grid * 365.25,  # show in days
                z=z,
                colorscale=colorscale,
                colorbar=dict(title=cbar_title, tickformat=".3f"),
                hovertemplate=(
                    "Strike: %{x:.1f}<br>"
                    "DTE: %{y:.0f} days<br>"
                    "Value: %{z:.4f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility" if "Residual" not in view_mode else "Residual",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)
