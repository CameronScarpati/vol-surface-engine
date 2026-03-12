"""
Local Volatility panel — Dupire's formula applied to the fitted SVI surface.

Local volatility σ_loc(K, T) is the unique diffusion coefficient consistent
with the observed European option prices.  Computing it from the SVI fit
demonstrates the connection between implied volatility (a quoting convention)
and the underlying risk-neutral dynamics.

Dupire's formula (1994):

    σ_loc²(K, T) = (∂w/∂T) / (1 - k·w'/w + (w'')/(2) - (w')²/4·(1/w + 1/4))

where w is total variance and k = ln(K/F).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.svi_fitter import (
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
)


def _compute_local_vol(
    k_grid: np.ndarray,
    T_vals: np.ndarray,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local volatility surface via Dupire's formula.

    Uses finite differences in T and analytical SVI derivatives in k.

    Returns
    -------
    strike_grid, T_grid, local_vol_grid : 2D arrays
    """
    sorted_sp = slice_params.sort_values("T")
    n_k = len(k_grid)
    n_T = len(T_vals)

    local_vol = np.full((n_T, n_k), np.nan)

    for i, T in enumerate(T_vals):
        sp_row = sorted_sp[np.isclose(sorted_sp["T"], T, atol=1e-6)]
        if sp_row.empty:
            continue
        sp = sp_row.iloc[0]

        # Total variance and derivatives w.r.t. k
        w = svi_total_variance(k_grid, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
        w_prime = svi_first_derivative(k_grid, sp["b"], sp["rho"], sp["m"], sp["sigma"])
        w_double_prime = svi_second_derivative(k_grid, sp["b"], sp["rho"], sp["m"], sp["sigma"])

        # dw/dT via finite difference between adjacent slices
        if i == 0 and len(T_vals) > 1:
            # Forward difference
            sp_next = sorted_sp[np.isclose(sorted_sp["T"], T_vals[min(i + 1, n_T - 1)], atol=1e-6)]
            if not sp_next.empty:
                sp_n = sp_next.iloc[0]
                w_next = svi_total_variance(k_grid, sp_n["a"], sp_n["b"], sp_n["rho"], sp_n["m"], sp_n["sigma"])
                dw_dT = (w_next - w) / (T_vals[min(i + 1, n_T - 1)] - T)
            else:
                dw_dT = w / T  # fallback: assume linear from origin
        elif i == n_T - 1 and len(T_vals) > 1:
            # Backward difference
            sp_prev = sorted_sp[np.isclose(sorted_sp["T"], T_vals[max(i - 1, 0)], atol=1e-6)]
            if not sp_prev.empty:
                sp_p = sp_prev.iloc[0]
                w_prev = svi_total_variance(k_grid, sp_p["a"], sp_p["b"], sp_p["rho"], sp_p["m"], sp_p["sigma"])
                dw_dT = (w - w_prev) / (T - T_vals[max(i - 1, 0)])
            else:
                dw_dT = w / T
        elif len(T_vals) > 2:
            # Central difference
            sp_prev = sorted_sp[np.isclose(sorted_sp["T"], T_vals[i - 1], atol=1e-6)]
            sp_next = sorted_sp[np.isclose(sorted_sp["T"], T_vals[i + 1], atol=1e-6)]
            if not sp_prev.empty and not sp_next.empty:
                sp_p = sp_prev.iloc[0]
                sp_n = sp_next.iloc[0]
                w_prev = svi_total_variance(k_grid, sp_p["a"], sp_p["b"], sp_p["rho"], sp_p["m"], sp_p["sigma"])
                w_next = svi_total_variance(k_grid, sp_n["a"], sp_n["b"], sp_n["rho"], sp_n["m"], sp_n["sigma"])
                dw_dT = (w_next - w_prev) / (T_vals[i + 1] - T_vals[i - 1])
            else:
                dw_dT = w / T
        else:
            dw_dT = w / T

        # Dupire denominator: 1 - k*w'/w + w''/2 - (w')²/4 * (1/w + 1/4)
        w_safe = np.maximum(w, 1e-10)
        denominator = (
            1.0
            - k_grid * w_prime / w_safe
            + w_double_prime / 2.0
            - (w_prime**2) / 4.0 * (1.0 / w_safe + 0.25)
        )

        # Local variance = dw/dT / denominator
        with np.errstate(divide="ignore", invalid="ignore"):
            local_var = np.where(
                denominator > 1e-8,
                dw_dT / denominator,
                np.nan,
            )

        local_vol[i, :] = np.sqrt(np.maximum(local_var, 0.0))

    # Convert k_grid to strikes for each T
    F_vals = spot * np.exp((risk_free - div_yield) * T_vals)
    strike_grid = np.outer(F_vals, np.exp(k_grid))
    T_grid = np.tile(T_vals[:, None], (1, n_k))

    return strike_grid, T_grid, local_vol


def render_local_vol(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the local volatility surface in Streamlit."""
    st.subheader("Local Volatility Surface (Dupire)")

    if slice_params.empty:
        st.warning("No fitted slices available.")
        return

    sorted_sp = slice_params.sort_values("T")
    T_vals = sorted_sp["T"].values

    k_grid = np.linspace(-0.35, 0.35, 100)

    strike_grid, T_grid, local_vol = _compute_local_vol(
        k_grid, T_vals, slice_params, spot, risk_free, div_yield,
    )

    col_3d, col_slice = st.columns([3, 2])

    with col_3d:
        fig = go.Figure(
            data=[
                go.Surface(
                    x=strike_grid,
                    y=T_grid * 365.25,
                    z=local_vol,
                    colorscale="Inferno",
                    colorbar=dict(title="σ_loc", tickformat=".2%"),
                    hovertemplate=(
                        "Strike: %{x:.1f}<br>"
                        "DTE: %{y:.0f}<br>"
                        "Local Vol: %{z:.2%}<extra></extra>"
                    ),
                )
            ]
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Days to Expiry",
                zaxis_title="Local Volatility",
                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_slice:
        # Local vol slices
        fig2 = go.Figure()
        for i, T in enumerate(T_vals):
            dte = round(T * 365.25)
            valid = np.isfinite(local_vol[i, :])
            if valid.any():
                fig2.add_trace(go.Scatter(
                    x=strike_grid[i, valid],
                    y=local_vol[i, valid],
                    mode="lines",
                    name=f"{dte}d",
                    line=dict(width=1.5),
                ))

        fig2.add_vline(
            x=spot, line_dash="dash", line_color="gray", line_width=1,
            annotation_text="ATM",
        )

        fig2.update_layout(
            xaxis_title="Strike",
            yaxis_title="Local Volatility",
            yaxis_tickformat=".1%",
            height=550,
            margin=dict(l=50, r=20, t=30, b=40),
            legend=dict(font=dict(size=10)),
        )

        st.plotly_chart(fig2, use_container_width=True)
