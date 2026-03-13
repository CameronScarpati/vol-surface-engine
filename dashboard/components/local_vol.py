"""
Local Volatility panel — Dupire's formula applied to the fitted SVI surface.

Local volatility σ_loc(K, T) is the unique diffusion coefficient consistent
with the observed European option prices.  Computing it from the SVI fit
demonstrates the connection between implied volatility (a quoting convention)
and the underlying risk-neutral dynamics.

Dupire's formula (1994):

    σ_loc²(K, T) = (∂w/∂T) / g(k)

where g(k) is the Durrleman condition:

    g(k) = (1 - k·w'/(2w))² - (w')²/4·(1/w + 1/4) + w''/2

and w is total variance and k = ln(K/F).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter

from src.svi_fitter import (
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
)


def _select_expiries(sorted_sp: pd.DataFrame, min_T: float = 0.04) -> np.ndarray:
    """Select well-spaced expiry slices for stable finite differences.

    Drops expiries shorter than *min_T* (~15 days) and ensures a minimum
    gap between consecutive expiries so that dw/dT is not dominated by
    noise from independently-calibrated SVI parameters.
    """
    T_all = sorted_sp["T"].values
    T_all = T_all[T_all >= min_T]
    if len(T_all) <= 1:
        return T_all

    total_range = T_all[-1] - T_all[0]
    min_gap = max(total_range / 20.0, 0.025)

    selected = [T_all[0]]
    for T in T_all[1:]:
        if T - selected[-1] >= min_gap:
            selected.append(T)
    return np.array(selected)


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
        # Find closest slice params for this T.
        idx = (sorted_sp["T"] - T).abs().argsort().values[:1]
        sp_row = sorted_sp.iloc[idx]
        if abs(sp_row.iloc[0]["T"] - T) > 0.02:
            continue
        sp = sp_row.iloc[0]

        w = svi_total_variance(k_grid, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
        w_prime = svi_first_derivative(k_grid, sp["b"], sp["rho"], sp["m"], sp["sigma"])
        w_double_prime = svi_second_derivative(k_grid, sp["b"], sp["rho"], sp["m"], sp["sigma"])

        # dw/dT via finite difference between adjacent selected slices.
        if i > 0 and i < n_T - 1:
            T_prev, T_next = T_vals[i - 1], T_vals[i + 1]
            idx_p = (sorted_sp["T"] - T_prev).abs().argsort().values[:1]
            idx_n = (sorted_sp["T"] - T_next).abs().argsort().values[:1]
            sp_p = sorted_sp.iloc[idx_p[0]]
            sp_n = sorted_sp.iloc[idx_n[0]]
            w_prev = svi_total_variance(k_grid, sp_p["a"], sp_p["b"], sp_p["rho"], sp_p["m"], sp_p["sigma"])
            w_next = svi_total_variance(k_grid, sp_n["a"], sp_n["b"], sp_n["rho"], sp_n["m"], sp_n["sigma"])
            dw_dT = (w_next - w_prev) / (T_next - T_prev)
        elif i == 0 and n_T > 1:
            idx_n = (sorted_sp["T"] - T_vals[1]).abs().argsort().values[:1]
            sp_n = sorted_sp.iloc[idx_n[0]]
            w_next = svi_total_variance(k_grid, sp_n["a"], sp_n["b"], sp_n["rho"], sp_n["m"], sp_n["sigma"])
            dw_dT = (w_next - w) / (T_vals[1] - T)
        elif i == n_T - 1 and n_T > 1:
            idx_p = (sorted_sp["T"] - T_vals[i - 1]).abs().argsort().values[:1]
            sp_p = sorted_sp.iloc[idx_p[0]]
            w_prev = svi_total_variance(k_grid, sp_p["a"], sp_p["b"], sp_p["rho"], sp_p["m"], sp_p["sigma"])
            dw_dT = (w - w_prev) / (T - T_vals[i - 1])
        else:
            dw_dT = w / T

        # Dupire denominator (Durrleman condition g(k)).
        w_safe = np.maximum(w, 1e-10)
        denominator = (
            (1.0 - k_grid * w_prime / (2.0 * w_safe)) ** 2
            - (w_prime**2) / 4.0 * (1.0 / w_safe + 0.25)
            + w_double_prime / 2.0
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            local_var = np.where(
                (denominator > 0.02) & (dw_dT > 0),
                dw_dT / denominator,
                np.nan,
            )

        local_vol[i, :] = np.sqrt(np.maximum(local_var, 0.0))

    # ── Post-processing ──────────────────────────────────────────────────
    # Cap extreme values before smoothing.
    local_vol = np.where(local_vol > 0.80, np.nan, local_vol)

    # Normalized-convolution Gaussian smoothing (handles NaN properly).
    valid = np.isfinite(local_vol)
    filled = np.where(valid, local_vol, 0.0)
    weights = valid.astype(float)
    sigma_smooth = (1.0, 1.5)
    smoothed_num = gaussian_filter(filled, sigma=sigma_smooth)
    smoothed_den = gaussian_filter(weights, sigma=sigma_smooth)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_vol = np.where(
            smoothed_den > 0.3, smoothed_num / smoothed_den, np.nan,
        )

    # Convert k_grid to strikes for each T.
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

    if slice_params.empty or len(slice_params) < 2:
        st.warning("Not enough fitted slices for local vol computation.")
        return

    sorted_sp = slice_params.sort_values("T")
    T_vals = _select_expiries(sorted_sp)

    if len(T_vals) < 2:
        st.warning("Not enough well-spaced expiry slices for local vol.")
        return

    k_grid = np.linspace(-0.15, 0.15, 60)

    strike_grid, T_grid, local_vol = _compute_local_vol(
        k_grid, T_vals, slice_params, spot, risk_free, div_yield,
    )

    valid_vals = local_vol[np.isfinite(local_vol)]
    if len(valid_vals) == 0:
        st.warning("Local vol computation produced no valid values.")
        return

    z_max = min(float(np.percentile(valid_vals, 97)) * 1.3, 0.80)
    z_max = max(z_max, 0.10)

    col_3d, col_slice = st.columns([3, 2])

    with col_3d:
        fig = go.Figure(
            data=[
                go.Surface(
                    x=strike_grid,
                    y=T_grid * 365.25,
                    z=local_vol,
                    colorscale="Inferno",
                    cmin=0,
                    cmax=z_max,
                    colorbar=dict(title="σ_loc", tickformat=".0%"),
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
                zaxis=dict(tickformat=".0%", range=[0, z_max]),
                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=550,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_slice:
        fig2 = go.Figure()
        n_show = min(8, len(T_vals))
        indices = np.linspace(0, len(T_vals) - 1, n_show, dtype=int)
        for i in indices:
            dte = round(T_vals[i] * 365.25)
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
            yaxis_tickformat=".0%",
            yaxis_range=[0, z_max],
            height=550,
            margin=dict(l=50, r=20, t=30, b=40),
            legend=dict(font=dict(size=10)),
        )

        st.plotly_chart(fig2, use_container_width=True)
