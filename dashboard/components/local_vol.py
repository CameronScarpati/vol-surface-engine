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

Implementation note: SVI parameters are interpolated smoothly across T
using cubic splines before computing Dupire derivatives.  This eliminates
the finite-difference noise that arises from independently-calibrated
SVI slices at closely-spaced weekly expiries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter

from src.svi_fitter import (
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
)


def _smooth_svi_params(
    slice_params: pd.DataFrame,
) -> dict[str, UnivariateSpline]:
    """Fit smooth cubic splines to each SVI parameter as a function of T.

    Returns a dict mapping parameter name → fitted spline.
    """
    sp = slice_params.sort_values("T")
    T = sp["T"].values

    # Smoothing factor: allow up to 5% residual variance per parameter.
    # With noisy independent fits this is critical — without smoothing
    # the spline would interpolate the noise exactly.
    n = len(T)
    splines = {}
    for name in ("a", "b", "rho", "m", "sigma"):
        vals = sp[name].values
        # s = n * variance * fraction — use a generous smoothing factor.
        # k=3 for cubic, s scaled to number of points.
        variance = np.var(vals) if np.var(vals) > 1e-12 else 1e-6
        s = n * variance * 0.1  # allow 10% of total variance as residual
        try:
            spl = UnivariateSpline(T, vals, k=min(3, n - 1), s=s)
        except Exception:
            # Fallback: linear interpolation if spline fitting fails.
            spl = UnivariateSpline(T, vals, k=1, s=0)
        splines[name] = spl
    return splines


def _compute_local_vol(
    k_grid: np.ndarray,
    T_grid_1d: np.ndarray,
    splines: dict[str, UnivariateSpline],
    spot: float,
    risk_free: float,
    div_yield: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local volatility surface via Dupire's formula.

    Uses smoothed SVI parameter splines for both w(k,T) evaluation and
    analytical dw/dT computation (via spline derivatives), eliminating
    finite-difference noise entirely.

    Returns
    -------
    strike_grid, T_grid, local_vol_grid : 2D arrays
    """
    n_k = len(k_grid)
    n_T = len(T_grid_1d)
    local_vol = np.full((n_T, n_k), np.nan)

    for i, T in enumerate(T_grid_1d):
        # Evaluate smoothed SVI parameters at this T.
        a = float(splines["a"](T))
        b = float(splines["b"](T))
        rho = float(splines["rho"](T))
        m = float(splines["m"](T))
        sigma = float(splines["sigma"](T))

        # Enforce SVI parameter constraints after interpolation.
        b = max(b, 1e-6)
        sigma = max(sigma, 1e-6)
        rho = np.clip(rho, -0.999, 0.999)

        # Total variance and analytical k-derivatives from SVI.
        w = svi_total_variance(k_grid, a, b, rho, m, sigma)
        w_prime = svi_first_derivative(k_grid, b, rho, m, sigma)
        w_double_prime = svi_second_derivative(k_grid, b, rho, m, sigma)

        # ── dw/dT via spline derivatives (analytical, no finite diff) ──
        # w(k, T) = a(T) + b(T) * [rho(T)*(k-m(T)) + sqrt((k-m(T))^2 + sigma(T)^2)]
        # dw/dT = da/dT + db/dT * [...] + b * d[...]/dT
        da = float(splines["a"].derivative()(T))
        db = float(splines["b"].derivative()(T))
        drho = float(splines["rho"].derivative()(T))
        dm = float(splines["m"].derivative()(T))
        dsigma = float(splines["sigma"].derivative()(T))

        u = k_grid - m
        v = np.sqrt(u**2 + sigma**2)
        bracket = rho * u + v

        # d(bracket)/dT = drho*u + rho*(-dm) + d(v)/dT
        # d(v)/dT = (u*(-dm) + sigma*dsigma) / v
        dv_dT = (-u * dm + sigma * dsigma) / np.maximum(v, 1e-10)
        dbracket_dT = drho * u + rho * (-dm) + dv_dT

        dw_dT = da + db * bracket + b * dbracket_dT

        # ── Dupire denominator (Durrleman condition g(k)) ──
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

    # ── Light post-processing ────────────────────────────────────────────
    # Cap outliers (the spline approach should produce very few).
    local_vol = np.where(local_vol > 1.0, np.nan, local_vol)

    # Light Gaussian smoothing for visual polish.
    valid = np.isfinite(local_vol)
    filled = np.where(valid, local_vol, 0.0)
    weights = valid.astype(float)
    sigma_smooth = (0.8, 1.2)
    smoothed_num = gaussian_filter(filled, sigma=sigma_smooth)
    smoothed_den = gaussian_filter(weights, sigma=sigma_smooth)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_vol = np.where(
            smoothed_den > 0.3, smoothed_num / smoothed_den, np.nan,
        )

    # Convert k_grid to strikes for each T.
    F_vals = spot * np.exp((risk_free - div_yield) * T_grid_1d)
    strike_grid = np.outer(F_vals, np.exp(k_grid))
    T_grid = np.tile(T_grid_1d[:, None], (1, n_k))

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

    if slice_params.empty or len(slice_params) < 3:
        st.warning("Not enough fitted slices for local vol computation.")
        return

    sorted_sp = slice_params.sort_values("T")

    # Smooth SVI parameters across T.
    splines = _smooth_svi_params(sorted_sp)

    # Build a regular T grid spanning the fitted expiry range.
    T_min = sorted_sp["T"].values[1]  # skip the very shortest slice
    T_max = sorted_sp["T"].values[-1]
    T_grid_1d = np.linspace(T_min, T_max, 30)

    k_grid = np.linspace(-0.15, 0.15, 60)

    strike_grid, T_grid, local_vol = _compute_local_vol(
        k_grid, T_grid_1d, splines, spot, risk_free, div_yield,
    )

    # Determine z-axis from data.
    valid_vals = local_vol[np.isfinite(local_vol)]
    if len(valid_vals) == 0:
        st.warning("Local vol computation produced no valid values.")
        return
    z_max = min(float(np.percentile(valid_vals, 97)) * 1.3, 1.0)
    z_max = max(z_max, 0.10)  # at least 10%

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
        # Show ~8 evenly-spaced slices to keep the legend readable.
        n_show = min(8, len(T_grid_1d))
        indices = np.linspace(0, len(T_grid_1d) - 1, n_show, dtype=int)
        for i in indices:
            dte = round(T_grid_1d[i] * 365.25)
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
