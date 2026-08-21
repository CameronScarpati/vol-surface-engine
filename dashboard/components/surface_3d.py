"""
3D Volatility Surface visualisation.

Renders an interactive Plotly 3D surface of implied volatility over the
(strike, time-to-expiry) grid, with a toggle between market IV and
SVI-fitted IV.

Market IV is shown as scatter markers overlaid on the fitted surface,
since raw market observations are inherently sparse and cannot form a
smooth surface on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.helpers import BLUES_SCALE
from src.svi_fitter import svi_total_variance


def _build_fitted_surface(
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
    strike_lo: float,
    strike_hi: float,
    n_strike: int = 80,
    band_by_T: dict[float, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a regular grid of SVI-fitted IV values.

    When *band_by_T* is given, each slice is evaluated only inside its own
    quoted strike band (NaN elsewhere), so the surface never shows a fit
    extrapolated far beyond the strikes that expiry actually quotes.

    Returns
    -------
    strikes_grid, T_grid, fitted_iv_grid : 2-D arrays
    """
    T_vals = np.sort(slice_params["T"].unique())

    strikes_grid = np.empty((len(T_vals), n_strike))
    T_grid = np.tile(T_vals[:, None], (1, n_strike))
    fitted_iv_grid = np.full_like(strikes_grid, np.nan)

    for i, T in enumerate(T_vals):
        sp_row = slice_params[np.isclose(slice_params["T"], T, atol=1e-6)]

        # Each slice gets its own strike axis spanning only its quoted
        # band, so short maturities are not extrapolated into wings they
        # never quoted and the surface edge stays smooth.
        row_lo, row_hi = strike_lo, strike_hi
        if band_by_T is not None:
            for band_T, band_range in band_by_T.items():
                if abs(band_T - T) < 1e-6:
                    row_lo, row_hi = band_range
                    break
        strikes = np.linspace(row_lo, row_hi, n_strike)
        strikes_grid[i, :] = strikes

        if sp_row.empty:
            continue
        sp = sp_row.iloc[0]

        F = spot * np.exp((risk_free - div_yield) * T)
        k = np.log(strikes / F)
        w = svi_total_variance(k, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
        fitted_iv_grid[i, :] = np.sqrt(np.maximum(w, 0.0) / T)

    return strikes_grid, T_grid, fitted_iv_grid


def _adaptive_iv_cap(fitted_iv_grid: np.ndarray) -> float:
    """Compute an adaptive IV cap based on the actual fitted IV range.

    Uses the 95th percentile × 1.5, with a floor of 0.80 and no ceiling.
    """
    valid = fitted_iv_grid[np.isfinite(fitted_iv_grid)]
    if len(valid) == 0:
        return 1.0
    p95 = float(np.percentile(valid, 95))
    return max(p95 * 1.5, 0.80)


def _get_market_iv_points(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract valid market IV data points from the chain.

    Returns
    -------
    strikes, T_days, market_ivs, residuals : 1-D arrays
        Only includes points that have valid IV and a matching SVI slice.
    """
    df = chain.dropna(subset=["iv"]).copy()
    if df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])

    strikes_list = []
    t_days_list = []
    mkt_iv_list = []
    resid_list = []

    for _, row in df.iterrows():
        T = row["T"]
        sp_match = slice_params[np.isclose(slice_params["T"], T, atol=1e-6)]
        if sp_match.empty:
            continue

        sp = sp_match.iloc[0]
        F = spot * np.exp((risk_free - div_yield) * T)
        k = np.log(row["strike"] / F)
        w = svi_total_variance(k, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
        fitted_iv = float(np.sqrt(max(float(np.squeeze(w)), 0.0) / T)) if T > 0 else np.nan

        strikes_list.append(row["strike"])
        t_days_list.append(T * 365.25)
        mkt_iv_list.append(row["iv"])
        resid_list.append(row["iv"] - fitted_iv if np.isfinite(fitted_iv) else np.nan)

    return (
        np.array(strikes_list),
        np.array(t_days_list),
        np.array(mkt_iv_list),
        np.array(resid_list),
    )


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

    # Determine strike range from chain data with valid IVs
    valid_iv_chain = chain.dropna(subset=["iv"])
    if valid_iv_chain.empty:
        st.warning("No valid implied volatilities to display.")
        return

    # Use the range of strikes that have valid IVs, with padding
    iv_strikes = valid_iv_chain["strike"]
    T_mid = float(np.median(slice_params["T"].unique()))
    F_mid = spot * np.exp((risk_free - div_yield) * T_mid)
    k_max = 0.35
    strike_lo = max(iv_strikes.min() * 0.95, F_mid * np.exp(-k_max))
    strike_hi = min(iv_strikes.max() * 1.05, F_mid * np.exp(k_max))

    # Build the fitted surface grid
    band_by_T = {
        float(t): (float(g["strike"].min()), float(g["strike"].max()))
        for t, g in valid_iv_chain.groupby("T")
    }
    strikes_grid, T_grid, fit_iv = _build_fitted_surface(
        slice_params,
        spot,
        risk_free,
        div_yield,
        strike_lo,
        strike_hi,
        band_by_T=band_by_T,
    )

    # Adaptive IV cap for the fitted surface to prevent extreme wing artifacts
    iv_cap = _adaptive_iv_cap(fit_iv)
    fit_iv_capped = np.where(fit_iv > iv_cap, np.nan, fit_iv)

    # Get market IV scatter points
    mkt_strikes, mkt_t_days, mkt_ivs, residuals = _get_market_iv_points(
        chain,
        slice_params,
        spot,
        risk_free,
        div_yield,
    )

    if view_mode == "SVI-fitted":
        _render_fitted_surface(
            fig_data=[], strikes_grid=strikes_grid, T_grid=T_grid, fit_iv=fit_iv_capped, st=st
        )
    elif view_mode == "Market IV":
        _render_market_iv(
            strikes_grid,
            T_grid,
            fit_iv_capped,
            mkt_strikes,
            mkt_t_days,
            mkt_ivs,
        )
    else:
        _render_residual(
            strikes_grid,
            T_grid,
            fit_iv_capped,
            mkt_strikes,
            mkt_t_days,
            mkt_ivs,
            residuals,
        )


def _render_fitted_surface(
    fig_data,
    strikes_grid,
    T_grid,
    fit_iv,
    st,
) -> None:
    """Render the SVI-fitted surface."""
    z_flat = fit_iv[np.isfinite(fit_iv)]
    if len(z_flat) == 0:
        st.warning("No fitted IV data available.")
        return

    z_lo = max(float(np.percentile(z_flat, 2)), 0.0)
    z_hi = float(np.percentile(z_flat, 98)) * 1.2
    z_hi = max(z_hi, 0.10)
    z_range = [z_lo, z_hi]

    fig = go.Figure(
        data=[
            go.Surface(
                x=strikes_grid,
                y=T_grid * 365.25,
                z=fit_iv,
                colorscale=BLUES_SCALE,
                colorbar=dict(title="IV", tickformat=".3f"),
                cmin=z_range[0],
                cmax=z_range[1],
                hovertemplate=(
                    "Strike: %{x:.1f}<br>DTE: %{y:.0f} days<br>IV: %{z:.4f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility",
            zaxis=dict(range=z_range),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    st.plotly_chart(fig, width="stretch")


def _render_market_iv(
    strikes_grid,
    T_grid,
    fit_iv,
    mkt_strikes,
    mkt_t_days,
    mkt_ivs,
) -> None:
    """Render market IV as scatter points overlaid on the fitted surface."""
    fig = go.Figure()

    # Add the fitted surface as a semi-transparent base
    fit_flat = fit_iv[np.isfinite(fit_iv)]
    if len(fit_flat) > 0:
        z_lo = max(float(np.percentile(fit_flat, 2)), 0.0)
        z_hi = float(np.percentile(fit_flat, 98)) * 1.2
    else:
        z_lo, z_hi = 0.0, 0.50

    # Extend range to include market IV values
    if len(mkt_ivs) > 0:
        mkt_lo = float(np.percentile(mkt_ivs, 2))
        mkt_hi = float(np.percentile(mkt_ivs, 98))
        z_lo = min(z_lo, max(mkt_lo * 0.9, 0.0))
        z_hi = max(z_hi, mkt_hi * 1.1)
    z_hi = max(z_hi, 0.10)
    z_range = [z_lo, z_hi]

    # Semi-transparent fitted surface for context
    fig.add_trace(
        go.Surface(
            x=strikes_grid,
            y=T_grid * 365.25,
            z=fit_iv,
            colorscale=BLUES_SCALE,
            opacity=0.4,
            showscale=False,
            name="SVI Fit",
            hovertemplate=(
                "Strike: %{x:.1f}<br>"
                "DTE: %{y:.0f} days<br>"
                "Fitted IV: %{z:.4f}<extra>SVI Fit</extra>"
            ),
        )
    )

    # Market IV scatter points
    if len(mkt_ivs) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=mkt_strikes,
                y=mkt_t_days,
                z=mkt_ivs,
                mode="markers",
                marker=dict(
                    size=4,
                    color=mkt_ivs,
                    colorscale=BLUES_SCALE,
                    colorbar=dict(title="Market IV", tickformat=".3f"),
                    cmin=z_range[0],
                    cmax=z_range[1],
                ),
                name="Market IV",
                hovertemplate=(
                    "Strike: %{x:.1f}<br>"
                    "DTE: %{y:.0f} days<br>"
                    "Market IV: %{z:.4f}<extra>Market</extra>"
                ),
            )
        )
    else:
        st.warning("No valid market IV data points to display.")

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility",
            zaxis=dict(range=z_range),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    st.plotly_chart(fig, width="stretch")

    if len(mkt_ivs) > 0:
        st.caption(
            f"Showing **{len(mkt_ivs)}** market IV observations "
            f"(range: {float(mkt_ivs.min()):.4f} – {float(mkt_ivs.max()):.4f}) "
            f"overlaid on the SVI-fitted surface."
        )


def _render_residual(
    strikes_grid,
    T_grid,
    fit_iv,
    mkt_strikes,
    mkt_t_days,
    mkt_ivs,
    residuals,
) -> None:
    """Render residuals as scatter points with fitted surface for reference."""
    fig = go.Figure()

    valid_resid = np.isfinite(residuals)
    resid_valid = residuals[valid_resid]

    if len(resid_valid) == 0:
        st.warning("No residual data available.")
        return

    abs_max = max(float(np.percentile(np.abs(resid_valid), 95)), 0.005)
    z_range = [-abs_max, abs_max]

    # Zero-plane for reference
    fit_flat = fit_iv[np.isfinite(fit_iv)]
    if len(fit_flat) > 0:
        zero_grid = np.zeros_like(fit_iv)
        fig.add_trace(
            go.Surface(
                x=strikes_grid,
                y=T_grid * 365.25,
                z=zero_grid,
                colorscale=[[0, "rgba(200,200,200,0.3)"], [1, "rgba(200,200,200,0.3)"]],
                showscale=False,
                opacity=0.3,
                name="Zero",
                hoverinfo="skip",
            )
        )

    # Residual scatter points
    fig.add_trace(
        go.Scatter3d(
            x=mkt_strikes[valid_resid],
            y=mkt_t_days[valid_resid],
            z=resid_valid,
            mode="markers",
            marker=dict(
                size=5,
                color=resid_valid,
                colorscale="RdBu_r",
                colorbar=dict(title="Residual", tickformat=".4f"),
                cmin=z_range[0],
                cmax=z_range[1],
            ),
            name="Residual",
            hovertemplate=(
                "Strike: %{x:.1f}<br>DTE: %{y:.0f} days<br>Residual: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Residual (Market − Fitted)",
            zaxis=dict(range=z_range),
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
    )
    st.plotly_chart(fig, width="stretch")

    n_pos = (resid_valid > 0).sum()
    n_neg = (resid_valid < 0).sum()
    st.caption(
        f"**{len(resid_valid)}** residual points | "
        f"Mean: {float(resid_valid.mean()):.4f} | "
        f"Std: {float(resid_valid.std()):.4f} | "
        f"{n_pos} above fit, {n_neg} below fit"
    )
