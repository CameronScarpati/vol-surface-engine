"""
Residual heatmap: (Market IV − Fitted IV) across the strike × expiry grid.

Highlights statistically significant mispricings using a diverging
blue-white-red colour scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.svi_fitter import svi_total_variance


def render_residual_heatmap(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the residual heatmap in Streamlit."""
    st.subheader("Residual Heatmap (Market IV − Fitted IV)")

    df = chain.dropna(subset=["iv"]).copy()
    if df.empty or slice_params.empty:
        st.warning("Insufficient data for residual heatmap.")
        return

    fitted_ivs = []
    for _, row in df.iterrows():
        T = row["T"]
        S = row["S"]
        r = row["r"]
        q = row["q"]
        K = row["strike"]
        F = S * np.exp((r - q) * T)
        k = np.log(K / F)

        sp_mask = np.isclose(slice_params["T"].values, T, atol=1e-6)
        if sp_mask.any():
            sp = slice_params[sp_mask].iloc[0]
            w = svi_total_variance(
                k, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"]
            )
            w_val = float(np.squeeze(w))
            fitted_ivs.append(float(np.sqrt(max(w_val, 0.0) / T)) if T > 0 else np.nan)
        else:
            fitted_ivs.append(np.nan)

    df["fitted_iv"] = fitted_ivs
    df["residual"] = df["iv"] - df["fitted_iv"]
    df = df.dropna(subset=["residual"])

    if df.empty:
        st.warning("No residuals to display.")
        return

    # Compute significance threshold (2σ of residuals)
    sigma_resid = df["residual"].std()

    # Create pivot for heatmap
    df["DTE"] = (df["T"] * 365.25).round().astype(int)
    df["strike_bucket"] = (df["strike"] / 2).round() * 2  # 2-point buckets

    pivot = df.pivot_table(
        values="residual",
        index="DTE",
        columns="strike_bucket",
        aggfunc="mean",
    )

    pivot = pivot.sort_index(ascending=False)

    # Color bounds symmetric around zero
    abs_max = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.005)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="RdBu_r",
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title="Residual IV"),
            hovertemplate=(
                "Strike: %{x:.0f}<br>"
                "DTE: %{y}d<br>"
                "Residual: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Strike",
        yaxis_title="Days to Expiry",
        height=450,
        margin=dict(l=50, r=20, t=30, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Significance summary
    n_sig = (df["residual"].abs() > 2 * sigma_resid).sum()
    st.caption(
        f"Residual σ = {sigma_resid:.4f} | "
        f"**{n_sig}** / {len(df)} points exceed 2σ threshold "
        f"(|residual| > {2 * sigma_resid:.4f})"
    )
