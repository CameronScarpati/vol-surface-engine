"""
Residual heatmap: (Market IV - Fitted IV) across the strike x expiry grid.

Highlights statistically significant mispricings using a diverging
blue-white-red colour scale.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.helpers import compute_chain_fitted_iv


def render_residual_heatmap(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> None:
    """Render the residual heatmap in Streamlit."""
    st.subheader("Residual Heatmap (Market IV - Fitted IV)")

    df = compute_chain_fitted_iv(chain, slice_params)
    df = df.dropna(subset=["residual"])

    if df.empty:
        st.warning("No residuals to display.")
        return

    # Compute significance threshold (2 sigma of residuals)
    sigma_resid = df["residual"].std()

    # Create pivot for heatmap
    df["DTE"] = (df["T"] * 365.25).round().astype(int)
    # Adaptive bucket size: ~40 buckets across the strike range.
    strike_range = df["strike"].max() - df["strike"].min()
    bucket_size = max(1.0, round(strike_range / 40.0))
    df["strike_bucket"] = (df["strike"] / bucket_size).round() * bucket_size

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
        f"Residual sigma = {sigma_resid:.4f} | "
        f"**{n_sig}** / {len(df)} points exceed 2 sigma threshold "
        f"(|residual| > {2 * sigma_resid:.4f})"
    )
