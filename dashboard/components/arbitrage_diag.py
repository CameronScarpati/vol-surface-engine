"""
Arbitrage diagnostics panel.

Visualises the Durrleman butterfly condition g(k) per expiry and the
calendar-spread total-variance monotonicity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.arbitrage import (
    ArbitrageDiagnostics,
    durrleman_condition,
)
from src.svi_fitter import SVIParams, svi_total_variance


def render_arbitrage_diagnostics(
    slice_params: pd.DataFrame,
    diagnostics: ArbitrageDiagnostics,
) -> None:
    """Render the arbitrage diagnostics panel in Streamlit."""
    st.subheader("Arbitrage Diagnostics")

    # Overall status
    all_butterfly = all(diagnostics.butterfly_free.values()) if diagnostics.butterfly_free else True
    is_arb_free = all_butterfly and diagnostics.calendar_free

    if is_arb_free:
        st.success("Surface is arbitrage-free (butterfly + calendar)")
    else:
        st.error("Arbitrage violations detected")

    tab_butterfly, tab_calendar = st.tabs(["Butterfly (Durrleman)", "Calendar Spread"])

    # --- Butterfly tab ---
    with tab_butterfly:
        _render_butterfly(slice_params, diagnostics)

    # --- Calendar tab ---
    with tab_calendar:
        _render_calendar(slice_params, diagnostics)


def _render_butterfly(
    slice_params: pd.DataFrame,
    diagnostics: ArbitrageDiagnostics,
) -> None:
    """Durrleman condition g(k) per expiry."""
    k_grid = np.linspace(-0.5, 0.5, 201)

    fig = go.Figure()

    for _, row in slice_params.iterrows():
        params = SVIParams(
            a=row["a"], b=row["b"], rho=row["rho"],
            m=row["m"], sigma=row["sigma"],
        )
        g = durrleman_condition(k_grid, params)
        dte = round(row["T"] * 365.25)
        label = str(row.get("expiry", f"T={row['T']:.4f}"))
        is_free = diagnostics.butterfly_free.get(label, True)

        fig.add_trace(go.Scatter(
            x=k_grid,
            y=g,
            mode="lines",
            name=f"{dte}d {'✓' if is_free else '✗'}",
            line=dict(width=1.5),
        ))

    # Zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color="red", line_width=1,
        annotation_text="g(k) = 0 (violation boundary)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis_title="Log-moneyness k",
        yaxis_title="g(k)   [Durrleman]",
        height=400,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font=dict(size=10)),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    rows = []
    for label, is_free in diagnostics.butterfly_free.items():
        # Compute min g(k) for display
        min_g_val = np.nan
        if label in diagnostics.butterfly_violations:
            min_g_val = float(np.min(diagnostics.butterfly_violations[label]))
        else:
            # Slice is arb-free; compute min g(k) for reference
            for _, row in slice_params.iterrows():
                if str(row.get("expiry", f"T={row['T']:.4f}")) == label:
                    params = SVIParams(
                        a=row["a"], b=row["b"], rho=row["rho"],
                        m=row["m"], sigma=row["sigma"],
                    )
                    g = durrleman_condition(k_grid, params)
                    min_g_val = float(np.min(g))
                    break

        rows.append({
            "Slice": label,
            "Status": "PASS" if is_free else "FAIL",
            "min g(k)": f"{min_g_val:.6f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_calendar(
    slice_params: pd.DataFrame,
    diagnostics: ArbitrageDiagnostics,
) -> None:
    """Calendar-spread total variance monotonicity."""
    if diagnostics.calendar_free:
        st.success("No calendar-spread arbitrage detected")
    else:
        n_violations = len(diagnostics.calendar_violation_expiries)
        st.error(f"{n_violations} calendar-spread violation(s) detected")
        if diagnostics.calendar_violation_expiries:
            violation_data = [
                {"Short expiry": short, "Long expiry": long}
                for short, long in diagnostics.calendar_violation_expiries
            ]
            st.dataframe(
                pd.DataFrame(violation_data),
                use_container_width=True,
                hide_index=True,
            )

    # Total variance vs T at several k values
    k_probes = [-0.3, -0.15, 0.0, 0.15, 0.3]
    sorted_slices = slice_params.sort_values("T")

    fig = go.Figure()
    for k_val in k_probes:
        w_vals = []
        T_vals = []
        for _, row in sorted_slices.iterrows():
            w = svi_total_variance(
                k_val, row["a"], row["b"], row["rho"], row["m"], row["sigma"]
            )
            w_vals.append(float(np.squeeze(w)))
            T_vals.append(row["T"])

        fig.add_trace(go.Scatter(
            x=[t * 365.25 for t in T_vals],
            y=w_vals,
            mode="lines+markers",
            name=f"k={k_val:.2f}",
            line=dict(width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        xaxis_title="Days to Expiry",
        yaxis_title="Total Variance w(k, T)",
        height=400,
        margin=dict(l=50, r=20, t=30, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Total variance must be non-decreasing in T for each fixed k "
        "to prevent calendar-spread arbitrage."
    )
