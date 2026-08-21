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

from dashboard.components.helpers import expiry_line_colors
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
        st.success("No arbitrage violations detected (butterfly + calendar)")
    else:
        failing = []
        if not all_butterfly:
            failing.append("butterfly")
        if not diagnostics.calendar_free:
            failing.append("calendar spread")
        st.error(
            f"Violations flagged by the {' and '.join(failing)} check"
            f"{'s' if len(failing) > 1 else ''}; details in the matching tab below"
        )

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

    g_curves: list[np.ndarray] = []
    line_colors = expiry_line_colors(len(slice_params))
    for trace_i, (_, row) in enumerate(slice_params.iterrows()):
        params = SVIParams(
            a=row["a"],
            b=row["b"],
            rho=row["rho"],
            m=row["m"],
            sigma=row["sigma"],
        )
        g = durrleman_condition(k_grid, params)
        g_curves.append(np.asarray(g))
        dte = round(row["T"] * 365.25)
        label = str(row.get("expiry", f"T={row['T']:.4f}"))
        is_free = diagnostics.butterfly_free.get(label, True)

        fig.add_trace(
            go.Scatter(
                x=k_grid,
                y=g,
                mode="lines",
                name=f"{dte}d {'✓' if is_free else '✗'}",
                line=dict(width=2, color=line_colors[trace_i]),
            )
        )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        line_width=1,
        annotation_text="g(k) = 0 (violation boundary)",
        annotation_position="bottom right",
    )

    # A fitted wing with near-zero total variance sends g(k) to
    # astronomical values; clamp the display to a robust range so the
    # curves near zero stay readable. Exact minima are in the table.
    g_all = np.concatenate(g_curves) if g_curves else np.array([0.0])
    finite = g_all[np.isfinite(g_all)]
    sane = finite[np.abs(finite) < 100]
    if sane.size:
        y_hi = float(np.percentile(sane, 99)) * 1.1
        y_lo = min(-0.05, float(sane.min()) * 1.3)
    else:
        y_hi, y_lo = 3.0, -0.5

    fig.update_layout(
        xaxis_title="Log-moneyness k",
        yaxis_title="g(k)   [Durrleman]",
        yaxis_range=[y_lo, y_hi],
        height=400,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(font=dict(size=10)),
    )

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The vertical axis is clamped to a readable range; g(k) can grow "
        "very large where a fitted wing approaches zero total variance. "
        "Exact per-slice minima are in the table below."
    )

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
                        a=row["a"],
                        b=row["b"],
                        rho=row["rho"],
                        m=row["m"],
                        sigma=row["sigma"],
                    )
                    g = durrleman_condition(k_grid, params)
                    min_g_val = float(np.min(g))
                    break

        rows.append(
            {
                "Slice": str(label)[:10],
                "Status": "PASS" if is_free else "FAIL",
                "min g(k)": f"{min_g_val:.6f}",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


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
                {"Short expiry": str(short)[:10], "Long expiry": str(long)[:10]}
                for short, long in diagnostics.calendar_violation_expiries
            ]
            st.dataframe(
                pd.DataFrame(violation_data),
                width="stretch",
                hide_index=True,
            )

    # Total variance vs T at several k values
    k_probes = [-0.3, -0.15, 0.0, 0.15, 0.3]
    sorted_slices = slice_params.sort_values("T")

    fig = go.Figure()
    probe_colors = expiry_line_colors(len(k_probes))
    for probe_i, k_val in enumerate(k_probes):
        w_vals = []
        T_vals = []
        for _, row in sorted_slices.iterrows():
            w = svi_total_variance(k_val, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
            w_vals.append(float(np.squeeze(w)))
            T_vals.append(row["T"])

        fig.add_trace(
            go.Scatter(
                x=[t * 365.25 for t in T_vals],
                y=w_vals,
                mode="lines+markers",
                name=f"k={k_val:.2f}",
                line=dict(width=2, color=probe_colors[probe_i]),
                marker=dict(size=6, color=probe_colors[probe_i]),
            )
        )

    fig.update_layout(
        xaxis_title="Days to Expiry",
        yaxis_title="Total Variance w(k, T)",
        height=400,
        margin=dict(l=50, r=20, t=30, b=40),
    )

    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Total variance must be non-decreasing in T for each fixed k "
        "to prevent calendar-spread arbitrage."
    )
