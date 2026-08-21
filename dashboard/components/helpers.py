"""Shared helper functions for dashboard components.

Consolidates repeated computation patterns (forward price, log-moneyness,
fitted IV from SVI params) so components do not re-implement them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.svi_fitter import svi_total_variance

# Tolerance for matching floating-point T values across DataFrames.
T_TOLERANCE = 1e-6

# Shared chart styling: one accent pair and a single-hue ramp so every
# panel reads as part of the same system.
ACCENT = "#2a78d6"
ACCENT_WARM = "#eb6834"
INK_SOFT = "#52514e"

# Blue ramp, light to dark, used both as a Plotly colorscale and for
# maturity-ordered line families.
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#1c5cab",
    "#104281",
    "#0d366b",
]
BLUES_SCALE = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]


def expiry_line_colors(n: int) -> list[str]:
    """Ordered colors for a family of expiry lines, light for the
    shortest maturity through dark for the longest, sampled from the
    visible part of the blue ramp."""
    if n <= 1:
        return [ACCENT]
    lo, hi = 1, len(BLUE_RAMP) - 1  # skip the lightest step on white
    idx = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return [BLUE_RAMP[round(i)] for i in idx]


def forward_price(S: float, r: float, q: float, T: float) -> float:
    """Compute the forward price F = S * exp((r - q) * T)."""
    return S * np.exp((r - q) * T)


def log_moneyness(strike: float | np.ndarray, F: float) -> float | np.ndarray:
    """Compute log-moneyness k = ln(K / F)."""
    return np.log(strike / F)


def get_slice_row(
    slice_params: pd.DataFrame,
    T: float,
) -> pd.Series | None:
    """Look up SVI parameters for the expiry closest to *T*.

    Returns ``None`` if no match is found within ``T_TOLERANCE``.
    """
    mask = np.isclose(slice_params["T"].values, T, atol=T_TOLERANCE)
    if not mask.any():
        return None
    return slice_params[mask].iloc[0]


def fitted_iv_from_svi(
    k: float | np.ndarray,
    sp: pd.Series,
    T: float,
) -> float | np.ndarray:
    """Compute implied volatility from SVI total-variance parameters.

    Parameters
    ----------
    k : log-moneyness value(s)
    sp : Series with SVI parameter columns (a, b, rho, m, sigma)
    T : time to expiry (years)

    Returns
    -------
    Implied volatility (scalar or array).
    """
    w = svi_total_variance(k, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"])
    w = np.maximum(np.squeeze(w), 0.0)
    if T <= 0:
        return np.nan
    return np.sqrt(w / T)


def compute_chain_fitted_iv(
    chain: pd.DataFrame,
    slice_params: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``fitted_iv`` and ``residual`` columns to a chain with IV data.

    This is the vectorized version of the pattern repeated in
    residual_heatmap.py, term_structure.py, and surface.py.
    """
    df = chain.dropna(subset=["iv"]).copy()
    if df.empty or slice_params.empty:
        df["fitted_iv"] = np.nan
        df["residual"] = np.nan
        return df

    fitted = np.full(len(df), np.nan)

    for i, (_, row) in enumerate(df.iterrows()):
        T = row["T"]
        sp = get_slice_row(slice_params, T)
        if sp is None:
            continue
        F = forward_price(row["S"], row["r"], row["q"], T)
        k = log_moneyness(row["strike"], F)
        fitted[i] = fitted_iv_from_svi(k, sp, T)

    df["fitted_iv"] = fitted
    df["residual"] = df["iv"] - df["fitted_iv"]
    return df
