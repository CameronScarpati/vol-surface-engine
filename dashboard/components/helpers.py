"""Shared helper functions for dashboard components.

Consolidates repeated computation patterns (forward price, log-moneyness,
fitted IV from SVI params) so each component doesn't re-implement them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.svi_fitter import svi_total_variance

# Tolerance for matching floating-point T values across DataFrames.
T_TOLERANCE = 1e-6


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
