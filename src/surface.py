"""
Surface construction and query layer.

Provides a unified ``VolSurface`` object that wraps the full pipeline:
data loading → IV extraction → SVI calibration → arbitrage diagnostics.

This module is the primary entry point for the dashboard and for
programmatic access to the fitted volatility surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.arbitrage import ArbitrageDiagnostics, generate_diagnostics
from src.iv_engine import compute_all_iv
from src.svi_fitter import (
    fit_all_slices,
    interpolate_surface,
    svi_total_variance,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VolSurface",
    "build_surface",
]


@dataclass
class VolSurface:
    """Fully calibrated volatility surface with diagnostics."""

    chain: pd.DataFrame          # options chain with 'iv' column
    slice_params: pd.DataFrame   # SVI params per expiry
    diagnostics: ArbitrageDiagnostics
    spot: float
    risk_free: float
    div_yield: float

    def iv(self, strike: float, T: float) -> float:
        """Query interpolated implied volatility at (strike, T)."""
        F = self.spot * np.exp((self.risk_free - self.div_yield) * T)
        k = np.log(strike / F)
        w = interpolate_surface(k, T, self.slice_params)
        w_scalar = float(np.squeeze(w))
        return float(np.sqrt(max(w_scalar, 0.0) / T)) if T > 0 else np.nan

    def fitted_iv_for_chain(self) -> pd.DataFrame:
        """Add ``fitted_iv`` and ``residual`` columns to the chain."""
        df = self.chain.copy()
        fitted = np.empty(len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            T = row["T"]
            S = row["S"]
            r = row["r"]
            q = row["q"]
            K = row["strike"]
            F = S * np.exp((r - q) * T)
            k = np.log(K / F)

            # Find the matching expiry slice
            mask = np.isclose(self.slice_params["T"].values, T, atol=1e-6)
            if mask.any():
                sp = self.slice_params[mask].iloc[0]
                w = svi_total_variance(
                    k, sp["a"], sp["b"], sp["rho"], sp["m"], sp["sigma"]
                )
                w_val = float(np.squeeze(w))
                fitted[i] = float(np.sqrt(max(w_val, 0.0) / T)) if T > 0 else np.nan
            else:
                w = interpolate_surface(k, T, self.slice_params)
                w_val = float(np.squeeze(w))
                fitted[i] = float(np.sqrt(max(w_val, 0.0) / T)) if T > 0 else np.nan

        df["fitted_iv"] = fitted
        df["residual"] = df["iv"] - df["fitted_iv"]
        return df

    @property
    def expiries(self) -> np.ndarray:
        return self.slice_params["T"].values

    @property
    def expiry_dates(self) -> list:
        if "expiry" in self.slice_params.columns:
            return sorted(self.slice_params["expiry"].unique())
        return []


def build_surface(
    chain: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
) -> VolSurface:
    """Full pipeline: IV extraction → SVI fit → arbitrage diagnostics.

    Parameters
    ----------
    chain : pd.DataFrame
        Clean options chain from ``data_loader``.
    spot, risk_free, div_yield : float
        Market parameters.

    Returns
    -------
    VolSurface
    """
    logger.info("Building volatility surface (%d options)", len(chain))

    # Step 0: Select OTM options only (calls K > F, puts K < F).
    # This is standard practice on derivatives desks — OTM options have
    # higher vega and more reliable IV extraction.  At-the-money options
    # (|k| < 0.005) are kept for both types.
    F = chain["S"] * np.exp((chain["r"] - chain["q"]) * chain["T"])
    k = np.log(chain["strike"] / F)
    is_call = chain["option_type"] == "call"
    atm_band = k.abs() < 0.005
    otm_mask = (is_call & (k >= 0)) | (~is_call & (k <= 0)) | atm_band
    chain = chain[otm_mask].copy()
    n_otm = otm_mask.sum()
    logger.info("OTM filter: kept %d / %d options", n_otm, len(otm_mask))

    # Step 1: extract implied volatilities
    chain_iv = compute_all_iv(chain)
    n_valid = chain_iv["iv"].notna().sum()
    logger.info("IV extraction complete: %d / %d valid", n_valid, len(chain_iv))

    # Step 1b: discard IVs that would poison the SVI fit.
    #
    # Two filters applied in sequence:
    # (a) Moneyness filter — remove deep ITM and far OTM options where IV
    #     extraction is unreliable.  Adaptive bounds ensure we keep at least
    #     5 valid IVs per slice for SVI fitting.
    # (b) Per-slice outlier removal — MAD-based filter catches stale quotes
    #     and data errors.
    has_iv = chain_iv["iv"].notna()
    F = chain_iv["S"] * np.exp((chain_iv["r"] - chain_iv["q"]) * chain_iv["T"])
    k = np.log(chain_iv["strike"] / F)

    # (a) Moneyness filter: discard far OTM and deep ITM.
    #     Uses asymmetric bounds per option type.  If the initial bounds
    #     would leave fewer than 5 options in a slice, widen them.
    is_call = chain_iv["option_type"] == "call"

    # Start with standard bounds but adapt per slice
    k_call_lo, k_call_hi = -0.20, 0.35
    k_put_lo, k_put_hi = -0.35, 0.20

    too_far = has_iv & (
        (is_call & ((k < k_call_lo) | (k > k_call_hi)))
        | (~is_call & ((k > k_put_hi) | (k < k_put_lo)))
    )

    # Check per-slice survival: if fewer than 5 options survive in a slice,
    # don't apply the moneyness filter for that slice.
    n_far_adjusted = 0
    for T_val in chain_iv["T"].unique():
        slice_mask = chain_iv["T"] == T_val
        slice_has_iv = slice_mask & has_iv
        slice_too_far = slice_mask & too_far
        n_surviving = slice_has_iv.sum() - slice_too_far.sum()
        if n_surviving < 5:
            # Keep all IVs for this slice — not enough survive the filter
            too_far = too_far & ~slice_mask
        else:
            n_far_adjusted += slice_too_far.sum()

    n_far = too_far.sum()
    if n_far > 0:
        chain_iv.loc[too_far, "iv"] = np.nan
        logger.info("Discarded %d far-from-ATM / deep-ITM IVs", n_far)

    # (b) Per-slice outlier removal using median absolute deviation.
    #     Within each expiry, remove IVs that deviate more than 3x MAD
    #     from the slice median.  Only remove if at least 5 IVs remain.
    n_outlier = 0
    for T_val in chain_iv["T"].unique():
        mask = (chain_iv["T"] == T_val) & chain_iv["iv"].notna()
        if mask.sum() < 5:
            continue
        iv_slice = chain_iv.loc[mask, "iv"]
        med = iv_slice.median()
        mad = (iv_slice - med).abs().median()
        if mad < 0.005:
            mad = 0.005  # floor to avoid rejecting everything

        # Remove IVs too far from the slice median.
        outlier = mask & (
            ((chain_iv["iv"] - med).abs() > 3.0 * mad)
            | (chain_iv["iv"] > max(3.0 * med, 1.5))
        )

        # Don't remove outliers if it would leave fewer than 5 points
        if mask.sum() - outlier.sum() < 5:
            continue

        n_out = outlier.sum()
        if n_out > 0:
            chain_iv.loc[outlier, "iv"] = np.nan
            n_outlier += n_out
    if n_outlier > 0:
        logger.info("Discarded %d per-slice outlier IVs", n_outlier)

    # Step 2: fit SVI per slice
    slice_params = fit_all_slices(chain_iv)
    logger.info("SVI calibration complete: %d slices", len(slice_params))

    # Step 3: arbitrage diagnostics
    diag = generate_diagnostics(slice_params)

    return VolSurface(
        chain=chain_iv,
        slice_params=slice_params,
        diagnostics=diag,
        spot=spot,
        risk_free=risk_free,
        div_yield=div_yield,
    )
