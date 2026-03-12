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
    SVIParams,
    fit_all_slices,
    interpolate_surface,
    svi_total_variance,
)

logger = logging.getLogger(__name__)


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

    # Step 1: extract implied volatilities
    chain_iv = compute_all_iv(chain)
    n_valid = chain_iv["iv"].notna().sum()
    logger.info("IV extraction complete: %d / %d valid", n_valid, len(chain_iv))

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
