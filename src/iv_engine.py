"""
Phase 2: Implied Volatility Engine

Black-Scholes pricing, vega, and implied-volatility extraction via
Newton-Raphson with Brent's method fallback.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

__all__ = [
    "bs_price",
    "bs_vega",
    "compute_all_iv",
    "implied_volatility",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IV_TOL_PRICE = 1e-8  # convergence in price space
IV_TOL_VOL = 1e-10  # convergence in vol space
IV_MAX_ITER = 100
IV_LOWER = 0.001
IV_UPPER = 5.0


# ---------------------------------------------------------------------------
# Black-Scholes closed-form
# ---------------------------------------------------------------------------
def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> float:
    """European Black-Scholes price with continuous dividend yield.

    Parameters
    ----------
    S : float – spot price
    K : float – strike price
    T : float – time to expiry in years  (must be > 0)
    r : float – risk-free rate (continuous)
    q : float – continuous dividend yield
    sigma : float – volatility (annualised)
    option_type : str – ``'call'`` or ``'put'``

    Returns
    -------
    float – option price
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return float(price)


def bs_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Black-Scholes vega: ∂C/∂σ = S·e^{-qT}·√T·n(d1).

    Vega is the same for calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    return float(S * np.exp(-q * T) * sqrt_T * norm.pdf(d1))


# ---------------------------------------------------------------------------
# Initial guess (Brenner-Subrahmanyam approximation)
# ---------------------------------------------------------------------------
def _initial_guess(market_price: float, S: float, T: float) -> float:
    """Brenner-Subrahmanyam (1988): σ₀ ≈ √(2π/T) · C/S."""
    guess = np.sqrt(2.0 * np.pi / T) * market_price / S
    # Clamp to sensible range
    return float(np.clip(guess, 0.05, 3.0))


# ---------------------------------------------------------------------------
# Newton-Raphson IV solver
# ---------------------------------------------------------------------------
def _newton_raphson(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    sigma0: float,
) -> float | None:
    """Attempt Newton-Raphson for implied volatility.

    Returns the IV on success, or ``None`` if it fails to converge.
    """
    sigma = sigma0

    for _ in range(IV_MAX_ITER):
        price = bs_price(S, K, T, r, q, sigma, option_type)
        diff = price - market_price

        if abs(diff) < IV_TOL_PRICE:
            return sigma

        vega = bs_vega(S, K, T, r, q, sigma)
        if vega < 1e-12:
            # Near-zero vega → NR step is unreliable
            return None

        sigma_new = sigma - diff / vega

        if abs(sigma_new - sigma) < IV_TOL_VOL:
            return sigma_new

        # Keep within bounds
        sigma = float(np.clip(sigma_new, IV_LOWER, IV_UPPER))

    return None  # did not converge


# ---------------------------------------------------------------------------
# Brent's method fallback
# ---------------------------------------------------------------------------
def _brent_fallback(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
) -> float | None:
    """Brent's method on the interval [IV_LOWER, IV_UPPER]."""

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, q, sigma, option_type) - market_price

    try:
        f_lo = objective(IV_LOWER)
        f_hi = objective(IV_UPPER)
        if f_lo * f_hi > 0:
            # No sign change → no root in interval
            return None
        iv = brentq(objective, IV_LOWER, IV_UPPER, xtol=IV_TOL_VOL, maxiter=IV_MAX_ITER)
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Public API: single-option IV
# ---------------------------------------------------------------------------
def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
) -> float:
    """Compute implied volatility for a single European option.

    Uses Newton-Raphson with Brenner-Subrahmanyam initial guess, falling
    back to Brent's method when NR fails.

    Returns
    -------
    float – implied volatility, or ``np.nan`` if extraction fails.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    # Intrinsic value floor check
    intrinsic = _intrinsic(S, K, T, r, q, option_type)
    if market_price < intrinsic - 1e-8:
        return np.nan

    sigma0 = _initial_guess(market_price, S, T)

    # Try Newton-Raphson first
    iv = _newton_raphson(market_price, S, K, T, r, q, option_type, sigma0)
    if iv is not None and IV_LOWER <= iv <= IV_UPPER:
        return iv

    # Fallback to Brent
    iv = _brent_fallback(market_price, S, K, T, r, q, option_type)
    if iv is not None:
        return iv

    return np.nan


def _intrinsic(S: float, K: float, T: float, r: float, q: float, option_type: str) -> float:
    """Discounted intrinsic value (lower bound for European option)."""
    fwd = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    if option_type == "call":
        return max(0.0, df * (fwd - K))
    else:
        return max(0.0, df * (K - fwd))


# ---------------------------------------------------------------------------
# Vectorised IV extraction over a full chain
# ---------------------------------------------------------------------------
def compute_all_iv(chain: pd.DataFrame) -> pd.DataFrame:
    """Add an ``iv`` column to *chain* by extracting IV for every row.

    Expected input columns: ``mid_price``, ``S``, ``strike``, ``T``,
    ``r``, ``q``, ``option_type``.

    Rows where extraction fails get ``iv = NaN``.
    """
    df = chain.copy()

    ivs = np.empty(len(df), dtype=np.float64)

    for idx in range(len(df)):
        row = df.iloc[idx]
        ivs[idx] = implied_volatility(
            market_price=row["mid_price"],
            S=row["S"],
            K=row["strike"],
            T=row["T"],
            r=row["r"],
            q=row["q"],
            option_type=row["option_type"],
        )

    df["iv"] = ivs

    n_ok = np.isfinite(ivs).sum()
    n_fail = len(ivs) - n_ok
    logger.info("IV extraction: %d succeeded, %d failed (NaN)", n_ok, n_fail)

    return df
