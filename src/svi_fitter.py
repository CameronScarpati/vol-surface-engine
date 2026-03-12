"""
Phase 3: SVI Parameterization

Fits the Stochastic Volatility Inspired (SVI) model (Gatheral 2004) to
implied-variance slices extracted by the IV engine.

The raw SVI parameterization models total implied variance as a function
of log-moneyness k = ln(K/F):

    w(k) = a + b * [rho * (k - m) + sqrt((k - m)^2 + sigma^2)]

Five parameters per expiry slice:
    a     – overall variance level
    b     – angle between put/call asymptotes (must be > 0)
    rho   – rotation / skew  (|rho| < 1)
    m     – horizontal translation
    sigma – smoothness at the vertex (must be > 0)

References
----------
Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility
    parameterization. Presentation at Global Derivatives & Risk Management.
Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    Quantitative Finance 14(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

__all__ = [
    "PARAM_BOUNDS",
    "SVIParams",
    "fit_all_slices",
    "fit_svi_slice",
    "interpolate_surface",
    "svi_first_derivative",
    "svi_second_derivative",
    "svi_total_variance",
]


# ---------------------------------------------------------------------------
# SVI parameter container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SVIParams:
    """Fitted SVI parameters for a single expiry slice."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    # Fit diagnostics
    rmse: float = 0.0
    r_squared: float = 0.0
    max_abs_error: float = 0.0
    n_points: int = 0

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, x: np.ndarray, **diagnostics) -> SVIParams:
        return cls(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4], **diagnostics)


# ---------------------------------------------------------------------------
# SVI function and analytical derivatives
# ---------------------------------------------------------------------------
def svi_total_variance(
    k: np.ndarray | float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Compute SVI total implied variance w(k).

    Parameters
    ----------
    k : array-like
        Log-moneyness values: k = ln(K / F).
    a, b, rho, m, sigma : float
        SVI parameters.

    Returns
    -------
    np.ndarray
        Total implied variance w = iv^2 * T.
    """
    k = np.asarray(k, dtype=np.float64)
    dk = k - m
    return a + b * (rho * dk + np.sqrt(dk**2 + sigma**2))


def svi_first_derivative(
    k: np.ndarray | float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Analytical first derivative dw/dk.

    w'(k) = b * [rho + (k - m) / sqrt((k - m)^2 + sigma^2)]
    """
    k = np.asarray(k, dtype=np.float64)
    dk = k - m
    return b * (rho + dk / np.sqrt(dk**2 + sigma**2))


def svi_second_derivative(
    k: np.ndarray | float,
    b: float,
    _rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Analytical second derivative d^2w/dk^2.

    w''(k) = b * sigma^2 / ((k - m)^2 + sigma^2)^{3/2}
    """
    k = np.asarray(k, dtype=np.float64)
    dk = k - m
    return b * sigma**2 / (dk**2 + sigma**2) ** 1.5


# ---------------------------------------------------------------------------
# Parameter bounds (Gatheral & Jacquier 2014)
# ---------------------------------------------------------------------------
PARAM_BOUNDS = [
    (-0.5, 0.5),    # a – variance level
    (1e-4, 2.0),    # b – slope (positive)
    (-0.999, 0.999), # rho – skew
    (-1.0, 1.0),    # m – translation
    (1e-4, 2.0),    # sigma – curvature (positive)
]


# ---------------------------------------------------------------------------
# Fitting a single slice
# ---------------------------------------------------------------------------
def _svi_objective(
    x: np.ndarray,
    k: np.ndarray,
    w_market: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    """Weighted sum of squared residuals."""
    w_model = svi_total_variance(k, *x)
    residuals = w_market - w_model
    if weights is not None:
        return float(np.sum(weights * residuals**2))
    return float(np.sum(residuals**2))


def fit_svi_slice(
    k_array: np.ndarray,
    w_array: np.ndarray,
    weights: np.ndarray | None = None,
    n_restarts: int = 8,
    penalty_fn=None,
    penalty_lambda: float = 0.0,
) -> SVIParams:
    """Fit raw SVI to a single expiry slice via multi-start L-BFGS-B.

    Parameters
    ----------
    k_array : np.ndarray
        Log-moneyness values (k = ln(K/F)).
    w_array : np.ndarray
        Observed total variance (w = iv^2 * T).
    weights : np.ndarray, optional
        Per-point weights (e.g., open interest or inverse bid-ask).
    n_restarts : int
        Number of random restarts (in addition to a heuristic seed).
    penalty_fn : callable, optional
        Additional penalty term f(x) added to the objective.
    penalty_lambda : float
        Multiplier for the penalty term.

    Returns
    -------
    SVIParams
        Best-fit parameters with diagnostics.
    """
    k = np.asarray(k_array, dtype=np.float64)
    w = np.asarray(w_array, dtype=np.float64)

    if len(k) < 5:
        raise ValueError(
            f"Need at least 5 data points to fit 5 SVI parameters, got {len(k)}"
        )

    # Normalise weights
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum() * len(weights)

    def objective(x: np.ndarray) -> float:
        obj = _svi_objective(x, k, w, weights)
        if penalty_fn is not None and penalty_lambda > 0:
            obj += penalty_lambda * penalty_fn(x)
        return obj

    # Heuristic initial guess from data moments
    w_atm = float(np.interp(0.0, k, w)) if k.min() <= 0 <= k.max() else float(np.median(w))
    x0_heuristic = np.array([w_atm, 0.1, -0.3, 0.0, 0.1])

    rng = np.random.default_rng(seed=42)
    best_result = None
    best_cost = np.inf

    # Generate start points: heuristic + random
    starts = [x0_heuristic]
    for _ in range(n_restarts):
        x0 = np.array([
            rng.uniform(-0.1, 0.3),
            rng.uniform(0.01, 0.5),
            rng.uniform(-0.8, 0.2),
            rng.uniform(-0.3, 0.3),
            rng.uniform(0.01, 0.5),
        ])
        starts.append(x0)

    for x0 in starts:
        # Clip to bounds
        x0_clipped = np.clip(
            x0,
            [b[0] for b in PARAM_BOUNDS],
            [b[1] for b in PARAM_BOUNDS],
        )
        try:
            result = minimize(
                objective,
                x0_clipped,
                method="L-BFGS-B",
                bounds=PARAM_BOUNDS,
                options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
            )
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
        except (ValueError, np.linalg.LinAlgError):
            continue

    if best_result is None:
        raise RuntimeError("SVI fitting failed: all restarts diverged")

    x_opt = best_result.x

    # Diagnostics
    w_fitted = svi_total_variance(k, *x_opt)
    residuals = w - w_fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((w - np.mean(w)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    max_abs_error = float(np.max(np.abs(residuals)))

    return SVIParams.from_array(
        x_opt,
        rmse=rmse,
        r_squared=r_squared,
        max_abs_error=max_abs_error,
        n_points=len(k),
    )


# ---------------------------------------------------------------------------
# Fit all expiry slices
# ---------------------------------------------------------------------------
def fit_all_slices(
    chain_with_iv: pd.DataFrame,
    weight_col: str | None = "open_interest",
) -> pd.DataFrame:
    """Fit SVI to every expiry slice in the chain.

    Parameters
    ----------
    chain_with_iv : pd.DataFrame
        Must contain columns: ``expiry``, ``strike``, ``T``, ``iv``,
        ``S``, ``r``, ``q``.  Optionally ``open_interest`` for weighting.
    weight_col : str or None
        Column to use as fitting weights. ``None`` for equal weights.

    Returns
    -------
    pd.DataFrame
        One row per expiry with columns:
        ``expiry``, ``T``, ``a``, ``b``, ``rho``, ``m``, ``sigma``,
        ``rmse``, ``r_squared``, ``max_abs_error``, ``n_points``.
    """
    df = chain_with_iv.dropna(subset=["iv"]).copy()

    # Compute forward price and log-moneyness
    df["F"] = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])
    df["k"] = np.log(df["strike"] / df["F"])
    df["w"] = df["iv"] ** 2 * df["T"]

    results: list[dict] = []

    for expiry, group in df.groupby("expiry"):
        # De-duplicate by strike (average calls/puts at same strike)
        slice_df = group.groupby("k").agg(
            w=("w", "mean"),
            weight=(weight_col, "sum") if weight_col and weight_col in group.columns else ("w", "count"),
        ).reset_index().sort_values("k")

        k_arr = slice_df["k"].values
        w_arr = slice_df["w"].values
        wts = slice_df["weight"].values if weight_col else None

        if len(k_arr) < 5:
            logger.warning(
                "Skipping expiry %s: only %d unique strikes", expiry, len(k_arr)
            )
            continue

        try:
            params = fit_svi_slice(k_arr, w_arr, weights=wts)
        except RuntimeError:
            logger.warning("SVI fit failed for expiry %s", expiry)
            continue

        T_val = float(group["T"].iloc[0])
        results.append({
            "expiry": expiry,
            "T": T_val,
            "a": params.a,
            "b": params.b,
            "rho": params.rho,
            "m": params.m,
            "sigma": params.sigma,
            "rmse": params.rmse,
            "r_squared": params.r_squared,
            "max_abs_error": params.max_abs_error,
            "n_points": params.n_points,
        })

        logger.info(
            "Expiry %s (T=%.3f): RMSE=%.6f  R²=%.4f  max|err|=%.6f",
            expiry, T_val, params.rmse, params.r_squared, params.max_abs_error,
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Surface interpolation
# ---------------------------------------------------------------------------
def interpolate_surface(
    k: float | np.ndarray,
    T: float,
    slice_params: pd.DataFrame,
) -> np.ndarray:
    """Interpolate the SVI surface at arbitrary (k, T) via linear
    interpolation of total variance between adjacent expiry slices.

    Uses variance-linear interpolation (linear in w) which preserves
    calendar-spread arbitrage-free property when individual slices are
    arbitrage-free.

    Parameters
    ----------
    k : float or np.ndarray
        Log-moneyness query point(s).
    T : float
        Time to expiry (years).
    slice_params : pd.DataFrame
        Output of ``fit_all_slices``.

    Returns
    -------
    np.ndarray
        Interpolated total variance w(k, T).
    """
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))

    T_values = slice_params["T"].values
    if T_values.min() >= T:
        row = slice_params.iloc[np.argmin(T_values)]
        return svi_total_variance(k, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
    if T_values.max() <= T:
        row = slice_params.iloc[np.argmax(T_values)]
        return svi_total_variance(k, row["a"], row["b"], row["rho"], row["m"], row["sigma"])

    # Find bracketing slices
    idx_upper = int(np.searchsorted(np.sort(T_values), T))
    sorted_idx = np.argsort(T_values)
    idx_lo = sorted_idx[idx_upper - 1]
    idx_hi = sorted_idx[idx_upper]

    row_lo = slice_params.iloc[idx_lo]
    row_hi = slice_params.iloc[idx_hi]

    T_lo, T_hi = row_lo["T"], row_hi["T"]

    w_lo = svi_total_variance(k, row_lo["a"], row_lo["b"], row_lo["rho"], row_lo["m"], row_lo["sigma"])
    w_hi = svi_total_variance(k, row_hi["a"], row_hi["b"], row_hi["rho"], row_hi["m"], row_hi["sigma"])

    # Linear interpolation in total variance
    alpha = (T - T_lo) / (T_hi - T_lo)
    return (1.0 - alpha) * w_lo + alpha * w_hi
