"""
Phase 4: No-Arbitrage Constraints

Enforces static-arbitrage-free conditions on the fitted SVI surface:

1. **Butterfly arbitrage** (Durrleman 2005) — the risk-neutral density must
   be non-negative everywhere.  This is equivalent to:

       g(k) = (1 - k w'/(2w))^2 - (w')^2/4 (1/w + 1/4) + w''/2  >= 0

   for all log-moneyness k, where w = w(k) is total variance and primes
   denote derivatives w.r.t. k.

2. **Calendar-spread arbitrage** — total variance must be non-decreasing
   in time to expiry for every strike:  dw/dT >= 0.

3. **Vertical-spread arbitrage** — automatically satisfied when the
   density is non-negative (condition 1).

References
----------
Durrleman, V. (2005). From implied to spot volatilities. PhD thesis,
    Princeton University.
Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility
    surfaces. Quantitative Finance 14(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.svi_fitter import (
    SVIParams,
    _PARAM_BOUNDS,
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
    fit_svi_slice,
)

logger = logging.getLogger(__name__)

# Default grid for checking arbitrage conditions
_DEFAULT_K_MIN = -0.5
_DEFAULT_K_MAX = 0.5
_DEFAULT_K_POINTS = 500


# ---------------------------------------------------------------------------
# Durrleman condition
# ---------------------------------------------------------------------------
def durrleman_condition(
    k: np.ndarray,
    svi_params: SVIParams,
) -> np.ndarray:
    """Evaluate the Durrleman function g(k) on a grid.

    g(k) >= 0 everywhere is necessary and sufficient for the absence
    of butterfly arbitrage in the given slice.

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness grid.
    svi_params : SVIParams
        Fitted SVI parameters for the slice.

    Returns
    -------
    np.ndarray
        Values of g(k).  Negative values indicate arbitrage.
    """
    a, b, rho, m, sigma = (
        svi_params.a, svi_params.b, svi_params.rho, svi_params.m, svi_params.sigma
    )

    w = svi_total_variance(k, a, b, rho, m, sigma)
    wp = svi_first_derivative(k, b, rho, m, sigma)
    wpp = svi_second_derivative(k, b, rho, m, sigma)

    # Protect against w <= 0 (pathological parameterization)
    w = np.maximum(w, 1e-14)

    term1 = (1.0 - k * wp / (2.0 * w)) ** 2
    term2 = wp**2 / 4.0 * (1.0 / w + 0.25)
    term3 = wpp / 2.0

    return term1 - term2 + term3


def check_butterfly_arbitrage(
    k_grid: np.ndarray,
    svi_params: SVIParams,
    tol: float = -1e-10,
) -> bool:
    """Check whether a single slice is free of butterfly arbitrage.

    Parameters
    ----------
    k_grid : np.ndarray
        Log-moneyness grid to evaluate.
    svi_params : SVIParams
        Fitted SVI parameters.
    tol : float
        Tolerance for g(k) negativity (to handle floating-point noise).

    Returns
    -------
    bool
        ``True`` if no butterfly arbitrage is detected.
    """
    g = durrleman_condition(k_grid, svi_params)
    return bool(np.all(g >= tol))


# ---------------------------------------------------------------------------
# Calendar-spread arbitrage
# ---------------------------------------------------------------------------
def check_calendar_arbitrage(
    slice_params_list: list[SVIParams] | pd.DataFrame,
    T_values: np.ndarray | None = None,
    k_grid: np.ndarray | None = None,
    tol: float = -1e-10,
) -> bool:
    """Check whether total variance is non-decreasing across expiries.

    For each k in the grid, verifies w(k, T_{i+1}) >= w(k, T_i).

    Parameters
    ----------
    slice_params_list : list[SVIParams] or pd.DataFrame
        Slices sorted by increasing T.
    T_values : np.ndarray, optional
        Expiry times (needed if passing a list of SVIParams).
    k_grid : np.ndarray, optional
        Log-moneyness grid; defaults to 500 points in [-0.5, 0.5].
    tol : float
        Tolerance for negativity.

    Returns
    -------
    bool
        ``True`` if no calendar-spread arbitrage is detected.
    """
    if k_grid is None:
        k_grid = np.linspace(_DEFAULT_K_MIN, _DEFAULT_K_MAX, _DEFAULT_K_POINTS)

    params_list = _to_params_list(slice_params_list)

    if T_values is not None:
        sort_idx = np.argsort(T_values)
        params_list = [params_list[i] for i in sort_idx]

    for i in range(len(params_list) - 1):
        p_short = params_list[i]
        p_long = params_list[i + 1]

        w_short = svi_total_variance(
            k_grid, p_short.a, p_short.b, p_short.rho, p_short.m, p_short.sigma
        )
        w_long = svi_total_variance(
            k_grid, p_long.a, p_long.b, p_long.rho, p_long.m, p_long.sigma
        )

        if np.any((w_long - w_short) < tol):
            return False

    return True


def _to_params_list(
    source: list[SVIParams] | pd.DataFrame,
) -> list[SVIParams]:
    """Convert DataFrame rows to SVIParams list if needed."""
    if isinstance(source, pd.DataFrame):
        params = []
        df = source.sort_values("T").reset_index(drop=True)
        for _, row in df.iterrows():
            params.append(SVIParams(
                a=row["a"], b=row["b"], rho=row["rho"],
                m=row["m"], sigma=row["sigma"],
            ))
        return params
    return list(source)


# ---------------------------------------------------------------------------
# Arbitrage-free SVI fitting (penalty method)
# ---------------------------------------------------------------------------
def _butterfly_penalty(
    x: np.ndarray,
    k_grid: np.ndarray,
) -> float:
    """Sum of squared Durrleman violations on the grid.

    Returns sum( max(0, -g(k_i))^2 ) which is zero when the surface
    is arbitrage-free and positive otherwise.
    """
    params = SVIParams.from_array(x)
    g = durrleman_condition(k_grid, params)
    violations = np.minimum(g, 0.0)
    return float(np.sum(violations**2))


def fit_svi_arbitrage_free(
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray | None = None,
    lambda_init: float = 1.0,
    lambda_max: float = 1e6,
    lambda_growth: float = 10.0,
    k_grid_points: int = _DEFAULT_K_POINTS,
    max_penalty_iters: int = 10,
) -> SVIParams:
    """Fit SVI with progressive penalty to enforce Durrleman condition.

    Algorithm:
        1. Fit unconstrained SVI.
        2. If butterfly violations exist, re-fit with penalty
           lambda * sum(max(0, -g(k_i))^2).
        3. Increase lambda until violations vanish or lambda_max is reached.

    Parameters
    ----------
    k, w : np.ndarray
        Market data (log-moneyness, total variance).
    weights : np.ndarray, optional
        Per-point fitting weights.
    lambda_init : float
        Starting penalty multiplier.
    lambda_max : float
        Maximum penalty multiplier before giving up.
    lambda_growth : float
        Factor by which lambda increases each iteration.
    k_grid_points : int
        Number of grid points for Durrleman evaluation.
    max_penalty_iters : int
        Maximum number of penalty escalation rounds.

    Returns
    -------
    SVIParams
        Arbitrage-free (or best-effort) fitted parameters.
    """
    k_check = np.linspace(
        float(np.min(k)) - 0.1,
        float(np.max(k)) + 0.1,
        k_grid_points,
    )

    # Step 1: unconstrained fit
    params = fit_svi_slice(k, w, weights=weights)

    if check_butterfly_arbitrage(k_check, params):
        logger.info("Unconstrained fit is already arbitrage-free")
        return params

    # Step 2: progressive penalty
    lam = lambda_init
    for iteration in range(max_penalty_iters):
        penalty_fn = lambda x: _butterfly_penalty(x, k_check)  # noqa: E731

        params = fit_svi_slice(
            k, w, weights=weights,
            penalty_fn=penalty_fn, penalty_lambda=lam,
            n_restarts=12,
        )

        if check_butterfly_arbitrage(k_check, params):
            logger.info(
                "Arbitrage-free fit achieved at lambda=%.1f (iteration %d)",
                lam, iteration + 1,
            )
            return params

        lam *= lambda_growth
        if lam > lambda_max:
            break

    logger.warning(
        "Could not fully eliminate butterfly arbitrage "
        "(lambda reached %.1e); returning best-effort fit", lam
    )
    return params


# ---------------------------------------------------------------------------
# Diagnostics report
# ---------------------------------------------------------------------------
@dataclass
class ArbitrageDiagnostics:
    """Summary of arbitrage conditions for the full surface."""

    butterfly_free: dict[str, bool] = field(default_factory=dict)
    calendar_free: bool = True
    butterfly_violations: dict[str, np.ndarray] = field(default_factory=dict)
    calendar_violation_expiries: list[tuple[str, str]] = field(default_factory=list)


def generate_diagnostics(
    slice_params: pd.DataFrame,
    k_grid: np.ndarray | None = None,
) -> ArbitrageDiagnostics:
    """Run all arbitrage checks and return a structured report.

    Parameters
    ----------
    slice_params : pd.DataFrame
        Output of ``fit_all_slices`` (must include ``expiry``, ``T``,
        and SVI parameter columns).
    k_grid : np.ndarray, optional
        Log-moneyness grid; defaults to 500 points in [-0.5, 0.5].

    Returns
    -------
    ArbitrageDiagnostics
    """
    if k_grid is None:
        k_grid = np.linspace(_DEFAULT_K_MIN, _DEFAULT_K_MAX, _DEFAULT_K_POINTS)

    diag = ArbitrageDiagnostics()
    df = slice_params.sort_values("T").reset_index(drop=True)

    # --- Butterfly (per slice) ---
    for _, row in df.iterrows():
        label = str(row["expiry"])
        params = SVIParams(
            a=row["a"], b=row["b"], rho=row["rho"],
            m=row["m"], sigma=row["sigma"],
        )
        g = durrleman_condition(k_grid, params)
        is_free = bool(np.all(g >= -1e-10))
        diag.butterfly_free[label] = is_free
        if not is_free:
            diag.butterfly_violations[label] = g

    # --- Calendar spread ---
    params_list = _to_params_list(df)

    for i in range(len(params_list) - 1):
        p_short = params_list[i]
        p_long = params_list[i + 1]

        w_short = svi_total_variance(
            k_grid, p_short.a, p_short.b, p_short.rho, p_short.m, p_short.sigma
        )
        w_long = svi_total_variance(
            k_grid, p_long.a, p_long.b, p_long.rho, p_long.m, p_long.sigma
        )

        if np.any((w_long - w_short) < -1e-10):
            label_short = str(df.iloc[i]["expiry"])
            label_long = str(df.iloc[i + 1]["expiry"])
            diag.calendar_violation_expiries.append((label_short, label_long))

    diag.calendar_free = len(diag.calendar_violation_expiries) == 0
    return diag
