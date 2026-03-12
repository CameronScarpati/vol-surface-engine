"""Shared test fixtures and constants for the vol-surface-engine test suite."""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import pytest

from src.iv_engine import bs_price
from src.svi_fitter import SVIParams

# ---------------------------------------------------------------------------
# Common market parameters (SPY-like)
# ---------------------------------------------------------------------------
SPOT = 560.0
RISK_FREE = 0.0435
DIV_YIELD = 0.013

# ---------------------------------------------------------------------------
# Reusable SVI parameter sets
# ---------------------------------------------------------------------------
KNOWN_SVI_PARAMS = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.0, sigma=0.1)
ARB_FREE_PARAMS = SVIParams(a=0.04, b=0.10, rho=-0.2, m=0.0, sigma=0.15)
ARB_FREE_PARAMS_LONG = SVIParams(a=0.08, b=0.08, rho=-0.15, m=0.0, sigma=0.20)
VIOLATING_PARAMS = SVIParams(a=0.001, b=1.5, rho=-0.9, m=0.0, sigma=0.01)

K_GRID = np.linspace(-0.5, 0.5, 500)


# ---------------------------------------------------------------------------
# Synthetic IV model (shared across integration & arbitrage tests)
# ---------------------------------------------------------------------------
def synthetic_iv(K: float, T: float, S: float = SPOT) -> float:
    """Realistic SPY-like IV model matching scripts/generate_synthetic_data.py."""
    k = np.log(K / S)
    atm = 0.16 + 0.03 * np.exp(-2.0 * T)
    skew_coeff = -0.12 * (1.0 + 0.5 / (T + 0.05))
    skew = skew_coeff * k
    smile = 0.15 * k**2
    return float(np.clip(atm + skew + smile, 0.05, 1.5))


# ---------------------------------------------------------------------------
# Synthetic chain generator
# ---------------------------------------------------------------------------
def make_synthetic_chain(
    dte_days: list[int] | None = None,
    n_strikes: int = 25,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic options chain with known IV characteristics."""
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now(tz=timezone.utc).normalize()

    if dte_days is None:
        dte_days = [14, 30, 60, 90, 180, 365]

    expiries = [now + pd.Timedelta(days=d) for d in dte_days]
    strikes = np.linspace(SPOT * 0.85, SPOT * 1.15, n_strikes)

    rows = []
    for exp, dte in zip(expiries, dte_days, strict=True):
        T = dte / 365.25
        for K in strikes:
            for otype in ["call", "put"]:
                iv = synthetic_iv(K, T)
                price = bs_price(SPOT, K, T, RISK_FREE, DIV_YIELD, iv, otype)
                noise = rng.normal(0, 0.002 * price + 0.01)
                mid = max(0.05, price + noise)
                spread_pct = rng.uniform(0.03, 0.15)
                spread = mid * spread_pct

                rows.append({
                    "expiry": exp,
                    "strike": round(K, 2),
                    "option_type": otype,
                    "mid_price": round(mid, 4),
                    "bid": round(mid - spread / 2, 4),
                    "ask": round(mid + spread / 2, 4),
                    "volume": int(rng.exponential(500)) + 1,
                    "open_interest": int(rng.exponential(3000)) + 1,
                    "S": SPOT,
                    "r": RISK_FREE,
                    "q": DIV_YIELD,
                    "T": round(T, 6),
                    "low_confidence": spread_pct > 0.10,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_chain() -> pd.DataFrame:
    """A default synthetic options chain for testing."""
    return make_synthetic_chain()
