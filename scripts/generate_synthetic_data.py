#!/usr/bin/env python3
"""Generate realistic synthetic SPY options data for offline development.

Produces a Parquet file with known IV characteristics (skew, term structure)
so that IV extraction and plotting can be validated without network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import timezone

from src.iv_engine import bs_price

SPOT = 560.0
R = 0.0435
Q = 0.013

# Realistic SPY IV surface: base vol + skew + smile
def synthetic_iv(K: float, T: float, S: float = SPOT) -> float:
    """Generate a realistic implied volatility for SPY.

    Models:
    - ATM vol term structure (higher short-term, decreasing)
    - Skew (puts more expensive than calls — negative skew)
    - Smile (far OTM options have higher IV)
    """
    k = np.log(K / S)  # log-moneyness

    # ATM vol: term structure (slight contango)
    atm = 0.16 + 0.03 * np.exp(-2.0 * T)

    # Skew: steeper for short-dated
    skew_coeff = -0.12 * (1.0 + 0.5 / (T + 0.05))
    skew = skew_coeff * k

    # Smile (curvature)
    smile = 0.15 * k**2

    iv = atm + skew + smile
    return float(np.clip(iv, 0.05, 1.5))


def main() -> None:
    np.random.seed(42)

    now = pd.Timestamp.now(tz=timezone.utc).normalize()

    # Expiries: 1w, 2w, 1m, 2m, 3m, 6m, 9m, 1y
    dte_days = [7, 14, 30, 60, 90, 180, 270, 365]
    expiries = [now + pd.Timedelta(days=d) for d in dte_days]

    # Strikes: from 0.80 S to 1.20 S
    n_strikes = 30
    strikes = np.linspace(SPOT * 0.80, SPOT * 1.20, n_strikes)

    rows = []
    for exp, dte in zip(expiries, dte_days):
        T = dte / 365.25
        for K in strikes:
            for otype in ["call", "put"]:
                iv = synthetic_iv(K, T)
                price = bs_price(SPOT, K, T, R, Q, iv, otype)

                # Add realistic noise & bid-ask
                noise = np.random.normal(0, 0.002 * price + 0.01)
                mid = max(0.05, price + noise)
                spread_pct = np.random.uniform(0.03, 0.15)
                spread = mid * spread_pct

                vol = int(np.random.exponential(500)) + 1
                oi = int(np.random.exponential(3000)) + 1

                rows.append({
                    "expiry": exp,
                    "strike": round(K, 2),
                    "option_type": otype,
                    "mid_price": round(mid, 4),
                    "bid": round(mid - spread / 2, 4),
                    "ask": round(mid + spread / 2, 4),
                    "volume": vol,
                    "open_interest": oi,
                    "S": SPOT,
                    "r": R,
                    "q": Q,
                    "T": round(T, 6),
                    "low_confidence": spread_pct > 0.10,
                })

    df = pd.DataFrame(rows)
    out_path = Path(__file__).resolve().parent.parent / "data" / "spy_options.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")

    print(f"Generated {len(df)} synthetic option rows")
    print(f"Expiries: {dte_days} days")
    print(f"Strikes:  {strikes[0]:.0f} – {strikes[-1]:.0f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
