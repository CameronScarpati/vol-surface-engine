#!/usr/bin/env python3
"""Download real SPY options data and build the volatility surface.

Usage
-----
    python data/download.py                     # Download SPY, build surface
    python data/download.py --symbol AAPL       # Use a different ticker
    python data/download.py --synthetic         # Generate synthetic data instead

This script fetches live options chains via yfinance, runs IV extraction,
SVI calibration, and arbitrage diagnostics, then caches the clean chain
to data/spy_options.parquet for fast reload by the dashboard.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_options, PARQUET_PATH
from src.surface import build_surface

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download options data and build volatility surface",
    )
    parser.add_argument(
        "--symbol", default="SPY", help="Ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic data instead of fetching live data",
    )
    args = parser.parse_args()

    if args.synthetic:
        logger.info("Generating synthetic data ...")
        from scripts.generate_synthetic_data import main as gen_main
        gen_main()
        return

    # Fetch real data
    logger.info("Fetching live options data for %s ...", args.symbol)
    opts = load_options(args.symbol, use_cache=False)

    print(f"\n{'='*60}")
    print(f"  {args.symbol} Options Data Summary")
    print(f"{'='*60}")
    print(f"  Spot price:     ${opts.spot:.2f}")
    print(f"  Risk-free rate: {opts.risk_free:.4f} ({opts.risk_free:.2%})")
    print(f"  Dividend yield: {opts.div_yield:.4f} ({opts.div_yield:.2%})")
    print(f"  Chain rows:     {len(opts.chains):,}")
    print(f"  Expiries:       {opts.chains['expiry'].nunique()}")
    print(f"  Saved to:       {PARQUET_PATH}")

    # Run full pipeline
    print(f"\n  Building volatility surface ...")
    surface = build_surface(opts.chains, opts.spot, opts.risk_free, opts.div_yield)

    n_valid = surface.chain["iv"].notna().sum()
    print(f"  IV extraction:  {n_valid} / {len(surface.chain)} valid")
    print(f"  SVI slices:     {len(surface.slice_params)}")

    # Fit quality
    print(f"\n  {'Expiry':<12} {'DTE':>5} {'RMSE':>10} {'R²':>8} {'Points':>7}")
    print(f"  {'-'*44}")
    for _, row in surface.slice_params.iterrows():
        dte = int(round(row["T"] * 365.25))
        print(f"  {'':12} {dte:>5}d {row['rmse']:>10.6f} {row['r_squared']:>8.4f} {int(row['n_points']):>7}")

    # Arbitrage status
    n_bf = sum(surface.diagnostics.butterfly_free.values())
    total = len(surface.diagnostics.butterfly_free)
    cal_status = "PASS" if surface.diagnostics.calendar_free else "FAIL"
    print(f"\n  Butterfly-free:  {n_bf}/{total} slices")
    print(f"  Calendar-free:   {cal_status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
