#!/usr/bin/env python3
"""Standalone script to download and cache SPY options data.

Usage
-----
    python data/download.py
    python data/download.py --symbol AAPL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_options

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download options chain data")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    args = parser.parse_args()

    opts = load_options(args.symbol, use_cache=False)
    print(f"Spot:       {opts.spot:.2f}")
    print(f"Risk-free:  {opts.risk_free:.4f}")
    print(f"Div yield:  {opts.div_yield:.4f}")
    print(f"Chain rows: {len(opts.chains)}")
    print(f"\nExpiries: {sorted(opts.chains['expiry'].unique())}")


if __name__ == "__main__":
    main()
