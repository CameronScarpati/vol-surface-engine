#!/usr/bin/env python3
"""Validation script: fetch options data, extract IVs, and plot raw IV smiles.

Usage
-----
    python scripts/plot_iv_smiles.py          # uses cached data if available
    python scripts/plot_iv_smiles.py --fresh   # re-downloads from yfinance

Generates ``plots/iv_smiles.html`` (interactive Plotly) and
``plots/iv_smiles.png`` (static image for README).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure `src` is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_options
from src.iv_engine import compute_all_iv

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"


def main(fresh: bool = False) -> None:
    # ---- 1. Load & compute IVs -------------------------------------------
    opts = load_options(use_cache=not fresh)
    logger.info(
        "Spot=%.2f  r=%.4f  q=%.4f  rows=%d",
        opts.spot,
        opts.risk_free,
        opts.div_yield,
        len(opts.chains),
    )

    chain = compute_all_iv(opts.chains)
    chain = chain.dropna(subset=["iv"])
    logger.info("Rows with valid IV: %d", len(chain))

    if chain.empty:
        logger.error("No valid IV rows — cannot plot.")
        sys.exit(1)

    # ---- 2. Build per-expiry smile plots ---------------------------------
    expiries = sorted(chain["expiry"].unique())
    n_exp = min(len(expiries), 9)  # cap at 9 subplots
    selected = expiries[:n_exp]

    ncols = 3
    nrows = (n_exp + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[str(e.date()) for e in selected],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for idx, exp in enumerate(selected):
        row = idx // ncols + 1
        col = idx % ncols + 1

        slc = chain[chain["expiry"] == exp]

        for otype, color, symbol in [
            ("call", "steelblue", "circle"),
            ("put", "tomato", "diamond"),
        ]:
            sub = slc[slc["option_type"] == otype].sort_values("strike")
            if sub.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=sub["strike"],
                    y=sub["iv"],
                    mode="markers",
                    marker=dict(size=5, color=color, symbol=symbol),
                    name=f"{otype} (T={sub['T'].iloc[0]:.3f}y)",
                    showlegend=(idx == 0),
                    hovertemplate=(
                        "K=%{x:.1f}<br>IV=%{y:.4f}<br>"
                        f"type={otype}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        fig.update_xaxes(title_text="Strike", row=row, col=col)
        fig.update_yaxes(title_text="IV", row=row, col=col)

    fig.update_layout(
        title_text=f"Raw IV Smiles — SPY (spot={opts.spot:.2f})",
        height=350 * nrows,
        width=1100,
        template="plotly_white",
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = PLOT_DIR / "iv_smiles.html"
    fig.write_html(str(html_path))
    logger.info("Saved interactive plot to %s", html_path)

    try:
        png_path = PLOT_DIR / "iv_smiles.png"
        fig.write_image(str(png_path), scale=2)
        logger.info("Saved static image to %s", png_path)
    except Exception:
        logger.warning("Could not write PNG (kaleido not installed); HTML is fine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot raw IV smiles for SPY options")
    parser.add_argument("--fresh", action="store_true", help="Re-download data (ignore cache)")
    args = parser.parse_args()
    main(fresh=args.fresh)
