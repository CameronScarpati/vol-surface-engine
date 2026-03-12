"""
Streamlit Dashboard — Arbitrage-Free Volatility Surface Engine

Run with:
    streamlit run dashboard/app.py

The dashboard supports two modes:
1. **Live mode**: fetches real SPY options data via yfinance and runs the
   full pipeline (data → IV → SVI → arbitrage).
2. **Placeholder mode**: uses synthetic data to demonstrate visualisations
   without any network dependency.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

if sys.version_info < (3, 10):  # noqa: UP036 — intentional guard for clear error message
    raise RuntimeError(
        f"Python ≥ 3.10 is required (running {sys.version}). "
        "Please recreate your virtualenv with Python 3.10+."
    )

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so `src.*` imports work when
# Streamlit is launched from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.components import (  # noqa: E402
    render_arbitrage_diagnostics,
    render_mispricing_table,
    render_residual_heatmap,
    render_smile_slices,
    render_surface_3d,
    render_term_structure,
)
from src.iv_engine import bs_price  # noqa: E402
from src.surface import VolSurface, build_surface  # noqa: E402
from src.svi_fitter import SVIParams, svi_total_variance  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Vol Surface Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.title("Vol Surface Engine")
st.sidebar.markdown("**Arbitrage-Free Implied Volatility Surface**")

data_source = st.sidebar.radio(
    "Data source",
    ["Placeholder (synthetic)", "Live (yfinance)"],
    index=0,
    help="Placeholder uses synthetic BS prices; Live fetches real SPY data.",
)

if data_source == "Live (yfinance)":
    symbol = st.sidebar.text_input("Ticker", value="SPY")
else:
    symbol = "SPY"

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Author:** Cameron Scarpati\n\n"
    "Vanderbilt CS + Applied Math"
)
st.sidebar.markdown(
    "Also see [LOB Regime Scanner]"
    "(https://github.com/CameronScarpati/lob-regime-scanner) — "
    "a Hidden Markov Model that classifies live order-book "
    "microstructure into Quiet, Trending, and Toxic regimes."
)
st.sidebar.markdown(
    "[Gatheral (2004)](https://doi.org/10.1002/wilm.10201) · "
    "[Durrleman (2005)](https://www.princeton.edu/~durrleman/)"
)


# ═══════════════════════════════════════════════════════════════════════════
# Placeholder data generator
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Generating synthetic surface …")
def _generate_placeholder() -> VolSurface:
    """Build a synthetic volatility surface from known SVI parameters.

    Uses realistic SPY-like parameters to produce a surface that looks
    plausible without requiring network access.
    """
    spot = 585.0
    r = 0.0435
    q = 0.013

    # Realistic expiry grid (7d to 365d)
    T_values = np.array([7, 14, 30, 60, 90, 120, 180, 270, 365]) / 365.25

    # SVI params that produce a realistic SPY smile per expiry
    # (steeper skew for short expiries, flattening out longer term)
    svi_configs = [
        SVIParams(a=0.005, b=0.20, rho=-0.70, m=0.00, sigma=0.10),  # 7d
        SVIParams(a=0.008, b=0.18, rho=-0.65, m=0.00, sigma=0.12),  # 14d
        SVIParams(a=0.015, b=0.15, rho=-0.55, m=-0.01, sigma=0.14),  # 30d
        SVIParams(a=0.025, b=0.12, rho=-0.50, m=-0.01, sigma=0.16),  # 60d
        SVIParams(a=0.035, b=0.10, rho=-0.45, m=-0.02, sigma=0.18),  # 90d
        SVIParams(a=0.045, b=0.09, rho=-0.42, m=-0.02, sigma=0.20),  # 120d
        SVIParams(a=0.060, b=0.08, rho=-0.38, m=-0.02, sigma=0.22),  # 180d
        SVIParams(a=0.085, b=0.07, rho=-0.35, m=-0.02, sigma=0.24),  # 270d
        SVIParams(a=0.110, b=0.06, rho=-0.32, m=-0.02, sigma=0.26),  # 365d
    ]

    rng = np.random.default_rng(42)
    rows: list[dict] = []
    now = pd.Timestamp.now().normalize()  # midnight today, consistent per expiry

    for T, svi in zip(T_values, svi_configs, strict=True):
        expiry_date = now + pd.Timedelta(days=int(T * 365.25))
        F = spot * np.exp((r - q) * T)
        # Generate strikes around ATM
        moneyness_range = min(0.15 + T * 0.3, 0.45)
        n_strikes = max(15, int(40 * np.sqrt(T)))
        k_values = np.linspace(-moneyness_range, moneyness_range, n_strikes)
        strikes = F * np.exp(k_values)

        w_true = svi_total_variance(
            k_values, svi.a, svi.b, svi.rho, svi.m, svi.sigma,
        )
        iv_true = np.sqrt(np.maximum(w_true, 1e-8) / T)

        for K, iv in zip(strikes, iv_true, strict=True):
            # Add realistic noise (wider for short expiry)
            noise = rng.normal(0, 0.002 + 0.001 / np.sqrt(T))
            iv_noisy = max(iv + noise, 0.03)

            for otype in ["call", "put"]:
                price = bs_price(spot, K, T, r, q, iv_noisy, otype)
                spread = max(0.02, price * rng.uniform(0.02, 0.08))
                bid = max(0.01, price - spread / 2)
                ask = price + spread / 2

                rows.append({
                    "expiry": expiry_date,
                    "strike": K,
                    "option_type": otype,
                    "mid_price": price,
                    "bid": bid,
                    "ask": ask,
                    "volume": int(rng.integers(10, 5000)),
                    "open_interest": int(rng.integers(100, 50000)),
                    "S": spot,
                    "r": r,
                    "q": q,
                    "T": T,
                    "low_confidence": spread / price > 0.20 if price > 0 else True,
                })

    chain = pd.DataFrame(rows)
    return build_surface(chain, spot, r, q)


# ═══════════════════════════════════════════════════════════════════════════
# Live data loader
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Fetching live options data …", ttl=300)
def _load_live(symbol: str) -> VolSurface:
    from src.data_loader import load_options

    opts = load_options(symbol, use_cache=False)
    return build_surface(opts.chains, opts.spot, opts.risk_free, opts.div_yield)


# ═══════════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("Arbitrage-Free Volatility Surface Engine")

    # Load data
    if data_source == "Live (yfinance)":
        try:
            surface = _load_live(symbol)
        except Exception as exc:
            st.error(f"Failed to load live data: {exc}")
            st.info("Falling back to placeholder data.")
            surface = _generate_placeholder()
    else:
        surface = _generate_placeholder()

    chain = surface.chain
    sp = surface.slice_params

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Spot", f"${surface.spot:.2f}")
    col2.metric("Risk-Free Rate", f"{surface.risk_free:.2%}")
    col3.metric("Div Yield", f"{surface.div_yield:.3%}")
    col4.metric("Expiry Slices", f"{len(sp)}")

    st.markdown("---")

    # Primary visualisations: 3D surface and smile slices side by side
    left, right = st.columns([3, 2])

    with left:
        render_surface_3d(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

    with right:
        render_smile_slices(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

    st.markdown("---")

    # Residual heatmap
    render_residual_heatmap(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

    st.markdown("---")

    # Arbitrage diagnostics
    render_arbitrage_diagnostics(sp, surface.diagnostics)

    st.markdown("---")

    # Term structure & mispricing table
    left2, right2 = st.columns([1, 1])

    with left2:
        render_term_structure(chain, sp)

    with right2:
        render_mispricing_table(
            chain, sp, surface.spot, surface.risk_free, surface.div_yield,
        )


if __name__ == "__main__":
    main()
