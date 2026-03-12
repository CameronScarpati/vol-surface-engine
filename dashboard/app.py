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
    render_delta_smile,
    render_greeks,
    render_local_vol,
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

st.sidebar.markdown("### Pipeline")
st.sidebar.markdown(
    "1. **Data** — Options chain ingestion & cleaning\n"
    "2. **IV Engine** — Newton-Raphson + Brent fallback\n"
    "3. **SVI Fit** — Multi-start L-BFGS-B calibration\n"
    "4. **Arbitrage** — Durrleman + calendar enforcement\n"
    "5. **Greeks** — BS sensitivities from fitted surface\n"
    "6. **Local Vol** — Dupire's formula"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Author:** Cameron Scarpati\n\n"
    "Vanderbilt CS + Applied Math"
)
st.sidebar.markdown(
    "[Gatheral (2004)](https://doi.org/10.1002/wilm.10201) · "
    "[Durrleman (2005)](https://www.princeton.edu/~durrleman/) · "
    "[Dupire (1994)](https://doi.org/10.3905/jod.1994.407887)"
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

    # Methodology overview
    st.markdown(
        "Constructs an arbitrage-free implied volatility surface by fitting the "
        "**SVI parameterization** (Gatheral 2004) to market options data, enforcing "
        "**no-butterfly arbitrage** via the Durrleman (2005) condition and "
        "**no-calendar-spread arbitrage** via total-variance monotonicity. "
        "The fitted surface is then used to derive **Black-Scholes Greeks** and "
        "**Dupire local volatility** — the complete toolkit for derivatives pricing "
        "and hedging."
    )

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

    # ── Summary metrics ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Spot", f"${surface.spot:.2f}")
    col2.metric("Risk-Free Rate", f"{surface.risk_free:.2%}")
    col3.metric("Div Yield", f"{surface.div_yield:.3%}")
    col4.metric("Expiry Slices", f"{len(sp)}")

    # Aggregate fit quality
    if not sp.empty:
        avg_r2 = sp["r_squared"].mean()
        avg_rmse = sp["rmse"].mean()
        col5.metric("Avg R²", f"{avg_r2:.6f}")
        col6.metric("Avg RMSE", f"{avg_rmse:.2e}")

    st.markdown("---")

    # ── Tab-based navigation for clean organization ──────────────────────
    tab_surface, tab_smiles, tab_greeks, tab_localvol, tab_arb, tab_term = st.tabs([
        "Volatility Surface",
        "Smile Analysis",
        "Greeks",
        "Local Volatility",
        "Arbitrage Diagnostics",
        "Term Structure",
    ])

    # ── Tab 1: Volatility Surface ────────────────────────────────────────
    with tab_surface:
        left, right = st.columns([3, 2])
        with left:
            st.caption(
                "**3-D Volatility Surface** — Implied volatility plotted against "
                "strike (moneyness) and time to expiry. The surface is built by "
                "fitting a Stochastic Volatility Inspired (SVI) model to each "
                "expiry slice, then interpolating across tenors. A smooth, "
                "well-behaved surface indicates consistent arbitrage-free pricing."
            )
            render_surface_3d(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

        with right:
            st.caption(
                "**Volatility Smile per Expiry** — Each curve shows the SVI fit "
                "for a single expiry overlaid on market-observed IVs. The "
                "characteristic 'smile' or 'skew' shape reflects how out-of-the-"
                "money puts trade at higher implied vols than ATM options, driven "
                "by demand for downside protection and the leverage effect."
            )
            render_smile_slices(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

        st.markdown("---")

        st.caption(
            "**Residual Heatmap** — Difference between market-observed IV and "
            "the SVI model fit, mapped across strike and expiry. Large residuals "
            "highlight options where the model deviates from the market — "
            "potential mispricings or areas where the SVI parameterization "
            "struggles (e.g. deep OTM wings, illiquid strikes)."
        )
        render_residual_heatmap(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

    # ── Tab 2: Smile Analysis (delta-space + mispricing) ─────────────────
    with tab_smiles:
        left2, right2 = st.columns([3, 2])
        with left2:
            st.caption(
                "**Delta-Space Smile** — IV plotted against Black-Scholes "
                "delta, the standard quoting convention on derivatives desks. "
                "Normalises across expiries so skew and convexity are directly "
                "comparable. The 25Δ risk-reversal measures skew direction "
                "while the 25Δ butterfly captures smile curvature."
            )
            render_delta_smile(chain, sp, surface.spot, surface.risk_free, surface.div_yield)
        with right2:
            st.caption(
                "**Mispricing Table** — Options with the largest absolute "
                "residuals between market IV and the SVI fit. These are "
                "candidates where the market price diverges most from the "
                "arbitrage-free model, potentially indicating trading "
                "opportunities or data quality issues."
            )
            render_mispricing_table(
                chain, sp, surface.spot, surface.risk_free, surface.div_yield,
            )

    # ── Tab 3: Greeks ────────────────────────────────────────────────────
    with tab_greeks:
        st.caption(
            "**Greeks Surface** — Black-Scholes sensitivities (Δ, Γ, ν, Θ) "
            "computed from the fitted SVI surface across the full (strike, T) "
            "grid. These are the quantities that drive hedging and risk "
            "management — the primary reason for constructing a volatility "
            "surface in production."
        )
        render_greeks(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

    # ── Tab 4: Local Volatility ──────────────────────────────────────────
    with tab_localvol:
        st.caption(
            "**Local Volatility (Dupire)** — The unique diffusion coefficient "
            "σ_loc(K, T) consistent with the fitted implied volatility surface, "
            "computed via Dupire's formula. Local vol reveals the instantaneous "
            "volatility structure that the market prices imply, bridging the "
            "quoting convention (implied vol) to the risk-neutral dynamics."
        )
        render_local_vol(chain, sp, surface.spot, surface.risk_free, surface.div_yield)

    # ── Tab 5: Arbitrage Diagnostics ─────────────────────────────────────
    with tab_arb:
        st.caption(
            "**Arbitrage Diagnostics** — Static no-arbitrage conditions verified "
            "across the surface. *Butterfly arbitrage* is checked via the "
            "Durrleman (2005) condition, which requires the risk-neutral density "
            "to be non-negative at every strike. *Calendar-spread arbitrage* "
            "ensures total variance is non-decreasing in time to expiry."
        )
        render_arbitrage_diagnostics(sp, surface.diagnostics)

    # ── Tab 6: Term Structure ────────────────────────────────────────────
    with tab_term:
        st.caption(
            "**ATM Term Structure** — At-the-money implied volatility as a "
            "function of time to expiry. An upward-sloping curve is typical "
            "in calm markets (mean-reversion expectation), while inversion "
            "signals near-term event risk or elevated short-dated demand."
        )
        render_term_structure(chain, sp)


if __name__ == "__main__":
    main()
