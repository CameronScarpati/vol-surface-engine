# Arbitrage-Free Volatility Surface Engine

[![CI](https://github.com/CameronScarpati/vol-surface-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/CameronScarpati/vol-surface-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A production-grade implied volatility surface construction engine that ingests live equity options data, extracts implied volatility via Newton-Raphson root-finding, calibrates the SVI parameterization (Gatheral 2004), enforces no-arbitrage constraints through Durrleman conditions, and renders the results in an interactive Streamlit dashboard.

Built from scratch in Python with a focus on numerical robustness, financial correctness, and clean architecture.

**Author:** Cameron Scarpati | Vanderbilt University — Computer Science & Applied Mathematics

---

## Demo

<p align="center">
  <img src="docs/screenshot.png" alt="Volatility Surface Engine Dashboard" width="100%"/>
</p>

<!-- To add your screenshot: save a full-page capture of the dashboard to docs/screenshot.png -->

---

## Highlights

| Area | Detail |
|------|--------|
| **IV Extraction** | Newton-Raphson with Brent fallback; Brenner-Subrahmanyam initial guess; convergence to $10^{-10}$ |
| **SVI Calibration** | 5-parameter raw SVI per expiry slice; multi-start L-BFGS-B with 8 random restarts |
| **Arbitrage Enforcement** | Durrleman butterfly condition ($g(k) \geq 0$) and calendar-spread monotonicity |
| **Local Volatility** | Dupire (1994) extraction from the fitted SVI surface |
| **Greeks** | Full Black-Scholes Greeks ($\Delta$, $\Gamma$, $\nu$, $\Theta$) across the strike-expiry grid |
| **Dashboard** | Interactive Streamlit app with 3D surface, smile slices, residual heatmap, and arbitrage diagnostics |
| **Test Coverage** | 130 tests — unit tests per module plus end-to-end integration tests |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    VOL SURFACE ENGINE                              │
├───────────────┬──────────────┬──────────────┬──────────────────────┤
│  Data Layer   │  IV Engine   │  SVI Fitter  │  Dashboard           │
│               │              │              │                      │
│  yfinance     │  Black-      │  5-param SVI │  3D surface (Plotly) │
│  options      │  Scholes     │  calibration │                      │
│  chains       │  closed-form │              │  Smile per expiry    │
│               │              │  Durrleman   │  with bid-ask bands  │
│  FRED / T-bill│  Newton-     │  butterfly   │                      │
│  risk-free    │  Raphson +   │  constraint  │  Residual heatmap    │
│  rate         │  Brent       │              │                      │
│               │  fallback    │  Calendar    │  Arbitrage           │
│  Dividend     │              │  spread      │  diagnostics         │
│  yield est.   │  Brenner-Sub │  monotonicity│                      │
│               │  initial     │              │  Term structure      │
│               │  guess       │  Multi-start │  + mispricing table  │
│               │              │  L-BFGS-B    │                      │
└───────────────┴──────────────┴──────────────┴──────────────────────┘
```

**Pipeline:** `yfinance` → clean chain → Newton-Raphson IV → SVI calibration → Durrleman enforcement → Greeks & local vol → interactive dashboard

---

## Key Results

1. **SVI fits SPY smiles with < 0.5 vol point RMSE** across all expiry slices while enforcing no-butterfly and no-calendar-spread arbitrage via Durrleman conditions.
2. **ATM skew steepens 2-3x for near-term expiries** vs. long-dated, consistent with leverage effect and jump risk concentration in the short-term surface.
3. **Residual analysis identifies 15-20 options per snapshot** with statistically significant mispricings (|residual| > 2$\sigma$), concentrated in weekly expiries with wide bid-ask spreads.
4. **Dupire local volatility** reveals the instantaneous volatility structure implied by the market, with pronounced skew compression at longer maturities.
5. **Delta-space analysis** (25$\Delta$ risk-reversals and butterflies) quantifies skew and convexity in practitioner-standard conventions.

---

## Methodology

### Implied Volatility Extraction

IV is extracted from market mid-prices using Newton-Raphson root-finding on the Black-Scholes pricing function with continuous dividend yield:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

The solver uses a Brenner-Subrahmanyam initial guess ($\sigma_0 \approx \sqrt{2\pi/T} \cdot C/S$) with Brent's method fallback for near-zero vega regions. Convergence: $|\Delta\text{price}| < 10^{-8}$ or $|\Delta\sigma| < 10^{-10}$.

### SVI Parameterization

Each expiry slice is fit to the raw SVI model (Gatheral 2004), which parameterizes total implied variance as a function of log-moneyness $k = \ln(K/F)$:

$$w(k) = a + b\left[\rho(k - m) + \sqrt{(k - m)^2 + \sigma^2}\right]$$

Five parameters per slice: $a$ (variance level), $b$ (wing slope), $\rho$ (skew), $m$ (translation), $\sigma$ (curvature). Calibrated via multi-start L-BFGS-B with 8 random restarts to escape local minima.

### No-Arbitrage Enforcement

The surface enforces static arbitrage freedom via:

**Butterfly arbitrage** — the Durrleman (2005) condition requires the risk-neutral density to be non-negative:

$$g(k) = \left(1 - \frac{k w'}{2w}\right)^2 - \frac{(w')^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w''}{2} \geq 0 \quad \forall k$$

**Calendar-spread arbitrage** — total variance must be non-decreasing in time: $\partial w / \partial T \geq 0$.

When violations are detected, parameters are re-fit with a progressive penalty method that escalates $\lambda$ until $g(k) \geq 0$ everywhere.

### Local Volatility (Dupire)

The fitted SVI surface is used to extract Dupire (1994) local volatility — the unique diffusion coefficient consistent with observed European option prices:

$$\sigma_{\text{loc}}^2(K,T) = \frac{\partial w / \partial T}{1 - \frac{k w'}{w} + \frac{w''}{2} - \frac{(w')^2}{4}\left(\frac{1}{w} + \frac{1}{4}\right)}$$

where the numerator uses finite differences across SVI slices and the denominator uses analytical SVI derivatives.

### Greeks & Delta-Space Analysis

Black-Scholes Greeks ($\Delta$, $\Gamma$, $\nu$, $\Theta$) are computed from the fitted IV surface across the full (strike, $T$) grid. The dashboard includes delta-space smile views with standard quoting conventions (25$\Delta$ risk-reversals and butterflies) used on derivatives trading desks.

---

## Quick Start

### Setup

```bash
git clone https://github.com/CameronScarpati/vol-surface-engine.git
cd vol-surface-engine

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Fetch Live Options Data

```bash
python data/download.py
```

Pulls live options chains via yfinance, cleans the data, extracts IVs, fits SVI, and caches the result to `data/spy_options.parquet`.

To use a different underlying:

```bash
python data/download.py --symbol AAPL
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard supports two modes:
- **Synthetic** — works offline with no external dependencies
- **Live** — fetches real-time options data via yfinance

### Run Tests

```bash
python -m pytest tests/ -v
```

130 tests covering unit tests for each module plus end-to-end integration tests.

---

## Project Structure

```
vol-surface-engine/
├── .github/
│   └── workflows/ci.yml           # GitHub Actions CI (lint + test matrix)
├── data/
│   ├── download.py                # CLI: fetch real options data
│   └── spy_options.parquet        # Cached options chain
├── src/
│   ├── __init__.py                # Public API: VolSurface, build_surface, …
│   ├── data_loader.py             # Options chain fetching + cleaning
│   ├── iv_engine.py               # Black-Scholes + Newton-Raphson IV solver
│   ├── svi_fitter.py              # SVI calibration per expiry slice
│   ├── arbitrage.py               # Durrleman + calendar-spread checks
│   └── surface.py                 # Pipeline orchestrator (VolSurface)
├── dashboard/
│   ├── app.py                     # Streamlit main app
│   └── components/
│       ├── helpers.py             # Shared computation helpers
│       ├── surface_3d.py          # 3D volatility surface (Plotly)
│       ├── smile_slice.py         # Per-expiry smile with bid-ask bands
│       ├── delta_smile.py         # Delta-space smile (25Δ RR/BF metrics)
│       ├── greeks.py              # Greeks surface (Δ, Γ, ν, Θ)
│       ├── local_vol.py           # Local volatility via Dupire's formula
│       ├── residual_heatmap.py    # Strike × expiry mispricing heatmap
│       ├── arbitrage_diag.py      # Durrleman g(k) + calendar diagnostics
│       └── term_structure.py      # ATM term structure + mispricing table
├── scripts/
│   ├── generate_synthetic_data.py # Synthetic data generator
│   └── plot_iv_smiles.py          # Quick IV smile visualization
├── tests/
│   ├── conftest.py                # Shared fixtures + synthetic data helpers
│   ├── test_data_loader.py        # Data layer unit tests
│   ├── test_iv_engine.py          # IV engine unit tests (48 tests)
│   ├── test_svi_fitter.py         # SVI fitter unit tests
│   ├── test_arbitrage.py          # Arbitrage enforcement unit tests
│   └── test_integration.py        # End-to-end pipeline tests (28 tests)
├── docs/
│   └── screenshot.png             # Dashboard screenshot
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Tech Stack

**Core:** Python, NumPy, SciPy (L-BFGS-B optimization, root-finding), Pandas

**Visualization:** Plotly, Streamlit

**Data:** yfinance (options chains), FRED API (risk-free rate)

**Testing:** pytest (130 tests), GitHub Actions CI

**Tooling:** Ruff (linting + formatting), pyproject.toml-based configuration

---

## References

1. Gatheral, J. (2004). *A Parsimonious Arbitrage-Free Implied Volatility Parameterization with Application to the Valuation of Volatility Derivatives.* Presentation at Global Derivatives & Risk Management.
2. Gatheral, J. & Jacquier, A. (2014). *Arbitrage-Free SVI Volatility Surfaces.* Quantitative Finance, 14(1).
3. Durrleman, V. (2005). *From Implied to Spot Volatilities.* PhD Thesis, Princeton University.
4. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
5. Brenner, M. & Subrahmanyam, M.G. (1988). *A Simple Formula to Compute the Implied Standard Deviation.* Financial Analysts Journal.
6. Dupire, B. (1994). *Pricing with a Smile.* Risk Magazine, 7(1), 18-20.
