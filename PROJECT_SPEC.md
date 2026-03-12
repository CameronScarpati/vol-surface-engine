# Arbitrage-Free Volatility Surface Engine

## Project Brief

Build an implied volatility surface construction tool that pulls live options data, extracts IV via Newton-Raphson, fits the SVI parameterization (Gatheral 2004), enforces no-arbitrage constraints, and identifies mispriced options. Renders everything in an interactive Plotly/Streamlit dashboard.

**Target audience:** Quant research recruiters at Two Sigma, DE Shaw, Citadel, Millennium.
**Author:** Cameron Scarpati (Vanderbilt CS + Applied Math, Morgan Stanley Equity Algorithms)
**Stack:** Python (NumPy, SciPy, Plotly, Streamlit)
**Companion project:** LOB Regime Scanner (microstructure → this bridges into derivatives)

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│              VOL SURFACE ENGINE                             │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Data Layer  │  IV Engine   │  SVI Fitter  │  Dashboard    │
│              │              │              │               │
│ yfinance     │ Black-Scholes│ SVI 5-param  │ 3D vol        │
│ options      │ closed-form  │ calibration  │ surface       │
│ chains       │              │              │               │
│              │ Newton-      │ Durrleman    │ Smile per     │
│ FRED         │ Raphson +    │ butterfly    │ expiry slice  │
│ risk-free    │ Brent        │ constraint   │               │
│ rate         │ fallback     │              │ Residual      │
│              │              │ Calendar     │ heatmap       │
│ Dividend     │ Greeks via   │ spread       │               │
│ yield        │ finite diff  │ monotonicity │ Mispricing    │
│ estimation   │              │              │ alerts        │
│              │              │ BIC/AIC      │               │
│              │              │ model select │ Term structure│
└──────────────┴──────────────┴──────────────┴───────────────┘
```

---

## Phase Layout for Parallel Sessions

```
SESSION A (Data + IV Engine)     SESSION B (SVI + Arbitrage)     SESSION C (Dashboard)
┌─────────────────────┐         ┌─────────────────────┐        ┌─────────────────────┐
│ Phase 1: Data       │         │                     │        │                     │
│ Pipeline            │         │   (waits for A)     │        │   (waits for A+B)   │
│                     │         │                     │        │                     │
│ Phase 2: IV Engine  │────────>│ Phase 3: SVI Fit    │───────>│ Phase 5: Dashboard  │
│                     │         │                     │        │                     │
│                     │         │ Phase 4: Arbitrage  │───────>│                     │
└─────────────────────┘         └─────────────────────┘        └─────────────────────┘
```

**Session A** and **Session B (Phase 3-4)** can overlap — B just needs the IV output format.
**Session C** can start scaffolding immediately, then wire in real data once A+B are done.

---

## SESSION A: Data Pipeline + IV Engine

### Phase 1: Data Pipeline

Pull options data and clean it for IV extraction.

**Tasks:**
1. Use `yfinance` to pull the full options chain for SPY (all available expiries)
2. Pull 3-month T-bill rate from FRED as the risk-free rate (or hardcode current rate with a TODO)
3. Estimate continuous dividend yield from SPY (trailing 12-month dividend / current price)
4. Clean the data:
   - Filter out options with zero volume or zero open interest
   - Filter out deep OTM options (delta < 0.05 equivalent, or moneyness |log(K/S)| > 0.5)
   - Filter out options expiring in < 3 days (noisy near-expiry behavior)
   - Compute mid-price as (bid + ask) / 2
   - Flag wide bid-ask spreads (spread > 20% of mid) as low-confidence
5. Store as a clean DataFrame with columns: `expiry`, `strike`, `option_type`, `mid_price`, `bid`, `ask`, `volume`, `open_interest`, `S` (spot), `r` (risk-free), `q` (div yield), `T` (time to expiry in years)
6. Save to Parquet for fast reload during development

**Data schema:**
```python
@dataclass
class OptionsData:
    spot: float              # Current SPY price
    risk_free: float         # Annualized risk-free rate
    div_yield: float         # Annualized continuous dividend yield
    chains: pd.DataFrame     # Columns: expiry, strike, type, mid, bid, ask, vol, oi, T
```

**Output:** `data/spy_options.parquet` + `src/data_loader.py`

### Phase 2: Implied Volatility Engine

Extract IV from market prices using numerical root-finding.

**Tasks:**
1. Implement Black-Scholes pricing function for European calls and puts:
   ```
   C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
   P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
   d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
   d2 = d1 - σ√T
   ```
2. Implement BS vega: `vega = S·e^(-qT)·√T·n(d1)` (needed for Newton-Raphson)
3. Implement Newton-Raphson IV solver:
   - Initial guess: σ₀ = 0.3 (or Brenner-Subrahmanyam approximation: σ₀ ≈ √(2π/T) · C/S)
   - Iterate: σ_{n+1} = σ_n - (BS(σ_n) - market_price) / vega(σ_n)
   - Convergence: |BS(σ_n) - market_price| < 1e-8 or |Δσ| < 1e-10
   - Max iterations: 100
   - Bounds: 0.001 < σ < 5.0
4. Implement Brent's method fallback for when Newton-Raphson fails (e.g., near zero vega)
5. Vectorize across the full options chain — compute IV for every option in the DataFrame
6. Add IV column to DataFrame, mark failed extractions as NaN
7. Validate: plot raw IV smile for each expiry slice — should see the characteristic skew

**Output:** `src/iv_engine.py` with functions:
```python
def bs_price(S, K, T, r, q, sigma, option_type) -> float
def bs_vega(S, K, T, r, q, sigma) -> float
def implied_volatility(market_price, S, K, T, r, q, option_type) -> float
def compute_all_iv(chain: pd.DataFrame) -> pd.DataFrame  # adds 'iv' column
```

**Tests:** `tests/test_iv_engine.py`
- Known-price roundtrip: BS(σ=0.2) → price → IV extraction → should recover 0.2
- Edge cases: deep ITM, deep OTM, near-expiry

---

## SESSION B: SVI Calibration + Arbitrage Enforcement

### Phase 3: SVI Parameterization

Fit the SVI model to each expiry slice.

**Background:** The SVI (Stochastic Volatility Inspired) parameterization models total implied variance `w(k) = σ²(k)·T` as a function of log-moneyness `k = ln(K/F)` where `F = S·e^((r-q)T)` is the forward price:

```
w(k) = a + b · [ρ(k - m) + √((k - m)² + σ²)]
```

Five parameters: `a` (overall variance level), `b` (angle between asymptotes), `ρ` (rotation/skew), `m` (translation), `σ` (smoothness at the vertex).

**Tasks:**
1. Convert IV data to total variance: `w = iv² · T`
2. Convert strikes to log-moneyness: `k = ln(K/F)`
3. Implement the SVI function `w(k; a, b, ρ, m, σ)`
4. Fit SVI per expiry slice using `scipy.optimize.minimize` (L-BFGS-B or trust-constr):
   - Objective: minimize sum of squared residuals Σ(w_market - w_svi)²
   - Optionally weight by vega or open interest (higher weight = more liquid options)
   - Parameter bounds:
     - `a` ∈ [-0.5, 0.5] (can be negative for short expiries)
     - `b` ∈ [0.001, 2.0] (must be positive)
     - `ρ` ∈ [-0.999, 0.999]
     - `m` ∈ [-1.0, 1.0]
     - `σ` ∈ [0.001, 2.0] (must be positive)
   - Multiple random restarts (5-10) to avoid local minima
5. Store fitted parameters per expiry in a DataFrame
6. Compute fit quality: R², RMSE, max absolute error per slice
7. Implement interpolation between expiry slices for a continuous surface

**Output:** `src/svi_fitter.py` with functions:
```python
def svi_total_variance(k, a, b, rho, m, sigma) -> float
def fit_svi_slice(k_array, w_array, weights=None) -> SVIParams
def fit_all_slices(chain_with_iv: pd.DataFrame) -> pd.DataFrame  # params per expiry
def interpolate_surface(k, T, slice_params) -> float  # continuous surface
```

### Phase 4: No-Arbitrage Constraints

Enforce that the fitted surface admits no static arbitrage.

**Background:** An implied volatility surface is arbitrage-free if and only if:
1. **Butterfly arbitrage** (no negative probability density): The Durrleman condition must hold:
   `g(k) = (1 - k·w'/(2w))² - w'²/4·(1/w + 1/4) + w''/2 ≥ 0`
   where `w' = dw/dk` and `w'' = d²w/dk²`
2. **Calendar spread arbitrage** (total variance non-decreasing in T): `∂w/∂T ≥ 0` for all k
3. **Vertical spread arbitrage** (call prices decreasing in K): automatically satisfied if density is non-negative

**Tasks:**
1. Implement analytical SVI derivatives:
   - `w'(k) = b · [ρ + (k-m)/√((k-m)² + σ²)]`
   - `w''(k) = b · σ² / ((k-m)² + σ²)^(3/2)`
2. Implement the Durrleman condition checker:
   - Evaluate `g(k)` on a fine grid of k values for each expiry
   - Flag any violations
3. If violations exist, re-fit with penalty:
   - Add penalty term: `λ · Σ max(0, -g(k_i))²` to the objective
   - Increase λ until no violations remain
4. Implement calendar spread checker:
   - For each k, verify total variance is non-decreasing across consecutive expiries
   - If violated, adjust parameters of the shorter-expiry slice
5. Generate an arbitrage diagnostic report per surface

**Output:** `src/arbitrage.py` with functions:
```python
def durrleman_condition(k, svi_params) -> np.ndarray  # g(k) values
def check_butterfly_arbitrage(k_grid, svi_params) -> bool
def check_calendar_arbitrage(slice_params_list) -> bool
def fit_svi_arbitrage_free(k, w, weights=None, lambda_init=1.0) -> SVIParams
```

**Tests:** `tests/test_arbitrage.py`
- Construct a known-arbitrage-free surface → verify passes
- Construct a known-violating surface → verify detection

---

## SESSION C: Dashboard + README

### Phase 5: Interactive Dashboard

Build a Streamlit app that renders the surface and diagnostics.

**Tasks:**
1. **3D Volatility Surface** (main panel):
   - X-axis: log-moneyness (or strike)
   - Y-axis: time to expiry
   - Z-axis: implied volatility
   - Color: residual (market IV - fitted IV), blue = cheap, red = rich
   - Use `plotly.graph_objects.Surface`
   - Toggle between market IV surface and SVI-fitted surface
   - Hover shows: strike, expiry, market IV, fitted IV, residual, bid-ask spread

2. **Smile Slices** (side panel):
   - 2D plot showing IV vs strike for a selected expiry
   - Market points as scatter, SVI fit as smooth line
   - Confidence bands from bid-ask IV range
   - Dropdown to select expiry

3. **Residual Heatmap**:
   - Strike × Expiry heatmap of (market IV - fitted IV)
   - Color scale: diverging blue-white-red
   - Highlights statistically significant mispricings (|residual| > 2σ)

4. **Arbitrage Diagnostics Panel**:
   - Durrleman condition plot per expiry (g(k) should be ≥ 0 everywhere)
   - Calendar spread total variance plot (should be monotonically increasing)
   - Green checkmarks / red flags for each constraint

5. **Term Structure Panel**:
   - ATM IV vs expiry (the term structure of volatility)
   - SVI parameter evolution across expiries (how skew, level, curvature change)

6. **Mispricing Table**:
   - Top 10 options where |residual| is largest
   - Columns: strike, expiry, market IV, fitted IV, residual, direction (cheap/rich), bid-ask

**Output:** `dashboard/app.py` + `dashboard/components/`

### Phase 6: README + Polish

1. Screenshot of the dashboard at the top
2. One-paragraph summary connecting to LOB scanner
3. Mathematical methodology section (SVI formula, Newton-Raphson, Durrleman condition)
4. Key findings (2-3 bullets about what the surface reveals)
5. Setup instructions
6. Architecture diagram
7. References (Gatheral 2004, Durrleman 2005)

---

## Project Structure

```
vol-surface-engine/
├── README.md
├── requirements.txt
├── data/
│   ├── download.py              # Fetch options data
│   └── spy_options.parquet      # Cached data
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Options chain fetching + cleaning
│   ├── iv_engine.py             # Black-Scholes + Newton-Raphson IV
│   ├── svi_fitter.py            # SVI calibration per slice
│   ├── arbitrage.py             # No-arbitrage constraint enforcement
│   └── surface.py               # Surface interpolation + query
├── dashboard/
│   ├── app.py                   # Streamlit main app
│   └── components/
│       ├── surface_3d.py        # 3D vol surface
│       ├── smile_slice.py       # Per-expiry smile plot
│       ├── residual_heatmap.py  # Mispricing heatmap
│       ├── arbitrage_diag.py    # Constraint diagnostics
│       └── term_structure.py    # ATM term structure
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_iv_extraction.ipynb
│   ├── 03_svi_fitting.ipynb
│   └── 04_arbitrage_analysis.ipynb
├── tests/
│   ├── test_iv_engine.py
│   ├── test_svi_fitter.py
│   └── test_arbitrage.py
└── docs/
    └── methodology.md
```

---

## Python Dependencies

```
numpy>=1.24
scipy>=1.11
pandas>=2.0
yfinance>=0.2
plotly>=5.18
streamlit>=1.28
pyarrow>=14.0
requests>=2.31
pytest>=7.4
```

---

## Key Findings

Frame results around questions a recruiter would find interesting:

1. "The SVI parameterization fits SPY option smiles with < 0.5 vol point RMSE across all expiries, while enforcing no-butterfly and no-calendar-spread arbitrage via Durrleman conditions"
2. "ATM skew steepens 2-3x for near-term expiries vs. long-dated, consistent with leverage effect and jump risk concentration"
3. "The residual analysis identifies 15-20 options per snapshot with statistically significant mispricings (|residual| > 2σ), concentrated in weekly expiries with wide bid-ask spreads"

---

## References

1. Gatheral, J. (2004). "A Parsimonious Arbitrage-Free Implied Volatility Parameterization with Application to the Valuation of Volatility Derivatives." Presentation, Global Derivatives & Risk Management.
2. Gatheral, J., Jacquier, A. (2014). "Arbitrage-Free SVI Volatility Surfaces." Quantitative Finance, 14(1).
3. Durrleman, V. (2005). "From Implied to Spot Volatilities." PhD Thesis, Princeton University.
4. Black, F., Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." Journal of Political Economy.
5. Brenner, M., Subrahmanyam, M.G. (1988). "A Simple Formula to Compute the Implied Standard Deviation." Financial Analysts Journal.
