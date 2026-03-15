# Technical Architecture Breakdown

## 1. Architecture & Data Pipeline

### Data Sources

**Primary**: SPY equity options chains via **yfinance** (`src/data_loader.py:136-178`). Fetches all available expiry dates for a ticker, concatenates calls and puts into a single DataFrame. Spot price sourced from `ticker.fast_info["lastPrice"]` with a 5-day history fallback for index tickers.

**Risk-free rate**: 3-month T-bill constant maturity rate (DGS3MO) from the **FRED API** (`src/data_loader.py:56-83`). Falls back to a hardcoded 4.35% if no API key is set or the call fails.

**Dividend yield**: Trailing 12-month dividends from yfinance, converted to a continuous rate via `q = ln(1 + D_12m / S)` (`src/data_loader.py:89-117`).

**Synthetic fallback**: `scripts/generate_synthetic_data.py` produces a Parquet file with known IV characteristics (ATM term structure + skew + smile) for offline development.

### Data Cleaning & Filtering (`src/data_loader.py:184-288`)

The `clean_chain` function applies a multi-stage filter pipeline:

1. **Zero volume/OI removal** — eliminates stale/illiquid contracts
2. **Near-expiry exclusion** — removes options with < 3 DTE (unstable IV near expiry)
3. **Log-moneyness filter** — removes |ln(K/S)| > 0.5 (deep OTM/ITM where BS model breaks down)
4. **Mid-price positivity** — removes zero/negative mid-prices
5. **Crossed-quote removal** — removes rows where bid >= ask (stale quotes)
6. **Low-confidence flagging** — marks rows where bid-ask spread > 20% of mid (not removed, just flagged)

### End-to-End Flow

```
yfinance options chain + FRED rate + dividend estimation
    → clean_chain() [filters, enriches with T, S, r, q]
    → Parquet cache
    → OTM-only selection (build_surface, src/surface.py:118-129)
    → compute_all_iv() [Newton-Raphson + Brent IV extraction]
    → Moneyness filter + MAD outlier removal (src/surface.py:143-209)
    → fit_all_slices() [SVI calibration per expiry]
    → generate_diagnostics() [butterfly + calendar checks]
    → VolSurface object (queryable for IV, Greeks, local vol)
    → Streamlit dashboard (8 interactive Plotly panels)
```

---

## 2. Core Mathematical/Statistical Methods

### (a) Black-Scholes Pricing with Continuous Dividends (`src/iv_engine.py:39-78`)

**What**: Closed-form European option pricing under geometric Brownian motion with continuous dividend yield:
- `C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)`
- `d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)`

**Why chosen**: The BS model is the universal quoting convention for equity options. It's not assumed to be "correct" — it's the inverting function used to translate prices into implied volatilities. The project uses BS as a mapping tool, not as a belief about actual dynamics.

**Assumptions**: Log-normal returns, constant rates/dividends, continuous trading, no frictions.

### (b) Newton-Raphson IV Solver with Brent Fallback (`src/iv_engine.py:115-226`)

**What**: Two-phase root-finding to invert BS price → σ:
1. **Brenner-Subrahmanyam initial guess**: `σ₀ ≈ √(2π/T) · C/S` (clamped to [0.05, 3.0])
2. **Newton-Raphson**: Uses BS vega as the Jacobian. Converges when |Δprice| < 10⁻⁸ or |Δσ| < 10⁻¹⁰
3. **Brent's method fallback**: Bracket-based root-finding on [0.001, 5.0] when NR fails (near-zero vega regions)

**Why NR + Brent**: NR gives quadratic convergence for well-behaved cases (ATM/near-ATM). Brent is guaranteed to converge when a sign change exists but doesn't need derivatives — critical for deep OTM where vega → 0 and NR steps become unreliable.

**Parameters estimated**: Implied volatility σ for each (K, T, option_type) triple. Max 100 iterations per option.

### (c) SVI Parameterization (`src/svi_fitter.py`)

**What**: The raw SVI model (Gatheral 2004) parameterizes total implied variance as a function of log-moneyness k = ln(K/F):

`w(k) = a + b·[ρ(k - m) + √((k - m)² + σ²)]`

Five parameters per expiry slice: `a` (variance level), `b` (wing slope), `ρ` (skew), `m` (horizontal translation), `σ` (vertex smoothness).

**Why SVI over alternatives**: SVI is the industry standard for single-slice smile parameterization because (1) it has closed-form analytical derivatives (critical for Dupire local vol), (2) its 5 parameters have direct financial interpretations, (3) Gatheral & Jacquier (2014) provide explicit no-arbitrage conditions. Alternatives like SABR require a separate stochastic model and can't easily enforce butterfly-free constraints; polynomial fits lack the asymptotic behavior needed for wing extrapolation.

**Optimizer**: Multi-start **L-BFGS-B** with 8 random restarts + 1 heuristic seed (`src/svi_fitter.py:237-278`). Bounds enforced: a ∈ [-0.5, 0.5], b ∈ [10⁻⁴, 2.0], ρ ∈ (-0.999, 0.999), m ∈ [-1, 1], σ ∈ [10⁻⁴, 2.0]. Convergence: ftol=10⁻¹⁵, gtol=10⁻¹². Loss function: weighted sum of squared residuals in total-variance space (optionally OI-weighted).

**Surface interpolation**: Linear interpolation in total-variance space between adjacent expiry slices (`src/svi_fitter.py:363-415`). This preserves calendar-spread arbitrage-free property.

### (d) Arbitrage Enforcement (`src/arbitrage.py`)

**Butterfly arbitrage** — checked via the **Durrleman (2005) condition** (`src/arbitrage.py:64-100`):

`g(k) = (1 - kw'/(2w))² - (w')²/4·(1/w + 1/4) + w''/2 ≥ 0 ∀k`

This is equivalent to requiring non-negative risk-neutral density. Evaluated on a 500-point grid over k ∈ [-0.5, 0.5].

**Calendar-spread arbitrage** — total variance must be non-decreasing in T for every k (`src/arbitrage.py:131-180`).

**Enforcement mechanism** — **progressive penalty method** (`src/arbitrage.py:217-296`):
1. Fit unconstrained SVI
2. If violations exist, re-fit with penalty `λ · Σ max(0, -g(kᵢ))²`
3. Escalate λ by 10x each iteration until violations vanish or λ reaches 10⁶

### (e) Dupire Local Volatility (`dashboard/components/local_vol.py:57-154`)

**What**: Extracts the instantaneous diffusion coefficient from the fitted SVI surface:

`σ_loc²(K,T) = (∂w/∂T) / g(k)`

where the numerator uses **central finite differences** across SVI slices and the denominator uses **analytical SVI derivatives** (Durrleman's g(k)).

**Post-processing**: Gaussian smoothing with normalized convolution (handles NaN), stronger smoothing along T-axis (σ = [1.5, 2.0]) to tame finite-difference noise. Values capped to [2%, 80%].

### (f) Greeks (`dashboard/components/greeks.py:21-46`)

Black-Scholes Delta, Gamma, Vega, Theta computed from the fitted IV surface across the full (strike, T) grid. Vega normalized per 1% vol move; Theta per calendar day.

---

## 3. Key Design Decisions

### 1. OTM-Only IV Extraction (`src/surface.py:118-129`)

**Decision**: Only use out-of-the-money options (calls where K > F, puts where K < F) with a narrow ATM band (|k| < 0.005) for both.

**Why**: OTM options have higher vega (sensitivity to vol), making IV extraction numerically more stable. Deep ITM options have most of their value from intrinsic — the time value (which encodes vol information) is a small residual prone to noise. This is standard practice on derivatives trading desks.

**What would go wrong**: Using ITM options would introduce noisy IV estimates from the intrinsic-value floor, corrupt the SVI fit, and potentially create spurious arbitrage violations.

### 2. Multi-Start L-BFGS-B with 8 Random Restarts (`src/svi_fitter.py:220-258`)

**Decision**: 9 total optimization runs (1 heuristic + 8 random) per expiry slice, keeping the best.

**Why**: The SVI objective is non-convex with multiple local minima. A single start can get trapped in a poor local minimum, especially with noisy data.

**What would go wrong**: A single-start fit could produce a slice with wildly wrong wing behavior (e.g., negative variance in the wings), triggering butterfly arbitrage violations and cascading errors into local vol computation.

### 3. Fitting in Total-Variance Space (w = σ²·T) Rather Than IV Space (`src/svi_fitter.py:310`)

**Decision**: The SVI model and all fitting is done in total implied variance space, not implied volatility space.

**Why**: Total variance is the natural quantity for SVI — it's what the model parameterizes, and calendar-spread arbitrage is simply w₂(k) ≥ w₁(k) for T₂ > T₁. Working in IV space would require dividing by √T at every step, introducing numerical instability for short-dated options.

**What would go wrong**: Fitting in IV space would make calendar-spread checks non-trivial (they'd depend on T ratios), and short-dated slices would dominate the loss function due to the 1/√T amplification.

### 4. MAD-Based Per-Slice Outlier Removal (`src/surface.py:184-209`)

**Decision**: Within each expiry, remove IVs deviating > 3x MAD from the slice median, with a MAD floor of 0.005 and a guard that at least 5 points survive.

**Why**: Market data contains stale quotes, data errors, and illiquid contracts that would poison the SVI fit. MAD (median absolute deviation) is robust to the very outliers it's trying to detect, unlike standard deviation.

**What would go wrong**: Without outlier removal, a single stale deep-OTM quote could pull the SVI wing parameters to extreme values, violating Durrleman's condition and producing nonsensical local vol.

### 5. Variance-Linear Interpolation Across Tenors (`src/svi_fitter.py:363-415`)

**Decision**: Interpolate between expiry slices linearly in total-variance space: `w(k,T) = (1-α)·w₁(k) + α·w₂(k)`.

**Why**: If individual slices satisfy w₁(k) ≤ w₂(k) ∀k (calendar-free), then any convex combination preserves this ordering. This guarantees the interpolated surface is also calendar-free — a property that wouldn't hold for interpolation in IV space.

**What would go wrong**: Interpolating in IV space could create calendar-spread arbitrage at intermediate tenors, meaning you could construct a risk-free profit by buying a long-dated option and selling a short-dated one at the same strike.

---

## 4. Results & Validation

### Metrics Computed

- **Per-slice RMSE** in total-variance space (`src/svi_fitter.py:266`)
- **R²** (coefficient of determination) per slice (`src/svi_fitter.py:268-269`)
- **Max absolute error** per slice (`src/svi_fitter.py:270`)
- **Butterfly-free boolean** per slice (Durrleman g(k) ≥ 0 on 500-point grid)
- **Calendar-free boolean** across all adjacent slice pairs
- **IV extraction success rate** (finite IV / total options)
- **Residual heatmap** (market IV − fitted IV across strike × expiry)

### Validation Approach

The test suite (130 tests) validates through **round-trip testing**: generate BS prices from known σ → extract IV → verify σ is recovered to < 10⁻⁶ tolerance. This is the gold standard for IV engine validation (`tests/test_iv_engine.py:102-139`).

Integration tests (`tests/test_integration.py`) verify the full pipeline: synthetic chain → IV → SVI fit → arbitrage diagnostics → surface queries. Key assertions: R² > 0.95 per slice, RMSE < 0.01, IV extraction rate > 90%, mean |residual| < 0.02.

### Skeptical Interviewer Questions

1. **"You're fitting to mid-prices — what about bid-ask bounce?"** Mid-prices are used as the input, but the low-confidence flag (spread > 20%) and MAD outlier removal partially address this. A more rigorous approach would fit to the bid-ask interval rather than the midpoint.

2. **"Why Black-Scholes for IV extraction when the market clearly isn't log-normal?"** BS is used purely as a *quoting convention* (the bijective map between price and σ), not as a model of reality. The SVI surface is the actual model.

3. **"How do you handle the smile-to-surface interpolation problem?"** Linear in total variance — but this assumes SVI parameters vary smoothly across tenors. Short-dated slices with few data points can still produce jumpy interpolated surfaces.

4. **"Your Dupire local vol uses finite differences in T — isn't that noisy?"** Yes. The code compensates with Gaussian smoothing (σ=[1.5, 2.0]) and expiry selection to ensure minimum gaps, but this is a fundamental limitation. A parametric model of T-dependence (e.g., SSVI) would be more stable.

5. **"What happens when the penalty method fails to eliminate butterfly arbitrage?"** It logs a warning and returns the best-effort fit (`src/arbitrage.py:292-296`). The surface is technically not arbitrage-free in that slice, meaning the implied risk-neutral density could go negative — this could produce negative local vol or mispriced exotics.

---

## 5. Resume Bullet Fodder

1. **Built an end-to-end volatility surface engine that ingests live SPY options data, extracts implied volatility via Newton-Raphson root-finding (10⁻⁸ price tolerance, 10⁻¹⁰ vol tolerance) with Brent's method fallback, and calibrates per-expiry SVI parameterizations using multi-start L-BFGS-B optimization with 8 random restarts.**

2. **Implemented static arbitrage enforcement using Durrleman's butterfly condition (g(k) ≥ 0 on a 500-point log-moneyness grid) and calendar-spread monotonicity, with a progressive penalty method that escalates λ from 1.0 to 10⁶ to eliminate risk-neutral density violations.**

3. **Designed an adaptive multi-stage data pipeline that filters raw options chains through 6 quality gates (volume, OI, moneyness, bid-ask validation, MAD-based outlier removal) and produces OI-weighted SVI fits achieving R² > 0.95 across all expiry slices.**

4. **Computed Dupire local volatility from the fitted SVI surface using analytical k-derivatives and finite-difference T-derivatives, with normalized-convolution Gaussian smoothing to handle sparse/noisy regions — bridging the implied-vol quoting convention to risk-neutral dynamics.**

5. **Developed a 130-test suite validating the full pipeline via round-trip IV recovery from synthetic Black-Scholes prices (σ recovered to < 10⁻⁶), SVI parameter recovery from noiseless data, Durrleman condition consistency checks, and end-to-end integration tests across 6 expiry slices.**

---

## 6. Interview Danger Zones (Priority Order)

1. **SVI parameterization and its financial interpretation** — You must be able to write w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)] from memory, explain what each parameter controls (a = level, b = wing slope, ρ = skew, m = translation, σ = curvature), and articulate why total variance (not IV) is the natural fitting space.

2. **Durrleman's butterfly condition** — You need to derive g(k) ≥ 0 from the requirement that the risk-neutral density (second derivative of call prices w.r.t. strike) is non-negative. Be ready to explain why g(k) < 0 means a butterfly spread has negative cost (arbitrage).

3. **Newton-Raphson convergence for IV extraction** — Explain why vega (the derivative dC/dσ) is the Jacobian, when NR fails (near-zero vega in deep OTM), why Brent's method is a safe fallback (guaranteed convergence with sign change), and what the Brenner-Subrahmanyam initial guess is.

4. **Calendar-spread arbitrage and variance-linear interpolation** — Explain why total variance must be non-decreasing in T (otherwise you can buy the cheaper long-dated option and sell the short-dated one for a guaranteed profit), and why linear interpolation in w-space preserves this property.

5. **Dupire's formula and the connection between implied and local volatility** — Be able to state the formula σ_loc² = (∂w/∂T) / g(k) and explain that local vol is the unique diffusion coefficient that reproduces all European option prices. Know that g(k) in the denominator is exactly the Durrleman condition — negative g means negative local variance, which is non-physical.

6. **OTM option selection and its numerical justification** — Explain why desks quote in OTM options (higher vega → more informative about vol), why ITM options have intrinsic-value contamination, and why you average calls and puts at the same strike (put-call parity means they should give the same IV).

7. **Non-convexity of SVI calibration** — Explain why multi-start optimization is necessary (the 5-parameter landscape has local minima), what L-BFGS-B is (limited-memory quasi-Newton with box constraints), and why 8 restarts was chosen as a cost/accuracy tradeoff.

8. **Black-Scholes Greeks from a fitted surface** — Be ready to compute d1, d2 by hand and explain why Greeks computed from the *fitted* surface (rather than per-contract) give smoother, more useful hedging quantities.
