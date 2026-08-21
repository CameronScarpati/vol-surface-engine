"""Golden-value tests for the numerical core.

Unlike the round-trip and property tests elsewhere in the suite, every
expected value in this file was computed OUTSIDE this codebase, so a
consistent error shared by a function and its inverse cannot pass here.

Provenance of the pinned constants:
- Computed 2026-08-21 with mpmath at 50 significant digits, using an
  erfc-based normal CDF (independent of scipy.stats), then rounded to
  float64 precision.
- The (S=42, K=40, r=0.10, sigma=0.20, T=0.5) Black-Scholes case is the
  standard worked example from Hull, "Options, Futures, and Other
  Derivatives" (call 4.76, put 0.81 at 2 decimal places); the pins below
  agree with those rounded values.
- The known butterfly-arbitrage SVI slice is the Vogt smile discussed in
  Gatheral & Jacquier, "Arbitrage-Free SVI Volatility Surfaces" (2014),
  with (a, b, rho, m, sigma) = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153).
  Its Durrleman function dips negative near k = 0.88, which sits OUTSIDE
  the [-0.5, 0.5] grid this repo checks by default, making it a sharp
  regression case for grid handling.

Tolerances: closed-form formulas are pinned to abs=1e-12 (float64
round-off level); iterative solver outputs to abs=1e-6 (the solver's own
convergence tolerance).
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.arbitrage import (
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    durrleman_condition,
)
from src.data_loader import clean_chain
from src.iv_engine import bs_price, bs_vega, compute_all_iv, implied_volatility
from src.svi_fitter import (
    SVIParams,
    fit_svi_slice,
    interpolate_surface,
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
)

# ---------------------------------------------------------------------------
# Reference SVI parameter sets
# ---------------------------------------------------------------------------
# Short-dated equity-like slice used for the value pins below.
_P1 = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.1)
# Longer-dated slice; w_P2(k) > w_P1(k) for all k in [-1, 1] (minimum gap
# 0.0270 at the left wing, verified with mpmath), so (P1, P2) is a
# calendar-ordered pair.
_P2 = SVIParams(a=0.06, b=0.5, rho=-0.3, m=0.0, sigma=0.15)
# The Vogt smile (Gatheral & Jacquier 2014): butterfly arbitrage near
# k = 0.88, outside the default diagnostic grid.
_VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


class TestBlackScholesGoldens:
    """bs_price and bs_vega against externally computed references."""

    def test_hull_call(self):
        # mpmath: 4.75942239287153343...; Hull rounds to 4.76.
        assert bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "call") == pytest.approx(
            4.7594223928715334, abs=1e-12
        )

    def test_hull_put(self):
        # mpmath: 0.80859937290009365...; Hull rounds to 0.81.
        assert bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "put") == pytest.approx(
            0.8085993729000936, abs=1e-12
        )

    def test_dividend_call(self):
        # S=100, K=100, T=1, r=0.05, q=0.02, sigma=0.25 (mpmath).
        assert bs_price(100, 100, 1.0, 0.05, 0.02, 0.25, "call") == pytest.approx(
            11.123761928058132, abs=1e-12
        )

    def test_dividend_put(self):
        assert bs_price(100, 100, 1.0, 0.05, 0.02, 0.25, "put") == pytest.approx(
            8.226837047454002, abs=1e-12
        )

    def test_deep_otm_call(self):
        # S=100, K=140, T=0.25, r=0.03, q=0, sigma=0.35 (mpmath): a small
        # price in the low-vega region the Brent fallback exists for.
        assert bs_price(100, 140, 0.25, 0.03, 0.0, 0.35, "call") == pytest.approx(
            0.2395482256355851, abs=1e-12
        )

    def test_vega_hull_case(self):
        # mpmath: S * exp(-qT) * sqrt(T) * n(d1) = 8.81341505960285...
        assert bs_vega(42, 40, 0.5, 0.10, 0.0, 0.20) == pytest.approx(8.813415059602851, abs=1e-12)

    def test_vega_dividend_case(self):
        assert bs_vega(100, 100, 1.0, 0.05, 0.02, 0.25) == pytest.approx(
            37.948089225445703, abs=1e-12
        )


class TestImpliedVolGoldens:
    """implied_volatility must recover the sigma behind externally
    computed prices, and refuse impossible prices."""

    def test_recovers_hull_sigma(self):
        iv = implied_volatility(4.7594223928715334, 42, 40, 0.5, 0.10, 0.0, "call")
        assert iv == pytest.approx(0.20, abs=1e-6)

    def test_recovers_dividend_sigma(self):
        iv = implied_volatility(8.226837047454002, 100, 100, 1.0, 0.05, 0.02, "put")
        assert iv == pytest.approx(0.25, abs=1e-6)

    def test_recovers_deep_otm_sigma(self):
        iv = implied_volatility(0.2395482256355851, 100, 140, 0.25, 0.03, 0.0, "call")
        assert iv == pytest.approx(0.35, abs=1e-6)

    def test_price_below_intrinsic_is_nan(self):
        # Discounted forward intrinsic for this call is
        # S - K * exp(-rT) = 12.22210791745006 (mpmath); a price of 5.0
        # is below it, so no implied volatility exists.
        iv = implied_volatility(5.0, 100, 90, 0.5, 0.05, 0.0, "call")
        assert math.isnan(iv)

    def test_price_above_upper_bound_is_nan(self):
        # A call can never be worth more than S * exp(-qT) = 100.
        iv = implied_volatility(101.0, 100, 90, 0.5, 0.05, 0.0, "call")
        assert math.isnan(iv)


class TestSVIGoldens:
    """svi_total_variance and its analytic derivatives at pinned points."""

    def test_w_at_vertex_is_hand_derivable(self):
        # At k = m the formula collapses to a + b * sigma = 0.08 exactly.
        w = svi_total_variance(np.array([0.05]), 0.04, 0.4, -0.4, 0.05, 0.1)
        assert w[0] == pytest.approx(0.08, abs=1e-12)

    def test_w_pins(self):
        k = np.array([-0.3, 0.0, 0.25])
        w = svi_total_variance(k, 0.04, 0.4, -0.4, 0.05, 0.1)
        # mpmath at 50 digits, rounded to float64.
        assert w[0] == pytest.approx(0.2416021977856104, abs=1e-12)
        assert w[1] == pytest.approx(0.0927213595499958, abs=1e-12)
        assert w[2] == pytest.approx(0.0974427190999916, abs=1e-12)

    def test_w_hand_checkable_pin(self):
        # For _P2 at k = -0.2: 0.06 + 0.5 * (0.06 + sqrt(0.04 + 0.0225))
        # = 0.06 + 0.5 * (0.06 + 0.25) = 0.215 with pen and paper.
        w = svi_total_variance(np.array([-0.2]), 0.06, 0.5, -0.3, 0.0, 0.15)
        assert w[0] == pytest.approx(0.215, abs=1e-12)

    def test_first_derivative_pin(self):
        # mpmath: b * (rho + dk / sqrt(dk^2 + sigma^2)) at k = 0.25.
        d1 = svi_first_derivative(np.array([0.25]), 0.4, -0.4, 0.05, 0.1)
        assert d1[0] == pytest.approx(0.1977708763999664, abs=1e-12)

    def test_second_derivative_pin(self):
        # mpmath: b * sigma^2 / (dk^2 + sigma^2)^1.5 at k = 0.25.
        d2 = svi_second_derivative(np.array([0.25]), 0.4, -0.4, 0.05, 0.1)
        assert d2[0] == pytest.approx(0.3577708763999664, abs=1e-12)


class TestDurrlemanGoldens:
    """durrleman_condition values and the Vogt regression case."""

    def test_g_pins_benign_slice(self):
        g = durrleman_condition(np.array([0.0, 0.25]), _P1)
        # mpmath from the published formula, independent of src/.
        assert g[0] == pytest.approx(2.1142593982569316, abs=1e-12)
        assert g[1] == pytest.approx(0.6330528570546797, abs=1e-12)

    def test_g_pins_vogt_slice(self):
        g = durrleman_condition(np.array([-0.5, 0.9]), _VOGT)
        assert g[0] == pytest.approx(0.3568535725377074, abs=1e-12)
        # Negative: the slice carries butterfly arbitrage at k = 0.9.
        assert g[1] == pytest.approx(-0.0326851307090229, abs=1e-12)

    def test_vogt_violation_detected_on_wide_grid(self):
        wide = np.linspace(-1.5, 1.5, 601)
        assert check_butterfly_arbitrage(wide, _VOGT) is False

    def test_vogt_violation_invisible_on_narrow_grid(self):
        # On [-0.5, 0.5] the Vogt g(k) stays positive; the violation near
        # k = 0.88 is only visible to a wider grid. This pins the fact
        # that grid choice is part of the check's meaning.
        narrow = np.linspace(-0.5, 0.5, 500)
        assert check_butterfly_arbitrage(narrow, _VOGT) is True

    def test_calendar_ordered_pair_passes(self):
        k_grid = np.linspace(-1.0, 1.0, 201)
        assert check_calendar_arbitrage([_P1, _P2], np.array([0.25, 0.75]), k_grid) is True

    def test_calendar_reversed_pair_fails(self):
        k_grid = np.linspace(-1.0, 1.0, 201)
        assert check_calendar_arbitrage([_P2, _P1], np.array([0.25, 0.75]), k_grid) is False


class TestInterpolationGoldens:
    """interpolate_surface against hand-assembled expectations."""

    def _slices(self):
        return pd.DataFrame(
            [
                {"T": 0.25, "a": 0.04, "b": 0.4, "rho": -0.4, "m": 0.05, "sigma": 0.1},
                {"T": 0.75, "a": 0.06, "b": 0.5, "rho": -0.3, "m": 0.0, "sigma": 0.15},
            ]
        )

    def test_midpoint_is_mean_of_slice_pins(self):
        # w_P1(0.1) = 0.0767213595499958 and w_P2(0.1) = 0.1351387818865997
        # (both mpmath); at T = 0.5 the variance-linear scheme returns
        # exactly their mean.
        w = interpolate_surface(0.1, 0.5, self._slices())
        assert w[0] == pytest.approx(0.1059300707182978, abs=1e-12)

    def test_at_first_knot_equals_slice_pin(self):
        w = interpolate_surface(0.1, 0.25, self._slices())
        assert w[0] == pytest.approx(0.0767213595499958, abs=1e-12)

    def test_flat_extrapolation_below(self):
        w = interpolate_surface(0.1, 0.05, self._slices())
        assert w[0] == pytest.approx(0.0767213595499958, abs=1e-12)

    def test_flat_extrapolation_above(self):
        w = interpolate_surface(0.1, 2.0, self._slices())
        assert w[0] == pytest.approx(0.1351387818865997, abs=1e-12)


class TestDeterminismAndConsistency:
    """Reproducibility and vectorized-equals-scalar guarantees."""

    def test_fit_svi_slice_is_bit_deterministic(self):
        # The fitter seeds its restart generator internally, so two calls
        # on identical input must agree to the last bit, not just within
        # tolerance.
        k = np.linspace(-0.4, 0.4, 25)
        w = svi_total_variance(k, 0.04, 0.4, -0.4, 0.05, 0.1)
        first = fit_svi_slice(k, w)
        second = fit_svi_slice(k, w)
        assert (first.a, first.b, first.rho, first.m, first.sigma) == (
            second.a,
            second.b,
            second.rho,
            second.m,
            second.sigma,
        )

    def test_compute_all_iv_equals_scalar_solver(self):
        # The vectorized path must produce bitwise the same result as
        # calling the scalar solver row by row, including the NaN row.
        rows = [
            # (mid_price, S, strike, T, r, q, option_type)
            (4.7594223928715334, 42.0, 40.0, 0.5, 0.10, 0.0, "call"),
            (8.226837047454002, 100.0, 100.0, 1.0, 0.05, 0.02, "put"),
            (0.2395482256355851, 100.0, 140.0, 0.25, 0.03, 0.0, "call"),
            (5.0, 100.0, 90.0, 0.5, 0.05, 0.0, "call"),  # below intrinsic
        ]
        chain = pd.DataFrame(
            rows, columns=["mid_price", "S", "strike", "T", "r", "q", "option_type"]
        )
        result = compute_all_iv(chain)
        expected = np.array(
            [implied_volatility(p, s, k, t, r, q, o) for p, s, k, t, r, q, o in rows]
        )
        np.testing.assert_array_equal(result["iv"].to_numpy(), expected)


class TestCleanChainGolden:
    """A fixture chain where each row targets exactly one filter, with the
    surviving set pinned. This fails if any filter silently stops firing
    or starts over-filtering."""

    def test_pinned_survivors(self):
        now = pd.Timestamp.now(tz="UTC").normalize()
        far = now + pd.Timedelta(days=60)
        near = now + pd.Timedelta(days=1)
        rows = [
            # keeper: tight spread, passes every filter
            {
                "expiry": far,
                "strike": 100.0,
                "option_type": "call",
                "bid": 5.0,
                "ask": 5.2,
                "volume": 10,
                "openInterest": 100,
            },
            # zero volume
            {
                "expiry": far,
                "strike": 100.0,
                "option_type": "call",
                "bid": 5.0,
                "ask": 5.2,
                "volume": 0,
                "openInterest": 100,
            },
            # zero open interest
            {
                "expiry": far,
                "strike": 100.0,
                "option_type": "call",
                "bid": 5.0,
                "ask": 5.2,
                "volume": 10,
                "openInterest": 0,
            },
            # near expiry (1 day < 3-day minimum)
            {
                "expiry": near,
                "strike": 100.0,
                "option_type": "call",
                "bid": 5.0,
                "ask": 5.2,
                "volume": 10,
                "openInterest": 100,
            },
            # deep OTM: ln(170/100) = 0.53 > 0.5
            {
                "expiry": far,
                "strike": 170.0,
                "option_type": "call",
                "bid": 0.5,
                "ask": 0.6,
                "volume": 10,
                "openInterest": 100,
            },
            # crossed quote: bid >= ask
            {
                "expiry": far,
                "strike": 100.0,
                "option_type": "call",
                "bid": 5.2,
                "ask": 5.0,
                "volume": 10,
                "openInterest": 100,
            },
            # zero mid-price
            {
                "expiry": far,
                "strike": 100.0,
                "option_type": "call",
                "bid": 0.0,
                "ask": 0.0,
                "volume": 10,
                "openInterest": 100,
            },
            # keeper with wide spread: (1.5 - 1.0) / 1.25 = 0.4 > 0.2,
            # so it survives but is flagged low-confidence
            {
                "expiry": far,
                "strike": 90.0,
                "option_type": "put",
                "bid": 1.0,
                "ask": 1.5,
                "volume": 5,
                "openInterest": 50,
            },
        ]
        cleaned = clean_chain(pd.DataFrame(rows), spot=100.0, risk_free=0.04, div_yield=0.01)

        # Exactly the two keepers survive, sorted by (expiry, type, strike).
        assert len(cleaned) == 2
        assert list(cleaned["option_type"]) == ["call", "put"]
        assert list(cleaned["strike"]) == [100.0, 90.0]
        assert list(cleaned["low_confidence"]) == [False, True]
        assert cleaned["mid_price"].iloc[0] == pytest.approx(5.1, abs=1e-12)
        assert cleaned["mid_price"].iloc[1] == pytest.approx(1.25, abs=1e-12)
        # Enrichment columns carry the inputs through unchanged.
        assert (cleaned["S"] == 100.0).all()
        assert (cleaned["r"] == 0.04).all()
        assert (cleaned["q"] == 0.01).all()
