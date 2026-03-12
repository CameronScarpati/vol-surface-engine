"""Tests for the Implied Volatility Engine (Phase 2)."""

import numpy as np
import pandas as pd
import pytest

from src.iv_engine import (
    IV_LOWER,
    IV_UPPER,
    bs_price,
    bs_vega,
    compute_all_iv,
    implied_volatility,
)


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------
class TestBSPrice:
    """Validate Black-Scholes closed-form pricing."""

    def test_call_put_parity(self):
        """C - P = S·e^{-qT} - K·e^{-rT}."""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
        call = bs_price(S, K, T, r, q, sigma, "call")
        put = bs_price(S, K, T, r, q, sigma, "put")
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=1e-10)

    def test_call_put_parity_otm(self):
        """Parity holds for OTM strikes too."""
        S, K, T, r, q, sigma = 100.0, 120.0, 0.5, 0.04, 0.01, 0.30
        call = bs_price(S, K, T, r, q, sigma, "call")
        put = bs_price(S, K, T, r, q, sigma, "put")
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert call - put == pytest.approx(parity, abs=1e-10)

    def test_atm_call_roughly_correct(self):
        """ATM call ≈ S·σ·√T / √(2π) for zero rates (Brenner-Subrahmanyam)."""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.0, 0.0, 0.20
        price = bs_price(S, K, T, r, q, sigma, "call")
        approx = S * sigma * np.sqrt(T) / np.sqrt(2 * np.pi)
        # Approximation is rough — within 5%
        assert abs(price - approx) / price < 0.05

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S·e^{-qT} - K·e^{-rT}."""
        S, K, T, r, q, sigma = 100.0, 50.0, 1.0, 0.05, 0.02, 0.20
        price = bs_price(S, K, T, r, q, sigma, "call")
        intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert price == pytest.approx(intrinsic, abs=0.01)

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be near zero."""
        price = bs_price(100.0, 200.0, 0.25, 0.05, 0.02, 0.20, "call")
        assert price < 0.01

    def test_put_positive_otm(self):
        """OTM put should still have positive value."""
        price = bs_price(100.0, 80.0, 1.0, 0.05, 0.02, 0.30, "put")
        assert price > 0

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            bs_price(100, 100, 0, 0.05, 0.02, 0.20, "call")

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            bs_price(100, 100, 1, 0.05, 0.02, 0.0, "call")

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            bs_price(100, 100, 1, 0.05, 0.02, 0.20, "straddle")


# ---------------------------------------------------------------------------
# Vega
# ---------------------------------------------------------------------------
class TestBSVega:
    def test_atm_vega_positive(self):
        v = bs_vega(100.0, 100.0, 1.0, 0.05, 0.02, 0.20)
        assert v > 0

    def test_vega_increases_with_T(self):
        """Longer expiry → higher vega (ATM)."""
        v_short = bs_vega(100.0, 100.0, 0.25, 0.05, 0.0, 0.20)
        v_long = bs_vega(100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
        assert v_long > v_short

    def test_vega_zero_for_zero_T(self):
        assert bs_vega(100.0, 100.0, 0.0, 0.05, 0.0, 0.20) == 0.0

    def test_deep_otm_vega_small(self):
        v = bs_vega(100.0, 200.0, 0.25, 0.05, 0.0, 0.20)
        assert v < 1.0


# ---------------------------------------------------------------------------
# Known-price roundtrip tests (core requirement from spec)
# ---------------------------------------------------------------------------
class TestImpliedVolRoundtrip:
    """BS(σ) → price → IV extraction → recover σ."""

    @pytest.mark.parametrize(
        "sigma_true",
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00],
    )
    def test_roundtrip_call(self, sigma_true):
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-6)

    @pytest.mark.parametrize(
        "sigma_true",
        [0.10, 0.20, 0.30, 0.50],
    )
    def test_roundtrip_put(self, sigma_true):
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "put")
        iv = implied_volatility(price, S, K, T, r, q, "put")
        assert iv == pytest.approx(sigma_true, abs=1e-6)

    @pytest.mark.parametrize("K", [80.0, 90.0, 100.0, 110.0, 120.0])
    def test_roundtrip_various_strikes(self, K):
        sigma_true = 0.25
        S, T, r, q = 100.0, 0.5, 0.04, 0.01
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-6)

    @pytest.mark.parametrize("T", [0.02, 0.08, 0.25, 0.5, 1.0, 2.0])
    def test_roundtrip_various_expiries(self, T):
        sigma_true = 0.20
        S, K, r, q = 100.0, 100.0, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_deep_itm_call(self):
        """Deep ITM call: IV should still be recoverable."""
        sigma_true = 0.20
        S, K, T, r, q = 100.0, 60.0, 1.0, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_deep_otm_call(self):
        """Deep OTM call: still extracts a valid IV."""
        sigma_true = 0.30
        S, K, T, r, q = 100.0, 140.0, 1.0, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_near_expiry(self):
        """Near-expiry (1 week): higher numerical sensitivity."""
        sigma_true = 0.20
        S, K, T, r, q = 100.0, 100.0, 7 / 365.25, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_zero_price_returns_nan(self):
        iv = implied_volatility(0.0, 100, 100, 1.0, 0.05, 0.02, "call")
        assert np.isnan(iv)

    def test_negative_price_returns_nan(self):
        iv = implied_volatility(-1.0, 100, 100, 1.0, 0.05, 0.02, "call")
        assert np.isnan(iv)

    def test_zero_T_returns_nan(self):
        iv = implied_volatility(5.0, 100, 100, 0.0, 0.05, 0.02, "call")
        assert np.isnan(iv)

    def test_high_vol_roundtrip(self):
        """Very high vol (like meme stocks): σ = 2.0."""
        sigma_true = 2.0
        S, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.0
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_low_vol_roundtrip(self):
        """Very low vol: σ = 0.02."""
        sigma_true = 0.02
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert iv == pytest.approx(sigma_true, abs=1e-4)

    def test_iv_within_bounds(self):
        """Extracted IV should always be within [IV_LOWER, IV_UPPER]."""
        sigma_true = 0.25
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        price = bs_price(S, K, T, r, q, sigma_true, "call")
        iv = implied_volatility(price, S, K, T, r, q, "call")
        assert IV_LOWER <= iv <= IV_UPPER


# ---------------------------------------------------------------------------
# Vectorised compute_all_iv
# ---------------------------------------------------------------------------
class TestComputeAllIV:
    def _make_chain(self) -> pd.DataFrame:
        """Create a synthetic chain with known IVs."""
        rows = []
        sigma_true = 0.25
        S, r, q = 100.0, 0.05, 0.02
        for K in [90.0, 95.0, 100.0, 105.0, 110.0]:
            for T in [0.25, 0.5, 1.0]:
                for otype in ["call", "put"]:
                    price = bs_price(S, K, T, r, q, sigma_true, otype)
                    rows.append(
                        {
                            "mid_price": price,
                            "S": S,
                            "strike": K,
                            "T": T,
                            "r": r,
                            "q": q,
                            "option_type": otype,
                        }
                    )
        return pd.DataFrame(rows)

    def test_all_ivs_recovered(self):
        chain = self._make_chain()
        result = compute_all_iv(chain)
        assert "iv" in result.columns
        assert result["iv"].notna().all()
        np.testing.assert_allclose(result["iv"].values, 0.25, atol=1e-5)

    def test_nan_for_bad_rows(self):
        chain = self._make_chain()
        # Inject an impossible price
        chain.loc[0, "mid_price"] = -1.0
        result = compute_all_iv(chain)
        assert np.isnan(result["iv"].iloc[0])
        # Rest should still be fine
        assert result["iv"].iloc[1:].notna().all()

    def test_preserves_other_columns(self):
        chain = self._make_chain()
        chain["extra_col"] = 42
        result = compute_all_iv(chain)
        assert "extra_col" in result.columns
        assert (result["extra_col"] == 42).all()
