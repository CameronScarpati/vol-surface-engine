"""Tests for the SVI Fitter (Phase 3)."""

import numpy as np
import pandas as pd
import pytest

from src.svi_fitter import (
    SVIParams,
    fit_all_slices,
    fit_svi_slice,
    interpolate_surface,
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
)


# ---------------------------------------------------------------------------
# Known SVI parameters for synthetic tests
# ---------------------------------------------------------------------------
# Typical SPY-like SVI parameters
_KNOWN_PARAMS = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.0, sigma=0.1)


# ---------------------------------------------------------------------------
# SVI function
# ---------------------------------------------------------------------------
class TestSVITotalVariance:
    """Verify the SVI functional form."""

    def test_atm_value(self):
        """At k=m, w = a + b * sigma."""
        k = np.array([0.0])
        w = svi_total_variance(k, 0.04, 0.15, -0.3, 0.0, 0.1)
        expected = 0.04 + 0.15 * 0.1  # a + b * sigma
        assert w[0] == pytest.approx(expected, abs=1e-12)

    def test_asymptotic_slopes(self):
        """For large |k|, w ~ a + b*(rho +/- 1)*(k-m) + b*sigma.

        Put wing slope: b*(rho - 1), call wing slope: b*(rho + 1).
        """
        a, b, rho, m, sigma = 0.04, 0.15, -0.3, 0.0, 0.1
        k_far_right = np.array([10.0])
        k_far_left = np.array([-10.0])

        w_right = svi_total_variance(k_far_right, a, b, rho, m, sigma)
        w_left = svi_total_variance(k_far_left, a, b, rho, m, sigma)

        # Right asymptote slope ~ b*(rho+1) = 0.15*0.7 = 0.105
        # Left asymptote slope  ~ b*(rho-1) = 0.15*(-1.3) = -0.195
        # (but it's a V-shape, so left wing goes *up* in total variance)
        assert w_right[0] > 1.0  # large positive
        assert w_left[0] > 1.0   # also large positive (put wing)

    def test_positive_variance_at_vertex(self):
        """Total variance must be positive for sensible parameters."""
        k = np.linspace(-0.5, 0.5, 200)
        w = svi_total_variance(k, 0.04, 0.15, -0.3, 0.0, 0.1)
        assert np.all(w > 0)

    def test_scalar_input(self):
        """Should accept scalar k and return a numeric result."""
        w = svi_total_variance(0.0, 0.04, 0.15, -0.3, 0.0, 0.1)
        assert np.isscalar(w) or (isinstance(w, np.ndarray) and w.ndim == 0)
        assert float(w) > 0

    def test_symmetry_when_rho_zero(self):
        """When rho=0, w(k) = w(-k) for m=0."""
        k = np.linspace(-0.5, 0.5, 101)
        w = svi_total_variance(k, 0.04, 0.15, 0.0, 0.0, 0.1)
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)


# ---------------------------------------------------------------------------
# Analytical derivatives
# ---------------------------------------------------------------------------
class TestSVIDerivatives:
    """Validate analytical derivatives against finite differences."""

    def test_first_derivative_fd(self):
        """w'(k) matches central finite difference."""
        k = np.linspace(-0.4, 0.4, 50)
        b, rho, m, sigma = 0.15, -0.3, 0.0, 0.1
        a = 0.04

        wp_analytical = svi_first_derivative(k, b, rho, m, sigma)

        h = 1e-6
        wp_fd = (
            svi_total_variance(k + h, a, b, rho, m, sigma)
            - svi_total_variance(k - h, a, b, rho, m, sigma)
        ) / (2 * h)

        np.testing.assert_allclose(wp_analytical, wp_fd, atol=1e-6)

    def test_second_derivative_fd(self):
        """w''(k) matches central finite difference of w'."""
        k = np.linspace(-0.4, 0.4, 50)
        b, rho, m, sigma = 0.15, -0.3, 0.0, 0.1

        wpp_analytical = svi_second_derivative(k, b, rho, m, sigma)

        h = 1e-5
        wpp_fd = (
            svi_first_derivative(k + h, b, rho, m, sigma)
            - svi_first_derivative(k - h, b, rho, m, sigma)
        ) / (2 * h)

        np.testing.assert_allclose(wpp_analytical, wpp_fd, atol=1e-4)

    def test_second_derivative_positive(self):
        """w''(k) = b * sigma^2 / (...) > 0 always (convexity)."""
        k = np.linspace(-1.0, 1.0, 200)
        wpp = svi_second_derivative(k, 0.15, -0.3, 0.0, 0.1)
        assert np.all(wpp > 0)


# ---------------------------------------------------------------------------
# SVIParams dataclass
# ---------------------------------------------------------------------------
class TestSVIParams:
    def test_roundtrip_array(self):
        p = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.0, sigma=0.1)
        arr = p.to_array()
        assert arr.shape == (5,)
        p2 = SVIParams.from_array(arr)
        assert p2.a == pytest.approx(p.a)
        assert p2.b == pytest.approx(p.b)
        assert p2.rho == pytest.approx(p.rho)

    def test_frozen(self):
        p = SVIParams(a=0.04, b=0.15, rho=-0.3, m=0.0, sigma=0.1)
        with pytest.raises(AttributeError):
            p.a = 0.05  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Slice fitting
# ---------------------------------------------------------------------------
class TestFitSVISlice:
    """Verify that the fitter can recover known SVI parameters."""

    def _generate_data(
        self,
        params: SVIParams,
        n_points: int = 50,
        noise_std: float = 0.0,
        k_range: tuple[float, float] = (-0.4, 0.4),
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        k = np.linspace(*k_range, n_points)
        w = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
        if noise_std > 0:
            w = w + rng.normal(0, noise_std, size=len(w))
        return k, np.maximum(w, 1e-6)

    def test_exact_recovery(self):
        """Recover known parameters from noiseless data."""
        k, w = self._generate_data(_KNOWN_PARAMS, noise_std=0.0)
        fitted = fit_svi_slice(k, w)

        assert fitted.a == pytest.approx(_KNOWN_PARAMS.a, abs=1e-4)
        assert fitted.b == pytest.approx(_KNOWN_PARAMS.b, abs=1e-4)
        assert fitted.rho == pytest.approx(_KNOWN_PARAMS.rho, abs=1e-3)
        assert fitted.m == pytest.approx(_KNOWN_PARAMS.m, abs=1e-3)
        assert fitted.sigma == pytest.approx(_KNOWN_PARAMS.sigma, abs=1e-3)

    def test_low_rmse_noiseless(self):
        """RMSE should be near zero for noiseless data."""
        k, w = self._generate_data(_KNOWN_PARAMS, noise_std=0.0)
        fitted = fit_svi_slice(k, w)
        assert fitted.rmse < 1e-6

    def test_high_r_squared(self):
        """R^2 should be near 1 for clean data."""
        k, w = self._generate_data(_KNOWN_PARAMS, noise_std=1e-5)
        fitted = fit_svi_slice(k, w)
        assert fitted.r_squared > 0.999

    def test_noisy_data_reasonable_fit(self):
        """With moderate noise, fit should still be decent."""
        k, w = self._generate_data(_KNOWN_PARAMS, noise_std=1e-3)
        fitted = fit_svi_slice(k, w)
        assert fitted.rmse < 0.005
        assert fitted.r_squared > 0.9

    def test_too_few_points_raises(self):
        """Need at least 5 points for 5 parameters."""
        k = np.array([0.0, 0.1, 0.2, 0.3])
        w = np.array([0.04, 0.05, 0.06, 0.07])
        with pytest.raises(ValueError, match="at least 5"):
            fit_svi_slice(k, w)

    def test_weighted_fit(self):
        """Weights should not break the fitter."""
        k, w = self._generate_data(_KNOWN_PARAMS, n_points=30)
        weights = np.ones(len(k))
        weights[len(k) // 2] = 10.0  # emphasise ATM
        fitted = fit_svi_slice(k, w, weights=weights)
        assert fitted.rmse < 1e-4


# ---------------------------------------------------------------------------
# Multi-slice fitting
# ---------------------------------------------------------------------------
class TestFitAllSlices:
    """Test fit_all_slices on a synthetic IV DataFrame."""

    def _make_chain(self) -> pd.DataFrame:
        """Create synthetic chain with known SVI smiles."""
        S, r, q = 560.0, 0.0435, 0.013
        rng = np.random.default_rng(42)
        rows = []

        T_values = [30 / 365.25, 90 / 365.25, 180 / 365.25, 365 / 365.25]
        strikes = np.linspace(S * 0.85, S * 1.15, 25)

        for T in T_values:
            F = S * np.exp((r - q) * T)
            for K in strikes:
                k = np.log(K / F)
                # Use known SVI to generate total variance
                w = float(svi_total_variance(
                    k, 0.04, 0.15, -0.3, 0.0, 0.1
                ))
                iv = np.sqrt(max(w / T, 1e-8))
                for otype in ["call", "put"]:
                    rows.append({
                        "expiry": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(T * 365.25)),
                        "strike": K,
                        "option_type": otype,
                        "iv": iv + rng.normal(0, 0.001),
                        "S": S,
                        "r": r,
                        "q": q,
                        "T": T,
                        "open_interest": rng.integers(100, 5000),
                    })

        return pd.DataFrame(rows)

    def test_returns_all_expiries(self):
        chain = self._make_chain()
        result = fit_all_slices(chain)
        assert len(result) == 4  # four expiry slices

    def test_required_columns(self):
        chain = self._make_chain()
        result = fit_all_slices(chain)
        for col in ["expiry", "T", "a", "b", "rho", "m", "sigma", "rmse", "r_squared"]:
            assert col in result.columns

    def test_good_fit_quality(self):
        chain = self._make_chain()
        result = fit_all_slices(chain)
        # All slices should fit well
        assert (result["r_squared"] > 0.95).all()
        assert (result["rmse"] < 0.01).all()


# ---------------------------------------------------------------------------
# Surface interpolation
# ---------------------------------------------------------------------------
class TestInterpolateSurface:
    def _make_slice_params(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"expiry": "2026-02-01", "T": 0.1, "a": 0.04, "b": 0.15, "rho": -0.3, "m": 0.0, "sigma": 0.1},
            {"expiry": "2026-06-01", "T": 0.5, "a": 0.05, "b": 0.12, "rho": -0.25, "m": 0.0, "sigma": 0.12},
            {"expiry": "2027-01-01", "T": 1.0, "a": 0.06, "b": 0.10, "rho": -0.20, "m": 0.0, "sigma": 0.15},
        ])

    def test_at_exact_expiry(self):
        """Interpolation at an exact slice T should match the slice."""
        sp = self._make_slice_params()
        k = np.array([0.0])
        w = interpolate_surface(k, 0.5, sp)
        expected = svi_total_variance(k, 0.05, 0.12, -0.25, 0.0, 0.12)
        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_midpoint_interpolation(self):
        """Between two slices, result should be between the slice values."""
        sp = self._make_slice_params()
        k = np.array([0.0])
        w_lo = svi_total_variance(k, 0.04, 0.15, -0.3, 0.0, 0.1).item()
        w_hi = svi_total_variance(k, 0.05, 0.12, -0.25, 0.0, 0.12).item()
        w_mid = interpolate_surface(k, 0.3, sp).item()
        assert w_lo <= w_mid <= w_hi or w_hi <= w_mid <= w_lo

    def test_extrapolation_below(self):
        """Below the shortest expiry, use the shortest slice."""
        sp = self._make_slice_params()
        k = np.array([0.0])
        w = interpolate_surface(k, 0.01, sp)
        expected = svi_total_variance(k, 0.04, 0.15, -0.3, 0.0, 0.1)
        np.testing.assert_allclose(w, expected, atol=1e-10)

    def test_extrapolation_above(self):
        """Above the longest expiry, use the longest slice."""
        sp = self._make_slice_params()
        k = np.array([0.0])
        w = interpolate_surface(k, 5.0, sp)
        expected = svi_total_variance(k, 0.06, 0.10, -0.20, 0.0, 0.15)
        np.testing.assert_allclose(w, expected, atol=1e-10)
