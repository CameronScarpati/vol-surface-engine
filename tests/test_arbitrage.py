"""Tests for the Arbitrage module (Phase 4)."""

import numpy as np
import pandas as pd
import pytest

from src.arbitrage import (
    ArbitrageDiagnostics,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    durrleman_condition,
    fit_svi_arbitrage_free,
    generate_diagnostics,
)
from src.svi_fitter import SVIParams, svi_total_variance


# ---------------------------------------------------------------------------
# Known arbitrage-free parameters
# ---------------------------------------------------------------------------
# These satisfy the Gatheral & Jacquier sufficient conditions:
#   a + b * sigma * sqrt(1 - rho^2) >= 0   (non-negative ATM variance)
#   b * (1 + |rho|) < 4                     (Roger Lee moment bound)
_AF_PARAMS = SVIParams(a=0.04, b=0.10, rho=-0.2, m=0.0, sigma=0.15)

# A well-behaved second slice with higher total variance (for calendar tests)
_AF_PARAMS_LONG = SVIParams(a=0.08, b=0.08, rho=-0.15, m=0.0, sigma=0.20)

# Pathological parameters that violate butterfly (extreme curvature)
_VIOLATING_PARAMS = SVIParams(a=0.001, b=1.5, rho=-0.9, m=0.0, sigma=0.01)


K_GRID = np.linspace(-0.5, 0.5, 500)


# ---------------------------------------------------------------------------
# Durrleman condition
# ---------------------------------------------------------------------------
class TestDurrlemanCondition:
    """Test the Durrleman function g(k)."""

    def test_returns_array(self):
        g = durrleman_condition(K_GRID, _AF_PARAMS)
        assert isinstance(g, np.ndarray)
        assert g.shape == K_GRID.shape

    def test_known_arb_free_positive(self):
        """Known good params should have g(k) >= 0 everywhere."""
        g = durrleman_condition(K_GRID, _AF_PARAMS)
        assert np.all(g >= -1e-10), f"Min g(k) = {g.min():.6e}"

    def test_violating_params_negative(self):
        """Pathological params should produce g(k) < 0 somewhere."""
        g = durrleman_condition(K_GRID, _VIOLATING_PARAMS)
        assert np.any(g < 0), "Expected butterfly violation not detected"

    def test_scalar_k(self):
        """Should handle scalar k input."""
        g = durrleman_condition(np.array([0.0]), _AF_PARAMS)
        assert g.shape == (1,)


# ---------------------------------------------------------------------------
# Butterfly arbitrage check
# ---------------------------------------------------------------------------
class TestCheckButterflyArbitrage:
    def test_arb_free_passes(self):
        assert check_butterfly_arbitrage(K_GRID, _AF_PARAMS) is True

    def test_violating_fails(self):
        assert check_butterfly_arbitrage(K_GRID, _VIOLATING_PARAMS) is False

    def test_tolerance_handling(self):
        """Tiny numerical noise should not trigger false violations."""
        # Use stricter tolerance to check default works
        assert check_butterfly_arbitrage(K_GRID, _AF_PARAMS, tol=-1e-8) is True


# ---------------------------------------------------------------------------
# Calendar-spread arbitrage
# ---------------------------------------------------------------------------
class TestCheckCalendarArbitrage:
    def test_increasing_variance_passes(self):
        """Total variance increasing across expiries → no calendar arb."""
        params = [_AF_PARAMS, _AF_PARAMS_LONG]
        T_values = np.array([0.25, 1.0])
        assert check_calendar_arbitrage(params, T_values, K_GRID) is True

    def test_reversed_variance_fails(self):
        """Longer expiry with lower variance → calendar arb detected."""
        # Swap order: long-dated params first (lower variance at short T)
        params = [_AF_PARAMS_LONG, _AF_PARAMS]
        T_values = np.array([0.25, 1.0])
        assert check_calendar_arbitrage(params, T_values, K_GRID) is False

    def test_single_slice_passes(self):
        """A single slice trivially has no calendar arbitrage."""
        assert check_calendar_arbitrage([_AF_PARAMS], np.array([0.5]), K_GRID) is True

    def test_dataframe_input(self):
        """Accepts DataFrame from fit_all_slices."""
        df = pd.DataFrame([
            {"T": 0.25, "a": _AF_PARAMS.a, "b": _AF_PARAMS.b,
             "rho": _AF_PARAMS.rho, "m": _AF_PARAMS.m, "sigma": _AF_PARAMS.sigma},
            {"T": 1.0, "a": _AF_PARAMS_LONG.a, "b": _AF_PARAMS_LONG.b,
             "rho": _AF_PARAMS_LONG.rho, "m": _AF_PARAMS_LONG.m,
             "sigma": _AF_PARAMS_LONG.sigma},
        ])
        assert check_calendar_arbitrage(df) is True


# ---------------------------------------------------------------------------
# Arbitrage-free fitting
# ---------------------------------------------------------------------------
class TestFitSVIArbitrageFree:
    """Test the penalty-method constrained fitter."""

    def _generate_smile(
        self, params: SVIParams, n: int = 40
    ) -> tuple[np.ndarray, np.ndarray]:
        k = np.linspace(-0.4, 0.4, n)
        w = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
        return k, w

    def test_already_arb_free(self):
        """When input is already arb-free, should return quickly."""
        k, w = self._generate_smile(_AF_PARAMS)
        fitted = fit_svi_arbitrage_free(k, w)
        assert check_butterfly_arbitrage(K_GRID, fitted) is True
        assert fitted.rmse < 1e-4

    def test_noisy_arb_free_data(self):
        """Noisy data from arb-free params should still yield arb-free fit."""
        k, w = self._generate_smile(_AF_PARAMS)
        rng = np.random.default_rng(99)
        w_noisy = w + rng.normal(0, 5e-4, size=len(w))
        w_noisy = np.maximum(w_noisy, 1e-6)

        fitted = fit_svi_arbitrage_free(k, w_noisy)
        assert check_butterfly_arbitrage(K_GRID, fitted) is True

    def test_returns_svi_params(self):
        k, w = self._generate_smile(_AF_PARAMS)
        fitted = fit_svi_arbitrage_free(k, w)
        assert isinstance(fitted, SVIParams)
        assert fitted.n_points == len(k)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
class TestGenerateDiagnostics:
    def _make_slice_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"expiry": "2026-04-01", "T": 0.25,
             "a": _AF_PARAMS.a, "b": _AF_PARAMS.b, "rho": _AF_PARAMS.rho,
             "m": _AF_PARAMS.m, "sigma": _AF_PARAMS.sigma},
            {"expiry": "2027-01-01", "T": 1.0,
             "a": _AF_PARAMS_LONG.a, "b": _AF_PARAMS_LONG.b,
             "rho": _AF_PARAMS_LONG.rho, "m": _AF_PARAMS_LONG.m,
             "sigma": _AF_PARAMS_LONG.sigma},
        ])

    def test_clean_surface_all_pass(self):
        df = self._make_slice_df()
        diag = generate_diagnostics(df)
        assert all(diag.butterfly_free.values())
        assert diag.calendar_free is True
        assert len(diag.calendar_violation_expiries) == 0

    def test_returns_diagnostics_type(self):
        df = self._make_slice_df()
        diag = generate_diagnostics(df)
        assert isinstance(diag, ArbitrageDiagnostics)

    def test_detects_butterfly_violation(self):
        """Inject a violating slice and verify detection."""
        df = pd.DataFrame([
            {"expiry": "2026-04-01", "T": 0.25,
             "a": _VIOLATING_PARAMS.a, "b": _VIOLATING_PARAMS.b,
             "rho": _VIOLATING_PARAMS.rho, "m": _VIOLATING_PARAMS.m,
             "sigma": _VIOLATING_PARAMS.sigma},
            {"expiry": "2027-01-01", "T": 1.0,
             "a": _AF_PARAMS_LONG.a, "b": _AF_PARAMS_LONG.b,
             "rho": _AF_PARAMS_LONG.rho, "m": _AF_PARAMS_LONG.m,
             "sigma": _AF_PARAMS_LONG.sigma},
        ])
        diag = generate_diagnostics(df)
        assert not all(diag.butterfly_free.values())

    def test_detects_calendar_violation(self):
        """Reversed variance ordering should flag calendar arb."""
        df = pd.DataFrame([
            {"expiry": "2026-04-01", "T": 0.25,
             "a": _AF_PARAMS_LONG.a, "b": _AF_PARAMS_LONG.b,
             "rho": _AF_PARAMS_LONG.rho, "m": _AF_PARAMS_LONG.m,
             "sigma": _AF_PARAMS_LONG.sigma},
            {"expiry": "2027-01-01", "T": 1.0,
             "a": _AF_PARAMS.a, "b": _AF_PARAMS.b,
             "rho": _AF_PARAMS.rho, "m": _AF_PARAMS.m,
             "sigma": _AF_PARAMS.sigma},
        ])
        diag = generate_diagnostics(df)
        assert diag.calendar_free is False
        assert len(diag.calendar_violation_expiries) > 0


# ---------------------------------------------------------------------------
# Integration: fit synthetic SPY data and check Durrleman
# ---------------------------------------------------------------------------
class TestSPYIntegration:
    """End-to-end test: generate synthetic SPY smiles, fit SVI,
    and verify the Durrleman condition holds."""

    @staticmethod
    def _synthetic_iv(k: float, T: float) -> float:
        """Same model as generate_synthetic_data.py."""
        atm = 0.16 + 0.03 * np.exp(-2.0 * T)
        skew_coeff = -0.12 * (1.0 + 0.5 / (T + 0.05))
        skew = skew_coeff * k
        smile = 0.15 * k**2
        return float(np.clip(atm + skew + smile, 0.05, 1.5))

    def test_fit_and_durrleman(self):
        """Fit SVI to realistic SPY smile and verify no butterfly arb."""
        S, r, q = 560.0, 0.0435, 0.013
        T = 90 / 365.25
        F = S * np.exp((r - q) * T)

        strikes = np.linspace(S * 0.85, S * 1.15, 40)
        k_arr = np.log(strikes / F)
        ivs = np.array([self._synthetic_iv(k, T) for k in k_arr])
        w_arr = ivs**2 * T

        fitted = fit_svi_arbitrage_free(k_arr, w_arr)
        k_check = np.linspace(-0.5, 0.5, 500)

        assert check_butterfly_arbitrage(k_check, fitted), (
            f"Durrleman violation on SPY 90d smile: "
            f"min g(k) = {durrleman_condition(k_check, fitted).min():.6e}"
        )
        assert fitted.r_squared > 0.95, f"Poor fit: R² = {fitted.r_squared:.4f}"
