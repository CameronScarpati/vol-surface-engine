"""
Integration tests covering the full pipeline:

    data (synthetic) -> IV extraction -> SVI calibration -> arbitrage diagnostics -> surface queries

These tests verify that all phases (built by separate sessions) work together
correctly as an end-to-end system.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.arbitrage import (
    check_butterfly_arbitrage,
    durrleman_condition,
    fit_svi_arbitrage_free,
    generate_diagnostics,
)
from src.iv_engine import bs_price, bs_vega, compute_all_iv, implied_volatility
from src.surface import VolSurface, build_surface
from src.svi_fitter import (
    SVIParams,
    fit_all_slices,
    interpolate_surface,
    svi_first_derivative,
    svi_second_derivative,
    svi_total_variance,
)
from tests.conftest import (
    DIV_YIELD,
    RISK_FREE,
    SPOT,
    make_synthetic_chain,
    synthetic_iv,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Test: IV extraction from synthetic chain
# ---------------------------------------------------------------------------
class TestIVExtractionIntegration:
    """Verify that IV extraction works on synthetic BS-generated prices."""

    def test_iv_recovery_from_synthetic_chain(self):
        """IVs extracted from BS prices should match the input IVs closely."""
        chain = make_synthetic_chain(dte_days=[90], n_strikes=15, seed=123)
        result = compute_all_iv(chain)

        assert "iv" in result.columns
        valid = result.dropna(subset=["iv"])
        assert len(valid) > 0

        for _, row in valid.iterrows():
            expected_iv = synthetic_iv(row["strike"], row["T"])
            assert abs(row["iv"] - expected_iv) < 0.03, (
                f"K={row['strike']:.1f} T={row['T']:.4f}: "
                f"extracted={row['iv']:.4f} vs expected={expected_iv:.4f}"
            )

    def test_high_extraction_rate(self):
        """Vast majority of synthetic options should yield a valid IV."""
        chain = make_synthetic_chain()
        result = compute_all_iv(chain)
        success_rate = result["iv"].notna().mean()
        assert success_rate > 0.90, f"IV extraction rate too low: {success_rate:.2%}"


# ---------------------------------------------------------------------------
# Test: IV -> SVI pipeline
# ---------------------------------------------------------------------------
class TestIVToSVIPipeline:
    """Verify that SVI fitting works on IV-enriched chain data."""

    def test_fit_all_slices_from_iv_chain(self):
        """fit_all_slices should produce params for every expiry in the chain."""
        chain = make_synthetic_chain()
        chain_iv = compute_all_iv(chain)
        slice_params = fit_all_slices(chain_iv)

        n_expiries = chain["expiry"].nunique()
        assert len(slice_params) == n_expiries

    def test_svi_fit_quality_on_synthetic(self):
        """SVI fits should have R^2 > 0.95 and RMSE < 0.01 on clean synthetic data."""
        chain = make_synthetic_chain()
        chain_iv = compute_all_iv(chain)
        slice_params = fit_all_slices(chain_iv)

        for _, row in slice_params.iterrows():
            assert row["r_squared"] > 0.95, (
                f"T={row['T']:.4f}: R^2={row['r_squared']:.4f}"
            )
            assert row["rmse"] < 0.01, (
                f"T={row['T']:.4f}: RMSE={row['rmse']:.6f}"
            )

    def test_svi_params_within_bounds(self):
        """Fitted SVI parameters should be within expected bounds."""
        chain = make_synthetic_chain()
        chain_iv = compute_all_iv(chain)
        slice_params = fit_all_slices(chain_iv)

        for _, row in slice_params.iterrows():
            assert -0.5 <= row["a"] <= 0.5
            assert 0 < row["b"] <= 2.0
            assert -1.0 < row["rho"] < 1.0
            assert -1.0 <= row["m"] <= 1.0
            assert 0 < row["sigma"] <= 2.0


# ---------------------------------------------------------------------------
# Test: SVI -> Arbitrage diagnostics pipeline
# ---------------------------------------------------------------------------
class TestSVIToArbitragePipeline:
    """Verify that arbitrage diagnostics work on fitted SVI parameters."""

    def test_diagnostics_from_fitted_slices(self):
        """generate_diagnostics should run without error on pipeline output."""
        chain = make_synthetic_chain()
        chain_iv = compute_all_iv(chain)
        slice_params = fit_all_slices(chain_iv)
        diag = generate_diagnostics(slice_params)

        assert len(diag.butterfly_free) == len(slice_params)
        assert isinstance(diag.calendar_free, bool)

    def test_arbitrage_free_fitting_improves_surface(self):
        """fit_svi_arbitrage_free should produce Durrleman-compliant slices."""
        chain = make_synthetic_chain(dte_days=[90], n_strikes=30, seed=77)
        chain_iv = compute_all_iv(chain)

        df = chain_iv.dropna(subset=["iv"]).copy()
        df["F"] = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])
        df["k"] = np.log(df["strike"] / df["F"])
        df["w"] = df["iv"] ** 2 * df["T"]

        grouped = df.groupby("k").agg(w=("w", "mean")).reset_index().sort_values("k")
        k_arr = grouped["k"].values
        w_arr = grouped["w"].values

        fitted = fit_svi_arbitrage_free(k_arr, w_arr)
        k_check = np.linspace(-0.5, 0.5, 500)
        assert check_butterfly_arbitrage(k_check, fitted)


# ---------------------------------------------------------------------------
# Test: Full pipeline via build_surface
# ---------------------------------------------------------------------------
class TestBuildSurfaceIntegration:
    """End-to-end test of the build_surface entry point."""

    @pytest.fixture
    def surface(self) -> VolSurface:
        chain = make_synthetic_chain()
        return build_surface(chain, SPOT, RISK_FREE, DIV_YIELD)

    def test_returns_vol_surface(self, surface):
        assert isinstance(surface, VolSurface)

    def test_chain_has_iv_column(self, surface):
        assert "iv" in surface.chain.columns
        assert surface.chain["iv"].notna().sum() > 0

    def test_slice_params_populated(self, surface):
        assert len(surface.slice_params) > 0
        for col in ["a", "b", "rho", "m", "sigma", "T", "rmse", "r_squared"]:
            assert col in surface.slice_params.columns

    def test_diagnostics_populated(self, surface):
        assert isinstance(surface.diagnostics.butterfly_free, dict)
        assert len(surface.diagnostics.butterfly_free) == len(surface.slice_params)

    def test_surface_query_returns_valid_iv(self, surface):
        """Query the fitted surface at ATM and verify it returns a sensible IV."""
        iv = surface.iv(SPOT, 0.25)
        assert np.isfinite(iv)
        assert 0.05 < iv < 1.0, f"ATM IV={iv:.4f} is outside expected range"

    def test_surface_query_monotone_in_strike(self, surface):
        """For a given T, put-wing IV should generally be higher than call-wing."""
        T = 0.25
        iv_low_strike = surface.iv(SPOT * 0.90, T)
        iv_atm = surface.iv(SPOT, T)
        if np.isfinite(iv_low_strike) and np.isfinite(iv_atm):
            assert iv_low_strike > iv_atm - 0.02

    def test_fitted_iv_for_chain(self, surface):
        """fitted_iv_for_chain should add fitted_iv and residual columns."""
        result = surface.fitted_iv_for_chain()
        assert "fitted_iv" in result.columns
        assert "residual" in result.columns

        valid = result.dropna(subset=["residual"])
        assert len(valid) > 0

        mean_abs = valid["residual"].abs().mean()
        assert mean_abs < 0.02, f"Mean |residual| = {mean_abs:.4f}"

    def test_expiries_property(self, surface):
        """VolSurface.expiries should return the T values from slice_params."""
        expiries = surface.expiries
        assert len(expiries) == len(surface.slice_params)
        assert np.all(expiries > 0)

    def test_surface_spot_and_rates(self, surface):
        assert surface.spot == SPOT
        assert surface.risk_free == RISK_FREE
        assert surface.div_yield == DIV_YIELD


# ---------------------------------------------------------------------------
# Test: Surface interpolation integration
# ---------------------------------------------------------------------------
class TestSurfaceInterpolationIntegration:
    """Verify that interpolate_surface works with fit_all_slices output."""

    def test_interpolate_between_slices(self):
        chain = make_synthetic_chain()
        chain_iv = compute_all_iv(chain)
        slice_params = fit_all_slices(chain_iv)

        T_vals = sorted(slice_params["T"].values)
        assert len(T_vals) >= 2

        T_mid = (T_vals[0] + T_vals[1]) / 2
        k = np.array([0.0])
        w = interpolate_surface(k, T_mid, slice_params)

        assert np.isfinite(w).all()
        assert float(np.squeeze(w)) > 0

    def test_interpolation_monotone_in_T_at_atm(self):
        """Total variance at ATM should generally increase with T."""
        chain = make_synthetic_chain()
        chain_iv = compute_all_iv(chain)
        slice_params = fit_all_slices(chain_iv)

        T_vals = sorted(slice_params["T"].values)
        k = np.array([0.0])

        w_prev = 0.0
        violations = 0
        for T in T_vals:
            w = float(np.squeeze(interpolate_surface(k, T, slice_params)))
            if w < w_prev - 1e-6:
                violations += 1
            w_prev = w

        assert violations <= 1, f"Too many calendar violations at ATM: {violations}"


# ---------------------------------------------------------------------------
# Test: Analytical derivative consistency across modules
# ---------------------------------------------------------------------------
class TestDerivativeConsistency:
    """Verify SVI derivatives are consistent with the SVI function."""

    def test_durrleman_uses_correct_derivatives(self):
        """Durrleman g(k) should use the same derivatives as svi_fitter."""
        params = SVIParams(a=0.04, b=0.10, rho=-0.2, m=0.0, sigma=0.15)
        k = np.linspace(-0.3, 0.3, 100)

        w = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
        wp = svi_first_derivative(k, params.b, params.rho, params.m, params.sigma)
        wpp = svi_second_derivative(k, params.b, params.rho, params.m, params.sigma)

        w = np.maximum(w, 1e-14)
        g_manual = (1 - k * wp / (2 * w)) ** 2 - wp**2 / 4 * (1 / w + 0.25) + wpp / 2

        g_module = durrleman_condition(k, params)
        np.testing.assert_allclose(g_manual, g_module, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: BS -> IV -> BS roundtrip across the full chain
# ---------------------------------------------------------------------------
class TestBSIVRoundtripIntegration:
    """Verify the BS pricing -> IV extraction roundtrip at scale."""

    def test_roundtrip_across_strikes_and_expiries(self):
        """For synthetic BS prices, IV extraction should recover original IV."""
        S, r, q = 560.0, 0.0435, 0.013
        sigma_true = 0.20

        strikes = np.linspace(S * 0.85, S * 1.15, 20)
        T_values = [0.04, 0.08, 0.25, 0.5, 1.0]

        for T in T_values:
            for K in strikes:
                for otype in ["call", "put"]:
                    price = bs_price(S, K, T, r, q, sigma_true, otype)
                    iv = implied_volatility(price, S, K, T, r, q, otype)
                    assert abs(iv - sigma_true) < 1e-5, (
                        f"Roundtrip fail: K={K:.0f} T={T} {otype}: iv={iv:.6f}"
                    )

    def test_vega_consistency_with_pricing(self):
        """Vega should approximate the finite-difference of BS price."""
        S, K, T, r, q, sigma = 560.0, 560.0, 0.25, 0.0435, 0.013, 0.20
        h = 1e-6
        price_up = bs_price(S, K, T, r, q, sigma + h, "call")
        price_down = bs_price(S, K, T, r, q, sigma - h, "call")
        vega_fd = (price_up - price_down) / (2 * h)
        vega_analytical = bs_vega(S, K, T, r, q, sigma)
        assert abs(vega_fd - vega_analytical) < 1e-4


# ---------------------------------------------------------------------------
# Test: Dashboard data flow (no Streamlit runtime needed)
# ---------------------------------------------------------------------------
class TestDashboardDataFlow:
    """Verify that the data structures produced by the pipeline match
    what the dashboard components expect."""

    @pytest.fixture
    def surface(self) -> VolSurface:
        chain = make_synthetic_chain()
        return build_surface(chain, SPOT, RISK_FREE, DIV_YIELD)

    def test_chain_has_columns_for_3d_surface(self, surface):
        """surface_3d.py expects: strike, T, iv in chain."""
        required = {"strike", "T", "iv"}
        assert required.issubset(surface.chain.columns)

    def test_chain_has_columns_for_smile_slice(self, surface):
        """smile_slice.py expects: strike, T, iv, bid, ask, S, r, q, option_type."""
        required = {"strike", "T", "iv", "bid", "ask", "S", "r", "q", "option_type"}
        assert required.issubset(surface.chain.columns)

    def test_chain_has_columns_for_residual_heatmap(self, surface):
        """residual_heatmap.py expects: strike, T, iv, S, r, q."""
        required = {"strike", "T", "iv", "S", "r", "q"}
        assert required.issubset(surface.chain.columns)

    def test_slice_params_has_columns_for_diagnostics(self, surface):
        """arbitrage_diag.py expects: T, a, b, rho, m, sigma, expiry."""
        required = {"T", "a", "b", "rho", "m", "sigma", "expiry"}
        assert required.issubset(surface.slice_params.columns)

    def test_slice_params_has_columns_for_term_structure(self, surface):
        """term_structure.py expects: T, a, b, rho, m, sigma, r_squared."""
        required = {"T", "a", "b", "rho", "m", "sigma", "r_squared"}
        assert required.issubset(surface.slice_params.columns)

    def test_diagnostics_structure(self, surface):
        """Dashboard expects butterfly_free dict and calendar_free bool."""
        diag = surface.diagnostics
        assert isinstance(diag.butterfly_free, dict)
        assert isinstance(diag.calendar_free, bool)
        assert isinstance(diag.calendar_violation_expiries, list)

    def test_chain_has_open_interest_for_mispricing(self, surface):
        """mispricing table needs open_interest for weighting."""
        assert "open_interest" in surface.chain.columns
