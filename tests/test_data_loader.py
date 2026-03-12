"""Tests for the Data Pipeline (Phase 1)."""

import numpy as np
import pandas as pd
import pytest
from datetime import timezone

from src.data_loader import (
    OptionsData,
    clean_chain,
    estimate_dividend_yield,
    load_parquet,
    save_parquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_raw_chain(
    n_strikes: int = 10,
    spot: float = 500.0,
) -> pd.DataFrame:
    """Build a realistic-looking raw options chain DataFrame."""
    now = pd.Timestamp.now(tz=timezone.utc).normalize()

    expiries = [
        now + pd.Timedelta(days=d) for d in [1, 7, 30, 60, 90, 180]
    ]

    rows = []
    for exp in expiries:
        for otype in ["call", "put"]:
            for i in range(n_strikes):
                K = spot * (0.85 + i * 0.03)
                mid = max(0.5, abs(spot - K) * 0.1 + np.random.uniform(0.1, 2.0))
                spread = mid * np.random.uniform(0.02, 0.30)
                rows.append(
                    {
                        "expiry": exp,
                        "strike": K,
                        "option_type": otype,
                        "bid": mid - spread / 2,
                        "ask": mid + spread / 2,
                        "volume": np.random.randint(0, 5000),
                        "openInterest": np.random.randint(0, 20000),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OptionsData dataclass
# ---------------------------------------------------------------------------
class TestOptionsData:
    def test_dataclass_fields(self):
        df = pd.DataFrame({"a": [1]})
        od = OptionsData(spot=500.0, risk_free=0.04, div_yield=0.01, chains=df)
        assert od.spot == 500.0
        assert od.risk_free == 0.04
        assert od.div_yield == 0.01
        assert len(od.chains) == 1


# ---------------------------------------------------------------------------
# clean_chain
# ---------------------------------------------------------------------------
class TestCleanChain:
    def test_filters_zero_volume(self):
        raw = _make_raw_chain()
        # Force all volume to zero → should be fully filtered out
        raw["volume"] = 0
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        assert len(result) == 0

    def test_filters_near_expiry(self):
        raw = _make_raw_chain()
        result = clean_chain(raw, 500.0, 0.04, 0.01, min_dte_days=3)
        # The 1-day expiry rows should be removed
        min_T = result["T"].min()
        assert min_T >= 3 / 365.25

    def test_filters_deep_otm(self):
        raw = _make_raw_chain()
        result = clean_chain(raw, 500.0, 0.04, 0.01, max_log_moneyness=0.5)
        log_m = np.log(result["strike"] / 500.0)
        assert (log_m.abs() <= 0.5).all()

    def test_output_columns(self):
        raw = _make_raw_chain()
        # Ensure some rows survive
        raw["volume"] = 100
        raw["openInterest"] = 100
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        expected_cols = {
            "expiry", "strike", "option_type", "mid_price",
            "bid", "ask", "volume", "open_interest",
            "S", "r", "q", "T", "low_confidence",
        }
        assert set(result.columns) == expected_cols

    def test_mid_price_positive(self):
        raw = _make_raw_chain()
        raw["volume"] = 100
        raw["openInterest"] = 100
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        if len(result) > 0:
            assert (result["mid_price"] > 0).all()

    def test_low_confidence_flag(self):
        raw = _make_raw_chain(n_strikes=5)
        raw["volume"] = 100
        raw["openInterest"] = 100
        # Make one row with a very wide spread
        raw.loc[raw.index[0], "bid"] = 0.01
        raw.loc[raw.index[0], "ask"] = 100.0
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        if len(result) > 0:
            assert result["low_confidence"].dtype == bool

    def test_sorted_output(self):
        raw = _make_raw_chain()
        raw["volume"] = 100
        raw["openInterest"] = 100
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        if len(result) > 1:
            # Should be sorted by expiry, option_type, strike
            is_sorted = (
                result[["expiry", "option_type", "strike"]]
                .apply(tuple, axis=1)
                .is_monotonic_increasing
            )
            assert is_sorted

    def test_S_r_q_columns(self):
        raw = _make_raw_chain()
        raw["volume"] = 100
        raw["openInterest"] = 100
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        if len(result) > 0:
            assert (result["S"] == 500.0).all()
            assert (result["r"] == 0.04).all()
            assert (result["q"] == 0.01).all()

    def test_crossed_quotes_removed(self):
        """Rows where bid >= ask should be removed."""
        raw = _make_raw_chain(n_strikes=5)
        raw["volume"] = 100
        raw["openInterest"] = 100
        # Force crossed quote
        raw.loc[raw.index[0], "bid"] = 10.0
        raw.loc[raw.index[0], "ask"] = 5.0
        result = clean_chain(raw, 500.0, 0.04, 0.01)
        if len(result) > 0:
            assert (result["bid"] < result["ask"]).all()


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------
class TestParquetIO:
    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame(
            {
                "strike": [100.0, 110.0],
                "mid_price": [5.0, 3.0],
            }
        )
        path = tmp_path / "test.parquet"
        save_parquet(df, path)
        loaded = load_parquet(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_parquet(tmp_path / "nonexistent.parquet")
