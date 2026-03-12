"""
Phase 1: Options Data Pipeline

Pulls SPY options chain data via yfinance, estimates risk-free rate and
dividend yield, cleans/filters the chain, and stores to Parquet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

__all__ = [
    "OptionsData",
    "clean_chain",
    "fetch_risk_free_rate",
    "load_options",
    "load_parquet",
    "save_parquet",
]

DEFAULT_RISK_FREE_RATE = 0.0435  # ~4.35 % 3-month T-bill (updated manually)
PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "spy_options.parquet"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class OptionsData:
    """Cleaned options snapshot ready for IV extraction."""

    spot: float  # Current SPY price
    risk_free: float  # Annualized risk-free rate
    div_yield: float  # Annualized continuous dividend yield
    chains: pd.DataFrame  # See column spec below

    # Required columns after cleaning:
    #   expiry (datetime), strike (float), option_type (str: 'call'|'put'),
    #   mid_price (float), bid (float), ask (float), volume (int),
    #   open_interest (int), S (float), r (float), q (float),
    #   T (float, years), low_confidence (bool)


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------
def fetch_risk_free_rate() -> float:
    """Fetch 3-month T-bill rate from FRED.

    Falls back to ``DEFAULT_RISK_FREE_RATE`` on failure.
    """
    try:
        import os

        from fredapi import Fred

        api_key = os.environ.get("FRED_API_KEY")
        if api_key is None:
            logger.info("FRED_API_KEY not set; using default risk-free rate")
            return DEFAULT_RISK_FREE_RATE

        fred = Fred(api_key=api_key)
        # DGS3MO = 3-Month Treasury Constant Maturity Rate (% annualized)
        series = fred.get_series("DGS3MO")
        latest = series.dropna().iloc[-1]
        rate = float(latest) / 100.0
        logger.info("Fetched risk-free rate from FRED: %.4f", rate)
        return rate
    except Exception:
        logger.warning(
            "Could not fetch risk-free rate from FRED; using default %.4f",
            DEFAULT_RISK_FREE_RATE,
        )
        return DEFAULT_RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Dividend yield
# ---------------------------------------------------------------------------
def estimate_dividend_yield(ticker: yf.Ticker, spot: float) -> float:
    """Trailing 12-month dividend yield as a continuous rate.

    dividend_yield = ln(1 + D_12m / S)  ≈ D_12m / S for small yields.
    """
    try:
        divs = ticker.dividends
        if divs.empty:
            logger.warning("No dividend data; using 0.0")
            return 0.0

        # Filter to last 12 months
        if divs.index.tz is not None:
            cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
        else:
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=1)

        trailing = divs.loc[divs.index >= cutoff]
        annual_div = float(trailing.sum())

        if annual_div <= 0 or spot <= 0:
            return 0.0

        q = np.log(1.0 + annual_div / spot)
        logger.info("Estimated continuous dividend yield: %.4f", q)
        return q
    except Exception:
        logger.warning("Dividend estimation failed; using 0.0")
        return 0.0


# ---------------------------------------------------------------------------
# Options chain fetching
# ---------------------------------------------------------------------------
def fetch_raw_chain(symbol: str = "SPY") -> tuple[yf.Ticker, pd.DataFrame, float]:
    """Download the full options chain for *symbol*.

    Returns
    -------
    ticker : yf.Ticker
    raw_chain : pd.DataFrame  (concatenation of all expiry chains)
    spot : float
    """
    ticker = yf.Ticker(symbol)
    spot = float(ticker.fast_info["lastPrice"])
    logger.info("Spot price for %s: %.2f", symbol, spot)

    expiries = ticker.options  # list of expiry date strings
    if not expiries:
        raise RuntimeError(f"No options expiries found for {symbol}")

    frames: list[pd.DataFrame] = []
    for exp_str in expiries:
        chain = ticker.option_chain(exp_str)
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["expiry"] = pd.to_datetime(exp_str)
            df["option_type"] = opt_type
            frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    logger.info("Fetched %d raw option rows across %d expiries", len(raw), len(expiries))
    return ticker, raw, spot


# ---------------------------------------------------------------------------
# Cleaning & filtering
# ---------------------------------------------------------------------------
def clean_chain(
    raw: pd.DataFrame,
    spot: float,
    risk_free: float,
    div_yield: float,
    *,
    min_dte_days: int = 3,
    max_log_moneyness: float = 0.5,
    wide_spread_pct: float = 0.20,
) -> pd.DataFrame:
    """Apply all Phase-1 filters and enrich the DataFrame.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw chain from ``fetch_raw_chain``.
    spot, risk_free, div_yield : float
        Market parameters.
    min_dte_days : int
        Exclude options expiring in fewer than this many days.
    max_log_moneyness : float
        Exclude options with |ln(K/S)| > this threshold.
    wide_spread_pct : float
        Flag rows where (ask - bid) / mid > this threshold.

    Returns
    -------
    pd.DataFrame
    """
    df = raw.copy()

    # --- Standardise column names ------------------------------------------
    rename_map = {
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "volume": "volume",
        "openInterest": "open_interest",
    }
    df = df.rename(columns=rename_map)

    # Ensure numeric types
    for col in ("bid", "ask", "strike", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open_interest" in df.columns:
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype(int)

    # --- Mid-price ---------------------------------------------------------
    df["mid_price"] = (df["bid"] + df["ask"]) / 2.0

    # --- Time to expiry (years) -------------------------------------------
    now = pd.Timestamp.now(tz=timezone.utc).normalize()
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True).dt.normalize()
    df["T"] = (df["expiry"] - now).dt.days / 365.25

    # --- Enrichment columns ------------------------------------------------
    df["S"] = spot
    df["r"] = risk_free
    df["q"] = div_yield

    # --- Filters -----------------------------------------------------------
    n_before = len(df)

    # Remove zero volume OR zero open interest
    df = df[(df["volume"] > 0) & (df["open_interest"] > 0)]

    # Remove near-expiry options
    df = df[df["T"] >= min_dte_days / 365.25]

    # Remove deep OTM (log-moneyness filter)
    df["log_moneyness"] = np.log(df["strike"] / spot)
    df = df[df["log_moneyness"].abs() <= max_log_moneyness]

    # Remove rows where mid_price <= 0 or NaN
    df = df[df["mid_price"] > 0]

    # Remove rows where bid >= ask (stale/crossed quotes)
    df = df[df["bid"] < df["ask"]]

    n_after = len(df)
    logger.info("Filtered from %d to %d rows", n_before, n_after)

    # --- Low-confidence flag -----------------------------------------------
    spread = df["ask"] - df["bid"]
    df["low_confidence"] = (spread / df["mid_price"]) > wide_spread_pct

    # --- Final column selection & sort -------------------------------------
    keep = [
        "expiry",
        "strike",
        "option_type",
        "mid_price",
        "bid",
        "ask",
        "volume",
        "open_interest",
        "S",
        "r",
        "q",
        "T",
        "low_confidence",
    ]
    df = df[keep].sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, path: Path = PARQUET_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), path)
    return path


def load_parquet(path: Path = PARQUET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No cached data at {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------
def load_options(
    symbol: str = "SPY",
    *,
    use_cache: bool = True,
    cache_path: Path = PARQUET_PATH,
) -> OptionsData:
    """End-to-end: fetch, clean, cache, and return ``OptionsData``.

    If *use_cache* is ``True`` and a Parquet file exists, it is loaded
    directly (fast reload during development).
    """
    if use_cache and cache_path.exists():
        logger.info("Loading cached options data")
        df = load_parquet(cache_path)
        spot = float(df["S"].iloc[0])
        risk_free = float(df["r"].iloc[0])
        div_yield = float(df["q"].iloc[0])
        return OptionsData(spot=spot, risk_free=risk_free, div_yield=div_yield, chains=df)

    ticker, raw, spot = fetch_raw_chain(symbol)
    risk_free = fetch_risk_free_rate()
    div_yield = estimate_dividend_yield(ticker, spot)

    chains = clean_chain(raw, spot, risk_free, div_yield)
    save_parquet(chains, cache_path)

    return OptionsData(spot=spot, risk_free=risk_free, div_yield=div_yield, chains=chains)
