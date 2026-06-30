"""Volatility Surface Engine.

Public API:
    - ``VolSurface`` / ``build_surface`` -- full pipeline entry point
    - ``OptionsData`` / ``load_options`` -- data fetching and cleaning
    - ``ArbitrageDiagnostics`` -- arbitrage diagnostic results
"""

from src.arbitrage import ArbitrageDiagnostics
from src.surface import VolSurface, build_surface


def __getattr__(name: str):
    """Lazy import for data_loader to avoid requiring yfinance at import time."""
    if name == "OptionsData":
        from src.data_loader import OptionsData
        return OptionsData
    if name == "load_options":
        from src.data_loader import load_options
        return load_options
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArbitrageDiagnostics",
    "OptionsData",
    "VolSurface",
    "build_surface",
    "load_options",
]
