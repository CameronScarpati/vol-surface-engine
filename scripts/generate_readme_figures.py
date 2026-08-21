"""Generate the static result figures embedded in the README.

Runs the full pipeline (clean chain -> IV extraction -> SVI calibration
-> arbitrage diagnostics) on the checked-in synthetic chain at
data/spy_options.parquet, entirely offline, and renders three PNGs into
docs/. The synthetic chain is deterministic (seeded generator), so the
figures are reproducible byte-for-byte in content.

Usage:
    pip install matplotlib  # dev extra: pip install -e ".[dev]"
    python scripts/generate_readme_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.arbitrage import durrleman_condition
from src.data_loader import load_options
from src.surface import build_surface
from src.svi_fitter import SVIParams, interpolate_surface, svi_total_variance

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Fixed categorical series colors (colorblind-validated order), plus
# neutral ink and surface tones shared by all three figures.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#ffffff"
GRID = "#e4e3e0"
# Single-hue blue ramp (light to dark) for the surface heatmap.
BLUES = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_SOFT)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _pick_expiries(slice_params, n=4):
    """Choose n expiries spread across the available maturity range."""
    df = slice_params.sort_values("T").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, num=min(n, len(df))).round().astype(int)
    return df.iloc[sorted(set(idx))]


def _params_from_row(row):
    return SVIParams(a=row["a"], b=row["b"], rho=row["rho"], m=row["m"], sigma=row["sigma"])


def fig_smile_fits(surface, chosen, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    chain = surface.chain.dropna(subset=["iv"])
    for color, (_, row) in zip(SERIES, chosen.iterrows(), strict=False):
        t = row["T"]
        f = surface.spot * np.exp((surface.risk_free - surface.div_yield) * t)
        slice_chain = chain[np.isclose(chain["T"], t, atol=1e-9)]
        label = f"{round(t * 365.25)} days"
        ax.scatter(
            slice_chain["strike"],
            100 * slice_chain["iv"],
            s=26,
            color=color,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
        k_grid = np.linspace(-0.22, 0.22, 200)
        w = svi_total_variance(k_grid, row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        iv_fit = 100 * np.sqrt(np.maximum(w, 0.0) / t)
        strikes = f * np.exp(k_grid)
        ax.plot(strikes, iv_fit, color=color, linewidth=2, zorder=4, label=label)

    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Per-expiry SVI fits against extracted implied volatilities", fontsize=11)
    ax.legend(
        title="Expiry",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        labelcolor=INK_SOFT,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def fig_surface_heatmap(surface, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.grid(False)

    t_min = surface.slice_params["T"].min()
    t_max = surface.slice_params["T"].max()
    k_grid = np.linspace(-0.25, 0.25, 121)
    t_grid = np.linspace(t_min, t_max, 121)
    iv = np.empty((len(t_grid), len(k_grid)))
    for i, t in enumerate(t_grid):
        w = interpolate_surface(k_grid, float(t), surface.slice_params)
        iv[i] = 100 * np.sqrt(np.maximum(w, 0.0) / t)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues_ramp", BLUES)
    # Cap the color range below the extreme short-expiry wings so the
    # smile and term structure stay readable; the colorbar arrow marks
    # the clip honestly.
    vmax = float(np.percentile(iv, 97))
    mesh = ax.pcolormesh(
        k_grid,
        t_grid * 365.25,
        iv,
        cmap=cmap,
        shading="gouraud",
        vmin=float(iv.min()),
        vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, extend="max")
    cbar.set_label("Implied volatility (%)", color=INK, fontsize=10)
    cbar.ax.tick_params(colors=INK_SOFT, labelsize=9)
    cbar.outline.set_visible(False)

    ax.set_xlabel("Log-moneyness k = ln(K / F)")
    ax.set_ylabel("Days to expiry")
    ax.set_title("Fitted implied volatility surface (variance-linear in maturity)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def fig_durrleman(chosen, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    k_grid = np.linspace(-0.5, 0.5, 500)
    for color, (_, row) in zip(SERIES, chosen.iterrows(), strict=False):
        g = durrleman_condition(k_grid, _params_from_row(row))
        ax.plot(
            k_grid,
            g,
            color=color,
            linewidth=2,
            label=f"{round(row['T'] * 365.25)} days",
        )
    ax.axhline(0.0, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        "g(k) below zero flags butterfly arbitrage",
        (0.02, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        color=INK_SOFT,
        va="top",
    )

    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Durrleman g(k)")
    ax.set_title("Butterfly diagnostic: Durrleman condition per fitted slice", fontsize=11)
    ax.legend(
        title="Expiry",
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        labelcolor=INK_SOFT,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    data = load_options(use_cache=True)
    surface = build_surface(data.chains, data.spot, data.risk_free, data.div_yield)
    chosen = _pick_expiries(surface.slice_params)

    fig_smile_fits(surface, chosen, DOCS_DIR / "smile_fits.png")
    fig_surface_heatmap(surface, DOCS_DIR / "surface_heatmap.png")
    fig_durrleman(chosen, DOCS_DIR / "durrleman_diagnostic.png")
    for name in ("smile_fits.png", "surface_heatmap.png", "durrleman_diagnostic.png"):
        print(f"wrote docs/{name}")


if __name__ == "__main__":
    main()
