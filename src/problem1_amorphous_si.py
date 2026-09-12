"""
MSE 710 — Problem 1
Radial distribution function, pair correlation function and
nearest-neighbour coordination number of amorphous silicon.

The configuration is a cube of side L = 30 Å (3 nm) with periodic
boundaries.  Pair distances are evaluated with the minimum-image
convention so that atoms near opposite faces are treated as neighbours
across the periodic wrap, not as a free cluster.

Definitions used here
---------------------
Number density
    ρ = N / V

Pair correlation function (normalised by spherical-shell volume)
    g(r) = n(r) / (N ρ ΔV(r))
    ΔV(r) = (4π/3) [(r + dr)³ − r³]

Radial distribution function (integrable neighbour density)
    RDF(r) = 4π r² ρ g(r)

Coordination number of the first shell
    CN = ∫₀^{r_min} RDF(r) dr
    with r_min = first local minimum of RDF after the first peak.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simpson

# ---------------------------------------------------------------------------
# Simulation / analysis parameters
# ---------------------------------------------------------------------------
BOX_LENGTH_A = 30.0  # Å  (3 nm cube specified in the assignment)
BIN_WIDTH_A = 0.05  # Å
PLOT_R_MAX_A = 10.0  # Å  (display window; analysis uses L/2)
DATA_PATH = Path(__file__).resolve().parent / "../data/amorphous.xlsx"
FIGURE_STEM = Path(__file__).resolve().parent / "../results/problem1_amorphous_si"


def load_positions(path: Path) -> np.ndarray:
    """Load Cartesian (x, y, z) coordinates from a header-less Excel file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Structure file not found: {path}\n"
            "Place amorphous.xlsx (1000 × 3, no header, Å) in ../data/."
        )
    positions = pd.read_excel(path, header=None).to_numpy(dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected an (N, 3) coordinate table, got shape {positions.shape}."
        )
    return positions


def pairwise_distances_pbc(positions: np.ndarray, box_length: float) -> np.ndarray:
    """
    Unique pair distances under the minimum-image convention.

    Broadcasting builds the (N, N, 3) separation tensor; wrapping
        Δ = Δ − L round(Δ / L)
    folds every component into (−L/2, L/2].  ``np.triu_indices`` then
    keeps only i < j so each physical pair is counted once.
    """
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    diff = diff - box_length * np.round(diff / box_length)
    dist = np.linalg.norm(diff, axis=-1)
    i_upper, j_upper = np.triu_indices(positions.shape[0], k=1)
    return dist[i_upper, j_upper]


def compute_gr_and_rdf(
    distances: np.ndarray,
    n_atoms: int,
    density: float,
    dr: float,
    r_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Histogram pair distances and convert counts into g(r) and RDF(r).

    Unique pairs (i < j) are doubled so that n(r) is the number of
    directed neighbours, consistent with the standard normalisation
    g(r) → 1 for an uncorrelated gas of density ρ.
    """
    bins = np.arange(0.0, r_max + dr, dr)
    r_centers = 0.5 * (bins[1:] + bins[:-1])
    hist, _ = np.histogram(distances, bins=bins)
    hist = hist.astype(float) * 2.0

    shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = hist / (n_atoms * density * shell_volumes)
    g_r = np.nan_to_num(g_r, nan=0.0, posinf=0.0, neginf=0.0)

    rdf_r = 4.0 * np.pi * (r_centers**2) * density * g_r
    return r_centers, g_r, rdf_r


def coordination_number(
    r: np.ndarray,
    rdf: np.ndarray,
    peak_search_max: float = 3.0,
    second_peak_window: tuple[float, float] = (3.0, 4.5),
) -> tuple[float, int, int]:
    """
    Integrate RDF(r) from the origin to the first coordination-shell minimum.

        CN = ∫_0^{r_min} RDF(r) dr

    ``r_min`` is the local minimum of RDF between the first and second
    peaks — the physical boundary of the nearest-neighbour shell — not
    a one-bin histogram flicker on the descending flank of the peak.
    """
    peak_window = r < peak_search_max
    if not np.any(peak_window):
        raise RuntimeError("No RDF samples in the first-peak search window.")
    peak_idx = int(np.argmax(rdf[peak_window]))

    lo, hi = second_peak_window
    second_window = (r > lo) & (r < hi)
    if not np.any(second_window):
        raise RuntimeError("No RDF samples in the second-peak search window.")
    second_peak_idx = int(np.where(second_window)[0][np.argmax(rdf[second_window])])
    if second_peak_idx <= peak_idx:
        raise RuntimeError("Second RDF peak was not found beyond the first peak.")

    min_idx = peak_idx + int(np.argmin(rdf[peak_idx : second_peak_idx + 1]))
    cn = float(simpson(rdf[: min_idx + 1], x=r[: min_idx + 1]))
    return cn, peak_idx, min_idx


def plot_rdf_and_gr(
    r: np.ndarray,
    rdf: np.ndarray,
    g_r: np.ndarray,
    peak_idx: int,
    min_idx: int,
    cn: float,
    dest: Path,
) -> Path:
    """Publication-style two-panel figure: RDF(r) and g(r)."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 1.0,
        }
    )

    r_min = r[min_idx]
    r_peak = r[peak_idx]
    display = r <= PLOT_R_MAX_A
    rdf_max = float(np.max(rdf[display])) if np.any(display) else float(np.max(rdf))
    g_max = float(np.max(g_r[display])) if np.any(display) else float(np.max(g_r))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    ax_rdf, ax_gr = axes
    navy, fill, teal, accent = "#1f4e79", "#5b9bd5", "#0d7377", "#c0392b"

    ax_rdf.plot(r, rdf, color=navy, lw=1.8, label=r"RDF$(r)$")
    ax_rdf.fill_between(
        r[: min_idx + 1],
        rdf[: min_idx + 1],
        color=fill,
        alpha=0.45,
        label="1st coordination shell",
    )
    ax_rdf.axvline(
        r_min,
        color=accent,
        ls="--",
        lw=1.2,
        label=fr"$r_{{\mathrm{{min}}}}={r_min:.2f}\,\mathrm{{\AA}}$",
    )
    ax_rdf.plot(r_peak, rdf[peak_idx], "o", color=navy, ms=6, zorder=5)
    ax_rdf.annotate(
        fr"CN $= {cn:.2f}$",
        xy=(r_peak, rdf[peak_idx]),
        xytext=(r_peak + 1.1, rdf[peak_idx] * 0.82),
        arrowprops=dict(arrowstyle="->", color=navy, lw=1.0),
        fontsize=10,
    )
    ax_rdf.set_xlim(0.0, PLOT_R_MAX_A)
    ax_rdf.set_ylim(0.0, rdf_max * 1.12)
    ax_rdf.set_xlabel(r"Distance $r$ ($\mathrm{\AA}$)")
    ax_rdf.set_ylabel(r"RDF$(r)$  ($\mathrm{\AA}^{-1}$)")
    ax_rdf.set_title("Radial distribution function")
    ax_rdf.legend(frameon=False, loc="upper right")
    ax_rdf.set_axisbelow(True)
    ax_rdf.grid(True, ls=":", lw=0.6, alpha=0.7)

    ax_gr.plot(r, g_r, color=teal, lw=1.8, label=r"$g(r)$")
    ax_gr.axhline(1.0, color="#d68910", ls="--", lw=1.1, label=r"$g(r)=1$ (ideal gas)")
    ax_gr.axvline(r_min, color=accent, ls="--", lw=1.2)
    ax_gr.set_xlim(0.0, PLOT_R_MAX_A)
    ax_gr.set_ylim(0.0, g_max * 1.12)
    ax_gr.set_xlabel(r"Distance $r$ ($\mathrm{\AA}$)")
    ax_gr.set_ylabel(r"$g(r)$")
    ax_gr.set_title("Pair correlation function")
    ax_gr.legend(frameon=False, loc="upper right")
    ax_gr.set_axisbelow(True)
    ax_gr.grid(True, ls=":", lw=0.6, alpha=0.7)

    fig.suptitle(
        r"Amorphous Si  |  $L = 30\,\mathrm{\AA}$  |  minimum-image PBC",
        fontsize=13,
        y=1.03,
    )

    png_path = dest.with_suffix(".png")
    pdf_path = dest.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    data_path = DATA_PATH.resolve()
    figure_stem = FIGURE_STEM.resolve()
    figure_stem.parent.mkdir(parents=True, exist_ok=True)

    positions = load_positions(data_path)
    n_atoms = positions.shape[0]
    volume = BOX_LENGTH_A**3
    density = n_atoms / volume
    r_max = BOX_LENGTH_A / 2.0

    distances = pairwise_distances_pbc(positions, BOX_LENGTH_A)
    r, g_r, rdf = compute_gr_and_rdf(
        distances, n_atoms, density, BIN_WIDTH_A, r_max
    )
    cn, peak_idx, min_idx = coordination_number(r, rdf)

    figure_path = plot_rdf_and_gr(
        r, rdf, g_r, peak_idx, min_idx, cn, figure_stem
    )

    width = 64
    print("=" * width)
    print("  MSE 710  |  Problem 1  —  Amorphous silicon")
    print("=" * width)
    print()
    print("Configuration")
    print(f"  Atoms N                 : {n_atoms}")
    print(f"  Box length L            : {BOX_LENGTH_A:.1f} Å")
    print(f"  Volume V                : {volume:.1f} Å³")
    print(f"  Number density ρ        : {density:.6f} atoms/Å³")
    print(f"  Histogram width dr      : {BIN_WIDTH_A:.3f} Å")
    print(f"  Analysis cutoff (L/2)   : {r_max:.1f} Å")
    print(f"  Unique pairs (i < j)    : {distances.size}")
    print()
    print("First coordination shell")
    print(f"  First RDF peak          : {r[peak_idx]:.3f} Å")
    print(f"  First local minimum     : {r[min_idx]:.3f} Å")
    print(f"  g(r) at first peak      : {g_r[peak_idx]:.3f}")
    print(f"  g(r) at large r (PBC)   : {g_r[r > 12.0].mean():.3f}  (should approach 1)")
    print(f"  Coordination number CN  : {cn:.4f}")
    print()
    print("Notes")
    print("  Crystalline Si is tetrahedral (CN = 4, d_Si–Si ≈ 2.35 Å).")
    print("  a-Si typically yields CN ≈ 3.5–4.0 because of 3- and 5-fold defects.")
    print()
    print(f"Figure written to {figure_path.name} and {figure_path.with_suffix('.pdf').name}")
    print("=" * width)


if __name__ == "__main__":
    main()
