"""
MSE 710 — Problem 5
Verification script for the rutile TiO₂ calculations of Problem 4.

The assignment PDF says “check problem 1”; that is a typo.  Problem 5
asks for a crystallographic utility module and a script that uses it to
confirm the hand calculations of Problem 4 (metric tensor, Ti–O bond
lengths and non-180° O–Ti–O angles of rutile).

Lattice parameters are the conventional tetragonal cell of rutile
(P4₂/mnm).  The oxygen 4f Wyckoff parameter u = 0.3053 is the value
paired with a = 4.5937 Å, c = 2.9581 Å in CrystalMaker / ICSD-type
entries.  Expected values below are closed-form expressions — they do
not call the module — so agreement is an independent check.
"""

from __future__ import annotations

import numpy as np

from crystallography_utils import (
    compute_angle,
    compute_distance,
    compute_dot_product,
    compute_metric_tensor,
)

# ---------------------------------------------------------------------------
# Rutile TiO₂ — conventional tetragonal cell (tI / P4₂/mnm)
# ---------------------------------------------------------------------------
A = B = 4.5937  # Å
C = 2.9581  # Å
ALPHA = BETA = GAMMA = 90.0  # deg
U = 0.3053  # oxygen 4f parameter

# Ti at the origin; the six nearest O complete a distorted octahedron:
#   2 apical  O at ±(u, u, 0)                 — longer Ti–O
#   4 equatorial O at the (½ ± u, ½ ∓ u, ±½) images that wrap nearest
TI = np.array([0.0, 0.0, 0.0])
O_APICAL = {
    "O_ap1": np.array([U, U, 0.0]),
    "O_ap2": np.array([-U, -U, 0.0]),
}
O_EQUATORIAL = {
    "O_eq1": np.array([U - 0.5, 0.5 - U, 0.5]),
    "O_eq2": np.array([0.5 - U, U - 0.5, 0.5]),
    "O_eq3": np.array([U - 0.5, 0.5 - U, -0.5]),
    "O_eq4": np.array([0.5 - U, U - 0.5, -0.5]),
}
OXYGENS = {**O_APICAL, **O_EQUATORIAL}

# Three crystallographically distinct non-180° O–Ti–O angles.
# The remaining pairs are either 180° (trans) or symmetry-equivalent.
ANGLE_PAIRS = (
    ("O_ap1–Ti–O_eq1  (apical–equatorial)", "O_ap1", "O_eq1"),
    ("O_eq1–Ti–O_eq2  (equatorial cis)", "O_eq1", "O_eq2"),
    ("O_eq1–Ti–O_eq3  (equatorial same-xy)", "O_eq1", "O_eq3"),
)

TOL_LENGTH = 1.0e-8
TOL_ANGLE = 1.0e-6
TOL_TENSOR = 1.0e-10


def _closed_form_references() -> dict:
    """Independent analytic values (no calls into crystallography_utils)."""
    g11 = A * A
    g22 = B * B
    g33 = C * C
    G = np.array(
        [
            [g11, 0.0, 0.0],
            [0.0, g22, 0.0],
            [0.0, 0.0, g33],
        ]
    )
    d_apical = A * np.sqrt(2.0) * U
    d_equatorial = np.sqrt(2.0 * A * A * (0.5 - U) ** 2 + (C / 2.0) ** 2)

    v_ap = np.array([U, U, 0.0])
    v_eq_cis = np.array([U - 0.5, 0.5 - U, 0.5])
    v_eq_adj = np.array([0.5 - U, U - 0.5, 0.5])
    v_eq_z = np.array([U - 0.5, 0.5 - U, -0.5])

    def _ang(u, v):
        dot = float(u @ G @ v)
        mu = float(np.sqrt(u @ G @ u))
        mv = float(np.sqrt(v @ G @ v))
        return float(np.rad2deg(np.arccos(np.clip(dot / (mu * mv), -1.0, 1.0))))

    return {
        "G": G,
        "d_apical": d_apical,
        "d_equatorial": d_equatorial,
        "angle_ap_eq": _ang(v_ap, v_eq_cis),
        "angle_eq_cis": _ang(v_eq_cis, v_eq_adj),
        "angle_eq_same_xy": _ang(v_eq_cis, v_eq_z),
    }


def _fmt_matrix(matrix: np.ndarray) -> str:
    lines = []
    for row in matrix:
        lines.append("    [" + "  ".join(f"{val:12.6f}" for val in row) + " ]")
    return "\n".join(lines)


def _fmt_frac(vec: np.ndarray) -> str:
    return "(" + ", ".join(f"{x:8.4f}" for x in vec) + ")"


def main() -> None:
    G = compute_metric_tensor(A, B, C, ALPHA, BETA, GAMMA)
    ref = _closed_form_references()

    bonds = {name: compute_distance(TI, xyz, G) for name, xyz in OXYGENS.items()}
    d_apical = bonds["O_ap1"]
    d_equatorial = bonds["O_eq1"]

    angles = []
    for label, name_i, name_j in ANGLE_PAIRS:
        vi = OXYGENS[name_i] - TI
        vj = OXYGENS[name_j] - TI
        theta = compute_angle(vi, vj, G)
        dot = compute_dot_product(vi, vj, G)
        angles.append((label, theta, dot))

    matches = [
        np.allclose(G, ref["G"], atol=TOL_TENSOR),
        abs(d_apical - ref["d_apical"]) < TOL_LENGTH,
        abs(d_equatorial - ref["d_equatorial"]) < TOL_LENGTH,
        abs(bonds["O_ap2"] - ref["d_apical"]) < TOL_LENGTH,
        all(abs(bonds[n] - ref["d_equatorial"]) < TOL_LENGTH for n in O_EQUATORIAL),
        abs(angles[0][1] - ref["angle_ap_eq"]) < TOL_ANGLE,
        abs(angles[1][1] - ref["angle_eq_cis"]) < TOL_ANGLE,
        abs(angles[2][1] - ref["angle_eq_same_xy"]) < TOL_ANGLE,
    ]
    verdict = "YES" if all(matches) else "NO"

    width = 72
    print("=" * width)
    print("  MSE 710  |  Problem 5  —  Verification of Problem 4")
    print("  Rutile TiO2   (P4_2/mnm, conventional tetragonal cell)")
    print("=" * width)
    print()
    print("Lattice parameters")
    print(f"  a = b = {A:.4f} Å")
    print(f"  c     = {C:.4f} Å")
    print(f"  α = β = γ = {ALPHA:.1f}°")
    print(f"  Oxygen Wyckoff 4f parameter  u = {U}")
    print()
    print("Fractional coordinates  (Ti at the origin)")
    print(f"  Ti      {_fmt_frac(TI)}")
    for name, xyz in OXYGENS.items():
        print(f"  {name:7s} {_fmt_frac(xyz)}")
    print()
    print("-" * width)
    print("(a)  Metric tensor G  (Å²)")
    print("     G_ij = a_i · a_j   →   G = diag(a², b², c²)  for 90° angles")
    print(_fmt_matrix(G))
    print()
    print("(b)  Ti–O bond distances")
    print(f"     {'Site':<10}{'Fractional vector':<28}{'d(Ti–O) / Å':>14}")
    for name, xyz in OXYGENS.items():
        print(f"     {name:<10}{_fmt_frac(xyz):<28}{bonds[name]:14.6f}")
    print()
    print(f"     Distinct lengths:  {d_equatorial:.4f} Å  (4× equatorial)")
    print(f"                        {d_apical:.4f} Å  (2× apical)")
    print()
    print("(c)  Non-180° O–Ti–O bond angles")
    print("     (trans pairs O_ap1–Ti–O_ap2 and O_eq1–Ti–O_eq4 are 180° and omitted)")
    print(f"     {'Angle':<42}{'θ / °':>8}  {'vᵢᵀ G vⱼ / Å²':>14}")
    for label, theta, dot in angles:
        print(f"     {label:<42}{theta:8.3f}  {dot:14.6f}")
    print("-" * width)
    print()
    print("Comparison with closed-form rutile geometry")
    print(f"  metric tensor G           match = {matches[0]}")
    print(f"  d_equatorial              {d_equatorial:.8f}  vs  {ref['d_equatorial']:.8f} Å")
    print(f"  d_apical                  {d_apical:.8f}  vs  {ref['d_apical']:.8f} Å")
    print(f"  ∠ apical–equatorial       {angles[0][1]:.6f}  vs  {ref['angle_ap_eq']:.6f}°")
    print(f"  ∠ equatorial cis          {angles[1][1]:.6f}  vs  {ref['angle_eq_cis']:.6f}°")
    print(f"  ∠ equatorial same-xy      {angles[2][1]:.6f}  vs  {ref['angle_eq_same_xy']:.6f}°")
    print()
    print(
        "Calculations match the expected theoretical values for Rutile TiO2: "
        f"{verdict}"
    )
    print("=" * width)


if __name__ == "__main__":
    main()
