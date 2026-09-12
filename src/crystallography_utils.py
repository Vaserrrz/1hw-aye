"""
Crystallographic geometry in fractional coordinates.

In a general triclinic cell the lattice vectors a, b, c are neither
orthogonal nor of equal length.  Cartesian identities such as
u · v = ux vx + uy vy + uz vz are therefore invalid when u and v are
written as fractions of the cell.  Distances, angles and volumes are
recovered by inserting the metric tensor G:

    u · v  =  uᵀ G v

This module implements that algebra so that the hand calculations of
MSE 710 Problem 4 (rutile TiO2) can be checked with NumPy.
"""

from __future__ import annotations

import numpy as np


def compute_metric_tensor(a, b, c, alpha, beta, gamma):
    """
    Build the metric tensor G from the six lattice parameters.

    Parameters
    ----------
    a, b, c : float
        Lattice-vector lengths (same length unit, e.g. Å).
    alpha, beta, gamma : float
        Cell angles in degrees.  IUCr convention:
            α = ∠(b, c),  β = ∠(a, c),  γ = ∠(a, b).

    Returns
    -------
    G : ndarray, shape (3, 3)
        Symmetric metric tensor.  G_ij = a_i · a_j, so a fractional
        separation Δx has squared length Δxᵀ G Δx.

    Notes
    -----
    With a, b, c the real-space lattice vectors,

        G = [[ a·a,  a·b,  a·c ],
             [ b·a,  b·b,  b·c ],
             [ c·a,  c·b,  c·c ]]

    and the inner products are

        a·a = a²,          a·b = ab cos γ,
        b·b = b²,          a·c = ac cos β,
        c·c = c²,          b·c = bc cos α.

    For orthogonal cells (α = β = γ = 90°) the off-diagonal terms
    vanish and G reduces to diag(a², b², c²).
    """
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    g11 = a * a
    g22 = b * b
    g33 = c * c
    g12 = a * b * np.cos(gamma_rad)
    g13 = a * c * np.cos(beta_rad)
    g23 = b * c * np.cos(alpha_rad)

    return np.array(
        [
            [g11, g12, g13],
            [g12, g22, g23],
            [g13, g23, g33],
        ],
        dtype=float,
    )


def compute_distance(p1, p2, metric_tensor):
    """
    Cartesian distance between two sites given in fractional coordinates.

    Parameters
    ----------
    p1, p2 : array-like, shape (3,)
        Fractional coordinates (x, y, z).  A component of 1.0 is one
        full translation along the corresponding lattice vector.
    metric_tensor : ndarray, shape (3, 3)
        Cell metric G from ``compute_metric_tensor``.

    Returns
    -------
    distance : float
        Interatomic distance in the same units as a, b, c.

    Notes
    -----
    With Δx = p2 − p1 the invariant length is

        d = √( Δxᵀ G Δx )

    The Euclidean norm ||Δx||₂ is correct only for a cubic cell of
    edge 1.  The ``@`` operator evaluates the quadratic form.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    delta = p2 - p1
    return float(np.sqrt(delta @ metric_tensor @ delta))


def compute_dot_product(v1, v2, metric_tensor):
    """
    Cartesian dot product of two vectors given in fractional coordinates.

    Parameters
    ----------
    v1, v2 : array-like, shape (3,)
        Vector components in the {a, b, c} basis (directions, not
        points).  Example: lattice vector a is (1, 0, 0); the bond
        from (0, 0, 0) to (x, y, z) is (x, y, z).
    metric_tensor : ndarray, shape (3, 3)
        Cell metric G.

    Returns
    -------
    dot : float
        v1 · v2 in units of length² (e.g. Å²).

    Notes
    -----
    In a non-orthonormal basis

        v1 · v2  =  v1ᵀ G v2

    If G = I (cubic cell of edge 1) this reduces to the ordinary
    Euclidean product.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    return float(v1 @ metric_tensor @ v2)


def compute_angle(v1, v2, metric_tensor):
    """
    Angle between two fractional vectors, in degrees.

    Parameters
    ----------
    v1, v2 : array-like, shape (3,)
        Vectors in the cell basis (e.g. two bonds that share an atom).
    metric_tensor : ndarray, shape (3, 3)
        Cell metric G.

    Returns
    -------
    angle_deg : float
        ∠(v1, v2) in degrees, in [0, 180].

    Raises
    ------
    ValueError
        If either vector has zero metric length.

    Notes
    -----
    The geometric definition

        cos θ  =  (v1 · v2) / (|v1| |v2|)

    remains valid provided both the numerator and the magnitudes are
    evaluated with G:

        v1 · v2  =  v1ᵀ G v2
        |v1|     =  √(v1ᵀ G v1)
        |v2|     =  √(v2ᵀ G v2)

    The cosine is clipped to [−1, 1] so that rounding of nearly
    collinear vectors cannot send ``arccos`` out of domain.
    """
    dot = compute_dot_product(v1, v2, metric_tensor)
    norm1 = np.sqrt(compute_dot_product(v1, v1, metric_tensor))
    norm2 = np.sqrt(compute_dot_product(v2, v2, metric_tensor))

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Cannot define an angle when either vector is null.")

    cosine = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))
