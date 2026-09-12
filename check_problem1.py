"""
Verificación numérica del Problema 1 (geometría de la celda unitaria).

Edita únicamente el bloque PLACEHOLDERS de abajo: introduce los seis
parámetros de red, las coordenadas fraccionales y los valores que
obtuviste a mano. El resto del script construye G, calcula distancia,
producto punto y ángulo, y declara si coinciden con tu solución.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from crystallography_utils import (
    compute_metric_tensor,
    compute_distance,
    compute_dot_product,
    compute_angle,
)


# =============================================================================
# AYELEN: edita SOLO este bloque (lineas de abajo). No toques el resto.
# Pegá acá los datos del Problema 1 y los numeros que calculaste a mano.
# =============================================================================

# ----- 1) Parametros de red del Problema 1 (reemplaza estos ejemplos) -----
a = 5.0          # longitud de a  (Angstroms)
b = 6.0          # longitud de b  (Angstroms)
c = 7.0          # longitud de c  (Angstroms)
alpha = 90.0     # angulo entre b y c  (grados)
beta = 90.0      # angulo entre a y c  (grados)
gamma = 90.0     # angulo entre a y b  (grados)

# ----- 2) Dos puntos en coordenadas fraccionales (x, y, z) -----
p1 = np.array([0.0, 0.0, 0.0])
p2 = np.array([0.5, 0.5, 0.5])

# ----- 3) Dos vectores fraccionales (para producto punto y angulo) -----
# Si el problema pide el angulo entre dos enlaces, pone aca
# (punto_B - punto_A) y (punto_C - punto_A).
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.0, 1.0, 0.0])

# ----- 4) TUS RESULTADOS A MANO  <-- esto es lo que falta cargar -----
# Reemplaza cada None por el numero (o la matriz) que obtuviste en papel.
# Hasta que no lo hagas, el script va a decir INCOMPLETE.
#
# Ejemplo de como se ve una vez cargado:
#   G_hand = np.array([
#       [25.0,  0.0,  0.0],
#       [ 0.0, 36.0,  0.0],
#       [ 0.0,  0.0, 49.0],
#   ])
#   distance_hand = 5.244044
#   dot_product_hand = 0.0
#   angle_hand = 90.0

G_hand = None            # matriz 3x3 del tensor metrico que calculaste
distance_hand = None     # distancia entre p1 y p2  (Angstroms)
dot_product_hand = None  # producto punto v1^T G v2  (Angstroms^2)
angle_hand = None        # angulo entre v1 y v2  (grados)

# Si tus cifras a mano tienen 2 decimales, podes aflojar esto a 1e-2.
TOLERANCE = 1e-3


# =============================================================================
# Cálculo con el módulo (las cuatro funciones)
# =============================================================================

# G convierte productos fraccionales en invariantes cartesianos:
# cualquier vector u = x a + y b + z c cumple  |u|^2 = [x, y, z] G [x, y, z]^T.
G = compute_metric_tensor(a, b, c, alpha, beta, gamma)

# Distancia interatómica: d = sqrt( (p2 - p1)^T G (p2 - p1) ).
# No uses la norma euclidiana de (p2 - p1): eso solo vale si a = b = c y
# los ángulos son 90° (celda cúbica de arista 1).
distance = compute_distance(p1, p2, G)

# Producto punto real: v1 · v2 = v1^T G v2  (unidades de longitud^2).
dot_product = compute_dot_product(v1, v2, G)

# Angulo de enlace: cos(theta) = (v1 · v2) / (|v1| |v2|), con normas
# tambien evaluadas via G. El modulo lo devuelve en grados.
angle = compute_angle(v1, v2, G)


# =============================================================================
# Impresión
# =============================================================================

def _fmt_matrix(M):
    """Formatea una matriz 3×3 alineada, útil para leer G en terminal."""
    rows = []
    for row in M:
        rows.append("  [" + "  ".join(f"{val:12.6f}" for val in row) + " ]")
    return "\n".join(rows)


def _close(computed, hand, tol):
    """True si el valor a mano está definido y coincide dentro de ``tol``."""
    if hand is None:
        return None
    computed = np.asarray(computed, dtype=float)
    hand = np.asarray(hand, dtype=float)
    return bool(np.allclose(computed, hand, atol=tol, rtol=0.0))


print("=" * 64)
print("  Problema 1  |  verificacion de geometria cristalina")
print("=" * 64)
print()
print("Parametros de red")
print(f"  a = {a}    b = {b}    c = {c}")
print(f"  alpha = {alpha} deg    beta = {beta} deg    gamma = {gamma} deg")
print()
print("Puntos (coordenadas fraccionales)")
print(f"  p1 = {p1}")
print(f"  p2 = {p2}")
print()
print("Vectores (base de la celda)")
print(f"  v1 = {v1}")
print(f"  v2 = {v2}")
print()
print("-" * 64)
print("A) Tensor metrico G   (G_ij = a_i · a_j)")
print(_fmt_matrix(G))
print()
print(f"B) Distancia |p2 - p1|         = {distance:12.6f}  A")
print(f"C) Producto punto  v1^T G v2   = {dot_product:12.6f}  A^2")
print(f"D) Angulo (v1, v2)             = {angle:12.6f}  deg")
print("-" * 64)
print()

# Comparación ítem a ítem con los valores a mano.
comparisons = {
    "G": _close(G, G_hand, TOLERANCE),
    "distance": _close(distance, distance_hand, TOLERANCE),
    "dot_product": _close(dot_product, dot_product_hand, TOLERANCE),
    "angle": _close(angle, angle_hand, TOLERANCE),
}

print("Comparación con resultados a mano")
labels = {
    "G": "Tensor métrico G",
    "distance": "Distancia",
    "dot_product": "Producto punto",
    "angle": "Ángulo",
}
for key, label in labels.items():
    status = comparisons[key]
    if status is None:
        verdict = "—  (no ingresado)"
    elif status:
        verdict = "YES"
    else:
        verdict = "NO"
    print(f"  {label:22s}  match: {verdict}")

# Veredicto global: YES solo si todos los valores ingresados coinciden.
entered = [v for v in comparisons.values() if v is not None]
if not entered:
    overall = "INCOMPLETE  (ingresa tus resultados a mano en los placeholders)"
elif all(entered):
    overall = "YES"
else:
    overall = "NO"

print()
print(f"Match with Problem 1: {overall}")
print("=" * 64)
