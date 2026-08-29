"""
Utilidades de geometría cristalina en coordenadas fraccionales.

En una celda unitaria triclínica (la más general), los vectores de red a, b y c
no son ortogonales ni de la misma longitud. El producto punto cartesiano
habitual, u·v = ux*vx + uy*vy + uz*vz, deja de ser válido si u y v se
expresan en fracciones de celda (x, y, z). Toda la geometría —distancias,
ángulos, volúmenes— se recupera insertando el tensor métrico G entre los
vectores fraccionales:

    u · v  =  uᵀ G v

Este módulo implementa esa álgebra para que un cálculo a mano (Problema 1)
pueda verificarse con NumPy.
"""

import numpy as np


def compute_metric_tensor(a, b, c, alpha, beta, gamma):
    """
    Construye el tensor métrico G a partir de los seis parámetros de red.

    Parámetros
    ----------
    a, b, c : float
        Longitudes de los vectores de red (misma unidad, p. ej. Å).
    alpha, beta, gamma : float
        Ángulos de celda en grados.
        Convención IUCr:
            α = ∠(b, c)
            β = ∠(a, c)
            γ = ∠(a, b)

    Retorna
    -------
    G : ndarray, shape (3, 3)
        Tensor métrico simétrico. G_ij = a_i · a_j, de modo que
        el cuadrado de una distancia fraccional Δx es Δxᵀ G Δx.

    Fundamento
    ----------
    Si a, b, c son los vectores de red en un marco cartesiano,

        G = [[ a·a,  a·b,  a·c ],
             [ b·a,  b·b,  b·c ],
             [ c·a,  c·b,  c·c ]]

    y los productos punto se escriben con los cosenos de los ángulos de celda:

        a·a = a²
        b·b = b²
        c·c = c²
        a·b = ab cos γ
        a·c = ac cos β
        b·c = bc cos α

    G es simétrico y definido positivo (celdas físicas). En redes ortorrómbicas
    o cúbicas los términos fuera de la diagonal se anulan (ángulos = 90°) y G
    se reduce a diag(a², b², c²).
    """
    # Los ángulos de celda se publican en grados; NumPy trabaja en radianes.
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    # Elementos de G = Aᵀ A, donde A es la matriz que apila a, b, c como filas.
    # No hace falta construir A: los productos punto bastan.
    g11 = a * a
    g22 = b * b
    g33 = c * c
    g12 = a * b * np.cos(gamma_rad)  # a · b
    g13 = a * c * np.cos(beta_rad)   # a · c
    g23 = b * c * np.cos(alpha_rad)  # b · c

    G = np.array([
        [g11, g12, g13],
        [g12, g22, g23],
        [g13, g23, g33],
    ], dtype=float)

    return G


def compute_distance(p1, p2, metric_tensor):
    """
    Distancia cartesiana real entre dos sitios en coordenadas fraccionales.

    Parámetros
    ----------
    p1, p2 : array-like, shape (3,)
        Coordenadas fraccionales (x, y, z) de cada punto. Un valor de 1.0
        significa un traslado completo a lo largo del vector de red
        correspondiente.
    metric_tensor : ndarray, shape (3, 3)
        Tensor G de la celda (salida de ``compute_metric_tensor``).

    Retorna
    -------
    distance : float
        Distancia interatómica en las mismas unidades que a, b, c.

    Fundamento
    ----------
    El vector diferencia en fracciones de celda es Δx = p2 − p1.
    Su longitud cartesiana no es ||Δx||₂, porque los ejes de la celda no
    son ortonormales. El invariante correcto es la forma cuadrática

        d² = Δxᵀ G Δx
        d  = √(Δxᵀ G Δx)

    En notación de índices: d² = Σᵢ Σⱼ Δxᵢ Gᵢⱼ Δxⱼ.
    El operador ``@`` realiza exactamente esos productos matriciales.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    # Vector de enlace (o de separación) en el sistema de la celda.
    delta = p2 - p1

    # Forma cuadrática: (Δxᵀ G) Δx  →  escalar.
    d_squared = delta @ metric_tensor @ delta
    return float(np.sqrt(d_squared))


def compute_dot_product(v1, v2, metric_tensor):
    """
    Producto punto cartesiano de dos vectores dados en coordenadas fraccionales.

    Parámetros
    ----------
    v1, v2 : array-like, shape (3,)
        Componentes de cada vector en la base {a, b, c}.
        Ejemplo: el vector de red a es (1, 0, 0); un enlace entre
        (0,0,0) y (x,y,z) es (x, y, z).
    metric_tensor : ndarray, shape (3, 3)
        Tensor G de la celda.

    Retorna
    -------
    dot : float
        v1 · v2 en unidades de longitud² (p. ej. Å²).

    Fundamento
    ----------
    En una base no ortonormal, el producto punto no es v1ᵀ v2 sino

        v1 · v2  =  v1ᵀ G v2

    G actúa como el "núcleo" que inyecta los ángulos y las longitudes de
    la celda. Si G = I (celda cúbica de arista 1), se recupera el producto
    euclidiano habitual.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    # v1ᵀ G v2  — el mismo patrón que en la distancia, sin la raíz.
    return float(v1 @ metric_tensor @ v2)


def compute_angle(v1, v2, metric_tensor):
    """
    Ángulo entre dos vectores fraccionales, en grados.

    Parámetros
    ----------
    v1, v2 : array-like, shape (3,)
        Vectores en la base de la celda (p. ej. dos enlaces que parten
        del mismo átomo).
    metric_tensor : ndarray, shape (3, 3)
        Tensor G de la celda.

    Retorna
    -------
    angle_deg : float
        Ángulo ∠(v1, v2) en grados, en [0, 180].

    Fundamento
    ----------
    La definición geométrica

        cos θ  =  (v1 · v2) / (|v1| |v2|)

    sigue siendo válida, pero tanto el numerador como los módulos deben
    evaluarse con G:

        v1 · v2  =  v1ᵀ G v2
        |v1|     =  √(v1ᵀ G v1)
        |v2|     =  √(v2ᵀ G v2)

    El recorte de cos θ al intervalo [−1, 1] evita errores de dominio en
    arccos por redondeo numérico cuando los vectores son casi colineales.
    """
    # Reutilizamos el producto punto métrico para numerador y normas.
    dot = compute_dot_product(v1, v2, metric_tensor)
    norm1 = np.sqrt(compute_dot_product(v1, v1, metric_tensor))
    norm2 = np.sqrt(compute_dot_product(v2, v2, metric_tensor))

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError(
            "No se puede definir un ángulo si alguno de los vectores es nulo."
        )

    cosine = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))
