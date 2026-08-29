# De la Física al Código: RDF, g(r) y Número de Coordinación en NumPy

Este documento traduce, paso a paso, las ecuaciones de la función de correlación de pares y la función de distribución radial en las operaciones matriciales implementadas en `app.py`. El objetivo es conectar el **razonamiento espacial** que ya dominas (celdas unitarias, vecinos, capas de coordinación) con la **lógica de arrays** de Python.

---

## Mapa general: del espacio físico al array

En simulación atómica, una configuración de \(N\) átomos en 3D se representa como una **matriz de coordenadas**:

\[
\text{positions} \in \mathbb{R}^{N \times 3}
\]

Cada fila es un átomo; cada columna es una componente cartesiana. Piensa en ello como una **tabla de sitios cristalográficos** donde, en lugar de índices de Miller, guardas coordenadas absolutas en Å.

```python
df = pd.read_excel('amorphous.xlsx', header=None)
positions = df.values  # Shape: (1000, 3)
```

| Concepto cristalográfico | Objeto NumPy | Shape |
|--------------------------|--------------|-------|
| 1000 átomos de Si | `positions` | `(1000, 3)` |
| Coordenada \(x_i\) del átomo \(i\) | `positions[i, 0]` | escalar |
| Todas las coordenadas \(x\) | `positions[:, 0]` | `(1000,)` |
| Posición vectorial del átomo \(i\) | `positions[i, :]` | `(3,)` |

La caja de simulación es un **paralelepípedo cúbico** de arista \(L = 30\) Å:

```python
L = 30.0
N = len(positions)
V = L**3
rho = N / V
```

Aquí \(\rho = N/V\) es la **densidad numérica** (átomos/Å³), análoga a la densidad de sitios en una red, pero continua porque el a-Si no tiene periodicidad translacional de largo alcance.

---

## Paso 1: Distancias entre todos los pares — Broadcasting

### Ecuación física

Para cada par de átomos \((i, j)\), necesitamos el vector separación:

\[
\Delta \vec{r}_{ij} = \vec{r}_j - \vec{r}_i
\]

y su magnitud \(r_{ij} = |\Delta \vec{r}_{ij}|\).

### Traducción a código

En lugar de dos bucles `for i ... for j ...`, usamos **broadcasting**: NumPy expande dimensiones para operar sobre todas las parejas simultáneamente.

```python
diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
```

**Analogía:** Imagina que colocas una copia de la estructura "desplazada" respecto a sí misma en cada dirección. `positions[:, np.newaxis, :]` tiene shape `(N, 1, 3)` — cada átomo \(i\) como "origen". `positions[np.newaxis, :, :]` tiene shape `(1, N, 3)` — todos los átomos \(j\) como "destino". La resta produce un **tensor de diferencias** `(N, N, 3)` donde `diff[i, j, :]` es el vector \(\vec{r}_j - \vec{r}_i\).

```
         j=0    j=1    j=2   ...  j=N-1
i=0   [ Δr₀₀   Δr₀₁   Δr₀₂  ...  Δr₀,ₙ₋₁ ]
i=1   [ Δr₁₀   Δr₁₁   Δr₁₂  ...  Δr₁,ₙ₋₁ ]
 ...
i=N-1 [ ...                              ]
```

La distancia euclidiana es la norma del vector:

```python
dist = np.sqrt(np.sum(diff**2, axis=-1))  # Shape: (N, N)
```

`dist[i, j]` es la distancia entre los átomos \(i\) y \(j\) **antes** de aplicar PBC.

---

## Paso 2: Condiciones de Contorno Periódicas (PBC) — Convención de Imagen Mínima

### Ecuación física

En una simulación bulk con PBC, la caja de \(L × L × L\) Å se **replica infinitamente** en el espacio (como un mosaico de celdas unitarias idénticas). Un átomo cerca de una cara "ve" copias de los átomos del lado opuesto. El vector separación correcto es el de **menor imagen**:

\[
\Delta \vec{r}_{ij}^{\text{PBC}} = \Delta \vec{r}_{ij} - L \cdot \text{round}\!\left(\frac{\Delta \vec{r}_{ij}}{L}\right)
\]

donde `round` se aplica componente a componente.

### Por qué importa

Sin PBC, un átomo en \(x \approx 0\) y otro en \(x \approx 29.5\) Å aparecerían separados ~29.5 Å — como si fueran átomos de **superficies opuestas de un clúster aislado**. Con PBC, la separación real es ~0.5 Å (a través del borde), reflejando que el material es **infinito y homogéneo**.

### Traducción a código

```python
diff = diff - L * np.round(diff / L)
```

Desglose componente a componente (eje \(x\), por ejemplo):

1. `diff / L` — expresa el desplazamiento en **unidades de caja** (¿cuántas celdas unitarias separan a los átomos?).
2. `np.round(...)` — identifica la imagen periódica más cercana (equivalente a elegir el traslado de red \(\vec{T} = n_x L \hat{x} + n_y L \hat{y} + n_z L \hat{z}\) que minimiza \(|\Delta \vec{r}|\)).
3. `L * np.round(...)` — ese traslato de red en Å.
4. Resta final — vector de imagen mínima.

**Analogía cristalográfica:** Es exactamente el mismo criterio que usas al calcular dist interatómicas en una celda unitaria convencional: si un átomo "sale" por la cara \(-x\), entra por la cara \(+x\).

### Límite de radio: \(r_{\max} = L/2\)

La convención de imagen mínima solo es válida para \(r < L/2\). Más allá, dos átomos pueden tener dos imágenes igualmente cercanas (ambigüedad). Por eso:

```python
r_max = L / 2.0  # 15 Å para L = 30 Å
```

---

## Paso 3: Evitar duplicados — `np.triu_indices`

### Problema

La matriz `dist` es **simétrica**: \(r_{ij} = r_{ji}\). Además, la diagonal \(r_{ii} = 0\) (auto-distancia). Contar todos los \(N^2\) elementos duplicaría cada par y contaminaría el histograma con ceros.

### Solución

Extraemos solo el **triángulo superior** (pares únicos con \(i < j\)):

```python
distances = dist[np.triu_indices(N, k=1)]
```

`np.triu_indices(N, k=1)` devuelve dos arrays de índices `(rows, cols)` donde `rows[k] < cols[k]`. El parámetro `k=1` excluye la diagonal (\(i = j\)).

**Analogía:** En una matriz de distancias interatómicas de un estudio de vecinos, solo necesitas la mitad superior: la distancia del átomo 5 al 12 es la misma que del 12 al 5.

| Enfoque | Pares contados | Memoria del histograma |
|---------|---------------|------------------------|
| Bucle doble completo | \(N(N-1)\) (con duplicados) | Inflado ×2 |
| `triu_indices` | \(N(N-1)/2\) (únicos) | Correcto tras factor ×2 |

El array `distances` tiene longitud \(N(N-1)/2 = 499{,}500\) para \(N = 1000\).

---

## Paso 4: Histograma radial y volumen de cascarones esféricos

### Ecuación física

Contamos cuántos pares interatómicos caen en un cascarón esférico de radio interno \(r\) y externo \(r + dr\):

\[
n(r) = \text{número de pares con } r \leq r_{ij} < r + dr
\]

El volumen de ese cascarón es:

\[
\Delta V(r) = \frac{4\pi}{3}\left[(r + dr)^3 - r^3\right] \approx 4\pi r^2 \, dr \quad \text{para } dr \ll r
\]

### Traducción a código

**Definición de bins:**

```python
dr = 0.05
bins = np.arange(0, r_max + dr, dr)       # Bordes: 0, 0.05, 0.10, ...
r_centers = 0.5 * (bins[1:] + bins[:-1])  # Centros: 0.025, 0.075, ...
```

**Histograma:**

```python
hist, _ = np.histogram(distances, bins=bins)
hist = hist * 2.0
```

`np.histogram` cuenta cuántos valores de `distances` caen en cada intervalo `[bins[k], bins[k+1])`. El factor `× 2` restaura la bidireccionalidad: si medimos solo \(i < j\), cada par físico contribuye en ambas direcciones (\(i \to j\) y \(j \to i\)), duplicando el conteo respecto a lo que necesita la normalización estándar.

**Volumen de cascarones:**

```python
shell_volumes = (4.0 / 3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)
```

Usamos la fórmula exacta del volumen entre dos esferas concéntricas, no la aproximación \(4\pi r^2 dr\). Esto es más preciso cuando \(dr = 0.05\) Å no es despreciable frente a \(r\).

**Analogía:** Piensa en dividir el espacio alrededor de un átomo de referencia en **capas esféricas concéntricas** de grosor \(dr\), como las capas de una cebolla. Cada bin del histograma es una capa; `hist[k]` cuenta cuántos vecinos hay en esa capa.

---

## Paso 5: Normalización — De conteos a \(g(r)\) y RDF\((r)\)

### Función de correlación de pares \(g(r)\)

La definición normalizada es:

\[
g(r) = \frac{n(r)}{N \cdot \rho \cdot \Delta V(r)}
\]

donde:
- \(n(r)\) = conteo de pares en el cascarón (nuestro `hist`),
- \(N\) = número total de átomos,
- \(\rho = N/V\) = densidad numérica,
- \(\Delta V(r)\) = volumen del cascarón.

```python
g_r = hist / (N * shell_volumes * rho)
```

**Interpretación física:**

| Valor de \(g(r)\) | Significado |
|-------------------|-------------|
| \(g(r) \gg 1\) | Mayor probabilidad de encontrar un par a distancia \(r\) que en un gas ideal de la misma densidad → **primer vecino, segundo vecino** |
| \(g(r) \approx 1\) | Correlaciones desvanecidas → comportamiento de **líquido/gas homogéneo** a larga distancia |
| \(g(r) < 1\) | "Zona prohibida" o correlación negativa (huecos estructurales entre capas) |

Para a-Si, el primer pico de \(g(r)\) cerca de 2.38 Å refleja la distancia Si–Si tetraédrica; los mínimos entre picos corresponden a **planos de exclusión** entre capas de coordinación.

### Función de distribución radial RDF\((r)\)

\[
\text{RDF}(r) = 4\pi r^2 \rho \, g(r)
\]

```python
rdf_r = 4.0 * np.pi * (r_centers**2) * rho * g_r
```

**Analogía:** Si \(g(r)\) es la "probabilidad relativa" de encontrar un vecino a distancia \(r\), RDF\((r)\) es la **distribución ponderada por el área de la esfera** a ese radio. El factor \(4\pi r^2\) convierte la densidad angular en una densidad radial integrable.

RDF\((r)\) tiene unidades de **probabilidad por Å** (densidad radial). Su integral respecto a \(r\) da directamente el número de vecinos acumulados.

---

## Paso 6: Número de Coordinación — Integración con Simpson

### Ecuación física

El número de coordinación de primer vecino es el número promedio de átomos en la primera capa esférica alrededor de un átomo de referencia:

\[
\text{CN} = \int_0^{r_{\min}} \text{RDF}(r) \, dr
\]

donde \(r_{\min}\) es la posición del **primer mínimo** de RDF\((r)\) después del primer pico — el límite natural de la primera capa de coordinación.

Para silicio covalente tetraédrico, el valor ideal es **CN = 4**. En a-Si, defectos topológicos (átomos sub-coordinados o sobre-coordinados) reducen el promedio a ~3.5.

### Identificación automática del pico y mínimo

```python
first_peak_idx = np.argmax(rdf_r[r_centers < 3.0])
```

Buscamos el máximo de RDF en la región \(r < 3\) Å (donde cae el primer vecino de Si).

```python
search_range = (r_centers > r_centers[first_peak_idx]) & (r_centers < 3.5)
first_min_idx = np.where(search_range)[0][np.argmin(rdf_r[search_range])]
r_min = r_centers[first_min_idx]
```

Después del pico, buscamos el mínimo local hasta 3.5 Å. Ese mínimo marca la frontera entre la primera y segunda capa.

**Analogía cristalográfica:** Es equivalente a trazar una esfera alrededor de un átomo central hasta el "valle" entre el primer y segundo shell de vecinos en una función radial de difracción.

### Integración numérica — Regla de Simpson

```python
from scipy.integrate import simpson

coord_num = simpson(rdf_r[:first_min_idx+1], x=r_centers[:first_min_idx+1])
```

La regla de Simpson aproxima la integral usando **parábolas** sobre puntos equiespaciados, con error \(O(dr^4)\). Para \(dr = 0.05\) Å y una curva suave como RDF, es más precisa que la regla trapezoidal.

\[
\text{CN} \approx \frac{dr}{3}\left[\text{RDF}(r_0) + 4\,\text{RDF}(r_1) + 2\,\text{RDF}(r_2) + 4\,\text{RDF}(r_3) + \cdots + \text{RDF}(r_n)\right]
\]

(con los pesos alternados 1-4-2-4-...-1 que implementa `simpson`).

**Por qué Simpson y no `np.trapz`?** Ambos funcionan; Simpson captura mejor la curvatura del pico principal (forma gaussiana/lorentziana del primer vecino), reduciendo el error de integración en ~1–2% para bins de 0.05 Å.

**Verificación física:** Con CN ≈ 3.53 y pico a 2.38 Å, la estructura es mayoritariamente tetraédrica con una fracción de defectos. Comparar con CN = 4 del c-Si y CN ≈ 3.67 del a-Si de Wooten-Winer-Weaire es un buen control de calidad.

---

## Resumen: correspondencia ecuación ↔ línea de código

| # | Física | Código | Shape / tamaño |
|---|--------|--------|----------------|
| 1 | Coordenadas atómicas | `positions = df.values` | `(N, 3)` |
| 2 | Densidad numérica \(\rho = N/V\) | `rho = N / V` | escalar |
| 3 | Vector separación \(\Delta \vec{r}_{ij}\) | `diff = pos[:,None,:] - pos[None,:,:]` | `(N, N, 3)` |
| 4 | PBC / imagen mínima | `diff -= L * np.round(diff / L)` | `(N, N, 3)` |
| 5 | Distancia \(r_{ij}\) | `dist = np.sqrt((diff**2).sum(-1))` | `(N, N)` |
| 6 | Pares únicos | `distances = dist[triu_indices(N,1)]` | `(N(N-1)/2,)` |
| 7 | Conteo en cascarón \(n(r)\) | `hist = np.histogram(...)*2` | `(n_bins,)` |
| 8 | Volumen \(\Delta V(r)\) | `(4π/3)(r_outer³ - r_inner³)` | `(n_bins,)` |
| 9 | PCF \(g(r)\) | `hist / (N * shell_volumes * rho)` | `(n_bins,)` |
| 10 | RDF\((r)\) | `4π r² ρ g(r)` | `(n_bins,)` |
| 11 | CN | `simpson(rdf_r[:r_min], x=r_centers[:r_min])` | escalar |

---

## Lecturas recomendadas para profundizar

- Allen & Tildesley, *Computer Simulation of Liquids* — Capítulo sobre funciones de correlación.
- Frenkel & Smit, *Understanding Molecular Simulation* — Sección de PBC y RDF en simulaciones Monte Carlo/MD.
- Wooten, Winer & Weaire, *Phys. Rev. Lett.* 54, 1392 (1985) — Modelo estructural de referencia para a-Si.

---

## Ejercicios sugeridos

1. **Cambiar `dr`:** Repite el cálculo con `dr = 0.01` y `dr = 0.10`. ¿Cómo cambia el CN? ¿El pico se desplaza?
2. **Sin PBC:** Comenta la línea de imagen mínima. ¿Qué ocurre con \(g(r)\) a distancias largas?
3. **Segundo vecino:** Modifica el script para calcular CN₂ integrando hasta el segundo mínimo.
4. **Exportar datos:** Guarda `r_centers`, `g_r` y `rdf_r` en un CSV para comparar con resultados de LAMMPS o OVITO.
