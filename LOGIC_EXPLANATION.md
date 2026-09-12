# De la física que ya dominas al álgebra que NumPy ejecuta

Esta guía no enseña cristalografía. Asume que la red, el tensor métrico y la función de distribución radial ya forman parte de tu vocabulario. El único salto que falta es mental: **cómo esas mismas invariantes se escriben como operaciones sobre arrays**, de modo que el código de `src/` no se lea como una receta de programación, sino como el modelo físico que ya calcularías a mano — evaluado de una sola vez sobre todo el *bulk*.

Los scripts viven en `src/problem1_amorphous_si.py` y `src/crystallography_utils.py`. Lo que sigue es el mapa entre tu razonamiento y esas líneas.

---

# 1. La Arquitectura del Proyecto (Ciencia Reproducible)

Un experimento de laboratorio no mezcla la muestra, el equipo y la bitácora en el mismo cajón. Un cálculo reproducible tampoco.

| Carpeta | Analogía de laboratorio | Rol en este proyecto |
|---------|-------------------------|----------------------|
| `data/` | La **muestra**: se recibe, se etiqueta y no se altera | `amorphous.xlsx` — 1000 sitios de Si en Å. Es el espécimen. El código lo lee; nunca lo reescribe. |
| `src/` | Los **equipos de medición**: goniómetro, espectrómetro, integrador | Los scripts. Aquí vive la física: PBC, $g(r)$, tensor métrico. Si cambias un parámetro, cambias el instrumento, no la muestra. |
| `results/` | La **bitácora**: curvas, números, figuras que se archivan | RDF, $g(r)$ y el PDF/PNG de dos paneles. Se regeneran; no se editan a mano. |

Esta separación no es cosmética. Te permite responder, meses después, a la pregunta que un revisor (o tu yo futuro) siempre hace: *¿estos números salieron de esta muestra, con este instrumento, en esta fecha?* El `README.md` fija el protocolo (`cd src` y ejecutar). `requirements.txt` fija la mesa óptica: las mismas librerías, las mismas versiones conceptuales.

Cuando leas el código, ten presente esa analogía. `DATA_PATH` apunta a la muestra. `FIGURE_STEM` apunta a la bitácora. Entre ambas, solo hay medición.

---

# 2. Física a Código: El Silicio Amorfo (Problema 1)

El modelo es un cubo de $L = 30\,\mathrm{\AA}$ (3 nm) con $N = 1000$ átomos. No hay red de Bravais; hay un *continuous random network*. Lo que sí hay —y lo que el código debe respetar— es **homogeneidad de un sólido infinito**, no de un nanocúmulo con superficies.

La configuración entra como una matriz $\mathbf{R}\in\mathbb{R}^{N\times 3}$. Cada fila es un sitio; cada columna, una componente cartesiana. Es la misma tabla que dibujarías en CrystalMaker, sin índices de Miller porque no hay celda convencional que traducir.

La densidad numérica $\rho = N/V = N/L^3$ no es un parámetro ajustable: es la densidad del *gas de referencia* contra el cual se normaliza $g(r)$. Si $g(r)\to 1$ a $r$ grande, el instrumento está bien calibrado.

## Vectorización vs. bucles: medir el *bulk* de una vez

El cálculo ingenuo recorre cada par $(i,j)$ con $j>i$, resta vectores y toma la norma. Eso es correcto físicamente y desastroso como modelo de un sólido: evalúa el material **átomo a átomo**, como si midieras 499 500 distancias con un calibre.

NumPy no “acelera un `for`”. Construye el **tensor de separaciones** de todo el sistema:

```python
diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 3)
```

`positions[:, None, :]` es cada átomo $i$ como origen; `positions[None, :, :]` es cada átomo $j$ como destino. La resta, por *broadcasting*, produce $\Delta\vec{r}_{ij}$ para todos los pares a la vez. Shape `(N, N, 3)`: una copia de la estructura desplazada respecto de sí misma.

Eso se parece más a lo que haces al pensar en un cristal: no recorres sitios, **aplicas una operación a la red**. El *bulk* se evalúa simultáneamente. `np.triu_indices(N, k=1)` extrae el triángulo superior ($i<j$) porque $r_{ij}=r_{ji}$ y la diagonal es la auto-distancia nula — el mismo recorte que harías en una matriz de distancias interatómicas para no contar dos veces el mismo enlace.

## Condiciones de contorno periódicas: la línea que inventa el infinito

Sin PBC, un átomo en $x\approx 0$ y otro en $x\approx 29.5\,\mathrm{\AA}$ aparecen separados 29.5 Å. El código habría modelado un **polvo de 3 nm** con superficies libres. El a-Si del enunciado no es eso: es un fragmento de un material que continúa.

La Convención de Imagen Mínima elige, entre las réplicas periódicas $\vec{T}=L(n_x,n_y,n_z)$, la que minimiza $|\Delta\vec{r}-\vec{T}|$:

```python
diff = diff - L * np.round(diff / L)
```

Lee esa línea como un cambio de celda, no como un truco numérico:

1. `diff / L` expresa la separación en **unidades de caja** — cuántas celdas de 30 Å median entre los dos sitios.
2. `np.round(...)` identifica el traslado de red más cercano, componente a componente. Es el mismo criterio que usas al plegar un vector fraccional al intervalo $(-1/2,1/2]$.
3. Restar $L\cdot\mathrm{round}(\Delta/L)$ pliega cada componente a $(-L/2,L/2]$.

El átomo del borde ya no “ve” el vacío: ve la imagen de su vecino al otro lado. El cubo se convierte en la celda primitiva de un mosaico infinito. Por eso el análisis se corta en $r_{\max}=L/2$: más allá, la imagen mínima deja de ser única (dos réplicas empatan) y el histograma mentiría.

Si comentas esa línea y vuelves a correr, $g(r)$ a $r$ grande **deja de tender a 1**. Ese es el sello experimental de haber medido un cúmulo en lugar de un *bulk*.

## Histogramas y cascarones: de vecinos a $g(r)$

Alrededor de cada átomo el espacio se parte en cascarones esféricos de espesor $dr=0.05\,\mathrm{\AA}$ — las capas de una cebolla, o las esferas de coordinación que dibujas sobre un RDF de polvo.

`np.histogram` no “hace un gráfico”: **cuenta cuántos pares caen en cada cascarón**. El factor $\times 2$ restaura la bidireccionalidad ($i\to j$ y $j\to i$) que perdimos al quedarnos con $i<j$. Sin ese 2, $g(r)$ convergería a $1/2$ y el CN saldría la mitad del valor físico.

El volumen del cascarón no se aproxima a ciegas. El código usa el volumen exacto entre dos esferas concéntricas

\[
\Delta V(r)=\frac{4\pi}{3}\bigl[(r+dr)^3-r^3\bigr],
\]

que es la integral de $4\pi r^2\,\mathrm{d}r$ sobre el bin. Para $dr\ll r$ recuperas $4\pi r^2\,dr$; cerca del origen, donde el primer vecino del Si vive (~2.35 Å) y $dr$ ya no es despreciable, la forma exacta evita sesgar la normalización.

Entonces

\[
g(r)=\frac{n(r)}{N\,\rho\,\Delta V(r)},\qquad
\mathrm{RDF}(r)=4\pi r^2\rho\,g(r).
\]

$g(r)$ es la probabilidad de encontrar un vecino a distancia $r$ **respecto de un gas ideal de la misma densidad**. El primer pico ($g\approx 8$ a $2.38\,\mathrm{\AA}$) es la distancia Si–Si tetraédrica. Los valles son planos de exclusión entre capas. RDF$(r)$ reintroduce el área de la esfera: es la densidad radial *integrable*, la que convierte “probabilidad relativa” en “número de átomos”.

Que $g(r)\to 1$ a $r\gtrsim 12\,\mathrm{\AA}$ no es un detalle de cómputo. Es la prueba de que las correlaciones de corto alcance —las únicas que sobreviven en un amorfo— se han extinguido, y de que las PBC no fabricaron un artefacto de superficie.

## Integración numérica: CN $\approx 3.53$ y los defectos de la red

El número de coordinación de la primera capa es el área bajo el primer pico de la RDF, hasta el mínimo que separa esa capa de la segunda:

\[
\mathrm{CN}=\int_0^{r_{\min}}\mathrm{RDF}(r)\,\mathrm{d}r.
\]

$r_{\min}$ no es “el bin más bajo después del máximo”. Es el valle estructural entre el primer y el segundo shell — en este modelo, $r_{\min}\approx 2.98\,\mathrm{\AA}$. Integrar un poco más o un poco menos es el equivalente computacional de elegir mal el radio de la esfera de coordinación en CrystalMaker.

`scipy.integrate.simpson` aproxima esa integral con parábolas sobre puntos equiespaciados (error $O(dr^4)$). El pico de primer vecino es suave, casi gaussiano; Simpson respeta esa curvatura mejor que un trapecio. No es un capricho de librería: es elegir la cuadratura adecuada para una densidad radial que ya conoces por forma.

**CN $= 3.53$** no es un fallo del código. El c-Si es tetraédrico (CN $= 4$, $d_{\mathrm{Si-Si}}\approx 2.35\,\mathrm{\AA}$). El a-Si de un *continuous random network* (Wooten–Winer–Weaire y sus herederos) admite átomos 3-coordinados —enlaces colgantes, $T_3$— y una fracción menor de sitios 5-coordinados. El promedio baja de 4 hacia $3.5$–$3.8$. Un valor de $3.53$ te está diciendo, en el lenguaje de defectos que usas para semiconductores: **la red es mayoritariamente tetraédrica, con una densidad apreciable de estados de coordinacion deficitaria**. Eso es coherente con colas de Urbach, centros $D$ y la electrónica del a-Si:H, aunque este modelo no lleve hidrógeno.

Si el CN saliera 2, habrías medido cadenas. Si saliera 4.00 exacto, habrías medido un cristal (o un amorfo demasiado perfecto para ser creíble). $3.53$ es el número de un sólido realista.

---

# 3. Álgebra Lineal: El Rutilo y el Tensor Métrico (Problemas 4 y 5)

El Problema 4 se calcula a mano. El Problema 5 pide un módulo (`crystallography_utils.py`) que **verifique** ese cálculo, no que lo reemplace. El PDF dice “check problem 1”; es un typo. Se verifica el rutilo.

Cuatro funciones, una por invariante: $G$, distancia, producto punto, ángulo. Ningún `if` por sistema de Bravais. La celda entra por sus seis parámetros; la geometría sale de $G$.

## El tensor métrico: por qué Pitágoras no sobrevive a las coordenadas fraccionales

En una base ortonormal, $d^2=\Delta x^2+\Delta y^2+\Delta z^2$. En coordenadas **fraccionales**, $\Delta\vec{x}=(x,y,z)$ no está escrito en ångströms: $x=1$ significa “un vector de red $\mathbf{a}$”. Los ejes $\mathbf{a},\mathbf{b},\mathbf{c}$ no son de longitud 1 ni, en general, ortogonales.

El rutilo es tetragonal ($P4_2/mnm$, red de Bravais tI). $\alpha=\beta=\gamma=90^\circ$, así que los ejes *sí* son ortogonales — pero $a=b=4.5937\,\mathrm{\AA}$ y $c=2.9581\,\mathrm{\AA}$. Pitágoras fraccional $x^2+y^2+z^2$ trataría $c$ como si midiera lo mismo que $a$. Un desplazamiento $(0,0,1)$ mediría $1$, no $2.9581\,\mathrm{\AA}$. El octaedro $\mathrm{TiO}_6$ saldría geométricamente falso.

El tensor métrico $G_{ij}=\mathbf{a}_i\cdot\mathbf{a}_j$ **encapsula esa deformación de la celda**. Es la Gram de la base $\{\mathbf{a},\mathbf{b},\mathbf{c}\}$:

\[
G=\begin{pmatrix}
a^2 & ab\cos\gamma & ac\cos\beta \\
ab\cos\gamma & b^2 & bc\cos\alpha \\
ac\cos\beta & bc\cos\alpha & c^2
\end{pmatrix}.
\]

Para el rutilo, los cosenos de $90^\circ$ anulan los términos fuera de la diagonal y $G=\mathrm{diag}(a^2,b^2,c^2)$. $G$ no “añade trigonometría”: **estira cada eje fraccional hasta su longitud real**. En una celda triclínica haría, además, el trabajo de los ángulos. Una sola matriz cubre los catorce tipos de Bravais.

`compute_metric_tensor` convierte $\alpha,\beta,\gamma$ de grados a radianes —porque así se publican los parámetros de red, y así trabaja el coseno— y arma esa matriz $3\times 3$. No hay ramas `if lattice == "tI"`. La física de la celda *es* $G$.

## El operador `@`: la fórmula de un renglón que sustituye páginas de trigonometría

La distancia cartesiana entre dos sitios fraccionales es la forma cuadrática

\[
d=\sqrt{\Delta\vec{x}^{\,T}\,G\,\Delta\vec{x}}.
\]

En el módulo eso es una línea:

```python
d = np.sqrt(delta @ metric_tensor @ delta)
```

`@` es el producto matricial. `delta @ G` contrae el primer índice; el segundo `@ delta` contrae el restante y deja un escalar. Es exactamente $\sum_{ij}\Delta x_i\,G_{ij}\,\Delta x_j$. El producto punto de dos *direcciones* (enlaces, no puntos) es el mismo patrón sin la raíz: `v1 @ G @ v2`. El ángulo recupera $\cos\theta=(\mathbf{u}\cdot\mathbf{v})/(|\mathbf{u}||\mathbf{v}|)$, evaluando numerador y normas con $G$.

A mano, para cada par Ti–O, expandirías $a^2\Delta x^2+b^2\Delta y^2+c^2\Delta z^2$ y, si la celda no fuera ortogonal, aparecerían los $2ab\Delta x\Delta y\cos\gamma$. Son las mismas páginas que llenaste en el Problema 4. El operador `@` no inventa un atajo: **es esa expansión**, escrita en la notación en la que $G$ ya era un objeto, no una lista de términos.

Por eso el script `check_problem4_rutile.py` puede declarar, al final, que los números coinciden. Coloca el Ti en el origen, los seis oxígenos del octaedro distorsionado (2 apicales a $\pm(u,u,0)$, 4 ecuatoriales en las imágenes más cercanas) y pide al módulo las dos longitudes Ti–O y los tres ángulos O–Ti–O que no son $180^\circ$. Las expresiones cerradas —$a\sqrt{2}\,u$ para el enlace apical, $\sqrt{2a^2(1/2-u)^2+(c/2)^2}$ para el ecuatorial— viven en el script de prueba, **no** en el módulo. Si ambas rutas coinciden, no te estás verificando a ti misma con tu propia algebra: estás contrastando la forma cuadrática contra la geometría clásica del rutilo.

Los valores que deben reaparecer en la bitácora:

| Invariante | Valor | Lectura cristalográfica |
|------------|-------|-------------------------|
| $G$ | $\mathrm{diag}(21.102,\,21.102,\,8.750)\,\mathrm{\AA}^2$ | Estiramiento tetragonal: $c^2\neq a^2$ |
| Ti–O ecuatorial ($\times 4$) | $1.946\,\mathrm{\AA}$ | Cuatro oxígenos del plano rectangular |
| Ti–O apical ($\times 2$) | $1.983\,\mathrm{\AA}$ | El eje largo del octaedro |
| O–Ti–O | $81.07^\circ$, $90.00^\circ$, $98.93^\circ$ | Distorsión del $\mathrm{TiO}_6$; los *trans* de $180^\circ$ se omiten |

Esa es la conexión que importa: **el cálculo a mano y el `@` son el mismo tensor**. Una vez que $G$ está en memoria, dejar de expandir a mano no es perder rigor; es reconocer que la invariante ya estaba escrita.

---

Cuando vuelvas a `src/`, no leas “funciones de Python”. Lee un densitómetro radial con PBC y un goniómetro métrico. La muestra está en `data/`. La bitácora, en `results/`. El instrumento eres tú, más el álgebra que ya sabías — ahora evaluada sobre el *bulk* entero, de una sola vez.
