# Análisis estructural de Silicio Amorfo: RDF, g(r) y Número de Coordinación

Script en Python para calcular la **Función de Distribución Radial** (RDF), la **Función de Correlación de Pares** (PCF, \(g(r)\)) y el **Número de Coordinación** (CN) a partir de coordenadas atómicas 3D de una simulación de silicio amorfo (a-Si) con condiciones de contorno periódicas.

> Para la traducción detallada de las ecuaciones físicas a operaciones de NumPy, consulta [`LOGIC_EXPLANATION.md`](LOGIC_EXPLANATION.md).

---

## Requisitos

| Paquete | Uso en el script |
|---------|------------------|
| `numpy` | Álgebra matricial, distancias, histogramas |
| `pandas` | Lectura del archivo Excel de coordenadas |
| `matplotlib` | Gráficas de RDF y g(r) |
| `scipy` | Integración numérica (regla de Simpson) |
| `openpyxl` | Motor de lectura de archivos `.xlsx` (requerido por pandas) |

### Instalación

Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install numpy pandas matplotlib scipy openpyxl
```

---

## Estructura del proyecto

```
1hw-aye/
├── app.py              # Script principal
├── amorphous.xlsx      # Coordenadas atómicas (entrada)
├── README.md           # Este archivo (uso y ejecución)
└── LOGIC_EXPLANATION.md # Física → código (documento pedagógico)
```

---

## Archivo de entrada

### `amorphous.xlsx`

| Propiedad | Valor |
|-----------|-------|
| Formato | Excel (`.xlsx`), **sin encabezados** |
| Dimensiones | 1000 filas × 3 columnas |
| Contenido | Coordenadas cartesianas \((x, y, z)\) en **Ångströms (Å)** |
| Convención | Cada fila es un átomo de Si; columnas 0, 1, 2 = \(x, y, z\) |

El script asume que las coordenadas están contenidas en una caja cúbica de arista \(L = 30\) Å (3 nm), coherente con la densidad típica de a-Si (~2.2 g/cm³).

---

## Ejecución

```bash
source .venv/bin/activate
python app.py
```

El script debe ejecutarse desde el directorio donde se encuentran `app.py` y `amorphous.xlsx`.

---

## Salidas

### 1. Métricas en terminal

Al finalizar el cálculo, se imprimen cuatro cantidades:

```
Density (ρ): 0.0370 atoms/Å³
First peak at: 2.38 Å
First minimum (Shell boundary) at: 2.98 Å
Nearest-Neighbor Coordination Number: 3.5255
```

| Métrica | Significado físico |
|---------|-------------------|
| **ρ** | Densidad numérica: \(N / V\) (átomos/Å³) |
| **First peak** | Posición del primer máximo de RDF(r), asociado a la distancia Si–Si de primer vecino (~2.35–2.40 Å) |
| **First minimum** | Límite superior de la primera capa de coordinación (\(r_{\min}\)) |
| **CN** | Número de coordinación integrando RDF(r) desde \(r = 0\) hasta \(r_{\min}\) |

Para silicio tetraédrico ideal se espera **CN ≈ 4**. Valores ligeramente inferiores (~3.5) son habituales en a-Si por defectos topológicos (átomos con coordinación 3 o 5).

### 2. Ventana gráfica (Matplotlib)

Se abre una figura con **dos paneles**:

| Panel | Eje Y | Contenido |
|-------|-------|-----------|
| Izquierdo | RDF(r) | Curva azul; área sombreada = región integrada para el CN; línea roja discontinua en \(r_{\min}\) |
| Derecho | g(r) | Curva azul; línea naranja en \(g(r) = 1\) (referencia de gas ideal / correlaciones ausentes a larga distancia) |

Ambos ejes X muestran la distancia en Å (rango visual: 0–10 Å).

La figura se guarda automáticamente como **`rdf_analysis.png`** en el directorio del proyecto. El script termina sin bloquear la terminal.

---

## Parámetros configurables

Edita las constantes al inicio de `app.py` si tu sistema difiere:

| Variable | Valor por defecto | Descripción |
|----------|-------------------|-------------|
| `L` | `30.0` | Arista de la caja cúbica (Å) |
| `dr` | `0.05` | Ancho de bin del histograma radial (Å) |
| `r_max` | `L / 2.0` | Radio máximo de análisis (Å); limitado por la convención de imagen mínima |

---

## Referencia rápida del flujo de cálculo

```
amorphous.xlsx  →  positions (N×3)
                        ↓
              Distancias pareadas + PBC
                        ↓
              Histograma radial → g(r) → RDF(r)
                        ↓
              Integración hasta r_min → CN
                        ↓
              Terminal + gráficas
```

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---------|---------------|----------|
| `FileNotFoundError: amorphous.xlsx` | Directorio de trabajo incorrecto | Ejecutar desde la carpeta del proyecto |
| `ModuleNotFoundError: openpyxl` | Falta dependencia para Excel | `pip install openpyxl` |
| CN ≈ 0 o muy alto | `L` incorrecto o coordenadas fuera de la caja | Verificar unidades (Å) y tamaño de caja |
| Gráfica vacía | Datos corruptos o `N = 0` | Inspeccionar `amorphous.xlsx` |

---

## Documentación complementaria

- **[`LOGIC_EXPLANATION.md`](LOGIC_EXPLANATION.md)** — Desglose paso a paso de cómo cada ecuación física se traduce en operaciones vectorizadas de NumPy/SciPy. Dirigido a conectar el razonamiento espacial cristalográfico con la programación matricial.
