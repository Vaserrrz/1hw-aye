# HW1 MSE 710 - Análisis Estructural Computacional

Este proyecto contiene el análisis computacional de estructuras amorfas (Silicio) y cristalinas (Rutilo TiO2) utilizando operaciones vectorizadas para garantizar alta eficiencia y exactitud física (aplicación de Condiciones de Contorno Periódicas).

## 🛠️ 1. Preparación del Entorno

Antes de ejecutar los scripts, asegúrate de tener las librerías científicas necesarias instaladas. Abre la terminal en esta carpeta y ejecuta:

`pip install -r requirements.txt`

## 🚀 2. Guía de Ejecución y Resultados Esperados

### Parte A: Análisis del Silicio Amorfo (Problema 1)
Este script lee las coordenadas atómicas, aplica la Convención de Imagen Mínima para simular un material *bulk* en una caja periódica de 30 Å, y calcula la distribución radial.

**Cómo ejecutar:**
1. Abre tu terminal.
2. Navega a la carpeta del código fuente: `cd src`
3. Ejecuta el script: `python problem1_amorphous_si.py`

**Resultados Visuales:**
El script generará automáticamente un archivo de imagen en la carpeta `results/` mostrando dos subplots: la RDF(r) y la Función de Correlación de Pares g(r).

### Parte B: Módulo de Tensor Métrico y Validación de Rutilo (Problemas 4 y 5)
El archivo `crystallography_utils.py` actúa como una "caja de herramientas" matemáticas. El script `check_problem4_rutile.py` lo importa para validar los cálculos del cristal tetragonal.

**Cómo ejecutar:**
1. Mantente en la carpeta `src`.
2. Ejecuta el script de prueba: `python check_problem4_rutile.py`
