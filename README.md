# Análisis Semántico de Perfiles de Egreso en Informática

## Resumen del Proyecto

Investigación basada en un **Enfoque Híbrido de Machine Learning** para medir la separabilidad y convergencia semántica en la oferta académica de Ingeniería en Informática en Chile. El sistema descubre patrones mediante métodos no supervisados (TF-IDF, PCA, Similitud Coseno de Centroides) y valida matemáticamente sus hallazgos con modelos supervisados (Regresión Logística, LDA, Test de Permutación), procesando 73 perfiles de egreso de diversas instituciones mediante Procesamiento de Lenguaje Natural (NLP) clásico.

**Autores:** Brayan Pineda Poblete · Walter Reyes Silva
**Institución:** Universidad de las Américas — Facultad de Ingeniería y Negocios
**Carrera:** Ingeniería de Ejecución en Informática

---

## Arquitectura del Sistema

El proyecto se divide en fases modulares para garantizar la integridad y trazabilidad de los datos. Cada fase opera de forma independiente y se comunica mediante un repositorio de datos centralizado ubicado en `src/data/`.

---

### Fase 1: Recolección y Extracción (Web Scraping)

Motor de scraping híbrido utilizando **BeautifulSoup** y **Selenium** para la extracción automatizada de 73 perfiles de egreso desde portales institucionales. El sistema aplica un fallback automático: intenta primero con `Requests` (sitios estáticos) y recurre a `Selenium` (sitios dinámicos con JavaScript) si la extracción falla.

- **Entrada:** Configuración de sitios en `config.py` con selectores CSS por institución.
- **Salida:** Dataset en bruto (`perfiles_egreso_raw.csv`).

---

### Fase 2: Procesamiento de Lenguaje Natural (NLP)

Pipeline lingüístico de limpieza en dos capas para preparar el corpus:

- **Capa Estructural:** Normalización de texto mediante expresiones regulares (Regex): minúsculas, remoción de símbolos y números, neutralización de género gramatical.
- **Capa Lingüística:** Uso de **spaCy** (`es_core_news_sm`) para tokenización, lematización y eliminación estricta de *Stopwords* académicas y **confusores institucionales** (términos como *alumno*, *universidad*, *malla*, *egresado* que distorsionan el análisis de competencias reales).

- **Entrada:** `perfiles_egreso_raw.csv`
- **Salida:** `perfiles_egreso_limpio_v1.csv`

---

### Fase 3: Vectorización y Diferenciación Léxica

Transformación del corpus limpio en representaciones matemáticas para el análisis de similitud y diferenciación:

- **TF-IDF (1500 features):** Vectorización del corpus completo mediante `TfidfVectorizer`, generando una matriz densa de 73 × 1500 dimensiones.
- **Análisis de Keyness (Z-Score):** Cálculo estadístico sobre frecuencias relativas (`CountVectorizer`) para identificar y extraer los **Top 15 términos distintivos** por grado académico — el ADN léxico de cada carrera.
- **PCA:** Reducción de dimensionalidad para proyección visual del espacio vectorial.
- **Similitud Coseno de Centroides:** Cálculo de la homogeneidad entre los 4 grados académicos.

> **Reproducibilidad garantizada:** Todos los scripts de esta fase (01 al 05) utilizan semillas fijas (`random_state=42`) para asegurar resultados idénticos en cada ejecución.

- **Salida principal:** `top15_palabras_clave.csv`, `resultados.json`

---

### Fase 4: Análisis y Modelado Matemático Supervisado

Validación matemática de los patrones descubiertos mediante Aprendizaje Supervisado:

- **Regresión Logística:** Clasificador principal entrenado con **Validación Cruzada Estratificada (K-Fold = 5)**, alcanzando un Accuracy del **57.7%** y F1-macro del **57.2%**. El bajo accuracy no es un error — es la evidencia cuantitativa de la alta convergencia semántica entre los grados.
- **SVC con kernel RBF:** Clasificador secundario de validación.
- **LDA (Análisis Discriminante Lineal):** Reducción supervisada a 2D para visualización de la separabilidad real entre grados.
- **Test de Permutación:** Validación estadística de la similitud coseno entre centroides de Ingeniería Civil e Ingeniería en Informática (**0.69**, con **p = 0.00**), confirmando que la convergencia es **estadísticamente significativa** (p < 0.05).

> **Reproducibilidad garantizada:** Todos los scripts de esta fase (01 al 05) utilizan semillas fijas (`random_state=42`).

- **Salida:** `resultados.json`, `matriz_confusion_LR.png`, `proyeccion_pca_vs_lda.png`, `similitud_centroides.png`

---

## Hallazgo Principal

> *"Al eliminar el ruido comercial y analizar puramente las competencias exigidas, la frontera semántica entre Ingeniería Civil e Ingeniería en Informática prácticamente ha desaparecido en el papel. La diferenciación real hoy solo sobrevive en los extremos: el grado Técnico (identidad operativa pura) y el grado de Ejecución (nicho puente)."*

---

## Instrucciones de Instalación y Despliegue

### Entorno Virtual

```bash
python -m venv venv
# En Windows:
.\venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### Dependencias

```bash
pip install -r requirements.txt
```

### Modelos de Lenguaje

Es indispensable descargar el modelo de spaCy para el procesamiento en español:

```bash
python -m spacy download es_core_news_sm
```

---

## Ejecución del Pipeline

Los scripts deben ejecutarse en el orden numérico establecido dentro de cada carpeta de fase en `src/`. Los datos generados se almacenan automáticamente en `src/data/`, carpeta excluida del repositorio según `.gitignore` para mantener la limpieza del control de versiones.

```
src/
├── Fase1_Recoleccion/    → main.py
├── Fase2_Procesamiento/  → 01_limpieza_nlp.py → 02_vectorizacion.py
├── Fase3_Analisis/       → 01_clustering.py → 02_analisis_clusters.py
│                            → 03_validacion_clusters.py → 04_reporte_distribucion.py
│                            → 05_visualizacion_interactiva.py
└── data/
    ├── raw/              → perfiles_egreso_raw.csv
    └── processed/        → perfiles_egreso_limpio_v1.csv · resultados.json · *.png
```