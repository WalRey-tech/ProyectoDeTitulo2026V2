# Descubrimiento de patrones en perfiles de Egreso de Informática mediante Aprendizaje no Supervisado

> Evolucionando hacia un modelo híbrido de validación estadística y clasificación supervisada.

**Proyecto de Título · Universidad de las Américas · Facultad de Ingeniería y Negocios · 2026**

[![Python](https://img.shields.io/badge/Python-3.13.7-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.x-09A3D5?logo=spacy&logoColor=white)](https://spacy.io)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/SVM%20Accuracy-87.58%25-8b5cf6)](.)
[![Convergencia](https://img.shields.io/badge/Convergencia%20L%C3%A9xica-76.8%25-10b981)](.)

**Autores:** Brayan Pineda Poblete · Walter Reyes Silva  
**Institución:** Universidad de las Américas — Facultad de Ingeniería y Negocios  
**Carrera:** Ingeniería de Ejecución en Informática  

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#-resumen-ejecutivo)
2. [Contexto y Problema de Investigación](#-contexto-y-problema-de-investigación)
3. [Pipeline Metodológico](#-pipeline-metodológico)
4. [Arquitectura del Sistema](#-arquitectura-del-sistema)
5. [Resultados Finales](#-resultados-finales)
6. [Evidencia Visual: Los 2 Gráficos Clave](#-evidencia-visual-los-2-gráficos-clave)
7. [Conclusiones del Estudio](#-conclusiones-del-estudio)
8. [Estructura del Repositorio](#-estructura-del-repositorio)
9. [Instalación y Ejecución](#-instalación-y-ejecución)
10. [Stack Tecnológico](#-stack-tecnológico)

---

## 🔬 Resumen Ejecutivo

Este proyecto implementa un **pipeline automatizado de Ciencia de Datos e Inteligencia Artificial** para el análisis semántico de perfiles de egreso en la educación superior informática chilena. A través de técnicas de **Procesamiento de Lenguaje Natural (NLP)**, **balanceo sintético de datos (SMOTE)** y un **benchmark competitivo de 11 algoritmos de Machine Learning**, el estudio cuantifica el nivel de homogeneidad curricular entre tres grados académicos en Informática.

| Métrica | Valor |
|---|---|
| **Muestra final depurada** | 63 perfiles de egreso |
| **Instituciones cubiertas** | 47 Instituciones de Educación Superior |
| **Grados analizados** | 3 (Ing. Civil, Ing. en Informática, Ing. de Ejecución) |
| **Algoritmos evaluados en benchmark** | 11 clasificadores |
| **Modelo ganador** | SVM con Kernel RBF |
| **Accuracy (CV Estratificada 5-Folds)** | **87.58%** |
| **Convergencia léxica Civil ↔ Informática** | **76.8%** (similitud coseno: 0.768) |

---

## 🎯 Contexto y Problema de Investigación

En Chile coexisten tres denominaciones para los programas de educación superior en informática:

- **Ingeniería Civil Informática**
- **Ingeniería en Informática**
- **Ingeniería de Ejecución en Informática**

La proliferación de estas denominaciones plantea una interrogante académica relevante: ¿reflejan estas distintas denominaciones competencias realmente diferenciadas, o responden a estrategias de posicionamiento institucional? La revisión cualitativa manual de perfiles de egreso carece de reproducibilidad científica y está sujeta al sesgo del evaluador. Este trabajo aplica métodos computacionales objetivos y reproducibles para cuantificar la **convergencia semántica** entre estos programas.

> *Objetivo: Transformar la incertidumbre cualitativa de las auditorías curriculares manuales en certeza cuantitativa y reproducible mediante el uso de Inteligencia Artificial.*

---

## ⚙️ Pipeline Metodológico

El sistema procesa texto crudo extraído de portales institucionales y lo transforma en evidencia estadística en 4 fases modulares:

```
[Portales Web IES]
       │
       ▼
┌─────────────────────────────────────┐
│  FASE 1: Web Scraping Híbrido       │
│  Requests + Selenium · BeautifulSoup│
│  OUTPUT: perfiles_egreso_raw.csv    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  FASE 2: NLP con spaCy              │
│  Tokenización · Lematización        │
│  Eliminación de confusores IES      │
│  OUTPUT: perfiles_egreso_limpio.csv │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  FASE 3: Vectorización + Análisis   │
│  TF-IDF · Z-Score Keyness           │
│  Similitud Coseno de Centroides     │
│  OUTPUT: top15_palabras_clave.csv   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  FASE 4: Benchmark ML + SMOTE       │
│  11 Clasificadores · SVM Ganador    │
│  PCA (no supervisado) + LDA         │
│  (supervisado) · 5-Fold CV          │
│  OUTPUT: resultados.json · *.png    │
└─────────────────────────────────────┘
```

---

## 🏗️ Arquitectura del Sistema

### Fase 1 — Recolección y Extracción (Web Scraping)

Motor de scraping **híbrido** con fallback automático entre extracción estática y dinámica:

- **`config.py`** — Selectores CSS configurados individualmente por institución
- **`scraper.py`** — Intenta extracción con `Requests`; escala a `Selenium` en sitios con JavaScript
- **`extractors.py`** — Parser BeautifulSoup con sanitización HTML
- **`main.py`** — Orquestador con validación de calidad mínima y guardado en CSV

> **Salida:** `perfiles_egreso_raw.csv` · 63 perfiles de egreso válidos de 47 IES chilenas

---

### Fase 2 — Procesamiento de Lenguaje Natural (NLP)

Pipeline lingüístico en **dos capas** de limpieza:

**Capa Estructural (Regex):** Normalización a minúsculas, remoción de símbolos y neutralización de género gramatical.

**Capa Lingüística (spaCy `es_core_news_sm`):**
- Tokenización y lematización morfológica
- Eliminación de stopwords estándar del español
- Eliminación de **confusores institucionales**: términos propios del contexto universitario (`alumno`, `universidad`, `malla`, `egresado`) que introducen ruido ajeno a las competencias técnicas declaradas
- Control de calidad: perfiles con corpus inferior a 150 caracteres post-limpieza son descartados

> **Salida:** `perfiles_egreso_limpio_v1.csv`

---

### Fase 3 — Vectorización y Análisis de Homogeneidad

Transformación del corpus limpio en representaciones matemáticas comparables:

- **TF-IDF:** Vectorización del corpus completo mediante `TfidfVectorizer` (1500 features), generando una representación densa del espacio léxico
- **Z-Score (Keyness):** Cálculo estadístico sobre frecuencias relativas para extraer los **Top 15 términos distintivos** por grado académico
- **Similitud Coseno de Centroides:** Cuantificación del grado de convergencia semántica entre los tres programas

> **Salida principal:** `top15_palabras_clave.csv`, `resultados.json`  
> **Reproducibilidad garantizada:** `random_state=42` en todos los scripts

---

### Fase 4 — Benchmark de Clasificadores con SMOTE

Validación supervisada con protocolo de evaluación riguroso:

1. **SMOTE** — Balanceo sintético de clases minoritarias para garantizar equidad en el entrenamiento
2. **Benchmark de 11 algoritmos** — Evaluación competitiva incluyendo: SVM (Kernel RBF), KNN, Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, MLP, Linear SVC y Ridge Classifier
3. **Validación Cruzada Estratificada (5-Folds)** — Protocolo de evaluación robusto y reproducible
4. **PCA + LDA** — Reducción a 2D para análisis comparativo de separabilidad (no supervisado vs. supervisado)

> **Salida:** `resultados.json`, `proyeccion_pca_vs_lda.png`, `similitud_centroides.png`

---

## 📊 Resultados Finales

### 🏆 Modelo Ganador: SVM con Kernel RBF

El benchmark de 11 clasificadores determinó al **Support Vector Machine con Kernel RBF** como el modelo de mayor rendimiento:

| Modelo | Accuracy (5-Fold CV) |
|---|---|
| **SVM · Kernel RBF** | **87.58%** 🏆 |
| Random Forest | — |
| Gradient Boosting | — |
| Otros 8 clasificadores | — |

> El rendimiento del SVM (87.58%) evidencia que la separabilidad entre grados existe, aunque no es linealmente distinguible. El kernel no lineal RBF logra capturar las fronteras semánticas latentes que la revisión cualitativa humana no puede discernir.

### 📐 Dataset Final Depurado

| Grado | Perfiles | % del corpus |
|---|---|---|
| Ingeniería Civil Informática | 32 | 50.8% |
| Ingeniería en Informática | 25 | 39.7% |
| Ingeniería de Ejecución | 6 | 9.5% |
| **Total** | **63** | **100%** |
| **Instituciones cubiertas** | **47 IES** | — |

---

## 📈 Evidencia Visual: Los 2 Gráficos Clave

### Gráfico 1 — Proyección 2D PCA vs LDA

`public/assets/proyeccion_pca_vs_lda.png`

Este gráfico de doble panel constituye la **evidencia visual central** del estudio:

**Panel izquierdo — PCA (no supervisado):**  
Los tres grados se proyectan formando un clúster central de alta densidad entre Civil e Informática, evidenciando una marcada homogeneidad léxica sin guía supervisada.

**Panel derecho — LDA (supervisado):**  
Al guiar el modelo para maximizar la separabilidad entre clases, Civil e Informática logran una separación parcial mientras que Ejecución se posiciona de forma claramente aislada. Este contraste explica el rendimiento del SVM (87.58%): la clasificación automática supera la capacidad de la auditoría cualitativa humana para identificar estas diferencias semánticas sutiles.

---

### Gráfico 2 — Mapa de Calor de Similitud Coseno

`public/assets/similitud_centroides.png`

Un **heatmap** que cuantifica la convergencia léxica entre los tres grados:

| Par de Grados | Similitud Coseno | Convergencia |
|---|---|---|
| **Civil ↔ Informática** | **0.768** | **76.8%** |
| Civil ↔ Ejecución | < 0.50 | < 50% |
| Informática ↔ Ejecución | < 0.50 | < 50% |

**Hallazgo:** Existe un **76.8% de convergencia léxica** entre la Ingeniería Civil e Ingeniería en Informática. La Ingeniería de Ejecución es el único grado con una identidad semántica verdaderamente diferenciada.

---

## 🧠 Conclusiones del Estudio

> *"El modelo híbrido demuestra científicamente una alta convergencia semántica en la educación superior TI en Chile. Las Ingenierías Civil e Informática comparten un núcleo léxico casi idéntico (76.8%), sugiriendo que la agregación de terminología de gestión corporativa se utiliza como estrategia de posicionamiento institucional. Por su parte, la Ingeniería de Ejecución se consolida como el único grado con una identidad semántica verdaderamente especializada y pragmática. Este ecosistema analítico automatizado supera las auditorías humanas, pasando de la incertidumbre cualitativa a la certeza cuantitativa."*

### Hallazgos Principales

1. **Alta homogeneidad curricular:** Un 76.8% de convergencia léxica entre la Ingeniería Civil e Ingeniería en Informática confirma que ambos programas comparten un núcleo de competencias declaradas casi idéntico.

2. **Estandarización lingüística como estrategia institucional:** La presencia de terminología de gestión corporativa en los perfiles de egreso de ambas ingenierías sugiere que su incorporación responde a estrategias de posicionamiento y atracción comercial, más que a una especialización técnica genuinamente diferenciada.

3. **Ingeniería de Ejecución como grado diferenciado:** Con una similitud coseno inferior al 50% respecto a los otros dos grados, la Ingeniería de Ejecución mantiene un vocabulario técnico y pragmático verdaderamente especializado, consolidándose como el grado con mayor identidad semántica propia.

4. **La clasificación automática supera la auditoría humana:** El SVM con Kernel RBF (87.58% de Accuracy) demuestra que los métodos computacionales son capaces de identificar fronteras semánticas que la revisión cualitativa manual no puede discernir con igual objetividad y reproducibilidad.

5. **Aporte metodológico:** El pipeline desarrollado constituye un ecosistema analítico reproducible que transforma la incertidumbre cualitativa de las auditorías curriculares manuales en evidencia cuantitativa verificable.

---

## 📁 Estructura del Repositorio

```
Proyecto_Titulo_2026/
│
├── README.md                        ← Este archivo
├── requirements.txt                 ← Dependencias Python
│
├── src/                             ← Pipeline de Ciencia de Datos
│   ├── Fase1_Recoleccion/
│   │   ├── config.py                ← Selectores CSS por IES
│   │   ├── scraper.py               ← Motor híbrido Requests/Selenium
│   │   ├── extractors.py            ← Parser BeautifulSoup
│   │   └── main.py                  ← Orquestador de scraping
│   │
│   ├── Fase2_Procesamiento/
│   │   ├── 01_limpieza_nlp.py       ← Pipeline NLP con spaCy
│   │   └── 02_vectorizacion.py      ← TF-IDF + Z-Score Keyness
│   │
│   ├── Fase3_Analisis/
│   │   ├── 01_clustering.py         ← Análisis de agrupamiento
│   │   ├── 02_analisis_clusters.py  ← Métricas de separabilidad
│   │   ├── 03_validacion.py         ← Similitud coseno de centroides
│   │   └── 04_visualizacion.py      ← Generación de gráficos PNG
│   │
│   └── data/                        ← [Excluido del repo por .gitignore]
│       ├── raw/
│       │   └── perfiles_egreso_raw.csv
│       └── processed/
│           ├── perfiles_egreso_limpio_v1.csv
│           ├── resultados.json
│           ├── proyeccion_pca_vs_lda.png
│           └── similitud_centroides.png
│
└── landing/                         ← Plataforma Web Interactiva (Next.js)
    ├── app/
    │   ├── components/
    │   │   ├── HeroSection.tsx
    │   │   ├── ChallengeSection.tsx
    │   │   ├── MethodologySection.tsx
    │   │   ├── ResultsSection.tsx
    │   │   └── Footer.tsx
    │   └── page.tsx
    └── public/
        └── assets/
            ├── proyeccion_pca_vs_lda.png    ← Gráfico 1: PCA vs LDA
            └── similitud_centroides.png      ← Gráfico 2: Heatmap similitud coseno
```

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/[usuario]/Proyecto_Titulo_2026.git
cd Proyecto_Titulo_2026
```

### 2. Entorno Virtual Python

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
.\venv\Scripts\activate

# Activar (Linux / macOS)
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar modelo NLP de spaCy

```bash
python -m spacy download es_core_news_sm
```

### 5. Ejecutar el Pipeline (en orden)

```bash
# Fase 1: Web Scraping
python src/Fase1_Recoleccion/main.py

# Fase 2: Procesamiento NLP y Vectorización
python src/Fase2_Procesamiento/01_limpieza_nlp.py
python src/Fase2_Procesamiento/02_vectorizacion.py

# Fase 3: Análisis y Benchmark ML
python src/Fase3_Analisis/01_clustering.py
python src/Fase3_Analisis/02_analisis_clusters.py
python src/Fase3_Analisis/03_validacion.py
python src/Fase3_Analisis/04_visualizacion.py
```

> **Reproducibilidad garantizada:** Todos los scripts utilizan `random_state=42`. Los datos procesados se almacenan en `src/data/` (excluido del control de versiones por `.gitignore`).

### 6. Ejecutar la Landing Page (opcional)

```bash
cd landing
npm install
npm run dev
# Disponible en: http://localhost:3000
```

---

## 🛠️ Stack Tecnológico

### Backend / Pipeline ML

| Herramienta | Uso |
|---|---|
| Python 3.13.7 | Lenguaje principal del pipeline |
| spaCy (`es_core_news_sm`) | NLP: tokenización y lematización |
| scikit-learn | TF-IDF, SVM, PCA, LDA, StratifiedKFold |
| imbalanced-learn | SMOTE para balanceo de clases |
| Selenium + BeautifulSoup4 | Web scraping estático y dinámico |
| Pandas / NumPy | Manipulación y análisis de datos |
| Matplotlib / Seaborn | Generación de gráficos de resultados |

### Frontend / Landing Page

| Herramienta | Uso |
|---|---|
| Next.js 15 | Framework React para la plataforma web |
| TypeScript 5.x | Tipado estático |
| Tailwind CSS | Estilos y diseño visual |

---

## 📄 Licencia

Distribuido bajo la licencia [MIT](LICENSE). Los perfiles de egreso corresponden a información de acceso público disponible en sitios web institucionales de educación superior chilena.

---

<div align="center">

**Universidad de las Américas · Facultad de Ingeniería y Negocios · 2026**

*Brayan Pineda Poblete · Walter Reyes Silva*

</div>