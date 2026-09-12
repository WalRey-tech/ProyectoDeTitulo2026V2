# Descubrimiento de patrones en perfiles de egreso de Informática mediante Machine Learning

**Proyecto de Título · Universidad de las Américas · Facultad de Ingeniería y Negocios · 2026**

**Autores:** Brayan Pineda Poblete · Walter Reyes Silva

---

## Descripción

Este proyecto analiza perfiles de egreso de carreras del área de Informática impartidas por Instituciones de Educación Superior chilenas, utilizando técnicas de procesamiento de lenguaje natural, análisis estadístico y Machine Learning.

El objetivo es estudiar hasta qué punto los perfiles de egreso presentan patrones semánticos diferenciables entre tres tipos de grado:

- Ingeniería Civil Informática.
- Ingeniería en Informática.
- Ingeniería de Ejecución en Informática o denominaciones equivalentes consideradas en el corpus.

El proyecto fue desarrollado como un pipeline reproducible dividido en tres etapas:

1. **Recolección y etiquetado de perfiles de egreso.**
2. **Preparación, control de calidad y auditoría del corpus.**
3. **Análisis semántico, selección de características y validación predictiva.**

---

## Corpus científico

El corpus oficial utilizado en los experimentos contiene:

| Grado | Perfiles |
|---|---:|
| Ingeniería Civil Informática | 31 |
| Ingeniería en Informática | 25 |
| Ingeniería de Ejecución | 5 |
| **Total** | **61** |

Durante la auditoría de datos se detectaron problemas de codificación de caracteres en dos perfiles.

Estos registros fueron corregidos mediante un procedimiento controlado que verificó:

- conservación de los 61 registros;
- conservación de la distribución de clases;
- ausencia de perfiles vacíos;
- ausencia de duplicados exactos;
- ausencia de grados fuera del catálogo;
- ausencia de problemas de encoding luego de la corrección.

La corrección se realizó antes de la ejecución científica final.

---

# Metodología

## Fase 1 — Recolección

La primera fase implementa un sistema de extracción de perfiles de egreso mediante:

- `requests`;
- BeautifulSoup;
- Selenium como mecanismo alternativo para páginas dinámicas;
- selectores específicos cuando están disponibles;
- extracción contextual cuando no existe un selector confiable.

El scraper incorpora controles de seguridad y trazabilidad:

- uso obligatorio de HTTPS;
- validación de destinos;
- rechazo de hosts locales o direcciones privadas;
- validación TLS;
- control de redirecciones;
- límites de tamaño de respuesta;
- timeouts;
- identificación estable del cliente;
- frecuencia reducida de solicitudes;
- revisión de `robots.txt`;
- registro del método y estrategia de extracción;
- separación entre registros correctos, registros que requieren revisión y errores.

La recolección en vivo se mantiene separada del corpus científico utilizado para los experimentos, evitando que cambios posteriores en sitios web modifiquen automáticamente los resultados reportados.

### Demostración

Para una ejecución breve:

```bash
python ejecutar_fase1.py --modo demo
```

Para ejecutar la recolección configurada completa:

```bash
python ejecutar_fase1.py --modo completo
```

---

## Fase 2 — Preparación y auditoría del corpus

La segunda fase no realiza clasificación ni selección de características.

Su función es verificar la calidad estructural y textual del corpus antes de utilizarlo en los experimentos.

Ejecutar:

```bash
python src/Fase2_Procesamiento/01_preparacion_corpus.py
```

Entre las comprobaciones realizadas se encuentran:

- cantidad de registros;
- distribución de clases;
- perfiles vacíos;
- grados inválidos;
- anomalías de encoding;
- perfiles duplicados;
- duplicados universidad-carrera;
- URLs duplicadas;
- longitud de los textos;
- normalización formal no destructiva;
- verificación de integridad mediante SHA-256.

También se implementó:

```bash
python src/Fase2_Procesamiento/02_corregir_encoding_v3.py
```

Este script fue desarrollado para realizar una corrección controlada de problemas de codificación detectados durante la auditoría, sin modificar silenciosamente el corpus de entrada.

La corrección permitió identificar y reparar dos registros afectados manteniendo inalterados el número de perfiles y sus etiquetas.

---

# Fase 3 — Análisis científico

La Fase 3 contiene el pipeline principal de análisis.

Puede ejecutarse desde:

```bash
python ejecutar_fase3.py
```

Por defecto se utiliza el modo:

```bash
python ejecutar_fase3.py --modo validar
```

Este modo reutiliza la selección GWO existente.

Para reproducir también la optimización Grey Wolf Optimizer desde cero:

```bash
python ejecutar_fase3.py --modo completo
```

El pipeline ejecuta siete etapas:

```text
01. PCA / LDA
02. Homogeneidad y significancia semántica
03. Diferenciación léxica
04. Selección de características mediante GWO
05. Validación exploratoria de GWO
06. Modelo final robusto
07. Reporte científico consolidado
```

---

# Representación textual

Los perfiles son representados mediante **TF-IDF** con:

- máximo de 400 características;
- unigramas y bigramas;
- eliminación de términos de muy baja frecuencia;
- lista de stopwords en español.

Se utilizan representaciones distintas según el objetivo del análisis.

La representación predictiva y la representación utilizada para interpretación léxica no deben confundirse: el ranking de términos distintivos constituye un análisis interpretativo y no equivale directamente al espacio utilizado por el modelo final.

---

# Homogeneidad semántica

El análisis de similitud coseno permite estudiar si los perfiles pertenecientes a un mismo grado son semánticamente más similares entre sí que respecto de perfiles de otros grados.

Resultados finales:

| Métrica | Resultado |
|---|---:|
| Similitud intra-clase media | **0.2370** |
| Similitud inter-clase media | **0.2165** |
| Δ intra - inter | **0.0205** |
| Permutaciones | **5000** |
| p-valor | **0.004799** |

El test de permutación detecta una estructura semántica estadísticamente significativa asociada al tipo de grado.

Sin embargo, la magnitud de la diferencia entre similitud intra e inter clase es pequeña.

Por tanto, el resultado se interpreta como evidencia de **diferenciación semántica moderada con un solapamiento importante entre los perfiles**.

---

# PCA y LDA

Se generan visualizaciones exploratorias mediante PCA y LDA.

Para PCA:

| Componente | Varianza explicada |
|---|---:|
| PC1 | **5.03 %** |
| PC2 | **4.56 %** |
| Total representación 2D | **9.59 %** |

PCA es una técnica no supervisada y la representación bidimensional explica solamente una fracción reducida de la variabilidad total.

Por esta razón, la figura PCA no se utiliza como prueba de separabilidad entre grados.

LDA es una técnica supervisada y se utiliza únicamente como herramienta exploratoria de visualización.

No constituye una estimación del rendimiento de generalización del clasificador.

---

# Selección de características con Grey Wolf Optimizer

Se utiliza **Grey Wolf Optimizer (GWO)** para buscar un subconjunto de características TF-IDF con alto poder discriminativo.

Configuración principal:

```text
Características iniciales : 400
Épocas                    : 100
Población                  : 30
Semilla                    : 42
Fitness                    : F1-macro
Clasificador               : Complement Naive Bayes
Balanceo                   : SMOTE
CV interna                 : 3 folds
```

Resultado:

| Métrica | Resultado |
|---|---:|
| Características iniciales | 400 |
| Características seleccionadas | **249** |
| Reducción | **37.75 %** |
| F1-macro exploratorio 5-fold | **0.8412** |
| Desviación estándar | **0.2179** |

El resultado GWO debe interpretarse como **exploratorio**.

El vectorizador TF-IDF y la selección de características preceden a la comparación externa de 5 folds. Por ello, el valor `0.8412` no se presenta como una estimación no sesgada de generalización.

GWO demuestra que existe un subconjunto compacto y altamente discriminativo, pero no reemplaza al protocolo robusto utilizado como resultado predictivo final.

---

# Diagnóstico 10-fold

También se conserva una ejecución de 10 folds para reproducir y analizar la sensibilidad del experimento GWO.

Resultados:

| Métrica | Resultado |
|---|---:|
| F1 histórico de referencia | **0.7316** |
| F1 con tres clases fijas | **0.6337** |
| Folds con las tres clases | **5 / 10** |

La clase Ingeniería de Ejecución contiene solamente cinco perfiles.

Por esta razón, una validación de 10 folds no puede incluir las tres clases en todos los conjuntos de prueba.

Este resultado se conserva como **diagnóstico de sensibilidad y reproducibilidad**, no como estimación final de generalización.

---

# Control de fuga léxica

Durante la revisión metodológica se observó que algunas características seleccionadas por GWO incluyen términos directamente relacionados con la denominación del grado, por ejemplo:

```text
ingeniería
informática
ingeniero
civil
```

Estos términos pueden facilitar artificialmente la predicción de la etiqueta.

Por esta razón se incorporó una evaluación adicional donde las denominaciones directas del grado son controladas antes del entrenamiento.

En la ejecución final:

```text
Documentos modificados      : 35 / 61
Denominaciones enmascaradas : 78
```

Esto permite separar el poder discriminativo producido por pistas explícitas del nombre de la carrera de la diferenciación semántica más general contenida en los perfiles.

---

# Modelo robusto final

El resultado predictivo principal del proyecto corresponde al protocolo robusto con:

- Complement Naive Bayes;
- control de denominaciones directamente asociadas a la etiqueta;
- TF-IDF ajustado dentro del entrenamiento de cada fold;
- SMOTE aplicado únicamente sobre los datos de entrenamiento;
- Stratified 5-Fold Cross Validation;
- F1-macro como métrica principal.

Resultados:

| Métrica | Resultado |
|---|---:|
| F1-macro medio | **0.5742** |
| Desviación estándar | **0.2019** |
| Accuracy media | **0.6551** |
| F1-macro OOF | **0.6040** |
| Accuracy OOF | **0.6557** |

Los cinco resultados F1-macro fueron:

```text
Fold 1 : 0.6444
Fold 2 : 0.7660
Fold 3 : 0.4051
Fold 4 : 0.7381
Fold 5 : 0.3175
```

El valor:

```text
F1-macro = 0.5742 ± 0.2019
```

se considera la **estimación principal y conservadora de generalización** del proyecto.

---

# Interpretación de los resultados

El proyecto utiliza diferentes experimentos para responder preguntas distintas.

No deben interpretarse todas las métricas como si pertenecieran al mismo protocolo.

| Resultado | Interpretación |
|---|---|
| **0.8412** | Resultado exploratorio utilizando características seleccionadas por GWO |
| **0.7316** | Reproducción histórica del diagnóstico 10-fold |
| **0.6337** | Diagnóstico 10-fold utilizando tres clases fijas |
| **0.5742 ± 0.2019** | Estimación robusta principal de generalización |

La conclusión general es que:

> Los perfiles de egreso presentan diferencias semánticas asociadas al tipo de grado, pero estas diferencias son moderadas y existe un solapamiento importante entre las clases. GWO identifica una representación compacta y altamente discriminativa, aunque parte de esa capacidad está asociada a términos directamente vinculados con las denominaciones de los grados. Al controlar esas pistas y aplicar un protocolo más estricto de validación, el rendimiento disminuye y entrega una estimación más conservadora de la diferenciación real entre los perfiles.

---

# Estructura del repositorio

```text
Proyecto_Titulo_2026/
│
├── README.md
├── requirements.txt
│
├── ejecutar_fase1.py
├── ejecutar_fase3.py
│
├── src/
│   │
│   ├── Fase1_Recoleccion/
│   │   ├── config.py
│   │   ├── etiquetador.py
│   │   ├── extractors.py
│   │   ├── main.py
│   │   ├── scraper.py
│   │   └── utils.py
│   │
│   ├── Fase2_Procesamiento/
│   │   ├── 01_preparacion_corpus.py
│   │   ├── 02_corregir_encoding_v3.py
│   │   └── legacy/
│   │
│   ├── Fase3_Analisis/
│   │   ├── 01_proyeccion_pca_lda.py
│   │   ├── 02_homogeneidad_significancia.py
│   │   ├── 03_diferenciacion_lexica.py
│   │   ├── 04_seleccion_caracteristicas_gwo.py
│   │   ├── 05_validacion_gwo.py
│   │   ├── 06_modelo_final_robusto.py
│   │   ├── 07_generar_reporte.py
│   │   ├── auditorias/
│   │   └── legacy/
│   │
│   └── data/
│       ├── processed/
│       │   └── perfiles_egreso_etiquetado_v2.csv
│       │
│       └── resultados_cientificos/
│
└── landing/
```

Los directorios `legacy/` conservan implementaciones anteriores con fines de trazabilidad, pero no forman parte del pipeline científico oficial.

La carpeta `landing/` corresponde a una interfaz complementaria del proyecto y no constituye la fuente oficial de las métricas científicas.

---

# Instalación

El entorno utilizado para la ejecución final fue **Python 3.12**.

Crear y activar un entorno virtual:

```bash
python -m venv .venv-gwo
```

En Windows PowerShell:

```powershell
.\.venv-gwo\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Ejecución recomendada

## 1. Demostración de recolección

```bash
python ejecutar_fase1.py --modo demo
```

## 2. Auditoría del corpus

```bash
python src/Fase2_Procesamiento/01_preparacion_corpus.py
```

## 3. Validación del análisis existente

```bash
python ejecutar_fase3.py --modo validar
```

## 4. Reproducción completa incluyendo GWO

```bash
python ejecutar_fase3.py --modo completo
```

---

# Tecnologías principales

| Área | Tecnologías |
|---|---|
| Lenguaje | Python 3.12 |
| Recolección | Requests, BeautifulSoup, Selenium |
| Procesamiento | Pandas, NumPy, ftfy |
| NLP | TF-IDF, unigramas y bigramas |
| Machine Learning | Scikit-learn, Complement Naive Bayes |
| Balanceo | imbalanced-learn / SMOTE |
| Optimización | Mealpy, Grey Wolf Optimizer |
| Estadística | Validación cruzada, test de permutación, similitud coseno |
| Reducción / visualización | PCA, LDA, Matplotlib |
| Frontend complementario | Next.js / TypeScript |

---

# Reproducibilidad y consideraciones metodológicas

Para facilitar la reproducibilidad:

- se utiliza una semilla fija (`42`) en los experimentos principales;
- el corpus científico se mantiene versionado;
- se registran resultados intermedios y finales;
- GWO puede reproducirse mediante el modo `completo`;
- SMOTE se mantiene dentro del entrenamiento en el protocolo robusto;
- el resultado GWO se reporta explícitamente como exploratorio;
- el resultado 10-fold se reporta como diagnóstico;
- el modelo robusto se utiliza como estimación predictiva principal;
- las visualizaciones PCA/LDA no se interpretan como métricas de generalización.

Debido al reducido número de ejemplos de Ingeniería de Ejecución (`n = 5`), los resultados asociados a la clase minoritaria deben interpretarse con especial cautela.

---

## Autores

**Brayan Pineda Poblete**  
**Walter Reyes Silva**

Proyecto de Título  
Universidad de las Américas  
2026