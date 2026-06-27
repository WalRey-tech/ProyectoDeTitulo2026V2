# Descubrimiento de patrones en perfiles de Egreso de Informática mediante Machine Learning

Ecosistema analítico automatizado para la validación estadística y clasificación supervisada de mallas curriculares en la educación superior chilena.

Proyecto de Título · Universidad de las Américas · Facultad de Ingeniería y Negocios · 2026

---

## 1. Resumen Ejecutivo

Este proyecto implementa un pipeline de Ciencia de Datos para el análisis semántico de perfiles de egreso de informática en Chile. Mediante técnicas de Procesamiento de Lenguaje Natural (NLP), balanceo sintético de datos (SMOTE) y un benchmark competitivo de 11 algoritmos de Machine Learning, el estudio cuantifica el nivel de homogeneidad y diferenciación curricular entre tres grados académicos.

### Métricas e Indicadores Clave
* **Muestra final depurada:** 64 perfiles de egreso reales (32 Civil, 25 Informática, 7 Ejecución) de 47 instituciones.
* **Algoritmo óptimo detectado:** Random Forest.
* **Rendimiento del modelo:** 92.79% de Accuracy (Validación Cruzada Estratificada de 5 pliegues).
* **Solapamiento semántico (Civil ↔ Informática):** 76.63% (Similitud Coseno: 0.7663).
* **Significancia estadística:** p-valor = 0.0000 (Hipótesis nula rechazada mediante Test de Permutación).

---

## 2. Estructura del Repositorio
Proyecto_Titulo_2026/
│
├── README.md                        # Descripción del proyecto
├── requirements.txt                 # Dependencias del entorno Python
│
├── src/                             # Pipeline de Ciencia de Datos
│   ├── Fase1_Recoleccion/           # Web Scraping híbrido (Requests/Selenium)
│   ├── Fase2_Procesamiento/         # Capa de limpieza lingüística (spaCy)
│   ├── Fase3_Analisis/              # Algoritmos, validación y reportabilidad
│   └── data/                        # Almacenamiento local de artefactos y datasets
│
└── landing/                         # Plataforma Web Interactiva
├── app/                         # Componentes de la interfaz de usuario
└── public/                      # Recursos estáticos y gráficos generados
---

## 3. Instalación y Ejecución

### Configuración del Entorno Virtual
```bash
# Clonar y acceder al directorio
git clone [https://github.com/](https://github.com/)[usuario]/Proyecto_Titulo_2026.git
cd Proyecto_Titulo_2026

# Instalar y activar el entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: .\venv\Scripts\activate

# Instalar dependencias del sistema y el modelo lingüístico
pip install -r requirements.txt
python -m spacy download es_core_news_sm
# 1. Extracción y normalización de perfiles
python src/Fase1_Recoleccion/main.py
python src/Fase2_Procesamiento/01_limpieza_nlp.py

# 2. Análisis de Machine Learning y visualización técnica
python src/Fase3_Analisis/01_separabilidad_supervisada.py
python src/Fase3_Analisis/02_proyeccion_lda.py
python src/Fase3_Analisis/03_homogeneidad_significancia.py
python src/Fase3_Analisis/04_diferenciacion_lexica.py
python src/Fase3_Analisis/05_generar_reporte.py
cd landing
npm install
npm run dev

4. Stack Tecnológico
Core Backend & Machine Learning: Python 3.13.7, scikit-learn, imbalanced-learn, spaCy (es_core_news_sm), Pandas, NumPy, joblib.

Gráficos e Informes: Matplotlib, Seaborn, JSON Serializer.

Frontend & Visualización B2B: Next.js 15, TypeScript 5.x, Tailwind CSS.

5. Conclusiones Principales
Convergencia curricular crítica: El solapamiento del 76.63% entre las Ingenierías Civil e Informática demuestra matemáticamente que comparten un núcleo de competencias casi idéntico en la oferta académica nacional.

Identidad de la Ingeniería de Ejecución: Este grado se consolida de forma aislada en las proyecciones (similitudes inferiores al 54%), manteniendo un vocabulario técnico orientado estrictamente a la eficiencia operacional y sistemas de información.

Validación metodológica: El modelo de producción basado en Random Forest (92.79% de precisión) y el p-valor obtenido demuestran que las auditorías curriculares automatizadas superan la subjetividad de la revisión cualitativa humana.

Universidad de las Américas · Facultad de Ingeniería y Negocios · 2026 Autores: Brayan Pineda Poblete · Walter Ignacio Reyes Silva