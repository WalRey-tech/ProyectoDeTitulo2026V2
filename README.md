Análisis Semántico de Perfiles de Egreso en Informática
Resumen del Proyecto
Investigación basada en aprendizaje no supervisado (Machine Learning) para la detección de patrones y convergencia semántica en la oferta académica de la educación superior en Chile. El sistema procesa perfiles de egreso de diversas instituciones mediante Procesamiento de Lenguaje Natural (NLP) para clasificar y visualizar la distribución de competencias en el mercado actual.

Arquitectura del Sistema
El proyecto se divide en fases modulares para garantizar la integridad y trazabilidad de los datos. Cada fase opera de forma independiente y se comunica mediante un repositorio de datos centralizado ubicado en src/data/.

Fase 1: Recolección y Extracción (Web Scraping)
Se implementa un motor de scraping utilizando BeautifulSoup y Selenium para la extracción automatizada de descriptores de perfiles de egreso desde portales institucionales.

Entrada: Configuración de sitios y selectores CSS.

Salida: Dataset en bruto (perfiles_egreso_raw.csv).

Fase 2: Procesamiento de Lenguaje Natural (NLP)
Aplicación de una arquitectura de limpieza en dos capas:

Capa Estructural: Normalización de texto mediante expresiones regulares (Regex).

Capa Lingüística: Uso de modelos Transformer (spaCy) para lematización y eliminación de ruido semántico (Stopwords académicas).

Vectorización: Generación de Embeddings semánticos de 384 dimensiones mediante el modelo paraphrase-multilingual-MiniLM-L12-v2.

Fase 3: Análisis y Modelado Matemático
Uso de algoritmos de Machine Learning para la interpretación de datos:

Clustering: Aplicación de K-Means para la agrupación de perfiles.

Reducción de Dimensionalidad: Implementación de PCA (Principal Component Analysis) para proyectar el espacio multidimensional en un mapa semántico 2D.

Validación: Evaluación de cohesión grupal mediante el Coeficiente de Silueta (Silhouette Score).

Fase 4: Integración y Visualización
Consolidación de resultados en un esquema JSON maestro preparado para su migración a bases de datos relacionales (PostgreSQL) y visualización en entorno web (Next.js).

Instrucciones de Instalación y Despliegue
Entorno Virtual:

Bash
python -m venv venv
source venv/bin/activate  # En Windows: .\venv\Scripts\activate
Dependencias:

Bash
pip install -r requirements.txt
Modelos de Lenguaje:
Es indispensable descargar el modelo de spaCy para el procesamiento en español:

Bash
python -m spacy download es_core_news_sm
Ejecución del Pipeline
Para generar los resultados locales, los scripts deben ejecutarse en el orden numérico establecido dentro de cada carpeta de fase en src/. Los datos generados se almacenarán automáticamente en src/data/, carpeta que se encuentra excluida del repositorio según el archivo .gitignore para mantener la limpieza del control de versiones.