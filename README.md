# Proyecto_Titulo_2026
**Proyecto Título: Descubrimiento de patrones en perfiles de Egreso de Informática mediante aprendizaje no supervisado.**

---

## 🏗️ Estado Actual y Arquitectura
Recientemente, el proyecto fue reestructurado por completo para pasar de scripts sueltos a un **Pipeline de Datos Industrial**. 

Se crearon 4 fases aisladas con **rutas absolutas**. Esto significa que ahora hay un Data Lake centralizado (`src/data/`) que se genera automáticamente al ejecutar el código. La IA lingüística (spaCy) y la IA matemática (Scikit-Learn) ahora trabajan en cadena hasta generar un archivo `master_data.json`, el cual será la base para nuestra futura aplicación web.

---

## ⚙️ 1. Preparación del Entorno (Instalación)
Antes de ejecutar cualquier código, debes preparar tu entorno local:

1. **Clona el repositorio** y abre la terminal en la raíz del proyecto.
2. **Crea y activa el entorno virtual:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate

Instala las dependencias:
pip install -r requirements.txt

Descarga el modelo de Inteligencia Artificial Lingüística (Obligatorio):
python -m spacy download es_core_news_sm

2. Guía de Ejecución Paso a Paso
Debido a que ignoramos la carpeta de datos en GitHub (.gitignore), debes ejecutar el proyecto en este orden estricto para generar los archivos locales.

Fase 1: Recolección
Entra a la carpeta y ejecuta el web scraper para extraer los datos de las universidades.

cd src/Fase1_Recoleccion
python main.py

Resultado: Se creará la carpeta src/data/raw/ con el archivo perfiles_egreso_raw.csv.

Fase 2: Procesamiento NLP
Vuelve a src y entra a la Fase 2. Aquí limpiamos el texto y lo convertimos a matemáticas.
cd ../Fase2_Procesamiento

python 01_limpieza_nlp.py
python 02_vectorizacion.py

Resultado: Se creará la carpeta src/data/processed/ con el texto limpio y los vectores .npy.

Fase 3: Análisis (Machine Learning)
Vuelve a src y entra a la Fase 3. Ejecuta los scripts en orden numérico para descubrir los clusters, generar los gráficos y empaquetar el JSON final.
cd ../Fase3_Analisis
python 01_clustering.py
python 02_analisis_clusters.py
python 03_validacion_clusters.py
python 04_reporte_distribucion.py
python 05_visualizacion_interactiva.py
python 06_preparacion_migracion.py

Resultado: Tendrás gráficos analíticos, reportes de silueta y el archivo master_data.json listo para usar.

Próximos Pasos (Fase 4)
La carpeta Fase4_Visualizacion actualmente se encuentra vacía. Nuestro siguiente objetivo es inicializar Next.js, configurar PostgreSQL y crear una Landing Page interactiva con scroll que consuma el master_data.json generado en la Fase 3.