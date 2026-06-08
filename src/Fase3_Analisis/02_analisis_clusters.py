"""
02_ANALISIS_CLUSTERS.PY
-----------------------------------------------------------------------
Fase: 3. Análisis
Propósito: 
    Interpretar los resultados matemáticos del agrupamiento (K-Means).
    Dado que el algoritmo agrupó las carreras basándose en vectores numéricos 
    incomprensibles para el humano, este script utiliza la técnica TF-IDF para 
    "traducir" esos grupos nuevamente a lenguaje natural, extrayendo las palabras 
    clave más representativas y distintivas de cada uno.

    Esto nos permite bautizar cada cluster empíricamente (ej. Perfil Orientado a Datos,
    Perfil Orientado a Ciberseguridad, Perfil de Desarrollo de Software, etc.).
"""

# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import pandas as pd
# TfidfVectorizer: Transforma texto en una matriz evaluando la importancia de cada palabra.
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# CONFIGURACIÓN DE RUTAS INTELIGENTES
# ==========================================
# Ubicamos la carpeta actual (Fase3_Analisis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta apuntando al Data Lake central donde el clustering guardó sus resultados
RUTA_ARCHIVO = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv"))

if __name__ == "__main__":
    # ==========================================
    # 1. CARGA DE DATOS AGRUPADOS
    # ==========================================
    print(f"📂 Buscando resultados de clustering en: {RUTA_ARCHIVO}")
    
    if not os.path.exists(RUTA_ARCHIVO):
        print("❌ Error: No se encontró el archivo de clusters. Ejecuta '01_clustering.py' primero.")
        exit()
        
    df = pd.read_csv(RUTA_ARCHIVO, sep=';', encoding='utf-8-sig')

    # Detectamos automáticamente cuántos clusters únicos creó el modelo
    num_clusters = df['Cluster'].nunique()

    # ==========================================
    # 2. CONFIGURACIÓN DEL EXTRACTOR DE PALABRAS (TF-IDF)
    # ==========================================
    # TF-IDF (Term Frequency - Inverse Document Frequency) es un cálculo estadístico.
    # Premia las palabras que se repiten mucho dentro de un grupo específico, 
    # pero castiga (ignora) las palabras que son muy comunes en todos los grupos a la vez.
    # max_features=8 le indica que solo extraiga las 8 palabras más "fuertes" de cada cluster.
    vectorizador = TfidfVectorizer(max_features=8)

    print("\n" + "="*60)
    print("🧠 ANÁLISIS SEMÁNTICO Y COMPOSICIÓN DE CLUSTERS")
    print("="*60)

    # ==========================================
    # 3. ANÁLISIS ITERATIVO POR GRUPO (CLUSTER)
    # ==========================================
    for i in range(num_clusters):
        # Filtramos el DataFrame para aislar solo las carreras de este cluster (i)
        filtro_cluster = df[df['Cluster'] == i]
        textos_cluster = filtro_cluster['perfil_final']
        
        # Verificamos que el cluster no esté vacío por precaución
        if not textos_cluster.empty:
            
            # 3.1. Extracción de Palabras Clave
            # Ajustamos el modelo TF-IDF a los textos de este cluster en particular
            matriz = vectorizador.fit_transform(textos_cluster)
            
            # Obtenemos el texto de las palabras que "ganaron" mayor puntaje
            palabras_clave = vectorizador.get_feature_names_out()
            
            # 3.2. Selección de Ejemplos Representativos
            # Concatenamos la Institución y la Carrera para tener contexto completo
            # Ej: "Duoc UC (Ingeniería en Informática)"
            instituciones_y_carreras = filtro_cluster['institucion_nombre'] + " (" + filtro_cluster['carrera_nombre'] + ")" \
                if 'institucion_nombre' in filtro_cluster.columns else filtro_cluster['universidad'] + " (" + filtro_cluster['carrera'] + ")"
            
            # Tomamos una muestra aleatoria para evidenciar la variedad
            cantidad_muestra = min(7, len(instituciones_y_carreras))
            ejemplos = instituciones_y_carreras.sample(n=cantidad_muestra, random_state=42).tolist()
            
            # 3.3. Impresión de Resultados en Consola
            print(f"📦 [CLUSTER {i}] - {len(textos_cluster)} programas formativos")
            print(f"🔑 Top 8 Palabras Clave : {', '.join(palabras_clave)}")
            print("🎓 Muestra de Ejemplos  :")
            for ej in ejemplos:
                print(f"   - {ej}")
            print("-" * 60)
            
    print("✅ Análisis finalizado. Revisa los resultados para 'bautizar' cada cluster.")