"""
01_CLUSTERING.PY
-----------------------------------------------------------------------
Fase: 3. Análisis
Propósito: 
    Utilizar algoritmos de Machine Learning (aprendizaje no supervisado) para 
    descubrir agrupaciones (clusters) naturales en los perfiles de egreso.
    Se utiliza K-Means para el agrupamiento matemático y PCA para la reducción 
    de dimensionalidad y visualización gráfica.
"""

# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import pandas as pd
import numpy as np

# Scikit-Learn: Librería estándar de Machine Learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Matplotlib y Seaborn: Librerías para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN DE RUTAS INTELIGENTES
# ==========================================
# Ubicamos la carpeta actual (Fase3_Analisis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas apuntando al Data Lake central (src/data/processed/)
RUTA_LIMPIOS = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_limpios.csv"))
RUTA_VECTORES = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "vectores_perfiles.npy"))

# Rutas de salida
RUTA_GRAFICO = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "mapa_clusters.png"))
RUTA_FINAL = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv"))

if __name__ == "__main__":
    # ==========================================
    # 1. CARGA DE DATOS (Texto + Vectores)
    # ==========================================
    print(f"📂 Verificando archivos base en {os.path.dirname(RUTA_LIMPIOS)}...")
    
    if not os.path.exists(RUTA_LIMPIOS) or not os.path.exists(RUTA_VECTORES):
        print("❌ Error: Faltan archivos de la Fase 2. Ejecuta '01_limpieza_nlp.py' y '02_vectorizacion.py' primero.")
        exit()

    print("Cargando matriz de vectores y dataset limpio...")
    df = pd.read_csv(RUTA_LIMPIOS, sep=';', encoding='utf-8')
    # Aseguramos que tenga la misma longitud que la matriz eliminando nulos
    df = df.dropna(subset=['perfil_final']).reset_index(drop=True)

    # Cargamos el espacio multidimensional generado por el Transformer
    vectores = np.load(RUTA_VECTORES)

    # ==========================================
    # 2. ALGORITMO DE CLUSTERING (K-MEANS)
    # ==========================================
    # Definimos cuántos grupos queremos buscar (4 perfiles detectados).
    NUM_CLUSTERS = 4 

    print(f"\n🧠 Aplicando algoritmo K-Means para descubrir {NUM_CLUSTERS} grupos ocultos...")
    # Inicializamos el modelo. random_state=42 asegura que el resultado sea reproducible.
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)

    # Entrenamos el modelo con nuestros vectores.
    df['Cluster'] = kmeans.fit_predict(vectores)

    # ==========================================
    # 3. REDUCCIÓN DE DIMENSIONALIDAD (PCA)
    # ==========================================
    print("📉 Comprimiendo 384 dimensiones a 2D para visualización gráfica...")
    # PCA (Principal Component Analysis) busca los ángulos con mayor varianza
    pca = PCA(n_components=2, random_state=42)
    vectores_2d = pca.fit_transform(vectores)

    # Guardamos estas coordenadas 2D en nuestro DataFrame
    df['Coordenada_X'] = vectores_2d[:, 0]
    df['Coordenada_Y'] = vectores_2d[:, 1]

    # ==========================================
    # 4. VISUALIZACIÓN DE LOS RESULTADOS (GRÁFICO)
    # ==========================================
    print("🎨 Generando gráfico de dispersión (Scatter Plot)...")
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Dibujamos los puntos
    grafico = sns.scatterplot(
        x='Coordenada_X', 
        y='Coordenada_Y',
        hue='Cluster',
        palette='viridis',
        data=df,
        s=100,
        alpha=0.8
    )

    plt.title('Mapa Semántico de Perfiles de Egreso (Informática en Chile)', fontsize=16, pad=20)
    plt.xlabel('Componente Principal 1 (Varianza Semántica)', fontsize=12)
    plt.ylabel('Componente Principal 2 (Varianza Semántica)', fontsize=12)
    plt.legend(title='Grupo (Cluster)')

    # ==========================================
    # 5. ALMACENAMIENTO DE RESULTADOS
    # ==========================================
    os.makedirs(os.path.dirname(RUTA_GRAFICO), exist_ok=True)
    
    # Guardar imagen
    plt.savefig(RUTA_GRAFICO, dpi=300, bbox_inches='tight')
    print(f"\n📸 Gráfico guardado en: {RUTA_GRAFICO}")

    # Guardar dataset con clusters asignados
    df.to_csv(RUTA_FINAL, index=False, sep=';', encoding='utf-8')
    print(f"💾 Dataset analítico guardado en: {RUTA_FINAL}")

    # Resumen por consola
    print("\n" + "="*40)
    print("=== RESUMEN DE UNIVERSIDADES POR CLUSTER ===")
    print(df['Cluster'].value_counts().sort_index())
    print("="*40)