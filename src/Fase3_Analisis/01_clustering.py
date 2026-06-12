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

# IMPORTACIÓN DE LIBRERÍAS
import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns



# CONFIGURACIÓN DE RUTAS INTELIGENTES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_LIMPIOS = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_limpios.csv")
)

RUTA_VECTORES = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "vectores_perfiles.npy")
)

RUTA_GRAFICO = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "mapa_clusters.png")
)

RUTA_FINAL = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv")
)


if __name__ == "__main__":

    # 1. CARGA DE DATOS
    print(f"Verificando archivos base en {os.path.dirname(RUTA_LIMPIOS)}...")

    if not os.path.exists(RUTA_LIMPIOS):
        print("Error: No se encontró perfiles_limpios.csv.")
        print("Ejecuta primero: python .\\01_limpieza_nlp.py")
        exit()

    if not os.path.exists(RUTA_VECTORES):
        print("Error: No se encontró vectores_perfiles.npy.")
        print("Ejecuta primero: python .\\02_vectorizacion.py")
        exit()

    print("Cargando dataset limpio...")
    df = pd.read_csv(
        RUTA_LIMPIOS,
        sep=";",
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    if "perfil_final" not in df.columns:
        print("Error: No existe la columna 'perfil_final'.")
        print("Columnas disponibles:")
        print(df.columns.tolist())
        exit()

    # Mismo filtro usado en la vectorización
    df["perfil_final"] = df["perfil_final"].fillna("").astype(str)
    df = df[df["perfil_final"].str.strip() != ""].reset_index(drop=True)

    print("Cargando matriz de vectores...")
    vectores = np.load(RUTA_VECTORES)

    print(f"Filas válidas en metadata: {len(df)}")
    print(f"Vectores cargados: {len(vectores)}")

    if len(df) != len(vectores):
        print("Error: El número de registros no coincide con la matriz de vectores.")
        print(f"Filas metadata: {len(df)}")
        print(f"Vectores: {len(vectores)}")
        print("Regenera primero la vectorización con 02_vectorizacion.py.")
        exit()


    # 2. ALGORITMO DE CLUSTERING K-MEANS
    NUM_CLUSTERS = 4

    print(f"\nAplicando algoritmo K-Means para descubrir {NUM_CLUSTERS} grupos ocultos...")

    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = kmeans.fit_predict(vectores)

    # 3. REDUCCIÓN DE DIMENSIONALIDAD PCA
    print("Comprimiendo 384 dimensiones a 2D para visualización gráfica...")

    pca = PCA(
        n_components=2,
        random_state=42
    )

    vectores_2d = pca.fit_transform(vectores)

    df["Coordenada_X"] = vectores_2d[:, 0]
    df["Coordenada_Y"] = vectores_2d[:, 1]

    print(f"Varianza explicada por PCA: {pca.explained_variance_ratio_}")

    # 4. VISUALIZACIÓN DE RESULTADOS
    print("Generando gráfico de dispersión...")

    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    sns.scatterplot(
        x="Coordenada_X",
        y="Coordenada_Y",
        hue="Cluster",
        palette="viridis",
        data=df,
        s=100,
        alpha=0.8
    )

    plt.title(
        "Mapa Semántico de Perfiles de Egreso Informáticos en Chile",
        fontsize=16,
        pad=20
    )

    plt.xlabel("Componente Principal 1", fontsize=12)
    plt.ylabel("Componente Principal 2", fontsize=12)
    plt.legend(title="Cluster")
    plt.tight_layout()

    # 5. GUARDADO DE RESULTADOS
    os.makedirs(os.path.dirname(RUTA_GRAFICO), exist_ok=True)

    plt.savefig(
        RUTA_GRAFICO,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nGráfico guardado en: {RUTA_GRAFICO}")

    df.to_csv(
        RUTA_FINAL,
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    print(f"Dataset analítico guardado en: {RUTA_FINAL}")

    # ==========================================
    # 6. RESUMEN POR CONSOLA
    # ==========================================
    print("\n" + "=" * 50)
    print("=== RESUMEN DE PROGRAMAS POR CLUSTER ===")
    print(df["Cluster"].value_counts().sort_index())
    print("=" * 50)

    print("\nFASE 3: CLUSTERING FINALIZADO CORRECTAMENTE")