import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_GRAFICO_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "proyeccion_pca_vs_lda.png"))

def main():
    print(" Cargando dataset limpio...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f" Error: No se encontró {RUTA_ENTRADA}.")
        return

    # Limpieza de seguridad
    df = df.dropna(subset=['perfil_limpio'])
    X = df['perfil_limpio']
    y = df['grado']

    # =============================================================================
    # 2. VECTORIZACIÓN (TF-IDF)
    # =============================================================================
    print(" Vectorizando textos...")
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    
    # TF-IDF genera una matriz dispersa (sparse). 
    # LDA exige una matriz densa (array normal), por eso usamos .toarray()
    X_tfidf_denso = vectorizer.fit_transform(X).toarray()

    # =============================================================================
    # 3. REDUCCIÓN DE DIMENSIONALIDAD (De 1500 dimensiones a 2)
    # =============================================================================
    print(" Aplicando PCA (Exploratorio no supervisado)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_tfidf_denso)

    print(" Aplicando LDA (Separabilidad supervisada)...")
    # LDA busca la máxima separación entre clases. 
    # Soporta máximo (N_clases - 1) dimensiones. Tenemos 4 clases, así que 2D es perfecto.
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_tfidf_denso, y)

    # =============================================================================
    # 4. VISUALIZACIÓN COMPARATIVA (Subplots)
    # =============================================================================
    print(" Generando gráficos de proyección 2D...")
    
    # Configuramos la paleta de colores para que sea consistente
    colores = {'Civil': '#1f77b4', 'Informática': '#ff7f0e', 'Ejecución': '#2ca02c', 'Técnico': '#d62728'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Proyección 2D de Perfiles de Egreso (PCA vs LDA)', fontsize=16, fontweight='bold')

    # Gráfico 1: PCA
    sns.scatterplot(
        ax=axes[0],
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=y, palette=colores,
        alpha=0.8, edgecolor='k', s=80
    )
    axes[0].set_title('PCA (Exploratorio - No Supervisado)')
    axes[0].set_xlabel('Componente Principal 1')
    axes[0].set_ylabel('Componente Principal 2')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Gráfico 2: LDA
    sns.scatterplot(
        ax=axes[1],
        x=X_lda[:, 0], y=X_lda[:, 1],
        hue=y, palette=colores,
        alpha=0.8, edgecolor='k', s=80
    )
    axes[1].set_title('LDA (Separabilidad - Supervisado)')
    axes[1].set_xlabel('Función Discriminante 1')
    axes[1].set_ylabel('Función Discriminante 2')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO_SALIDA, dpi=300)
    
    print(f" ¡Punto 5 completado exitosamente!")
    print(f" Gráfico guardado en: {RUTA_GRAFICO_SALIDA}")

if __name__ == "__main__":
    main()