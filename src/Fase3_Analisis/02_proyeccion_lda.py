import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================
# Localización de directorios para asegurar compatibilidad entre sistemas operativos
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_GRAFICO_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "proyeccion_pca_vs_lda.png"))

# Configuración estética de los gráficos (Seaborn)
sns.set_theme(style="whitegrid")

def main():
    """
    Función principal para la generación de proyecciones espaciales de los perfiles de egreso.
    Compara la distribución natural (PCA) frente a la separabilidad supervisada (LDA).
    """
    print("Iniciando procesamiento de visualización espacial...")

    # Carga de datos procesados
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el archivo en la ruta {RUTA_ENTRADA}.")
        return

    # Limpieza de valores nulos en el campo de texto procesado
    df = df.dropna(subset=['perfil_limpio'])
    X_raw = df['perfil_limpio']
    y = df['grado']

    # =============================================================================
    # 2. VECTORIZACIÓN (TF-IDF)
    # =============================================================================
    # Se utiliza unigramas y bigramas para capturar términos compuestos específicos de ingeniería
    print("Ejecutando vectorización TF-IDF (1500 dimensiones)...")
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    
    # LDA requiere una matriz densa para el cálculo de autovectores y autovalores
    X_tfidf_dense = vectorizer.fit_transform(X_raw).toarray()

    # =============================================================================
    # 3. REDUCCIÓN DE DIMENSIONALIDAD
    # =============================================================================
    
    # PCA: Análisis de Componentes Principales (Enfoque Exploratorio No Supervisado)
    # Muestra cómo se agrupan los datos sin conocer las etiquetas de carrera
    print("Calculando reducción de dimensiones mediante PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_tfidf_dense)

    # LDA: Análisis Discriminante Lineal (Enfoque de Separabilidad Supervisado)
    # Busca maximizar la varianza entre las clases (Grados Académicos)
    print("Calculando reducción de dimensiones mediante LDA...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_tfidf_dense, y)

    # =============================================================================
    # 4. GENERACIÓN DE RESULTADOS GRÁFICOS
    # =============================================================================
    print("Generando visualización comparativa...")

    # Definición de paleta de colores institucional para mayor claridad
    colores_dict = {
        'Civil': '#1f77b4',       # Azul
        'Informática': '#ff7f0e', # Naranja
        'Ejecución': '#2ca02c'    # Verde
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Proyección Espacial de Perfiles de Egreso: PCA vs LDA', fontsize=18, fontweight='bold')

    # Subplot 1: PCA (Estado natural de los datos)
    sns.scatterplot(
        ax=axes[0],
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=y, palette=colores_dict,
        alpha=0.7, edgecolor='k', s=100
    )
    axes[0].set_title('PCA (Distribución Natural - No Supervisado)', fontsize=14)
    axes[0].set_xlabel('Componente Principal 1')
    axes[0].set_ylabel('Componente Principal 2')

    # Subplot 2: LDA (Separabilidad algorítmica)
    sns.scatterplot(
        ax=axes[1],
        x=X_lda[:, 0], y=X_lda[:, 1],
        hue=y, palette=colores_dict,
        alpha=0.7, edgecolor='k', s=100
    )
    axes[1].set_title('LDA (Separabilidad Forzada - Supervisado)', fontsize=14)
    axes[1].set_xlabel('Función Discriminante 1')
    axes[1].set_ylabel('Función Discriminante 2')

    # Ajuste de diseño y exportación de archivo
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(RUTA_GRAFICO_SALIDA, dpi=300)
    
    print(f"Proceso finalizado. Gráfico técnico guardado en: {RUTA_GRAFICO_SALIDA}")

if __name__ == "__main__":
    main()