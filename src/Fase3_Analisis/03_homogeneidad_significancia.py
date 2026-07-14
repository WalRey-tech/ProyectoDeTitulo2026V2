import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_GRAFICO_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "similitud_centroides.png"))

def calcular_diferencia_intra_inter(sim_matrix, labels):
    n = sim_matrix.shape[0]
    indices_superiores = np.triu_indices(n, k=1)
    sim_plana = sim_matrix[indices_superiores]

    labels_arr = np.array(labels)
    matriz_misma_etiqueta = (labels_arr[:, None] == labels_arr[None, :])
    misma_etiqueta_plana = matriz_misma_etiqueta[indices_superiores]

    sim_intra = sim_plana[misma_etiqueta_plana].mean()
    sim_inter = sim_plana[~misma_etiqueta_plana].mean()
    
    return sim_intra - sim_inter

def main():
    print("Iniciando Análisis de Homogeneidad Semántica (Corpus Real)...")
    print("-" * 70)
    
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {RUTA_ENTRADA}.")
        return

    df = df.dropna(subset=['perfil_limpio'])
    X = df['perfil_limpio'].values
    y = df['grado'].values
    
    # =============================================================================
    # 2. VECTORIZACIÓN (TF-IDF) SOBRE DATOS REALES
    # =============================================================================
    print("Ejecutando vectorización TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X_tfidf_denso = vectorizer.fit_transform(X).toarray()
    
    grados_unicos = sorted(np.unique(y))

    # =============================================================================
    # 3. CENTROIDES Y SIMILITUD COSENO (DATOS REALES)
    # =============================================================================
    centroides = []
    for grado in grados_unicos:
        vectores_grado = X_tfidf_denso[y == grado]
        centroides.append(vectores_grado.mean(axis=0))
    
    similitud_centroides = cosine_similarity(centroides)

    print("\n" + "="*70)
    print("MATRIZ DE SIMILITUD COSENO ENTRE CENTROIDES (Corpus Real)")
    print("="*70)
    print(f"{'':>15} | " + " | ".join([f"{g:>12}" for g in grados_unicos]))
    print("-" * 70)
    for i, grado1 in enumerate(grados_unicos):
        fila = [f"{similitud_centroides[i, j]:.4f}" for j in range(len(grados_unicos))]
        print(f"{grado1:>15} | " + " | ".join([f"{val:>12}" for val in fila]))
    
    idx_civil = grados_unicos.index('Civil')
    idx_info = grados_unicos.index('Informática')
    print(f"\n[HALLAZGO ESTRUCTURAL] Similitud entre Civil e Informática: {similitud_centroides[idx_civil, idx_info]:.4f}")

    # (Heatmap igual que antes...)
    plt.figure(figsize=(8, 6))
    sns.heatmap(similitud_centroides, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=grados_unicos, yticklabels=grados_unicos, vmin=0, vmax=1)
    plt.title('Similitud Coseno (Datos Reales)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO_SALIDA, dpi=300)

    # =============================================================================
    # 4. TEST DE PERMUTACIÓN (SOBRE DATOS REALES)
    # =============================================================================
    print("\nEjecutando Test de Permutación sobre datos reales...")
    similitud_total = cosine_similarity(X_tfidf_denso)
    diferencia_observada = calcular_diferencia_intra_inter(similitud_total, y)

    N_PERMUTACIONES = 1000
    np.random.seed(42) 
    diferencias_permutadas = np.array([calcular_diferencia_intra_inter(similitud_total, np.random.permutation(y)) 
                                       for _ in range(N_PERMUTACIONES)])
        
    p_valor = np.sum(diferencias_permutadas >= diferencia_observada) / N_PERMUTACIONES
    print(f"p-valor calculado: {p_valor:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()