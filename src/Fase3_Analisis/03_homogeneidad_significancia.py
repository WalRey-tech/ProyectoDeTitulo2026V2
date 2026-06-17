import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_GRAFICO_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "similitud_centroides.png"))

def calcular_diferencia_intra_inter(sim_matrix, labels):
    """
    Calcula la diferencia entre la similitud intra-grupo (misma carrera)
    y la similitud inter-grupo (distinta carrera).
    """
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
    print("Cargando dataset para el Análisis de Homogeneidad...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: No se encontró {RUTA_ENTRADA}.")
        return

    df = df.dropna(subset=['perfil_limpio'])
    X = df['perfil_limpio'].values
    y = df['grado'].values
    
    # =============================================================================
    # 2. VECTORIZACIÓN TF-IDF (Sincronizado con Benchmark de 87%)
    # =============================================================================
    print("Vectorizando textos...")
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X_tfidf_denso = vectorizer.fit_transform(X).toarray()

    # =============================================================================
    # 3. BALANCEO SINTÉTICO (SMOTE)
    # =============================================================================
    print("⚖️ Aplicando SMOTE para balancear métricas estadísticas...")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf_denso, y)
    
    grados_unicos = sorted(np.unique(y_resampled))

    # =============================================================================
    # 4. CÁLCULO DE CENTROIDES Y SIMILITUD COSENO (Homogeneidad)
    # =============================================================================
    print("📐 Calculando los centroides de cada grado...")
    centroides = []
    for grado in grados_unicos:
        vectores_grado = X_resampled[y_resampled == grado]
        centroide_promedio = vectores_grado.mean(axis=0)
        centroides.append(centroide_promedio)
    
    similitud_centroides = cosine_similarity(centroides)

    print("\n" + "="*50)
    print(" MATRIZ DE SIMILITUD COSENO ENTRE CENTROIDES (Con SMOTE)")
    print("="*50)
    print(f"{'':>15} | " + " | ".join([f"{g:>12}" for g in grados_unicos]))
    print("-" * 70)
    for i, grado1 in enumerate(grados_unicos):
        fila = [f"{similitud_centroides[i, j]:.4f}" for j in range(len(grados_unicos))]
        print(f"{grado1:>15} | " + " | ".join([f"{val:>12}" for val in fila]))
    
    idx_civil = grados_unicos.index('Civil')
    idx_info = grados_unicos.index('Informática')
    sim_civil_info = similitud_centroides[idx_civil, idx_info]
    print(f"\n💡 [HALLAZGO TESIS] Similitud entre Civil e Informática: {sim_civil_info:.4f}")

    # Graficamos el Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(similitud_centroides, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=grados_unicos, yticklabels=grados_unicos, vmin=0, vmax=1)
    plt.title('Similitud Coseno entre Centroides de Grados', fontweight='bold')
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO_SALIDA, dpi=300)
    print(f"✅ Heatmap actualizado y guardado en: {RUTA_GRAFICO_SALIDA}")

    # =============================================================================
    # 5. TEST DE PERMUTACIÓN (Significancia Estadística)
    # =============================================================================
    print("\n" + "="*50)
    print(" INICIANDO TEST DE PERMUTACIÓN (Significancia)")
    print("="*50)
    
    similitud_total = cosine_similarity(X_resampled)
    diferencia_observada = calcular_diferencia_intra_inter(similitud_total, y_resampled)
    print(f"Diferencia observada (Intra - Inter) real: {diferencia_observada:.4f}")

    N_PERMUTACIONES = 1000
    np.random.seed(42) 
    diferencias_permutadas = np.zeros(N_PERMUTACIONES)
    
    print(f"Revolviendo etiquetas {N_PERMUTACIONES} veces para validar significancia...")
    for i in range(N_PERMUTACIONES):
        y_revuelto = np.random.permutation(y_resampled)
        diferencias_permutadas[i] = calcular_diferencia_intra_inter(similitud_total, y_revuelto)
        
    casos_extremos = np.sum(diferencias_permutadas >= diferencia_observada)
    p_valor = casos_extremos / N_PERMUTACIONES

    print("\n" + "="*50)
    print(" RESULTADO FINAL DEL TEST ESTADÍSTICO")
    print("="*50)
    print(f"   - Casos al azar que superaron la realidad: {casos_extremos} de {N_PERMUTACIONES}")
    print(f"   - p-valor final: {p_valor:.4f}")
    
    if p_valor < 0.05:
        print("    CONCLUSIÓN: p < 0.05. Se rechaza la hipótesis nula ($H_0$).")
        print("      La separabilidad de los perfiles de egreso, impulsada por")
        print("      el balanceo sintético y las diferencias en Ingeniería de Ejecución,")
        print("      es ESTADÍSTICAMENTE SIGNIFICATIVA y no producto del azar.")
    else:
        print("    CONCLUSIÓN: p >= 0.05. La estructura es aleatoria.")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()