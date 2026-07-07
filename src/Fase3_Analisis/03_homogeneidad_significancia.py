import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_GRAFICO_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "similitud_centroides.png"))

def calcular_diferencia_intra_inter(sim_matrix, labels):
    """
    Calcula la diferencia matemática entre la similitud intra-grupo (misma categoría)
    y la similitud inter-grupo (distinta categoría).
    
    Args:
        sim_matrix (np.ndarray): Matriz de similitud coseno de los vectores.
        labels (np.ndarray o list): Etiquetas correspondientes a cada vector.
        
    Returns:
        float: Diferencia entre el promedio de similitud intra-grupo y el inter-grupo.
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
    print("Iniciando Análisis de Homogeneidad Semántica y Significancia Estadística...")
    print("-" * 70)
    
    # =============================================================================
    # 2. CARGA Y PREPARACIÓN DE DATOS
    # =============================================================================
    print("Cargando dataset estructurado...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error crítico: No se encontró el archivo de datos en {RUTA_ENTRADA}.")
        return

    # Depuración de registros nulos
    df = df.dropna(subset=['perfil_limpio'])
    X = df['perfil_limpio'].values
    y = df['grado'].values
    
    # =============================================================================
    # 3. VECTORIZACIÓN SEMÁNTICA (TF-IDF)
    # =============================================================================
    # Sincronizado con los parámetros del Benchmark óptimo (92.79%)
    print("Ejecutando vectorización TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X_tfidf_denso = vectorizer.fit_transform(X).toarray()

    # =============================================================================
    # 4. BALANCEO SINTÉTICO (SMOTE)
    # =============================================================================
    print("Aplicando algoritmo SMOTE para estabilización de métricas estadísticas...")
    smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_tfidf_denso, y)
    
    grados_unicos = sorted(np.unique(y_resampled))

    # =============================================================================
    # 5. CÁLCULO DE CENTROIDES ESPACIALES Y SIMILITUD COSENO
    # =============================================================================
    print("Calculando vectores promedio (centroides) por grado académico...")
    centroides = []
    for grado in grados_unicos:
        vectores_grado = X_resampled[y_resampled == grado]
        centroide_promedio = vectores_grado.mean(axis=0)
        centroides.append(centroide_promedio)
    
    similitud_centroides = cosine_similarity(centroides)

    # Reporte de matriz de similitud por consola
    print("\n" + "="*70)
    print("MATRIZ DE SIMILITUD COSENO ENTRE CENTROIDES (Dataset Balanceado)")
    print("="*70)
    print(f"{'':>15} | " + " | ".join([f"{g:>12}" for g in grados_unicos]))
    print("-" * 70)
    for i, grado1 in enumerate(grados_unicos):
        fila = [f"{similitud_centroides[i, j]:.4f}" for j in range(len(grados_unicos))]
        print(f"{grado1:>15} | " + " | ".join([f"{val:>12}" for val in fila]))
    
    # Extracción de métrica crítica para la tesis
    idx_civil = grados_unicos.index('Civil')
    idx_info = grados_unicos.index('Informática')
    sim_civil_info = similitud_centroides[idx_civil, idx_info]
    print(f"\n[HALLAZGO ESTRUCTURAL] Similitud entre Civil e Informática: {sim_civil_info:.4f}")

    # Generación y exportación de Mapa de Calor (Heatmap)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similitud_centroides, annot=True, fmt='.3f', cmap='YlOrRd', 
        xticklabels=grados_unicos, yticklabels=grados_unicos, vmin=0, vmax=1
    )
    plt.title('Similitud Coseno entre Centroides de Grados', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICO_SALIDA, dpi=300)
    print(f"Mapa de calor exportado exitosamente en: {RUTA_GRAFICO_SALIDA}")

    # =============================================================================
    # 6. VALIDACIÓN ESTADÍSTICA: TEST DE PERMUTACIÓN
    # =============================================================================
    print("\n" + "="*70)
    print("EJECUCIÓN DE TEST DE PERMUTACIÓN (Análisis de Significancia)")
    print("="*70)
    
    similitud_total = cosine_similarity(X_resampled)
    diferencia_observada = calcular_diferencia_intra_inter(similitud_total, y_resampled)
    print(f"Diferencia observada (Intra - Inter) empírica: {diferencia_observada:.4f}")

    N_PERMUTACIONES = 1000
    np.random.seed(42) 
    diferencias_permutadas = np.zeros(N_PERMUTACIONES)
    
    print(f"Ejecutando {N_PERMUTACIONES} iteraciones de permutación de etiquetas...")
    for i in range(N_PERMUTACIONES):
        y_revuelto = np.random.permutation(y_resampled)
        diferencias_permutadas[i] = calcular_diferencia_intra_inter(similitud_total, y_revuelto)
        
    casos_extremos = np.sum(diferencias_permutadas >= diferencia_observada)
    p_valor = casos_extremos / N_PERMUTACIONES

    # Reporte de resultados estadísticos
    print("\n" + "="*70)
    print("RESULTADO DE VALIDACIÓN ESTADÍSTICA")
    print("="*70)
    print(f"Iteraciones nulas que superaron la métrica observada: {casos_extremos} de {N_PERMUTACIONES}")
    print(f"p-valor calculado: {p_valor:.4f}")
    
    if p_valor < 0.05:
        print("\nCONCLUSIÓN: p-valor < 0.05. Se rechaza la Hipótesis Nula (H0).")
        print("La estructura de separabilidad de los perfiles de egreso analizados")
        print("es estadísticamente significativa y no responde a una distribución aleatoria.")
    else:
        print("\nCONCLUSIÓN: p-valor >= 0.05. No existe evidencia suficiente para rechazar H0.")
        print("La estructura de distribución léxica actual podría ser producto del azar.")
        
    print("="*70 + "\n")

if __name__ == "__main__":
    main()