import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import warnings

# Inhabilitación de advertencias
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_SALIDA_CSV = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "top15_palabras_clave.csv"))

def main():
    print("Iniciando análisis de diferenciación léxica (Sincronizado con Benchmark)...")
    
    # Carga de datos
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {RUTA_ENTRADA}")
        return

    df = df.dropna(subset=['perfil_limpio'])

    # =============================================================================
    # 2. CUANTIFICACIÓN (Consistencia con el modelo de ML)
    # =============================================================================
    # Se mantienen max_features=400 y ngram_range=(1, 2) 
    # para asegurar que los términos clave expliquen la capacidad predictiva del modelo.
    vectorizador = CountVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X_counts = vectorizador.fit_transform(df['perfil_limpio'])
    vocabulario = vectorizador.get_feature_names_out()

    df_counts = pd.DataFrame(X_counts.toarray(), columns=vocabulario)
    df_counts['grado'] = df['grado'].values

    # Agregación de frecuencias
    frecuencias_por_grado = df_counts.groupby('grado').sum()

    # =============================================================================
    # 3. ANÁLISIS DE KEYNESS: CÁLCULO DE Z-SCORE
    # =============================================================================
    # Cálculo de frecuencias relativas para normalizar el impacto del desbalance (N=63)
    totales_por_grado = frecuencias_por_grado.sum(axis=1)
    frecuencia_relativa = frecuencias_por_grado.div(totales_por_grado, axis=0)

    # Cálculo de media y desviación estándar para determinar especificidad léxica
    media_corpus = frecuencia_relativa.mean(axis=0)
    desviacion_corpus = frecuencia_relativa.std(axis=0).replace(0, 1e-9)

    # Z-Score: cuántas desviaciones estándar se aleja un término de la media del corpus
    z_scores = (frecuencia_relativa - media_corpus) / desviacion_corpus

    # =============================================================================
    # 4. EXTRACCIÓN Y EXPORTACIÓN
    # =============================================================================
    resultados_analiticos = []
    print("\n" + "="*70)
    print("RANKING TOP 15: TÉRMINOS DISTINTIVOS POR GRADO (Consistente con Modelo)")
    print("="*70)

    for grado in z_scores.index:
        top_terminos = z_scores.loc[grado].sort_values(ascending=False).head(15)
        print(f"\nIDENTIDAD LÉXICA: {grado.upper()}")
        print("-" * 35)
        
        for i, (termino, valor_z) in enumerate(top_terminos.items(), 1):
            print(f"{i:>2}. {termino:<25} | Z-Score: {valor_z:.2f}")
            resultados_analiticos.append({
                'Grado': grado, 
                'Ranking': i, 
                'Termino': termino, 
                'Z_Score': round(valor_z, 4)
            })

    # Exportación
    df_final = pd.DataFrame(resultados_analiticos)
    df_final.to_csv(RUTA_SALIDA_CSV, index=False, encoding='utf-8-sig')
    
    print("\nAnálisis finalizado. Resultados exportados a:", RUTA_SALIDA_CSV)

if __name__ == "__main__":
    main()