import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_SALIDA_CSV = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "top15_palabras_clave.csv"))

def main():
    print(" Cargando dataset para el Análisis de Diferenciación Léxica...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f" Error: No se encontró {RUTA_ENTRADA}.")
        return

    df = df.dropna(subset=['perfil_limpio'])

    # =============================================================================
    # 2. EXTRACCIÓN DE FRECUENCIAS ABSOLUTAS
    # =============================================================================
    print(" Contando palabras (Bigramas y Monogramas)...")
    # Usamos CountVectorizer (no TF-IDF) porque el profesor pide frecuencias reales
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2)
    X_counts = vec.fit_transform(df['perfil_limpio'])
    vocabulario = vec.get_feature_names_out()

    # Convertimos a DataFrame para agrupar por carrera
    df_counts = pd.DataFrame(X_counts.toarray(), columns=vocabulario)
    df_counts['grado'] = df['grado'].values

    # Sumamos cuántas veces se dijo cada palabra en cada grado
    frecuencias_por_grado = df_counts.groupby('grado').sum()

    # =============================================================================
    # 3. CÁLCULO DE FRECUENCIA RELATIVA Y Z-SCORE (Keyness)
    # =============================================================================
    print(" Calculando Keyness mediante Z-Score...")
    
    # 1. Frecuencia Relativa: porcentaje de uso de una palabra dentro de ese grado
    totales_por_grado = frecuencias_por_grado.sum(axis=1)
    freq_relativa = frecuencias_por_grado.div(totales_por_grado, axis=0)

    # 2. Medidas globales: Promedio (mu) y Desviación Estándar (sigma) de cada palabra
    mu = freq_relativa.mean(axis=0)
    sigma = freq_relativa.std(axis=0).replace(0, 1e-9) # replace evita división por cero

    # 3. Cálculo del Z-Score: (Frecuencia Relativa - Promedio Global) / Desviación Estándar
    z_scores = (freq_relativa - mu) / sigma

    # =============================================================================
    # 4. EXTRACCIÓN Y REPORTE DEL TOP-15
    # =============================================================================
    print("\n" + "="*60)
    print(" TOP 15 TÉRMINOS DISTINTIVOS POR GRADO (Z-Score)")
    print("="*60)

    resultados = []

    for grado in z_scores.index:
        # Tomamos los 15 términos con el Z-score más alto para este grado
        top_terms = z_scores.loc[grado].sort_values(ascending=False).head(15)
        
        print(f"\n {grado.upper()}:")
        for i, (termino, puntaje_z) in enumerate(top_terms.items(), 1):
            print(f"   {i}. {termino:<25} (Z-Score: {puntaje_z:.2f})")
            resultados.append({
                'Grado': grado, 
                'Ranking': i, 
                'Término distintivo': termino, 
                'Z-Score': round(puntaje_z, 4)
            })

    # Guardamos el reporte para la tesis
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(RUTA_SALIDA_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f" ¡Punto 7 completado! Datos exportados a: {RUTA_SALIDA_CSV}")

if __name__ == "__main__":
    main()