import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_SALIDA_CSV = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "top15_palabras_clave.csv"))

def main():
    print("📝 Cargando dataset para el Análisis de Diferenciación Léxica...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {RUTA_ENTRADA}.")
        return

    df = df.dropna(subset=['perfil_limpio'])

    # =============================================================================
    # 2. EXTRACCIÓN DE FRECUENCIAS (Sincronizado con el Benchmark de 87%)
    # =============================================================================
    print("🧮 Contando palabras (Bigramas y Monogramas)...")
    # Ajustamos los parámetros EXACTAMENTE como en el modelo ganador de la tesis
    # para que la IA extraiga el "ADN léxico" del mismo vocabulario validado.
    vec = CountVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X_counts = vec.fit_transform(df['perfil_limpio'])
    vocabulario = vec.get_feature_names_out()

    df_counts = pd.DataFrame(X_counts.toarray(), columns=vocabulario)
    df_counts['grado'] = df['grado'].values

    frecuencias_por_grado = df_counts.groupby('grado').sum()

    # =============================================================================
    # 3. CÁLCULO DE Z-SCORE (Keyness)
    # Nota: El Z-Score normaliza automáticamente el desbalance de clases al usar
    # frecuencias relativas, por lo que no es necesario (ni posible) aplicar SMOTE aquí.
    # =============================================================================
    print("📊 Calculando Keyness mediante Z-Score...")
    
    totales_por_grado = frecuencias_por_grado.sum(axis=1)
    freq_relativa = frecuencias_por_grado.div(totales_por_grado, axis=0)

    mu = freq_relativa.mean(axis=0)
    sigma = freq_relativa.std(axis=0).replace(0, 1e-9)

    z_scores = (freq_relativa - mu) / sigma

    # =============================================================================
    # 4. EXTRACCIÓN Y REPORTE DEL TOP-15
    # =============================================================================
    print("\n" + "="*60)
    print("🏆 TOP 15 TÉRMINOS DISTINTIVOS POR GRADO (Z-Score)")
    print("="*60)

    resultados = []

    for grado in z_scores.index:
        top_terms = z_scores.loc[grado].sort_values(ascending=False).head(15)
        
        print(f"\n📌 {grado.upper()}:")
        for i, (termino, puntaje_z) in enumerate(top_terms.items(), 1):
            print(f"   {i}. {termino:<25} (Z-Score: {puntaje_z:.2f})")
            resultados.append({
                'Grado': grado, 
                'Ranking': i, 
                'Término distintivo': termino, 
                'Z-Score': round(puntaje_z, 4)
            })

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(RUTA_SALIDA_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print(f"✅ ¡Extracción exitosa! Datos exportados a: {RUTA_SALIDA_CSV}")

if __name__ == "__main__":
    main()