import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
import warnings

# Inhabilitación de advertencias para optimizar la claridad de la salida en consola
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
RUTA_SALIDA_CSV = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "top15_palabras_clave.csv"))

def main():
    """
    Ejecuta el análisis de diferenciación léxica utilizando la métrica de Keyness (Z-Score).
    Identifica los términos estadísticamente significativos que definen la identidad de cada grado.
    """
    print("Iniciando análisis de diferenciación léxica por grado académico...")
    
    # Ingesta de datos procesados
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error crítico: El archivo fuente no existe en la ruta {RUTA_ENTRADA}")
        return

    # Limpieza de registros con campos de texto nulos
    df = df.dropna(subset=['perfil_limpio'])

    # =============================================================================
    # 2. CUANTIFICACIÓN DE FRECUENCIAS (Sincronización con Parámetros de Tesis)
    # =============================================================================
    print("Generando matriz de conteo (Unigramas y Bigramas)...")
    
    # Se mantienen los parámetros de vectorización validados en el benchmark de Machine Learning
    # (max_features=400) para asegurar que el análisis léxico explique la capacidad predictiva.
    vectorizador = CountVectorizer(max_features=400, ngram_range=(1, 2), max_df=0.85, min_df=2)
    X_counts = vectorizador.fit_transform(df['perfil_limpio'])
    vocabulario = vectorizador.get_feature_names_out()

    # Estructuración de datos para análisis grupal
    df_counts = pd.DataFrame(X_counts.toarray(), columns=vocabulario)
    df_counts['grado'] = df['grado'].values

    # Agregación de frecuencias absolutas por categoría de grado
    frecuencias_por_grado = df_counts.groupby('grado').sum()

    # =============================================================================
    # 3. ANÁLISIS DE KEYNESS: CÁLCULO DE Z-SCORE
    # =============================================================================
    # El uso de Z-Score sobre frecuencias relativas permite normalizar el impacto 
    # del desbalance de clases (diferente número de mallas por carrera).
    print("Ejecutando normalización estadística y cálculo de puntajes Z-Score...")
    
    # Cálculo de frecuencias relativas para mitigar el sesgo de volumen
    totales_por_grado = frecuencias_por_grado.sum(axis=1)
    frecuencia_relativa = frecuencias_por_grado.div(totales_por_grado, axis=0)

    # Cálculo de media y desviación estándar para determinar la especificidad léxica
    media_corpus = frecuencia_relativa.mean(axis=0)
    desviacion_corpus = frecuencia_relativa.std(axis=0).replace(0, 1e-9)

    # El Z-Score resultante indica cuántas desviaciones estándar se aleja un término 
    # de su frecuencia media en el corpus para un grado específico.
    z_scores = (frecuencia_relativa - media_corpus) / desviacion_corpus

    # =============================================================================
    # 4. EXTRACCIÓN DE HALLAZGOS Y EXPORTACIÓN
    # =============================================================================
    print("\n" + "="*70)
    print("RANKING TOP 15: TÉRMINOS DISTINTIVOS POR GRADO ACADÉMICO")
    print("="*70)

    resultados_analiticos = []

    for grado in z_scores.index:
        # Selección de los 15 términos con mayor capacidad de diferenciación
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

    # Persistencia de los resultados en formato estructurado
    df_final = pd.DataFrame(resultados_analiticos)
    df_final.to_csv(RUTA_SALIDA_CSV, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print(f"Análisis finalizado exitosamente. Archivo exportado en: {RUTA_SALIDA_CSV}")
    print("="*70)

if __name__ == "__main__":
    main()