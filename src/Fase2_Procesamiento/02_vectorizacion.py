"""
02_VECTORIZACION.PY
-----------------------------------------------------------------------
Fase: 2. Procesamiento
Propósito: 
    Convertir los perfiles de egreso (limpios y lematizados) en representaciones 
    numéricas o "vectores" (Embeddings). Esto permite que el algoritmo matemático 
    K-Means pueda calcular qué tan similares son dos textos basándose en su 
    significado profundo (semántica), y no solo en coincidencias exactas de palabras.
"""

# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import pandas as pd
import numpy as np
# SentenceTransformer: Librería basada en la arquitectura BERT de Google.
# Diseñada específicamente para entender el contexto de frases o párrafos enteros.
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURACIÓN DE RUTAS INTELIGENTES
# ==========================================
# Ubicamos la carpeta donde vive este script (Fase2_Procesamiento)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Definimos rutas apuntando al Data Lake central (src/data/processed/)
RUTA_LIMPIOS = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_limpios.csv"))
RUTA_VECTORES = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "vectores_perfiles.npy"))

# ==========================================
# 1. CARGA DE DATOS PREPROCESADOS
# ==========================================
if __name__ == "__main__":
    print(f"📂 Cargando dataset limpio desde: {RUTA_LIMPIOS}")
    
    if not os.path.exists(RUTA_LIMPIOS):
        print("❌ Error: No se encontró el archivo limpio. Ejecuta '01_limpieza_nlp.py' primero.")
        exit()
        
    df = pd.read_csv(RUTA_LIMPIOS, sep=';', encoding='utf-8')

    # Control de Calidad: Eliminar filas donde el perfil quedó vacío tras la limpieza.
    # Un texto nulo ('NaN') generaría un error matemático crítico en la IA.
    df = df.dropna(subset=['perfil_final'])

    # ==========================================
    # 2. CARGA DEL MODELO DE INTELIGENCIA ARTIFICIAL
    # ==========================================
    print("\n🧠 Inicializando modelo de Inteligencia Artificial...")

    # Modelo: 'paraphrase-multilingual-MiniLM-L12-v2'
    # Justificación técnica:
    # - 'paraphrase': Entrenado para identificar si dos frases significan lo mismo.
    # - 'multilingual': Capta perfectamente la sintaxis del español chileno.
    # - 'MiniLM': Versión optimizada y rápida de BERT.
    # - 'L12': Genera un espacio vectorial de 384 dimensiones.
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # ==========================================
    # 3. VECTORIZACIÓN (TRANSFORMACIÓN A EMBEDDINGS)
    # ==========================================
    print("⚙️ Vectorizando los perfiles de egreso (Convirtiendo texto a matemáticas)...")

    # .encode() lee las raíces de las palabras y su contexto para ubicarlas en el espacio
    vectores = modelo.encode(df['perfil_final'].tolist(), show_progress_bar=True)

    # ==========================================
    # 4. VALIDACIÓN DE LA MATRIZ
    # ==========================================
    print("\n=== RESULTADO DE LA VECTORIZACIÓN ===")
    # .shape devuelve (filas, columnas). Esperamos (N_carreras, 384) dimensiones.
    print(f"Forma de la matriz matemática: {vectores.shape}")
    print(f"Ejemplo de las primeras 5 coordenadas del primer perfil:\n{vectores[0][:5]}")

    # ==========================================
    # 5. ALMACENAMIENTO BINARIO
    # ==========================================
    # Guardamos los vectores en formato binario nativo (.npy).
    # Es vital porque guarda con precisión decimal exacta y carga mucho más rápido que un CSV.
    os.makedirs(os.path.dirname(RUTA_VECTORES), exist_ok=True)
    np.save(RUTA_VECTORES, vectores)

    print("\n" + "="*40)
    print("✅ FASE 2: VECTORIZACIÓN FINALIZADA")
    print(f"💾 Matriz de vectores lista para Clustering en: {RUTA_VECTORES}")
    print("="*40)