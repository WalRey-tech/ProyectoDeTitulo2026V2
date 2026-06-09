"""
02_VECTORIZACION.PY
-----------------------------------------------------------------------
Fase: 2. Procesamiento

Propósito:
    Convertir los perfiles de egreso procesados en embeddings semánticos.
    Estos vectores serán utilizados posteriormente en la Fase 3 para aplicar
    algoritmos de clustering como K-Means o DBSCAN.
"""

# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_LIMPIOS = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_limpios.csv")
)

RUTA_VECTORES = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "vectores_perfiles.npy")
)

RUTA_METADATA = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "metadata_vectores.csv")
)


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print(f"📂 Cargando dataset limpio desde: {RUTA_LIMPIOS}")

    # 1. Validar existencia del archivo limpio
    if not os.path.exists(RUTA_LIMPIOS):
        print("❌ Error: No se encontró el archivo limpio.")
        print("➡️ Ejecuta primero: python .\\01_limpieza_nlp.py")
        exit()

    # 2. Leer dataset limpio
    try:
        df = pd.read_csv(
            RUTA_LIMPIOS,
            sep=";",
            encoding="utf-8-sig"
        )
    except Exception as e:
        print("❌ Error al leer perfiles_limpios.csv")
        print(f"Detalle técnico: {e}")
        print("\n➡️ Revisa que 01_limpieza_nlp.py esté guardando el CSV con:")
        print('df.to_csv(RUTA_SALIDA, index=False, sep=";", encoding="utf-8-sig")')
        exit()

    # 3. Normalizar nombres de columnas
    df.columns = df.columns.str.strip()

    print("\n📌 Columnas detectadas:")
    print(df.columns.tolist())

    print(f"\n📊 Filas cargadas: {len(df)}")

    # 4. Definir columna de texto para embeddings
    columna_texto = "perfil_final"

    if columna_texto not in df.columns:
        print(f"\n❌ Error: No existe la columna '{columna_texto}' en el archivo limpio.")
        print("Columnas disponibles:")
        print(df.columns.tolist())
        print("\n➡️ Verifica que 01_limpieza_nlp.py esté creando la columna 'perfil_final'.")
        exit()

    # 5. Control de calidad del texto
    df[columna_texto] = df[columna_texto].fillna("").astype(str)

    df = df[df[columna_texto].str.strip() != ""]

    if df.empty:
        print("\n❌ Error: No hay perfiles válidos para vectorizar.")
        print("La columna 'perfil_final' está vacía o solo contiene espacios.")
        exit()

    print(f"\n✅ Perfiles válidos para vectorización: {len(df)}")

    # 6. Inicializar modelo de embeddings
    print("\n🧠 Inicializando modelo de embeddings semánticos...")

    try:
        modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print("❌ Error al cargar el modelo SentenceTransformer.")
        print(f"Detalle técnico: {e}")
        print("\n➡️ Verifica que sentence-transformers esté instalado:")
        print("pip install sentence-transformers")
        exit()

    # 7. Vectorizar perfiles
    print("\n⚙️ Vectorizando perfiles de egreso...")

    try:
        vectores = modelo.encode(
            df[columna_texto].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )
    except Exception as e:
        print("❌ Error durante la vectorización.")
        print(f"Detalle técnico: {e}")
        exit()

    # 8. Mostrar resultado
    print("\n=== RESULTADO DE LA VECTORIZACIÓN ===")
    print(f"Forma de la matriz: {vectores.shape}")
    print("Primeras 5 coordenadas del primer perfil:")
    print(vectores[0][:5])

    # 9. Guardar resultados
    os.makedirs(os.path.dirname(RUTA_VECTORES), exist_ok=True)

    np.save(RUTA_VECTORES, vectores)

    df.to_csv(
        RUTA_METADATA,
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 50)
    print("✅ FASE 2: VECTORIZACIÓN FINALIZADA")
    print(f"💾 Vectores guardados en: {RUTA_VECTORES}")
    print(f"💾 Metadata alineada guardada en: {RUTA_METADATA}")
    print("=" * 50)