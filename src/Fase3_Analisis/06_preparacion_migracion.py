"""
06_PREPARACION_MIGRACION.PY
-----------------------------------------------------------------------
Fase: 3. Análisis (Cierre del Pipeline de Datos)
Propósito: 
    Este es el archivo final del motor de datos. Su función es consolidar 
    todos los hallazgos matemáticos, coordenadas espaciales y descriptivos 
    en un único archivo maestro (JSON).
    
    Este archivo servirá como el "Contrato de Datos" o la "Única Fuente de Verdad" 
    para alimentar la base de datos PostgreSQL y la interfaz web en Next.js (Fase 4).
"""

# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import json
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURACIÓN DE RUTAS INTELIGENTES
# ==========================================
# Ubicamos la carpeta actual (Fase3_Analisis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas apuntando al Data Lake central
RUTA_ENTRADA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv"))
RUTA_JSON = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "master_data.json"))

if __name__ == "__main__":
    print("📦 Iniciando empaquetamiento y sellado del Data Pipeline...")

    # ==========================================
    # 1. CARGA DE DATOS DE DIVERSAS FUENTES
    # ==========================================
    if not os.path.exists(RUTA_ENTRADA):
        print("❌ Error: Faltan los datos de clustering. Ejecuta '01_clustering.py' primero.")
        exit()

    df = pd.read_csv(RUTA_ENTRADA, sep=';', encoding='utf-8')

    # Bautizo oficial de clusters (consistente con el resto de la Fase 3)
    nombres_perfiles = {
        0: "Ingeniería de Software y Desarrollo",
        1: "Ciencia de Datos y Analytics",
        2: "Arquitectura y Gestión de Sistemas",
        3: "Ciberseguridad y Control de Riesgos"
    }

    # ==========================================
    # 2. CONSOLIDACIÓN Y LIMPIEZA FINAL
    # ==========================================
    # Mapeamos los nombres de los clusters
    df['perfil_nombre'] = df['Cluster'].map(nombres_perfiles)

    # Seleccionamos y renombramos las columnas para que coincidan con 
    # el estándar de nombres de tablas en PostgreSQL (snake_case)
    df_final = df[[
        'universidad', 'carrera', 'tipo_institucion', 
        'Coordenada_X', 'Coordenada_Y', 'Cluster', 'perfil_nombre'
    ]].copy()

    df_final.columns = [
        'institucion_nombre', 'carrera_nombre', 'institucion_tipo', 
        'coord_x', 'coord_y', 'cluster_id', 'perfil_nombre'
    ]

    # ==========================================
    # 3. GENERACIÓN DE INSIGHTS (PARA EL SCROLL DE NEXT.JS)
    # ==========================================
    # Estos datos nutrirán los textos destacados (storytelling) de la Landing Page
    insights = {
        "total_carreras": int(len(df_final)),
        "convergencia_global": 0.074,  # Resultado fijo de nuestro Silhouette Score
        "distribucion_por_perfil": df_final['perfil_nombre'].value_counts().to_dict(),
        "dominancia_universitaria": round((df_final[df_final['institucion_tipo'] == 'Universidad'].shape[0] / len(df_final)) * 100, 1),
        "hallazgo_ciberseguridad": "El 100% de la oferta pura en Ciberseguridad es liderada por Institutos Profesionales."
    }

    # ==========================================
    # 4. EXPORTACIÓN MAESTRA (JSON)
    # ==========================================
    # Convertimos el DataFrame a una lista de diccionarios (Formato ideal para APIs y DBs)
    data_maestra = {
        "proyecto": "Análisis Semántico Informática Chile 2026",
        "fase_pipeline": "Completada",
        "metadatos": insights,
        "registros": df_final.to_dict(orient='records')
    }

    os.makedirs(os.path.dirname(RUTA_JSON), exist_ok=True)

    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(data_maestra, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ ÉXITO: FASE 3 Y PIPELINE DE DATOS SELLADO.")
    print(f"📄 Archivo Maestro generado: {RUTA_JSON}")
    print(f"📊 Total de registros empaquetados: {len(df_final)}")
    print("="*60)
    print("🚀 Siguiente paso: Fase 4 -> Diseño de Tablas en PostgreSQL y Frontend en Next.js.")