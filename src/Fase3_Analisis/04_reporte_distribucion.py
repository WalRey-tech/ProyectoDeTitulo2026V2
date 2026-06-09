"""
04_REPORTE_DISTRIBUCION.PY
-----------------------------------------------------------------------
Fase: 3. Análisis
Propósito: 
    Generar una tabla cruzada (contingencia) que muestre la relación estadística 
    entre el tipo de institución (Universidad, IP, CFT) y el Cluster asignado. 
    
    Esto permite concluir empíricamente si ciertos perfiles (ej. Ciberseguridad) 
    son exclusivos de las universidades o si están democratizados/liderados 
    por otras entidades en la educación superior chilena.
"""


# IMPORTACIÓN DE LIBRERÍAS
import os
import pandas as pd

# CONFIGURACIÓN DE RUTAS INTELIGENTES
# Ubicamos la carpeta actual (Fase3_Analisis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas apuntando al Data Lake central
RUTA_ENTRADA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv"))
RUTA_SALIDA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "reporte_final_distribucion.csv"))

if __name__ == "__main__":
    # 1. CARGA DEL DATASET ANALÍTICO
    print(f"Leyendo datos para el cruce estadístico desde: {RUTA_ENTRADA}")
    if not os.path.exists(RUTA_ENTRADA):
        print("Error: Falta el archivo de clusters. Ejecuta '01_clustering.py' primero.")
        exit()

    df = pd.read_csv(RUTA_ENTRADA, sep=";", encoding="utf-8-sig")

    # 2. MAPEO DE NOMBRES HUMANOS
    # Bautizamos los clusters según los hallazgos del análisis TF-IDF (Script 02)
    nombres_oficiales = {
    0: "Gestión Tecnológica y Sistemas Organizacionales",
    1: "Ciencia de Datos y Desarrollo de Software",
    2: "Ingeniería de Software y Computación Aplicada",
    3: "Ciberseguridad y Gestión de Riesgos"
}
    df['Perfil'] = df['Cluster'].map(nombres_oficiales)

    # 3. TABLA DE CONTINGENCIA (VOLUMEN)
    # Contamos cuántas instituciones de cada tipo hay en cada cluster
    reporte = pd.crosstab(df['tipo_institucion'], df['Perfil'], margins=True, margins_name="Total")

    print("\n" + "="*70)
    print("DISTRIBUCIÓN DE PERFILES POR TIPO DE INSTITUCIÓN (VOLUMEN)")
    print("="*70)
    print(reporte)


    # 4. ANÁLISIS DE CONCENTRACIÓN (PORCENTAJE)
    # normalize='index' calcula el porcentaje por fila (qué % de la oferta de IPs va a cada perfil)
    reporte_pct = pd.crosstab(df['tipo_institucion'], df['Perfil'], normalize='index') * 100
    
    print("\n" + "="*70)
    print("ANÁLISIS DE CONCENTRACIÓN (%)")
    print("="*70)
    print(reporte_pct.round(1).astype(str) + '%')
    print("-" * 70)

    # 5. EXPORTACIÓN PARA EXCEL / PRESENTACIÓN
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    reporte.to_csv(RUTA_SALIDA, sep=';', encoding='utf-8-sig')
    
    print(f"\nReporte estadístico guardado con éxito en: {RUTA_SALIDA}")