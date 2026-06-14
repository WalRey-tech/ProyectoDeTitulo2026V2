import pandas as pd
import os
import csv

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS (Arquitectura de Datos)
# =============================================================================
# Obtenemos la ruta exacta donde estamos ejecutando el script
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# Ruta de entrada: Buscamos el archivo crudo que acaba de generar tu main.py
RUTA_RAW = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "raw", "perfiles_egreso_raw.csv"))

# Ruta de salida: Creamos la ruta para el archivo versionado en la carpeta 'processed'
# Cumplimos con la exigencia del profesor de "Versionar el dataset"
RUTA_VERSIONADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_etiquetado_v1.csv"))

# =============================================================================
# 2. MOTOR DE CLASIFICACIÓN (Reglas del Profesor)
# =============================================================================
def clasificar_grado(nombre_carrera):
    """
    Esta función recibe el nombre de la carrera y aplica estrictamente 
    las reglas de texto solicitadas para determinar el grado académico.
    """
    # Convertimos todo a minúsculas para que no falle si dice "Civil" o "CIVIL"
    carrera = str(nombre_carrera).lower()
    
    # Regla 1: 'Civil' si contiene «civil»
    if 'civil' in carrera:
        return 'Civil'
    # Regla 2: 'Ejecución' si contiene «ejecu»
    elif 'ejecu' in carrera:
        return 'Ejecución'
    # Regla 3: 'Técnico' si contiene «técnico» o «analista» (cubrimos con y sin tilde)
    elif 'técnico' in carrera or 'tecnico' in carrera or 'analista' in carrera:
        return 'Técnico'
    # Regla 4: 'Informática' en el resto de los casos
    else:
        return 'Informática'

# =============================================================================
# 3. FLUJO PRINCIPAL
# =============================================================================
def main():
    print("⏳ Leyendo el dataset crudo generado por el scraper...")
    try:
        # Leemos el archivo original. 
        # Pandas por defecto maneja bien las comillas si el archivo está limpio.
        df = pd.read_csv(RUTA_RAW, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {RUTA_RAW}. Asegúrate de que main.py terminó.")
        return

    print("🏷️ Creando la nueva columna 'grado'...")
    # .apply() toma nuestra función clasificar_grado y la ejecuta fila por fila
    # sobre la columna 'carrera', guardando el resultado en la nueva columna 'grado'.
    df['grado'] = df['carrera'].apply(clasificar_grado)

    # Reordenamos las columnas visualmente. 
    # Movemos 'grado' para que quede inmediatamente después de 'carrera'.
    # Esto te facilitará muchísimo la validación manual en Excel.
    columnas = list(df.columns)
    columnas.insert(columnas.index('carrera') + 1, columnas.pop(columnas.index('grado')))
    df = df[columnas]

    # Nos aseguramos de que la carpeta 'processed' exista antes de guardar
    os.makedirs(os.path.dirname(RUTA_VERSIONADA), exist_ok=True)

    print("💾 Guardando y versionando el dataset...")
    # Guardamos el nuevo archivo con la etiqueta _v1
    # Usamos quoting=csv.QUOTE_ALL para mantener la misma estructura de seguridad del raw
    df.to_csv(RUTA_VERSIONADA, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f"✅ ¡Etiquetado exitoso!")
    print(f"📁 Archivo versionado guardado en: {RUTA_VERSIONADA}")
    print("\n📊 Distribución final del dataset (Revisa si cuadra con lo esperado):")
    print(df['grado'].value_counts())

if __name__ == "__main__":
    main()