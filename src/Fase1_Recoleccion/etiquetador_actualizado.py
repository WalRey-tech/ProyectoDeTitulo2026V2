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
RUTA_VERSIONADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_etiquetado_v1.csv"))

# =============================================================================
# 2. MOTOR DE CLASIFICACIÓN
# =============================================================================
def clasificar_grado(nombre_carrera, tipo_institucion=""):
    """
    Clasifica cada registro según el grado académico definido para el análisis.

    Reglas metodológicas:
    1. Si la institución es Instituto Profesional, se clasifica como 'Ejecución',
       porque se considera un perfil técnico-operativo equivalente para efectos
       del análisis de separabilidad.
    2. Si el nombre de la carrera contiene 'civil', se clasifica como 'Civil'.
    3. Si el nombre de la carrera contiene 'ejecu', se clasifica como 'Ejecución'.
    4. El resto se clasifica como 'Informática'.
    """
    carrera = str(nombre_carrera).lower()
    institucion = str(tipo_institucion).lower()

    # Regla 1: Institutos Profesionales se agrupan como perfil técnico-operativo / Ejecución
    if "instituto profesional" in institucion:
        return "Ejecución"

    # Regla 2: 'Civil' si contiene «civil»
    if "civil" in carrera:
        return "Civil"

    # Regla 3: 'Ejecución' si contiene «ejecu»
    if "ejecu" in carrera:
        return "Ejecución"

    # Regla 4: 'Informática' en el resto de los casos
    return "Informática"


# =============================================================================
# 3. FLUJO PRINCIPAL
# =============================================================================
def main():
    print("Leyendo el dataset crudo generado por el scraper...")
    try:
        df = pd.read_csv(RUTA_RAW, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Error: No se encontró {RUTA_RAW}. Asegúrate de que main.py terminó.")
        return

    # --- FILTRO DE SEGURIDAD AUTOMÁTICO ---
    # Se eliminan carreras técnicas o analistas cuando son programas técnicos.
    # Importante: NO se elimina una Ingeniería de Ejecución "para Técnicos de Nivel Superior",
    # porque sigue siendo una carrera profesional y debe entrar al análisis.
    carrera_lower = df["carrera"].astype(str).str.lower()

    filtro_tecnicos = (
        carrera_lower.str.match(r"^\s*t[eé]cnico\b", na=False) |
        carrera_lower.str.contains(r"\banalista\b", na=False)
    )

    tecnicos_encontrados = filtro_tecnicos.sum()

    if tecnicos_encontrados > 0:
        print(f"Se detectaron {tecnicos_encontrados} carreras técnicas o analistas. Eliminándolas del dataset...")
        df = df[~filtro_tecnicos].copy()

    print("Creando la nueva columna 'grado'...")

    # Ahora la clasificación usa carrera + tipo_institucion.
    df["grado"] = df.apply(
        lambda fila: clasificar_grado(
            fila.get("carrera", ""),
            fila.get("tipo_institucion", "")
        ),
        axis=1
    )

    # Reordenamos las columnas visualmente.
    columnas = list(df.columns)
    columnas.insert(columnas.index("carrera") + 1, columnas.pop(columnas.index("grado")))
    df = df[columnas]

    # Nos aseguramos de que la carpeta 'processed' exista antes de guardar
    os.makedirs(os.path.dirname(RUTA_VERSIONADA), exist_ok=True)

    print("Guardando y versionando el dataset...")
    df.to_csv(RUTA_VERSIONADA, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    print("¡Etiquetado exitoso!")
    print(f"Archivo versionado guardado en: {RUTA_VERSIONADA}")
    print("\nDistribución final del dataset:")
    print(df["grado"].value_counts())

    print("\nDistribución por grado y tipo de institución:")
    print(pd.crosstab(df["grado"], df["tipo_institucion"]))


if __name__ == "__main__":
    main()
