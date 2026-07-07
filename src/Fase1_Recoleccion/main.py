import os
import csv
import pandas as pd
from config import SITES
from scraper import scrapear_sitio

try:
    from ftfy import fix_text
except ImportError:
    fix_text = None

# CONFIGURACIÓN DE RUTA
# 1. Obtenemos la ruta exacta de la carpeta donde está este script (Fase1_Recoleccion)
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

# 2. Construimos la ruta: Subimos un nivel ("..") hacia 'src', y luego a 'data/raw'
# NOTA: Asegúrate de que la carpeta data/raw esté lista en tu proyecto
RUTA_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "raw", "perfiles_egreso_raw.csv"))


def corregir_mojibake(texto):
    """
    Corrige errores de codificación:
    InformÃ¡tica -> Informática
    diseÃ±ar -> diseñar
    tecnolÃ³gicas -> tecnológicas
    """
    if not isinstance(texto, str):
        return texto

    texto = texto.strip()

    if fix_text is not None:
        return fix_text(texto)

    try:
        if any(marca in texto for marca in ["Ã", "Â", "â", "�"]):
            return texto.encode("latin1").decode("utf-8")
    except Exception:
        pass

    return texto


def limpiar_dataframe(df):
    df = df.fillna("")

    # Corrige caracteres dañados en todas las columnas de texto
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(corregir_mojibake)

    # Limpieza secundaria por seguridad en Pandas
    if "perfil_egreso" in df.columns:
        df["perfil_egreso"] = (
            df["perfil_egreso"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    return df

def ordenar_columnas(df):
    columnas_ordenadas = [
        "universidad",
        "tipo_institucion",
        "carrera",
        "tipo_carrera",
        "url",
        "selector",
        "metodo_usado",
        "perfil_egreso",
        "error"
    ]
    columnas_presentes = [col for col in columnas_ordenadas if col in df.columns]
    return df[columnas_presentes]

def validar_dataframe(df):
    print("\n" + "="*40)
    print("VALIDACIÓN PARA EL MODELO NLP")
    print("="*40)
    print(f"Total de instituciones procesadas: {len(df)}")

    # Agregamos largo del perfil para revisar fácilmente
    df["largo_perfil"] = df["perfil_egreso"].astype(str).str.len()

    perfiles_vacios_df = df[df["perfil_egreso"].str.strip() == ""]

    perfiles_cortos_df = df[
        (df["perfil_egreso"].str.len() > 0) &
        (df["perfil_egreso"].str.len() < 150)
    ]

    perfiles_vacios = len(perfiles_vacios_df)
    perfiles_cortos = len(perfiles_cortos_df)

    print(f"Perfiles listos para análisis: {len(df) - perfiles_vacios - perfiles_cortos}")
    print(f"Perfiles vacíos (Revisar selector CSS): {perfiles_vacios}")
    print(f"Perfiles muy cortos (< 150 caracteres): {perfiles_cortos}")
    print("="*40)

    if perfiles_vacios > 0:
        print("\nPERFILES VACÍOS DETECTADOS:")
        columnas = ["universidad", "carrera", "url", "selector", "metodo_usado", "error"]
        columnas = [col for col in columnas if col in perfiles_vacios_df.columns]

        print(
            perfiles_vacios_df[columnas]
            .to_string(index=False)
        )

    if perfiles_cortos > 0:
        print("\nPERFILES MUY CORTOS DETECTADOS:")
        columnas = ["universidad", "carrera", "url", "selector", "metodo_usado", "largo_perfil", "perfil_egreso"]
        columnas = [col for col in columnas if col in perfiles_cortos_df.columns]

        print(
            perfiles_cortos_df[columnas]
            .to_string(index=False)
        )

    print("="*40)

def validar_calidad_dataset(df):
    print("\n" + "="*40)
    print("VALIDACIÓN DE CALIDAD DEL DATASET")
    print("="*40)

    # 1. Duplicados por universidad + carrera
    if {"universidad", "carrera"}.issubset(df.columns):
        duplicados_carrera = df[
            df.duplicated(subset=["universidad", "carrera"], keep=False)
        ].sort_values(by=["universidad", "carrera"])

        print(f"Duplicados por universidad + carrera: {len(duplicados_carrera)}")

        if not duplicados_carrera.empty:
            print(duplicados_carrera[["universidad", "carrera", "url"]].to_string(index=False))

    # 2. Duplicados por URL
    if "url" in df.columns:
        duplicados_url = df[
            df.duplicated(subset=["url"], keep=False)
        ].sort_values(by=["url"])

        print(f"\nDuplicados por URL: {len(duplicados_url)}")

        if not duplicados_url.empty:
            print(duplicados_url[["universidad", "carrera", "url"]].to_string(index=False))

    # 3. Caracteres sospechosos de encoding
    patron_encoding = r"Ã|Â|â|�"

    problemas_encoding = df[
        df.apply(
            lambda fila: fila.astype(str).str.contains(patron_encoding, regex=True).any(),
            axis=1
        )
    ]

    print(f"\nFilas con posibles problemas de codificación: {len(problemas_encoding)}")

    if not problemas_encoding.empty:
        columnas_mostrar = [col for col in ["universidad", "carrera", "perfil_egreso"] if col in df.columns]
        print(problemas_encoding[columnas_mostrar].head(10).to_string(index=False))

    # 4. Perfiles muy cortos
    if "perfil_egreso" in df.columns:
        perfiles_cortos = df[
            (df["perfil_egreso"].str.len() > 0) &
            (df["perfil_egreso"].str.len() < 150)
        ]

        print(f"\nPerfiles muy cortos: {len(perfiles_cortos)}")

        if not perfiles_cortos.empty:
            print(perfiles_cortos[["universidad", "carrera", "perfil_egreso"]].head(10).to_string(index=False))

    print("="*40)

def guardar_csv(df):
    # Usar os.path.dirname previene errores si la ruta cambia en el futuro
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    
    df.to_csv(
        RUTA_SALIDA,
        index=False,
        sep=";",                 # <- ESTO FALTABA
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )

    print(f"\nArchivo maestro guardado en: {RUTA_SALIDA}\n")

def main():
    resultados = []

    for site in SITES:
        print(f"\nScrapeando: {site['universidad']} - {site['carrera']}")

        data = scrapear_sitio(site)

        if data.get("error"):
            print(f"   ↳ Estado: ERROR ({data.get('error')})")
        # CORRECCIÓN: Actualizado a 'perfil_egreso'
        elif not data.get("perfil_egreso"):
            print("   ↳ Estado: ADVERTENCIA Sin texto extraído")
        else:
            print(f"   ↳ Estado: OK usando {data.get('metodo_usado')}")

        resultados.append(data)

    df = pd.DataFrame(resultados)

    df = limpiar_dataframe(df)
    df = ordenar_columnas(df)

    if "universidad" in df.columns:
        df = df.sort_values(by="universidad")

    validar_dataframe(df)
    guardar_csv(df)

if __name__ == "__main__":
    main()