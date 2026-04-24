import os
import pandas as pd
from config import SITES
from scraper import scrapear_sitio


RUTA_SALIDA = "data/raw/perfiles_egreso_raw.csv"


def limpiar_dataframe(df):
    # Limpiar espacios y saltos de línea
    df["perfil"] = df["perfil"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Rellenar valores nulos
    df = df.fillna("")

    return df


def ordenar_columnas(df):
    columnas_ordenadas = [
        "universidad",
        "tipo_institucion",
        "carrera",
        "tipo_carrera",
        "url",
        "selector",
        "perfil",
        "error"
    ]

    # Solo mantiene las que existen
    columnas_presentes = [col for col in columnas_ordenadas if col in df.columns]

    return df[columnas_presentes]


def validar_dataframe(df):
    print("\n--- VALIDACIÓN DEL DATASET ---")
    print(f"Cantidad de registros: {len(df)}")

    perfiles_vacios = (df["perfil"].str.strip() == "").sum()
    perfiles_cortos = (df["perfil"].str.len() < 50).sum()

    print(f"Perfiles vacíos: {perfiles_vacios}")
    print(f"Perfiles muy cortos: {perfiles_cortos}")


def guardar_csv(df):
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(
        RUTA_SALIDA,
        index=False,
        encoding="utf-8-sig",  # importante para Excel
        sep=";"  # asegura columnas separadas
    )

    print("\nArchivo guardado en:")
    print(RUTA_SALIDA)


def main():
    resultados = []

    for site in SITES:
        try:
            print(f"\nScrapeando: {site['universidad']} - {site['carrera']}")

            data = scrapear_sitio(site)
            resultados.append(data)

            print("Estado: OK")

        except Exception as e:
            print("Estado: ERROR")

            resultados.append({
                "universidad": site.get("universidad", ""),
                "carrera": site.get("carrera", ""),
                "url": site.get("url", ""),
                "perfil": "",
                "error": str(e)
            })

    df = pd.DataFrame(resultados)

    # 🔥 PROCESAMIENTO
    df = limpiar_dataframe(df)
    df = ordenar_columnas(df)

    # Orden opcional por universidad
    df = df.sort_values(by="universidad")

    validar_dataframe(df)
    guardar_csv(df)

    print("\n--- VISTA PREVIA ---")
    print(df.head())


if __name__ == "__main__":
    main()