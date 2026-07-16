import csv
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

RUTA_ENTRADA = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "perfiles_egreso_limpio_v1.csv",
    )
)

RUTA_SALIDA_CSV = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "top15_palabras_clave.csv",
    )
)

RUTA_SALIDA_JSON = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "terminos_distintivos_v1.json",
    )
)

DIRECTORIO_GRAFICOS = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "graficos_terminos_distintivos",
    )
)

TOP_N = 15

# Para que un término sea considerado distintivo debe aparecer en varios
# perfiles reales del grado, no solo repetirse muchas veces en un único texto.
MIN_DOCUMENTOS_ABSOLUTO = 2
SOPORTE_MINIMO_PROPORCIONAL = 0.10

VECTORIZADOR_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "max_df": 0.85,
    "min_df": 2,
}


# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def validar_dataset(df):
    """Valida que el corpus contenga las columnas y registros requeridos."""
    columnas_obligatorias = {"perfil_limpio", "grado"}
    columnas_faltantes = columnas_obligatorias - set(df.columns)

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas obligatorias: "
            f"{sorted(columnas_faltantes)}"
        )

    df = df.dropna(
        subset=["perfil_limpio", "grado"]
    ).copy()

    df = df[
        df["perfil_limpio"].astype(str).str.strip() != ""
    ].copy()

    if df.empty:
        raise ValueError(
            "El dataset no contiene perfiles válidos."
        )

    if df["grado"].nunique() < 2:
        raise ValueError(
            "Se requieren al menos dos grados para comparar."
        )

    return df


def calcular_z_scores(frecuencias_por_grado):
    """
    Calcula frecuencia relativa y Z-Score por término.

    Cada fila representa un grado y cada columna un término.
    La frecuencia relativa corrige el efecto del diferente volumen
    de palabras de cada clase. El Z-Score indica cuánto se aleja la
    frecuencia relativa de un grado respecto del promedio entre grados.
    """
    totales_por_grado = frecuencias_por_grado.sum(axis=1)

    if (totales_por_grado == 0).any():
        grados_sin_terminos = totales_por_grado[
            totales_por_grado == 0
        ].index.tolist()

        raise ValueError(
            "Los siguientes grados no contienen términos válidos: "
            f"{grados_sin_terminos}"
        )

    frecuencia_relativa = frecuencias_por_grado.div(
        totales_por_grado,
        axis=0,
    )

    media_corpus = frecuencia_relativa.mean(axis=0)

    # Se mantiene ddof=1 para conservar el criterio estadístico
    # utilizado en la versión original del análisis.
    desviacion_corpus = frecuencia_relativa.std(
        axis=0,
        ddof=1,
    )

    desviacion_corpus = desviacion_corpus.replace(
        0,
        np.nan,
    )

    z_scores = (
        frecuencia_relativa - media_corpus
    ).div(desviacion_corpus, axis=1)

    # Los términos sin variación entre grados no son distintivos.
    z_scores = z_scores.fillna(0.0)

    return frecuencia_relativa, z_scores


def generar_graficos(df_resultados):
    """Genera un gráfico independiente para cada grado académico."""
    os.makedirs(
        DIRECTORIO_GRAFICOS,
        exist_ok=True,
    )

    rutas_generadas = []

    for grado in df_resultados["Grado"].unique():
        datos = (
            df_resultados[
                df_resultados["Grado"] == grado
            ]
            .sort_values(
                by="Z_Score",
                ascending=True,
            )
            .tail(10)
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            datos["Termino"],
            datos["Z_Score"],
        )

        plt.title(
            f"Términos distintivos — {grado}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xlabel("Z-Score")
        plt.ylabel("Término")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        nombre_archivo = (
            "terminos_distintivos_"
            + grado.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace(" ", "_")
            + ".png"
        )

        ruta_grafico = os.path.join(
            DIRECTORIO_GRAFICOS,
            nombre_archivo,
        )

        plt.savefig(
            ruta_grafico,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()
        rutas_generadas.append(ruta_grafico)

    return rutas_generadas


# =============================================================================
# 3. FLUJO PRINCIPAL
# =============================================================================

def main():
    print("=" * 76)
    print("ANÁLISIS DE DIFERENCIACIÓN LÉXICA POR GRADO")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # Carga y validación
    # -------------------------------------------------------------------------

    try:
        df = pd.read_csv(
            RUTA_ENTRADA,
            encoding="utf-8-sig",
        )
    except FileNotFoundError:
        print(
            "Error: no se encontró el archivo de entrada:\n"
            f"{RUTA_ENTRADA}"
        )
        return
    except Exception as error:
        print(f"Error al leer el dataset: {error}")
        return

    try:
        df = validar_dataset(df)
    except ValueError as error:
        print(f"Error de validación: {error}")
        return

    print(f"Total de perfiles reales: {len(df)}")

    print("\nDistribución por grado:")
    print(df["grado"].value_counts())

    # -------------------------------------------------------------------------
    # Matriz de conteo
    # -------------------------------------------------------------------------

    print("\nGenerando matriz de conteo...")
    print(f"Configuración: {VECTORIZADOR_CONFIG}")

    vectorizador = CountVectorizer(
        **VECTORIZADOR_CONFIG
    )

    X_counts = vectorizador.fit_transform(
        df["perfil_limpio"].astype(str)
    )

    vocabulario = vectorizador.get_feature_names_out()

    print(
        "Dimensiones de la matriz léxica: "
        f"{X_counts.shape}"
    )

    df_counts = pd.DataFrame(
        X_counts.toarray(),
        columns=vocabulario,
        index=df.index,
    )

    df_counts["grado"] = df["grado"].values

    frecuencias_por_grado = (
        df_counts.groupby("grado").sum()
    )

    # Matriz binaria: 1 si el término aparece al menos una vez en el perfil.
    X_binario = (X_counts > 0).astype(int)

    df_documentos = pd.DataFrame(
        X_binario.toarray(),
        columns=vocabulario,
        index=df.index,
    )

    df_documentos["grado"] = df["grado"].values

    documentos_por_grado = (
        df_documentos.groupby("grado").sum()
    )

    cantidad_documentos_grado = (
        df["grado"].value_counts().to_dict()
    )

    # -------------------------------------------------------------------------
    # Z-Score sobre frecuencia relativa
    # -------------------------------------------------------------------------

    print(
        "\nCalculando frecuencias relativas y Z-Score..."
    )

    frecuencia_relativa, z_scores = calcular_z_scores(
        frecuencias_por_grado
    )

    frecuencia_total_corpus = (
        frecuencias_por_grado.sum(axis=0)
    )

    # -------------------------------------------------------------------------
    # Extracción de resultados
    # -------------------------------------------------------------------------

    resultados = []

    print("\n" + "=" * 76)
    print(
        f"TOP {TOP_N}: TÉRMINOS DISTINTIVOS "
        "POR GRADO ACADÉMICO"
    )
    print("=" * 76)

    for grado in z_scores.index:
        total_documentos_grado = int(
            cantidad_documentos_grado[grado]
        )

        umbral_documentos = max(
            MIN_DOCUMENTOS_ABSOLUTO,
            int(
                np.ceil(
                    total_documentos_grado
                    * SOPORTE_MINIMO_PROPORCIONAL
                )
            ),
        )

        terminos_validos = documentos_por_grado.loc[grado]
        terminos_validos = terminos_validos[
            terminos_validos >= umbral_documentos
        ].index

        z_scores_validos = z_scores.loc[
            grado,
            terminos_validos,
        ]

        # Solo se conservan términos con diferenciación positiva.
        z_scores_validos = z_scores_validos[
            z_scores_validos > 0
        ]

        top_terminos = (
            z_scores_validos
            .sort_values(ascending=False)
            .head(TOP_N)
        )

        print(f"\n{grado.upper()}")
        print("-" * 48)
        print(
            "Umbral de soporte documental: "
            f"{umbral_documentos} de "
            f"{total_documentos_grado} perfiles"
        )

        for ranking, (termino, z_score) in enumerate(
            top_terminos.items(),
            start=1,
        ):
            frecuencia_absoluta = int(
                frecuencias_por_grado.loc[
                    grado,
                    termino,
                ]
            )

            frecuencia_rel = float(
                frecuencia_relativa.loc[
                    grado,
                    termino,
                ]
            )

            frecuencia_corpus = int(
                frecuencia_total_corpus.loc[termino]
            )

            documentos_grado = int(
                documentos_por_grado.loc[
                    grado,
                    termino,
                ]
            )

            soporte_documental = (
                documentos_grado
                / total_documentos_grado
            )

            print(
                f"{ranking:>2}. "
                f"{termino:<28} "
                f"Z={z_score:>6.3f} | "
                f"f={frecuencia_absoluta} | "
                f"docs={documentos_grado}/"
                f"{total_documentos_grado}"
            )

            resultados.append(
                {
                    "Grado": grado,
                    "Ranking": ranking,
                    "Termino": termino,
                    "Z_Score": round(
                        float(z_score),
                        6,
                    ),
                    "Frecuencia_Absoluta_Grado":
                        frecuencia_absoluta,
                    "Frecuencia_Relativa_Grado":
                        round(frecuencia_rel, 8),
                    "Frecuencia_Total_Corpus":
                        frecuencia_corpus,
                    "Documentos_Grado":
                        documentos_grado,
                    "Total_Documentos_Grado":
                        total_documentos_grado,
                    "Soporte_Documental_Grado":
                        round(soporte_documental, 6),
                    "Umbral_Documentos":
                        umbral_documentos,
                }
            )

    df_resultados = pd.DataFrame(resultados)

    os.makedirs(
        os.path.dirname(RUTA_SALIDA_CSV),
        exist_ok=True,
    )

    df_resultados.to_csv(
        RUTA_SALIDA_CSV,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )

    # -------------------------------------------------------------------------
    # Exportación JSON
    # -------------------------------------------------------------------------

    terminos_json = {}

    for grado in df_resultados["Grado"].unique():
        registros_grado = df_resultados[
            df_resultados["Grado"] == grado
        ]

        terminos_json[grado] = (
            registros_grado[
                [
                    "Ranking",
                    "Termino",
                    "Z_Score",
                    "Frecuencia_Absoluta_Grado",
                    "Frecuencia_Relativa_Grado",
                    "Frecuencia_Total_Corpus",
                    "Documentos_Grado",
                    "Total_Documentos_Grado",
                    "Soporte_Documental_Grado",
                    "Umbral_Documentos",
                ]
            ]
            .to_dict(orient="records")
        )

    with open(
        RUTA_SALIDA_JSON,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            {
                "total_perfiles": int(len(df)),
                "distribucion_grados": {
                    str(grado): int(cantidad)
                    for grado, cantidad in df["grado"]
                    .value_counts()
                    .to_dict()
                    .items()
                },
                "configuracion_vectorizador": {
                    "max_features":
                        VECTORIZADOR_CONFIG[
                            "max_features"
                        ],
                    "ngram_range": list(
                        VECTORIZADOR_CONFIG[
                            "ngram_range"
                        ]
                    ),
                    "max_df":
                        VECTORIZADOR_CONFIG["max_df"],
                    "min_df":
                        VECTORIZADOR_CONFIG["min_df"],
                },
                "metodo": (
                    "Z-Score sobre frecuencia relativa por grado, "
                    "con filtro de soporte documental"
                ),
                "filtro_soporte_documental": {
                    "minimo_absoluto":
                        MIN_DOCUMENTOS_ABSOLUTO,
                    "proporcion_minima":
                        SOPORTE_MINIMO_PROPORCIONAL,
                    "regla": (
                        "max(minimo_absoluto, "
                        "ceil(n_documentos_grado * proporcion_minima))"
                    ),
                },
                "terminos_distintivos": terminos_json,
            },
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # Gráfico
    # -------------------------------------------------------------------------

    rutas_graficos = generar_graficos(df_resultados)

    print("\n" + "=" * 76)
    print("PROCESO COMPLETADO")
    print("=" * 76)
    print(f"CSV:\n  {RUTA_SALIDA_CSV}")
    print(f"JSON:\n  {RUTA_SALIDA_JSON}")
    print("Gráficos:")
    for ruta in rutas_graficos:
        print(f"  {ruta}")


if __name__ == "__main__":
    main()
