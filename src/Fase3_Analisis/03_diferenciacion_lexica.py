# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


# =============================================================================
# 1. RUTAS
# =============================================================================

DIRECTORIO_ACTUAL = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_ROOT = os.path.abspath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
    )
)

RUTA_ENTRADA = os.path.join(
    SRC_ROOT,
    "data",
    "processed",
    "perfiles_egreso_etiquetado_v2.csv",
)

RESULTADOS_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "diferenciacion_lexica",
)

RUTA_TERMINOS = os.path.join(
    RESULTADOS_DIR,
    "terminos_distintivos_v2.csv",
)

RUTA_AUDITORIA = os.path.join(
    RESULTADOS_DIR,
    "auditoria_terminos_excluidos_v2.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_diferenciacion_lexica_v2.json",
)

RUTA_GRAFICO = os.path.join(
    RESULTADOS_DIR,
    "terminos_distintivos_v2.png",
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

TOP_N = 15

# Suavizado para evitar divisiones por cero.
ALPHA = 0.5

# Un término debe aparecer en al menos dos perfiles
# de la clase para entrar en el ranking.
MIN_DOCUMENTOS_CLASE = 2

VECTORIZADOR_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "lowercase": True,
    "strip_accents": "unicode",
    "binary": True,
}


# =============================================================================
# 3. STOPWORDS PARA INTERPRETACIÓN
# =============================================================================
#
# Se utilizan SOLO en este análisis descriptivo.
#
# NO modifican:
# - el corpus V2;
# - GWO;
# - ComplementNB;
# - las validaciones.
# =============================================================================

STOPWORDS_ES = {
    "a", "al", "algo", "algunas", "algunos",
    "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde",
    "donde", "dos", "durante", "e", "el",
    "ella", "ellas", "ellos", "en", "entre",
    "era", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estas", "este", "esto",
    "estos", "fue", "ha", "hacia", "hasta",
    "hay", "la", "las", "le", "les", "lo",
    "los", "mas", "me", "mi", "mis", "muy",
    "no", "nos", "o", "para", "pero", "por",
    "porque", "que", "se", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tambien",
    "tanto", "te", "tiene", "todo", "todos",
    "tras", "tu", "un", "una", "uno", "unos",
    "y", "ya",
}


# =============================================================================
# 4. TÉRMINOS EXCLUIDOS DE LA INTERPRETACIÓN
# =============================================================================

TOKENS_GRADO = {
    "civil",
    "ejecucion",
    "informatica",
    "informatico",
    "informaticos",
    "informaticas",
    "ingenieria",
    "ingeniero",
    "ingenieros",
    "ingeniera",
    "ingenieras",
}


TOKENS_INSTITUCIONALES = {
    "universidad",
    "instituto",
    "magister",
    "valparaiso",
    "chile",
}


# =============================================================================
# 5. UTILIDADES
# =============================================================================

def normalizar_texto(
    texto: str,
) -> str:

    texto = str(texto).lower()

    texto = unicodedata.normalize(
        "NFD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return texto


def clasificar_exclusion(
    termino: str,
) -> str | None:

    normalizado = normalizar_texto(
        termino
    )

    tokens = set(
        normalizado.split()
    )

    if tokens.intersection(
        TOKENS_GRADO
    ):
        return "denominacion_grado"

    if tokens.intersection(
        TOKENS_INSTITUCIONALES
    ):
        return "contexto_institucional_geografico"

    return None


# =============================================================================
# 6. CARGA DEL CORPUS
# =============================================================================

def cargar_corpus() -> pd.DataFrame:

    if not os.path.exists(
        RUTA_ENTRADA
    ):
        raise FileNotFoundError(
            "No se encontró el corpus V2:\n"
            f"{RUTA_ENTRADA}"
        )

    df = pd.read_csv(
        RUTA_ENTRADA,
        encoding="utf-8-sig",
    )

    columnas_requeridas = {
        "perfil_egreso",
        "grado",
    }

    faltantes = (
        columnas_requeridas
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{sorted(faltantes)}"
        )

    df = df.dropna(
        subset=[
            "perfil_egreso",
            "grado",
        ]
    ).copy()

    df = df[
        df["perfil_egreso"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    return df


# =============================================================================
# 7. MATRIZ DE PRESENCIA DOCUMENTAL
# =============================================================================

def construir_matriz(
    df: pd.DataFrame,
):

    config = dict(
        VECTORIZADOR_CONFIG
    )

    config["stop_words"] = sorted(
        STOPWORDS_ES
    )

    vectorizador = CountVectorizer(
        **config
    )

    X = vectorizador.fit_transform(
        df["perfil_egreso"]
        .astype(str)
    )

    vocabulario = (
        vectorizador
        .get_feature_names_out()
    )

    return (
        vectorizador,
        X,
        vocabulario,
    )


# =============================================================================
# 8. RAZÓN DE PREVALENCIA DOCUMENTAL
# =============================================================================

def calcular_puntuacion(
    documentos_clase: np.ndarray,
    documentos_resto: np.ndarray,
    n_clase: int,
    n_resto: int,
) -> np.ndarray:

    prevalencia_clase = (
        documentos_clase
        + ALPHA
    ) / (
        n_clase
        + 2 * ALPHA
    )

    prevalencia_resto = (
        documentos_resto
        + ALPHA
    ) / (
        n_resto
        + 2 * ALPHA
    )

    return np.log2(
        prevalencia_clase
        / prevalencia_resto
    )


# =============================================================================
# 9. GENERAR RANKINGS
# =============================================================================

def generar_rankings(
    df: pd.DataFrame,
    X,
    vocabulario: np.ndarray,
):

    etiquetas = (
        df["grado"]
        .astype(str)
        .to_numpy()
    )

    clases = sorted(
        np.unique(
            etiquetas
        )
    )

    resultados = []
    excluidos = []

    for clase in clases:

        mascara = (
            etiquetas == clase
        )

        n_clase = int(
            mascara.sum()
        )

        n_resto = int(
            (~mascara).sum()
        )

        documentos_clase = np.asarray(
            X[mascara]
            .sum(axis=0)
        ).ravel()

        documentos_resto = np.asarray(
            X[~mascara]
            .sum(axis=0)
        ).ravel()

        puntuaciones = calcular_puntuacion(
            documentos_clase,
            documentos_resto,
            n_clase,
            n_resto,
        )

        candidatos = []

        for indice, termino in enumerate(
            vocabulario
        ):

            n_docs_clase = int(
                documentos_clase[
                    indice
                ]
            )

            n_docs_resto = int(
                documentos_resto[
                    indice
                ]
            )

            if (
                n_docs_clase
                < MIN_DOCUMENTOS_CLASE
            ):
                continue

            motivo = clasificar_exclusion(
                termino
            )

            fila = {
                "grado":
                    clase,

                "termino":
                    str(
                        termino
                    ),

                "log2_ratio_prevalencia":
                    float(
                        puntuaciones[
                            indice
                        ]
                    ),

                "documentos_grado":
                    n_docs_clase,

                "total_documentos_grado":
                    n_clase,

                "documentos_resto":
                    n_docs_resto,

                "total_documentos_resto":
                    n_resto,

                "prevalencia_grado":
                    float(
                        n_docs_clase
                        / n_clase
                    ),

                "prevalencia_resto":
                    float(
                        n_docs_resto
                        / n_resto
                    ),
            }

            if motivo is not None:

                fila[
                    "motivo_exclusion"
                ] = motivo

                excluidos.append(
                    fila
                )

                continue

            candidatos.append(
                fila
            )

        candidatos = sorted(
            candidatos,
            key=lambda x: (
                x[
                    "log2_ratio_prevalencia"
                ],
                x[
                    "documentos_grado"
                ],
            ),
            reverse=True,
        )

        for ranking, fila in enumerate(
            candidatos[:TOP_N],
            start=1,
        ):

            fila = fila.copy()

            fila[
                "ranking"
            ] = ranking

            resultados.append(
                fila
            )

    return (
        clases,
        pd.DataFrame(
            resultados
        ),
        pd.DataFrame(
            excluidos
        ),
    )


# =============================================================================
# 10. GRÁFICO
# =============================================================================

def generar_grafico(
    df_resultados: pd.DataFrame,
) -> None:

    clases = sorted(
        df_resultados[
            "grado"
        ].unique()
    )

    fig, axes = plt.subplots(
        len(clases),
        1,
        figsize=(
            11,
            5 * len(clases),
        ),
    )

    if len(clases) == 1:
        axes = [
            axes
        ]

    for ax, clase in zip(
        axes,
        clases,
    ):

        datos = (
            df_resultados[
                df_resultados[
                    "grado"
                ] == clase
            ]
            .sort_values(
                "log2_ratio_prevalencia",
                ascending=True,
            )
            .tail(10)
        )

        ax.barh(
            datos[
                "termino"
            ],
            datos[
                "log2_ratio_prevalencia"
            ],
        )

        ax.set_title(
            (
                f"{clase} — "
                "Términos distintivos"
            ),
            fontweight="bold",
        )

        ax.set_xlabel(
            (
                "Log2 razón de prevalencia "
                "documental (grado vs. resto)"
            )
        )

        ax.set_ylabel(
            "Término"
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

    fig.suptitle(
        (
            "Diferenciación léxica — Corpus V2\n"
            "Presencia documental; términos de grado e "
            "institucionales excluidos de la interpretación"
        ),
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    fig.savefig(
        RUTA_GRAFICO,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 11. RESUMEN
# =============================================================================

def guardar_resumen(
    df: pd.DataFrame,
    vectorizador: CountVectorizer,
    df_resultados: pd.DataFrame,
    df_excluidos: pd.DataFrame,
) -> dict:

    distribucion = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    top_por_grado = {}

    for clase in sorted(
        df_resultados[
            "grado"
        ].unique()
    ):

        datos = df_resultados[
            df_resultados[
                "grado"
            ] == clase
        ]

        top_por_grado[
            clase
        ] = []

        for _, fila in datos.iterrows():

            top_por_grado[
                clase
            ].append(
                {
                    "ranking":
                        int(
                            fila[
                                "ranking"
                            ]
                        ),

                    "termino":
                        str(
                            fila[
                                "termino"
                            ]
                        ),

                    "log2_ratio_prevalencia":
                        round(
                            float(
                                fila[
                                    "log2_ratio_prevalencia"
                                ]
                            ),
                            6,
                        ),

                    "documentos_grado":
                        int(
                            fila[
                                "documentos_grado"
                            ]
                        ),

                    "total_documentos_grado":
                        int(
                            fila[
                                "total_documentos_grado"
                            ]
                        ),

                    "documentos_resto":
                        int(
                            fila[
                                "documentos_resto"
                            ]
                        ),

                    "total_documentos_resto":
                        int(
                            fila[
                                "total_documentos_resto"
                            ]
                        ),
                }
            )

    motivos = {}

    if not df_excluidos.empty:

        motivos = (
            df_excluidos[
                "motivo_exclusion"
            ]
            .value_counts()
            .astype(int)
            .to_dict()
        )

    resumen = {

        "corpus": {

            "total":
                int(
                    len(df)
                ),

            "distribucion": {

                str(clase):
                    int(cantidad)

                for clase, cantidad
                in distribucion.items()
            },
        },

        "metodo": {

            "unidad_analisis":
                (
                    "presencia del término "
                    "por documento"
                ),

            "vectorizador":
                "CountVectorizer(binary=True)",

            "max_features":
                400,

            "features_generadas":
                int(
                    len(
                        vectorizador
                        .get_feature_names_out()
                    )
                ),

            "ngram_range": [
                1,
                2,
            ],

            "min_df":
                2,

            "max_df":
                0.90,

            "min_documentos_clase":
                MIN_DOCUMENTOS_CLASE,

            "metrica":
                (
                    "log2 razón de prevalencia "
                    "documental clase vs. resto"
                ),

            "suavizado_alpha":
                ALPHA,
        },

        "filtrado_interpretativo": {

            "stopwords_espanol":
                True,

            "exclusion_denominacion_grado":
                True,

            "exclusion_contexto_institucional":
                True,

            "afecta_corpus_original":
                False,

            "afecta_gwo":
                False,

            "registros_excluidos":
                int(
                    len(
                        df_excluidos
                    )
                ),

            "exclusiones_por_motivo":
                motivos,
        },

        "top_terminos_por_grado":
            top_por_grado,

        "interpretacion": {

            "uso":
                (
                    "Identificar vocabulario "
                    "relativamente más extendido "
                    "entre los perfiles de cada grado."
                ),

            "advertencia":
                (
                    "Un término distintivo describe "
                    "una diferencia léxica en este "
                    "corpus y no implica que sea una "
                    "competencia exclusiva del grado."
                ),
        },

        "limitaciones": [

            (
                "La clase Ejecución contiene únicamente "
                "cinco perfiles."
            ),

            (
                "Los términos institucionales y de "
                "denominación del grado se excluyen "
                "solo del análisis interpretativo."
            ),

            (
                "Este análisis no modifica ni sustituye "
                "el experimento predictivo GWO."
            ),
        ],
    }

    with open(
        RUTA_RESUMEN,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    return resumen


# =============================================================================
# 12. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 72
    )

    print(
        "DIFERENCIACIÓN LÉXICA INTERPRETABLE — CORPUS V2"
    )

    print(
        "=" * 72
    )

    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True,
    )

    df = cargar_corpus()

    print(
        f"\nCorpus: {len(df)} perfiles"
    )

    print(
        "Distribución:"
    )

    print(
        df[
            "grado"
        ].value_counts()
    )

    (
        vectorizador,
        X,
        vocabulario,
    ) = construir_matriz(
        df
    )

    print(
        f"\nVocabulario interpretable: "
        f"{len(vocabulario)} características"
    )

    (
        clases,
        df_resultados,
        df_excluidos,
    ) = generar_rankings(
        df,
        X,
        vocabulario,
    )

    df_resultados.to_csv(
        RUTA_TERMINOS,
        index=False,
        encoding="utf-8-sig",
    )

    df_excluidos.to_csv(
        RUTA_AUDITORIA,
        index=False,
        encoding="utf-8-sig",
    )

    generar_grafico(
        df_resultados
    )

    resumen = guardar_resumen(
        df,
        vectorizador,
        df_resultados,
        df_excluidos,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TOP TÉRMINOS DISTINTIVOS"
    )

    print(
        "=" * 72
    )

    for clase in clases:

        print(
            f"\n{clase.upper()}"
        )

        print(
            "-" * 60
        )

        datos = df_resultados[
            df_resultados[
                "grado"
            ] == clase
        ]

        for _, fila in datos.iterrows():

            print(
                f"{int(fila['ranking']):>2}. "
                f"{fila['termino']:<28} "
                f"log2={fila['log2_ratio_prevalencia']:.3f} "
                f"| grado "
                f"{int(fila['documentos_grado'])}/"
                f"{int(fila['total_documentos_grado'])} "
                f"| resto "
                f"{int(fila['documentos_resto'])}/"
                f"{int(fila['total_documentos_resto'])}"
            )

    print(
        "\nAuditoría interpretativa:"
    )

    print(
        f"  Registros excluidos: "
        f"{len(df_excluidos)}"
    )

    for motivo, cantidad in (
        resumen[
            "filtrado_interpretativo"
        ][
            "exclusiones_por_motivo"
        ].items()
    ):

        print(
            f"  {motivo}: {cantidad}"
        )

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Ranking:\n  "
        f"{RUTA_TERMINOS}"
    )

    print(
        f"  Auditoría:\n  "
        f"{RUTA_AUDITORIA}"
    )

    print(
        f"  Resumen:\n  "
        f"{RUTA_RESUMEN}"
    )

    print(
        f"  Gráfico:\n  "
        f"{RUTA_GRAFICO}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "El filtrado se utiliza únicamente para "
        "interpretación léxica."
    )

    print(
        "No modifica el corpus V2 ni los "
        "resultados GWO."
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ANÁLISIS LÉXICO COMPLETADO CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )