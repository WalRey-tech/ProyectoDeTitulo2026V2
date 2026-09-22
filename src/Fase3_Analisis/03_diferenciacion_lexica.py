# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
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

CORPUS_SELECCIONADO = "actual"


def configurar_corpus(corpus: str) -> None:
    global CORPUS_SELECCIONADO, RUTA_ENTRADA, RESULTADOS_DIR
    global RUTA_TERMINOS, RUTA_AUDITORIA, RUTA_RESUMEN, RUTA_GRAFICO
    if corpus not in {"actual", "v2"}:
        raise ValueError("Corpus no válido: usa actual o v2.")
    CORPUS_SELECCIONADO = corpus
    nombre = ("perfiles_egreso_etiquetado_actual_corregido.csv"
              if corpus == "actual" else "perfiles_egreso_etiquetado_v2.csv")
    RUTA_ENTRADA = os.path.join(SRC_ROOT, "data", "processed", nombre)
    RESULTADOS_DIR = os.path.join(SRC_ROOT, "data", "resultados_cientificos", "diferenciacion_lexica")
    def ruta(nombre, extension):
        return os.path.join(RESULTADOS_DIR, f"{nombre}_{corpus}.{extension}")
    RUTA_TERMINOS = ruta("terminos_distintivos", "csv")
    RUTA_AUDITORIA = ruta("auditoria_terminos_excluidos", "csv")
    RUTA_RESUMEN = ruta("resumen_diferenciacion_lexica", "json")
    RUTA_GRAFICO = ruta("terminos_distintivos", "png")


def solicitar_corpus() -> str:
    print("\nSelecciona el corpus:")
    print("1. Actual — salida corregida del encoding")
    print("2. V2 — corpus histórico")
    opciones = {"1": "actual", "actual": "actual", "2": "v2", "v2": "v2"}
    while True:
        try:
            respuesta = input("Opción [1/2]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Selección cancelada. Usa --corpus actual o --corpus v2.") from None
        if respuesta in opciones:
            return opciones[respuesta]
        print("Opción no válida. Escribe 1 o 2.")


def sha256_archivo(ruta: str) -> str:
    with open(ruta, "rb") as archivo:
        return hashlib.sha256(archivo.read()).hexdigest()


configurar_corpus("actual")


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
# - el corpus de entrada;
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
    if not os.path.isfile(RUTA_ENTRADA):
        raise FileNotFoundError(
            f"No se encontró el corpus seleccionado ({CORPUS_SELECCIONADO}):\n"
            f"{RUTA_ENTRADA}\n"
            "Para actual, ejecuta primero el encoding con --corpus actual."
        )
    huella = sha256_archivo(RUTA_ENTRADA)
    # Admite los CSV históricos con comas y los actuales con punto y coma.
    df = pd.read_csv(RUTA_ENTRADA, sep=None, engine="python",
                     encoding="utf-8-sig", keep_default_na=False)
    if sha256_archivo(RUTA_ENTRADA) != huella:
        raise ValueError("El CSV cambió durante la lectura. Repite la ejecución.")
    faltantes = {"perfil_egreso", "grado"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    if df.empty:
        raise ValueError("El corpus está vacío.")
    for columna in ["perfil_egreso", "grado"]:
        vacias = df[columna].astype(str).str.strip().eq("")
        if vacias.any():
            raise ValueError(
                f"Hay {int(vacias.sum())} filas sin {columna}; "
                "corrige el corpus en fase 2. No se eliminaron filas."
            )
    desconocidas = set(df["grado"]) - {"Civil", "Informática", "Ejecución"}
    if desconocidas:
        raise ValueError(f"Grados fuera del catálogo: {sorted(desconocidas)}")
    for columna in ["estado_registro", "estado_etiquetado"]:
        if columna in df:
            pendientes = df[columna].astype(str).str.strip().str.upper().isin(["REVISAR", "ERROR"])
            if pendientes.any():
                raise ValueError(f"El CSV contiene filas REVISAR/ERROR en {columna}.")
    conteos = df["grado"].value_counts()
    if len(conteos) != 3 or conteos.min() < 2:
        raise ValueError("El análisis requiere las tres clases, con al menos dos perfiles cada una.")
    df.attrs["sha256_entrada"] = huella
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

    if ALPHA <= 0 or n_clase <= 0 or n_resto <= 0:
        raise ValueError("El suavizado y los tamaños de clase deben ser positivos.")
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

            if motivo is None and puntuaciones[indice] <= 0:
                motivo = "prevalencia_suavizada_no_superior_al_resto"

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
            key=lambda x: (-x["log2_ratio_prevalencia"],
                           -x["documentos_grado"], x["termino"]),
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

    columnas = ["grado", "termino", "log2_ratio_prevalencia", "documentos_grado",
                "total_documentos_grado", "documentos_resto", "total_documentos_resto",
                "prevalencia_grado", "prevalencia_resto"]
    return (clases, pd.DataFrame(resultados, columns=columnas + ["ranking"]),
            pd.DataFrame(excluidos, columns=columnas + ["motivo_exclusion"]))


# =============================================================================
# 10. GRÁFICO
# =============================================================================

def generar_grafico(
    df_resultados: pd.DataFrame,
) -> None:

    clases = ["Civil", "Ejecución", "Informática"]

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
            .tail(TOP_N)
        )

        if datos.empty:
            ax.text(0.5, 0.5, "Sin términos que cumplan los criterios", ha="center",
                    va="center", transform=ax.transAxes)
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
            f"Diferenciación léxica — Corpus {CORPUS_SELECCIONADO.upper()}\n"
            "Presencia documental; exclusión de términos de grado e "
            "institucionales según listas declaradas"
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

    top_por_grado = {str(clase): [] for clase in sorted(distribucion)}

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

        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
        "version_analisis": "corpus_seleccionable_ranking_positivo_v1",
        "corpus": {
            "seleccion": CORPUS_SELECCIONADO,
            "archivo_entrada": RUTA_ENTRADA,
            "sha256": df.attrs["sha256_entrada"],
            "filas_descartadas": 0,

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
            "top_maximo_por_grado": TOP_N,
            "criterio_puntuacion": "log2_ratio_prevalencia > 0",
            "formula": "log2(((docs_grado+alpha)/(n_grado+2*alpha))/((docs_resto+alpha)/(n_resto+2*alpha)))",
            "prevalencias_csv": "proporciones sin suavizar; la puntuación aplica alpha",
            "desempate": "puntuación descendente, documentos del grado descendente, término alfabético",

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
            "tokens_grado": sorted(TOKENS_GRADO),
            "tokens_institucionales": sorted(TOKENS_INSTITUCIONALES),
            "lista_stopwords": sorted(STOPWORDS_ES),
            "alcance_auditoria": "pares grado-término del vocabulario con soporte mínimo en el grado, descartados por listas o puntuación no positiva",
            "orden": "stopwords antes de vectorizar; exclusiones por tokens después de seleccionar el vocabulario",

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
                f"La clase Ejecución contiene {distribucion.get('Ejecución', 0)} perfiles."
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

    resumen["grupos_perfil_repetidos"] = {}
    if "grupo_perfil" in df:
        grupos = df["grupo_perfil"].astype(str).str.strip()
        conteos = grupos[grupos.ne("")].value_counts()
        resumen["grupos_perfil_repetidos"] = {str(k): int(v) for k,v in conteos.items() if v > 1}
    resumen["limitaciones"].extend([
        "Cada perfil cuenta como documento, también los de modalidades similares. No se agrupan en este análisis descriptivo.",
        "Los perfiles similares pueden elevar la prevalencia de términos de una plantilla institucional; el ranking no es una prueba de significancia.",
        "El filtrado institucional se limita a la lista declarada y no garantiza eliminar todos los nombres propios.",
        "Los términos excluidos después de vectorizar pueden ocupar parte del límite de 400 características.",
        "Los bigramas se forman tras retirar stopwords y pueden conectar palabras no contiguas en el original.",
    ])
    if sha256_archivo(RUTA_ENTRADA) != df.attrs["sha256_entrada"]:
        raise ValueError("El CSV cambió durante el análisis. Repite la ejecución.")
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
    parser = argparse.ArgumentParser(description="Diferenciación léxica descriptiva del corpus seleccionado.")
    parser.add_argument("--corpus", choices=["actual", "v2"], help="Si se omite, muestra el menú.")
    args = parser.parse_args()
    configurar_corpus(args.corpus or solicitar_corpus())

    print(
        "=" * 72
    )

    print(
        f"DIFERENCIACIÓN LÉXICA INTERPRETABLE — {CORPUS_SELECCIONADO.upper()}"
    )

    print(
        "=" * 72
    )

    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True,
    )

    print(f"Entrada seleccionada: {RUTA_ENTRADA}")
    df = cargar_corpus()
    print(f"SHA-256: {df.attrs['sha256_entrada']}")

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
        f"\nVocabulario antes del filtrado interpretativo: "
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

    if resumen["grupos_perfil_repetidos"]:
        print("Grupos compartidos:", resumen["grupos_perfil_repetidos"])
        print("Este ranking descriptivo cuenta cada perfil por separado.")
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

        if datos.empty:
            print("Sin términos que cumplan los criterios de frecuencia, exclusión y puntuación positiva.")
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
        f"  Pares grado-término excluidos: "
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
        "No modifica el corpus de entrada ni los "
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
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None