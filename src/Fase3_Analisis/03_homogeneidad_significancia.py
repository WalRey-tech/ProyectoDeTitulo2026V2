# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    "homogeneidad_semantica",
)

RUTA_MATRIZ_CENTROIDES = os.path.join(
    RESULTADOS_DIR,
    "similitud_centroides_v2.csv",
)

RUTA_HOMOGENEIDAD_CLASE = os.path.join(
    RESULTADOS_DIR,
    "homogeneidad_por_clase_v2.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_homogeneidad_v2.json",
)

RUTA_GRAFICO_CENTROIDES = os.path.join(
    RESULTADOS_DIR,
    "similitud_centroides_v2.png",
)

RUTA_GRAFICO_PERMUTACION = os.path.join(
    RESULTADOS_DIR,
    "test_permutacion_homogeneidad_v2.png",
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

RANDOM_STATE = 42

N_PERMUTACIONES = 5000

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
}


# =============================================================================
# 3. CARGA Y VALIDACIÓN
# =============================================================================

def cargar_corpus() -> pd.DataFrame:

    if not os.path.exists(RUTA_ENTRADA):
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
            "El corpus V2 no contiene las columnas "
            f"requeridas: {sorted(faltantes)}"
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
# 4. VECTORIZACIÓN
# =============================================================================

def vectorizar(
    textos: pd.Series,
):

    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X = vectorizador.fit_transform(
        textos.astype(str)
    )

    return (
        vectorizador,
        X.toarray(),
    )


# =============================================================================
# 5. SIMILITUD INTRA / INTER
# =============================================================================

def obtener_similitudes_intra_inter(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
):

    n = len(etiquetas)

    indices = np.triu_indices(
        n,
        k=1,
    )

    similitudes = matriz_similitud[
        indices
    ]

    mismas_clases = (
        etiquetas[:, None]
        == etiquetas[None, :]
    )[indices]

    intra = similitudes[
        mismas_clases
    ]

    inter = similitudes[
        ~mismas_clases
    ]

    return (
        intra,
        inter,
    )


def calcular_estadistico(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
) -> float:

    intra, inter = obtener_similitudes_intra_inter(
        matriz_similitud,
        etiquetas,
    )

    return float(
        intra.mean()
        - inter.mean()
    )


# =============================================================================
# 6. HOMOGENEIDAD POR CLASE
# =============================================================================

def calcular_homogeneidad_por_clase(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
) -> pd.DataFrame:

    resultados = []

    clases = sorted(
        np.unique(etiquetas)
    )

    for clase in clases:

        indices = np.where(
            etiquetas == clase
        )[0]

        n = len(indices)

        submatriz = matriz_similitud[
            np.ix_(
                indices,
                indices,
            )
        ]

        pares = np.triu_indices(
            n,
            k=1,
        )

        if len(pares[0]) > 0:
            valores = submatriz[
                pares
            ]

            media = float(
                valores.mean()
            )

            desviacion = float(
                valores.std(
                    ddof=1
                )
            ) if len(valores) > 1 else 0.0

            n_pares = int(
                len(valores)
            )

        else:
            media = float(
                "nan"
            )

            desviacion = float(
                "nan"
            )

            n_pares = 0

        resultados.append(
            {
                "grado": clase,
                "n_perfiles": int(n),
                "n_pares": n_pares,
                "similitud_intra_media":
                    media,
                "similitud_intra_std":
                    desviacion,
            }
        )

    return pd.DataFrame(
        resultados
    )


# =============================================================================
# 7. CENTROIDES
# =============================================================================

def calcular_centroides(
    X: np.ndarray,
    etiquetas: np.ndarray,
):

    clases = sorted(
        np.unique(etiquetas)
    )

    centroides = []

    for clase in clases:

        vectores = X[
            etiquetas == clase
        ]

        centroides.append(
            vectores.mean(
                axis=0
            )
        )

    centroides = np.asarray(
        centroides
    )

    matriz = cosine_similarity(
        centroides
    )

    return (
        clases,
        centroides,
        matriz,
    )


# =============================================================================
# 8. TEST DE PERMUTACIÓN
# =============================================================================

def ejecutar_test_permutacion(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    observado = calcular_estadistico(
        matriz_similitud,
        etiquetas,
    )

    permutados = np.empty(
        N_PERMUTACIONES,
        dtype=float,
    )

    for i in range(
        N_PERMUTACIONES
    ):

        etiquetas_permutadas = (
            rng.permutation(
                etiquetas
            )
        )

        permutados[i] = calcular_estadistico(
            matriz_similitud,
            etiquetas_permutadas,
        )

    # Test unilateral:
    # H1: similitud intra-clase > similitud inter-clase.
    #
    # La corrección +1 evita reportar un p-valor
    # exactamente igual a cero.
    extremos = int(
        np.sum(
            permutados
            >= observado
        )
    )

    p_valor = (
        extremos + 1
    ) / (
        N_PERMUTACIONES + 1
    )

    return (
        observado,
        permutados,
        extremos,
        float(p_valor),
    )


# =============================================================================
# 9. GRÁFICO DE CENTROIDES
# =============================================================================

def generar_grafico_centroides(
    matriz: np.ndarray,
    clases: list[str],
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    imagen = ax.imshow(
        matriz,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )

    ax.set_xticks(
        np.arange(
            len(clases)
        )
    )

    ax.set_yticks(
        np.arange(
            len(clases)
        )
    )

    ax.set_xticklabels(
        clases,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        clases
    )

    for i in range(
        len(clases)
    ):

        for j in range(
            len(clases)
        ):

            ax.text(
                j,
                i,
                f"{matriz[i, j]:.3f}",
                ha="center",
                va="center",
            )

    ax.set_title(
        (
            "Similitud coseno entre centroides\n"
            "Perfiles de egreso — Corpus V2"
        ),
        fontweight="bold",
    )

    barra = fig.colorbar(
        imagen,
        ax=ax,
    )

    barra.set_label(
        "Similitud coseno"
    )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_CENTROIDES,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 10. GRÁFICO DEL TEST DE PERMUTACIÓN
# =============================================================================

def generar_grafico_permutacion(
    observado: float,
    permutados: np.ndarray,
    p_valor: float,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        permutados,
        bins=40,
        alpha=0.8,
        edgecolor="black",
    )

    ax.axvline(
        observado,
        linestyle="--",
        linewidth=2,
        label=(
            "Δ observado = "
            f"{observado:.4f}"
        ),
    )

    ax.set_title(
        (
            "Test de permutación — "
            "Homogeneidad semántica"
        ),
        fontweight="bold",
    )

    ax.set_xlabel(
        (
            "Δ similitud "
            "(intra-clase − inter-clase)"
        )
    )

    ax.set_ylabel(
        "Frecuencia"
    )

    ax.legend()

    ax.text(
        0.98,
        0.95,
        f"p = {p_valor:.5f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.15,
        },
    )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_PERMUTACION,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 11. RESUMEN JSON
# =============================================================================

def guardar_resumen(
    df: pd.DataFrame,
    vectorizador: TfidfVectorizer,
    matriz_centroides: np.ndarray,
    clases: list[str],
    intra: np.ndarray,
    inter: np.ndarray,
    observado: float,
    extremos: int,
    p_valor: float,
    homogeneidad_clase: pd.DataFrame,
) -> dict:

    distribucion = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    matriz_dict = {}

    for i, clase_i in enumerate(
        clases
    ):

        matriz_dict[
            clase_i
        ] = {}

        for j, clase_j in enumerate(
            clases
        ):

            matriz_dict[
                clase_i
            ][
                clase_j
            ] = round(
                float(
                    matriz_centroides[
                        i,
                        j,
                    ]
                ),
                6,
            )

    macro_intra = float(
        homogeneidad_clase[
            "similitud_intra_media"
        ].mean()
    )

    significativo = bool(
        p_valor < 0.05
    )

    resumen = {

        "corpus": {
            "total": int(
                len(df)
            ),

            "distribucion": {
                str(clase):
                    int(cantidad)

                for clase, cantidad
                in distribucion.items()
            },
        },

        "representacion": {
            "metodo":
                "TF-IDF",

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

            "sublinear_tf":
                True,
        },

        "similitud_global": {

            "intra_clase_media":
                round(
                    float(
                        intra.mean()
                    ),
                    6,
                ),

            "inter_clase_media":
                round(
                    float(
                        inter.mean()
                    ),
                    6,
                ),

            "diferencia_intra_inter":
                round(
                    float(
                        observado
                    ),
                    6,
                ),

            "homogeneidad_intra_macro":
                round(
                    macro_intra,
                    6,
                ),
        },

        "test_permutacion": {

            "hipotesis_nula":
                (
                    "La asociación entre perfiles y "
                    "grados no produce una diferencia "
                    "intra/inter mayor a la esperada "
                    "por azar."
                ),

            "hipotesis_alternativa":
                (
                    "La similitud intra-grado es mayor "
                    "que la similitud inter-grado."
                ),

            "tipo":
                "unilateral",

            "n_permutaciones":
                N_PERMUTACIONES,

            "semilla":
                RANDOM_STATE,

            "permutaciones_extremas":
                extremos,

            "p_valor":
                round(
                    p_valor,
                    8,
                ),

            "alpha":
                0.05,

            "significativo":
                significativo,
        },

        "similitud_centroides":
            matriz_dict,

        "interpretacion": {

            "resultado":
                (
                    "Los perfiles del mismo grado "
                    "presentan una similitud promedio "
                    "mayor que los perfiles de grados "
                    "diferentes."
                    if observado > 0
                    else
                    "No se observó una similitud "
                    "intra-grado superior a la "
                    "similitud inter-grado."
                ),

            "significancia":
                (
                    "La diferencia observada es "
                    "estadísticamente significativa "
                    "bajo el test de permutación."
                    if significativo
                    else
                    "La diferencia observada no "
                    "alcanza significancia estadística "
                    "bajo el test de permutación."
                ),

            "advertencia":
                (
                    "Este es un análisis descriptivo "
                    "y de asociación del corpus. "
                    "No constituye una estimación "
                    "de rendimiento predictivo."
                ),
        },

        "limitaciones": [

            (
                "La clase Ejecución contiene solo "
                "cinco perfiles, por lo que sus "
                "estimaciones presentan mayor "
                "incertidumbre."
            ),

            (
                "El corpus utiliza el texto original "
                "de los perfiles de egreso. Algunos "
                "documentos pueden contener términos "
                "directamente asociados al nombre "
                "del programa o grado."
            ),

            (
                "La similitud global intra-clase está "
                "influida por el número de pares de "
                "cada clase; por ello también se "
                "reporta homogeneidad por clase."
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
        "HOMOGENEIDAD Y SIGNIFICANCIA SEMÁNTICA — CORPUS V2"
    )

    print(
        "=" * 72
    )

    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True,
    )

    df = cargar_corpus()

    etiquetas = (
        df["grado"]
        .astype(str)
        .to_numpy()
    )

    print(
        f"\nCorpus: {len(df)} perfiles"
    )

    print(
        "Distribución:"
    )

    print(
        df["grado"]
        .value_counts()
    )

    vectorizador, X = vectorizar(
        df["perfil_egreso"]
    )

    print(
        f"\nTF-IDF generado: "
        f"{X.shape[1]} características"
    )

    matriz_similitud = cosine_similarity(
        X
    )

    intra, inter = obtener_similitudes_intra_inter(
        matriz_similitud,
        etiquetas,
    )

    clases, _, matriz_centroides = calcular_centroides(
        X,
        etiquetas,
    )

    df_centroides = pd.DataFrame(
        matriz_centroides,
        index=clases,
        columns=clases,
    )

    df_centroides.index.name = (
        "grado"
    )

    df_centroides.to_csv(
        RUTA_MATRIZ_CENTROIDES,
        encoding="utf-8-sig",
    )

    homogeneidad_clase = calcular_homogeneidad_por_clase(
        matriz_similitud,
        etiquetas,
    )

    homogeneidad_clase.to_csv(
        RUTA_HOMOGENEIDAD_CLASE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nSimilitud entre centroides:"
    )

    print(
        df_centroides.round(
            4
        )
    )

    print(
        "\nHomogeneidad por clase:"
    )

    print(
        homogeneidad_clase.round(
            4
        ).to_string(
            index=False
        )
    )

    print(
        "\nEjecutando test de permutación..."
    )

    (
        observado,
        permutados,
        extremos,
        p_valor,
    ) = ejecutar_test_permutacion(
        matriz_similitud,
        etiquetas,
    )

    generar_grafico_centroides(
        matriz_centroides,
        clases,
    )

    generar_grafico_permutacion(
        observado,
        permutados,
        p_valor,
    )

    resumen = guardar_resumen(
        df,
        vectorizador,
        matriz_centroides,
        clases,
        intra,
        inter,
        observado,
        extremos,
        p_valor,
        homogeneidad_clase,
    )

    print(
        "\nResultados globales:"
    )

    print(
        "  Similitud intra-clase media : "
        f"{intra.mean():.4f}"
    )

    print(
        "  Similitud inter-clase media : "
        f"{inter.mean():.4f}"
    )

    print(
        "  Δ intra - inter             : "
        f"{observado:.4f}"
    )

    print(
        "  Homogeneidad intra macro    : "
        f"{resumen['similitud_global']['homogeneidad_intra_macro']:.4f}"
    )

    print(
        "\nTest de permutación:"
    )

    print(
        f"  Permutaciones : "
        f"{N_PERMUTACIONES}"
    )

    print(
        f"  Extremas      : "
        f"{extremos}"
    )

    print(
        f"  p-valor       : "
        f"{p_valor:.6f}"
    )

    print(
        "  Significativo : "
        f"{'Sí' if p_valor < 0.05 else 'No'}"
    )

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Matriz centroides:\n  "
        f"{RUTA_MATRIZ_CENTROIDES}"
    )

    print(
        f"  Homogeneidad por clase:\n  "
        f"{RUTA_HOMOGENEIDAD_CLASE}"
    )

    print(
        f"  Resumen:\n  "
        f"{RUTA_RESUMEN}"
    )

    print(
        f"  Gráfico centroides:\n  "
        f"{RUTA_GRAFICO_CENTROIDES}"
    )

    print(
        f"  Gráfico permutación:\n  "
        f"{RUTA_GRAFICO_PERMUTACION}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Este análisis describe estructura semántica "
        "del corpus; no mide rendimiento predictivo."
    )

    print(
        "El texto original puede contener términos "
        "asociados directamente al nombre del grado."
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ANÁLISIS DE HOMOGENEIDAD COMPLETADO CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )