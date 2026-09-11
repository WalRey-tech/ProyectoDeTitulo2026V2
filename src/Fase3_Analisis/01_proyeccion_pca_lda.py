# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer


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
    "visualizaciones_exploratorias",
)

RUTA_GRAFICO = os.path.join(
    RESULTADOS_DIR,
    "proyeccion_pca_vs_lda_v2.png",
)

RUTA_COORDENADAS = os.path.join(
    RESULTADOS_DIR,
    "coordenadas_pca_lda_v2.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_pca_lda_v2.json",
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

RANDOM_STATE = 42

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
}


# =============================================================================
# 3. COLORES PARA VISUALIZACIÓN
# =============================================================================

COLORES = {
    "Civil": "#1f77b4",
    "Informática": "#ff7f0e",
    "Ejecución": "#2ca02c",
}


# =============================================================================
# 4. CARGA Y VALIDACIÓN
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
            "El corpus no contiene las columnas "
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
# 5. VECTORIZACIÓN
# =============================================================================

def vectorizar(
    textos: pd.Series,
):

    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X_tfidf = vectorizador.fit_transform(
        textos.astype(str)
    )

    return (
        vectorizador,
        X_tfidf.toarray(),
    )


# =============================================================================
# 6. PROYECCIONES
# =============================================================================

def calcular_proyecciones(
    X: np.ndarray,
    y: pd.Series,
):

    # PCA:
    # proyección no supervisada.
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    X_pca = pca.fit_transform(
        X
    )

    # LDA:
    # proyección supervisada.
    #
    # IMPORTANTE:
    # usa las etiquetas de grado y, por tanto,
    # solo se interpreta como visualización
    # descriptiva/exploratoria.
    lda = LinearDiscriminantAnalysis(
        n_components=2
    )

    X_lda = lda.fit_transform(
        X,
        y,
    )

    return (
        pca,
        X_pca,
        lda,
        X_lda,
    )


# =============================================================================
# 7. GUARDAR COORDENADAS
# =============================================================================

def guardar_coordenadas(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    X_lda: np.ndarray,
) -> None:

    coordenadas = pd.DataFrame(
        {
            "grado":
                df["grado"].astype(str).values,

            "PCA_1":
                X_pca[:, 0],

            "PCA_2":
                X_pca[:, 1],

            "LDA_1":
                X_lda[:, 0],

            "LDA_2":
                X_lda[:, 1],
        }
    )

    coordenadas.to_csv(
        RUTA_COORDENADAS,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 8. GENERACIÓN DEL GRÁFICO
# =============================================================================

def generar_grafico(
    y: pd.Series,
    X_pca: np.ndarray,
    X_lda: np.ndarray,
    varianza_pca: np.ndarray,
) -> None:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
    )

    fig.suptitle(
        (
            "Perfiles de egreso V2 — "
            "Proyección PCA vs. LDA"
        ),
        fontsize=16,
        fontweight="bold",
    )

    clases = [
        "Civil",
        "Informática",
        "Ejecución",
    ]

    # -------------------------------------------------------------------------
    # PCA
    # -------------------------------------------------------------------------

    for clase in clases:

        mascara = (
            y.astype(str).values
            == clase
        )

        axes[0].scatter(
            X_pca[mascara, 0],
            X_pca[mascara, 1],
            label=clase,
            alpha=0.75,
            s=70,
            edgecolors="black",
            linewidths=0.4,
            color=COLORES.get(
                clase
            ),
        )

    axes[0].set_title(
        "PCA — Proyección no supervisada"
    )

    axes[0].set_xlabel(
        (
            "Componente principal 1 "
            f"({varianza_pca[0] * 100:.1f}% var.)"
        )
    )

    axes[0].set_ylabel(
        (
            "Componente principal 2 "
            f"({varianza_pca[1] * 100:.1f}% var.)"
        )
    )

    axes[0].legend(
        title="Grado"
    )

    axes[0].grid(
        alpha=0.25
    )

    # -------------------------------------------------------------------------
    # LDA
    # -------------------------------------------------------------------------

    for clase in clases:

        mascara = (
            y.astype(str).values
            == clase
        )

        axes[1].scatter(
            X_lda[mascara, 0],
            X_lda[mascara, 1],
            label=clase,
            alpha=0.75,
            s=70,
            edgecolors="black",
            linewidths=0.4,
            color=COLORES.get(
                clase
            ),
        )

    axes[1].set_title(
        (
            "LDA — Proyección supervisada\n"
            "(visualización exploratoria)"
        )
    )

    axes[1].set_xlabel(
        "Función discriminante 1"
    )

    axes[1].set_ylabel(
        "Función discriminante 2"
    )

    axes[1].legend(
        title="Grado"
    )

    axes[1].grid(
        alpha=0.25
    )

    fig.text(
        0.5,
        0.01,
        (
            "Nota: la proyección LDA utiliza las etiquetas "
            "de clase y no representa una estimación "
            "de rendimiento predictivo."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            0.94,
        ]
    )

    plt.savefig(
        RUTA_GRAFICO,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 9. RESUMEN DE RESULTADOS
# =============================================================================

def guardar_resumen(
    df: pd.DataFrame,
    vectorizador: TfidfVectorizer,
    pca: PCA,
    lda: LinearDiscriminantAnalysis,
) -> dict:

    distribucion = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    varianza_pca = (
        pca.explained_variance_ratio_
    )

    if hasattr(
        lda,
        "explained_variance_ratio_",
    ):
        varianza_lda = (
            lda.explained_variance_ratio_
        )
    else:
        varianza_lda = np.array(
            []
        )

    resumen = {

        "corpus": {
            "total": int(len(df)),
            "distribucion": {
                str(clase): int(cantidad)
                for clase, cantidad
                in distribucion.items()
            },
        },

        "representacion": {
            "metodo": "TF-IDF",
            "max_features": 400,
            "ngram_range": [
                1,
                2,
            ],
            "min_df": 2,
            "max_df": 0.90,
            "sublinear_tf": True,
            "features_generadas":
                int(
                    len(
                        vectorizador
                        .get_feature_names_out()
                    )
                ),
        },

        "pca": {
            "tipo":
                "no supervisado",

            "varianza_explicada_componente_1":
                round(
                    float(
                        varianza_pca[0]
                    ),
                    6,
                ),

            "varianza_explicada_componente_2":
                round(
                    float(
                        varianza_pca[1]
                    ),
                    6,
                ),

            "varianza_explicada_total_2d":
                round(
                    float(
                        varianza_pca.sum()
                    ),
                    6,
                ),
        },

        "lda": {
            "tipo":
                "supervisado",

            "uso":
                (
                    "Visualización exploratoria "
                    "de separabilidad entre clases."
                ),

            "advertencia":
                (
                    "LDA utiliza las etiquetas reales; "
                    "no debe interpretarse como "
                    "rendimiento predictivo."
                ),

            "varianza_discriminante":
                [
                    round(
                        float(valor),
                        6,
                    )
                    for valor
                    in varianza_lda
                ],
        },
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
# 10. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 72
    )

    print(
        "VISUALIZACIÓN EXPLORATORIA PCA vs. LDA — CORPUS V2"
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
        df["grado"].value_counts()
    )

    vectorizador, X = vectorizar(
        df["perfil_egreso"]
    )

    print(
        f"\nTF-IDF generado: "
        f"{X.shape[1]} características"
    )

    (
        pca,
        X_pca,
        lda,
        X_lda,
    ) = calcular_proyecciones(
        X,
        df["grado"],
    )

    guardar_coordenadas(
        df,
        X_pca,
        X_lda,
    )

    generar_grafico(
        df["grado"],
        X_pca,
        X_lda,
        pca.explained_variance_ratio_,
    )

    resumen = guardar_resumen(
        df,
        vectorizador,
        pca,
        lda,
    )

    print(
        "\nPCA:"
    )

    print(
        "  Varianza PC1: "
        f"{resumen['pca']['varianza_explicada_componente_1'] * 100:.2f}%"
    )

    print(
        "  Varianza PC2: "
        f"{resumen['pca']['varianza_explicada_componente_2'] * 100:.2f}%"
    )

    print(
        "  Varianza total 2D: "
        f"{resumen['pca']['varianza_explicada_total_2d'] * 100:.2f}%"
    )

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Gráfico:\n  {RUTA_GRAFICO}"
    )

    print(
        f"  Coordenadas:\n  {RUTA_COORDENADAS}"
    )

    print(
        f"  Resumen:\n  {RUTA_RESUMEN}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "La proyección LDA es supervisada y se utiliza "
        "únicamente como visualización exploratoria."
    )

    print(
        "No representa una estimación de generalización "
        "del modelo."
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PROYECCIÓN PCA/LDA GENERADA CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )