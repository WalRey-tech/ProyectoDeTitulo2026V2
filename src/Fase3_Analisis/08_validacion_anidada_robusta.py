# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import time
import warnings

import ftfy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from mealpy import GWO, FloatVar, Problem

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB


warnings.filterwarnings("ignore")


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

RUTA_CORPUS = os.path.join(
    SRC_ROOT,
    "data",
    "processed",
    "perfiles_egreso_etiquetado_v2.csv",
)

RESULTADOS_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "validacion_robusta",
)

RUTA_FOLDS = os.path.join(
    RESULTADOS_DIR,
    "validacion_anidada_folds.csv",
)

RUTA_FEATURES = os.path.join(
    RESULTADOS_DIR,
    "features_gwo_por_fold.csv",
)

RUTA_METRICAS_CLASE = os.path.join(
    RESULTADOS_DIR,
    "metricas_clase_gwo_anidado.csv",
)

RUTA_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_gwo_anidado.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_validacion_anidada.json",
)

RUTA_GRAFICO_COMPARACION = os.path.join(
    RESULTADOS_DIR,
    "comparacion_validacion_robusta.png",
)

RUTA_GRAFICO_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_gwo_anidado.png",
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

SEED = 42

OUTER_SPLITS = 5
INNER_SPLITS = 3

GWO_EPOCHS = 100
GWO_POP_SIZE = 30

CLASES = [
    "Civil",
    "Ejecución",
    "Informática",
]


# =============================================================================
# 3. STOPWORDS
# =============================================================================
#
# Se conserva la lista utilizada por la metodología GWO de referencia.
# =============================================================================

STOPWORDS_ES = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes",
    "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "durante", "e", "el", "ella", "ellas",
    "ellos", "en", "entre", "era", "erais", "eran", "eras",
    "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estaban", "estado", "estar", "estas", "este",
    "esto", "estos", "estoy", "fue", "fueron", "fui", "ha",
    "han", "has", "hasta", "hay", "he", "hun", "la", "las",
    "le", "les", "lo", "los", "mas", "me", "mi", "mia",
    "mias", "mientras", "mis", "mo", "mucho", "muchos", "muy",
    "más", "mí", "nada", "ni", "no", "nos", "nosotras",
    "nosotros", "nuestra", "nuestras", "nuestro", "nuestros",
    "o", "os", "otra", "otras", "otro", "otros", "para",
    "pero", "poco", "por", "porque", "que", "quien", "quienes",
    "qué", "se", "sea", "seais", "sean", "seas", "ser", "si",
    "sin", "sobre", "sois", "somos", "son", "su", "sus",
    "también", "tanto", "te", "tenemos", "tengo", "ti",
    "tiene", "tienen", "toda", "todas", "todo", "todos",
    "tu", "tus", "un", "una", "unas", "uno", "unos",
    "vosotras", "vosotros", "vuestra", "vuestras", "vuestro",
    "vuestros", "y", "ya", "yo", "él", "ésta", "éstas",
    "éste", "éstos", "última", "últimas", "último", "últimos",
]


TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
    "stop_words": STOPWORDS_ES,
}


# =============================================================================
# 4. PROXIES DIRECTOS DEL GRADO
# =============================================================================

PATRONES_MASCARA = [

    r"\bingenier[ií]a\s+civil"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[ií]a\s+de\s+ejecuci[oó]n"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[ií]a\s+en\s+inform[aá]tica\b",

    r"\bingenier[oa]\s+civil"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[oa]\s+de\s+ejecuci[oó]n"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[oa]\s+en\s+inform[aá]tica\b",

    r"\bcivil\b",

    r"\bejecuci[oó]n\b",

    r"\binform[aá]tica\b",

    r"\binform[aá]tico\b",

    r"\binform[aá]ticos\b",

    r"\binform[aá]ticas\b",
]


# =============================================================================
# 5. UTILIDADES
# =============================================================================

def enmascarar_texto(
    texto: str,
) -> str:

    resultado = ftfy.fix_text(
        str(texto)
    )

    for patron in PATRONES_MASCARA:

        resultado = re.sub(
            patron,
            " ",
            resultado,
            flags=re.IGNORECASE,
        )

    resultado = re.sub(
        r"\s+",
        " ",
        resultado,
    )

    return resultado.strip()


def crear_smote(
    etiquetas: np.ndarray,
):

    conteos = pd.Series(
        etiquetas
    ).value_counts()

    minimo = int(
        conteos.min()
    )

    if minimo < 2:
        return None

    k = min(
        2,
        minimo - 1,
    )

    return SMOTE(
        k_neighbors=k,
        random_state=SEED,
    )


def aplicar_smote(
    X: np.ndarray,
    y: np.ndarray,
):

    smote = crear_smote(
        y
    )

    if smote is None:
        return X, y

    return smote.fit_resample(
        X,
        y,
    )


def metricas_prediccion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    resultado = {

        "F1_macro":
            f1_score(
                y_true,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0,
            ),

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),
    }

    for clase in CLASES:

        nombre = (
            clase
            .lower()
            .replace(
                "ó",
                "o",
            )
            .replace(
                "á",
                "a",
            )
        )

        resultado[
            f"F1_{nombre}"
        ] = f1_score(
            y_true,
            y_pred,
            labels=[
                clase
            ],
            average="macro",
            zero_division=0,
        )

    return resultado


# =============================================================================
# 6. CARGA DEL CORPUS
# =============================================================================

def cargar_corpus() -> pd.DataFrame:

    if not os.path.exists(
        RUTA_CORPUS
    ):
        raise FileNotFoundError(
            f"No se encontró:\n{RUTA_CORPUS}"
        )

    df = pd.read_csv(
        RUTA_CORPUS,
        encoding="utf-8-sig",
    )

    requeridas = {
        "perfil_egreso",
        "grado",
    }

    faltantes = (
        requeridas
        - set(df.columns)
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas: "
            f"{sorted(faltantes)}"
        )

    df = df.dropna(
        subset=[
            "perfil_egreso",
            "grado",
        ]
    ).copy()

    df[
        "perfil_egreso"
    ] = df[
        "perfil_egreso"
    ].apply(
        lambda x: ftfy.fix_text(
            str(x)
        )
    )

    return df


# =============================================================================
# 7. BASELINE EXTERNO
# =============================================================================

def evaluar_baseline_fold(
    textos_train: np.ndarray,
    textos_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):

    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X_train = vectorizador.fit_transform(
        textos_train
    ).toarray()

    X_test = vectorizador.transform(
        textos_test
    ).toarray()

    X_balanceado, y_balanceado = aplicar_smote(
        X_train,
        y_train,
    )

    modelo = ComplementNB()

    modelo.fit(
        X_balanceado,
        y_balanceado,
    )

    pred = modelo.predict(
        X_test
    )

    return (
        metricas_prediccion(
            y_test,
            pred,
        ),
        pred,
        len(
            vectorizador
            .get_feature_names_out()
        ),
    )


# =============================================================================
# 8. GWO ANIDADO
# =============================================================================

def seleccionar_gwo_en_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: np.ndarray,
    fold_externo: int,
):

    n_features = int(
        X_train.shape[
            1
        ]
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


    def fitness_fn(
        solution_bin,
    ) -> float:

        mascara = (
            solution_bin
            .astype(bool)
        )

        n_seleccionadas = int(
            mascara.sum()
        )

        if (
            n_seleccionadas == 0
            or n_seleccionadas
            == n_features
        ):
            return 0.0

        X_sel = X_train[
            :,
            mascara,
        ]

        f1s = []

        for (
            idx_train_inner,
            idx_valid_inner,
        ) in inner_cv.split(
            X_sel,
            y_train,
        ):

            X_tr = X_sel[
                idx_train_inner
            ]

            X_va = X_sel[
                idx_valid_inner
            ]

            y_tr = y_train[
                idx_train_inner
            ]

            y_va = y_train[
                idx_valid_inner
            ]

            try:

                X_r, y_r = aplicar_smote(
                    X_tr,
                    y_tr,
                )

                modelo = ComplementNB()

                modelo.fit(
                    X_r,
                    y_r,
                )

                pred = modelo.predict(
                    X_va
                )

                f1s.append(
                    f1_score(
                        y_va,
                        pred,
                        labels=CLASES,
                        average="macro",
                        zero_division=0,
                    )
                )

            except Exception:

                f1s.append(
                    0.0
                )

        return float(
            np.mean(
                f1s
            )
        )


    class GWOFeatureSelection(
        Problem
    ):

        def __init__(
            self,
            bounds,
            minmax,
            **kwargs,
        ):

            super().__init__(
                bounds,
                minmax,
                **kwargs,
            )


        def obj_func(
            self,
            solution,
        ):

            binary = (
                np.abs(
                    (2 / np.pi)
                    * np.arctan(
                        (np.pi / 2)
                        * solution
                    )
                )
                > 0.5
            ).astype(
                float
            )

            return fitness_fn(
                binary
            )


    bounds = FloatVar(
        lb=(-6.0,) * n_features,
        ub=(6.0,) * n_features,
        name=(
            f"features_fold_"
            f"{fold_externo}"
        ),
    )

    problema = GWOFeatureSelection(
        bounds=bounds,
        minmax="max",
        log_to=None,
        seed=SEED,
    )

    optimizador = GWO.OriginalGWO(
        epoch=GWO_EPOCHS,
        pop_size=GWO_POP_SIZE,
    )

    inicio = time.time()

    optimizador.solve(
        problema,
        seed=SEED,
    )

    duracion = (
        time.time()
        - inicio
    )

    mejor_solucion = (
        optimizador
        .g_best
        .solution
    )

    mascara = (
        np.abs(
            (2 / np.pi)
            * np.arctan(
                (np.pi / 2)
                * mejor_solucion
            )
        )
        > 0.5
    )

    if not mascara.any():
        raise RuntimeError(
            "GWO no seleccionó ninguna "
            f"característica en fold {fold_externo}."
        )

    seleccionadas = (
        feature_names[
            mascara
        ]
    )

    fitness = float(
        optimizador
        .g_best
        .target
        .fitness
    )

    return (
        mascara,
        seleccionadas,
        fitness,
        duracion,
    )


# =============================================================================
# 9. EVALUACIÓN GWO ANIDADA
# =============================================================================

def evaluar_gwo_anidado_fold(
    textos_train: np.ndarray,
    textos_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    fold_externo: int,
):

    # El vectorizador se ajusta SOLO
    # sobre el entrenamiento externo.
    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X_train = vectorizador.fit_transform(
        textos_train
    ).toarray()

    X_test = vectorizador.transform(
        textos_test
    ).toarray()

    feature_names = np.asarray(
        vectorizador
        .get_feature_names_out()
    )


    (
        mascara,
        seleccionadas,
        fitness_inner,
        duracion,
    ) = seleccionar_gwo_en_train(
        X_train,
        y_train,
        feature_names,
        fold_externo,
    )


    X_train_sel = X_train[
        :,
        mascara,
    ]

    X_test_sel = X_test[
        :,
        mascara,
    ]


    X_balanceado, y_balanceado = aplicar_smote(
        X_train_sel,
        y_train,
    )


    modelo = ComplementNB()

    modelo.fit(
        X_balanceado,
        y_balanceado,
    )

    pred = modelo.predict(
        X_test_sel
    )


    metricas = metricas_prediccion(
        y_test,
        pred,
    )


    return (
        metricas,
        pred,
        seleccionadas,
        fitness_inner,
        duracion,
        len(feature_names),
    )


# =============================================================================
# 10. GRÁFICO COMPARATIVO
# =============================================================================

def generar_grafico_comparacion(
    resultados: pd.DataFrame,
) -> None:

    orden = [
        "Original_sin_GWO",
        "Enmascarado_sin_GWO",
        "Enmascarado_GWO_anidado",
    ]

    medias = []

    desvios = []

    for condicion in orden:

        valores = resultados[
            resultados[
                "Condicion"
            ] == condicion
        ][
            "F1_macro"
        ]

        medias.append(
            valores.mean()
        )

        desvios.append(
            valores.std()
        )

    etiquetas = [
        "Original\nsin GWO",
        "Enmascarado\nsin GWO",
        "Enmascarado\nGWO anidado",
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        etiquetas,
        medias,
        yerr=desvios,
        capsize=6,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "F1-macro medio"
    )

    ax.set_title(
        (
            "Validación externa 5-fold\n"
            "Auditoría y selección GWO anidada"
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for indice, media in enumerate(
        medias
    ):

        ax.text(
            indice,
            media + 0.03,
            f"{media:.3f}",
            ha="center",
            fontweight="bold",
        )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_COMPARACION,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 11. MATRIZ DE CONFUSIÓN
# =============================================================================

def generar_matriz_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    matriz = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASES,
    )

    matriz_normalizada = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASES,
        normalize="true",
    )

    df_matriz = pd.DataFrame(
        matriz,
        index=CLASES,
        columns=CLASES,
    )

    df_matriz.index.name = (
        "Real"
    )

    df_matriz.columns.name = (
        "Predicho"
    )

    df_matriz.to_csv(
        RUTA_MATRIZ,
        encoding="utf-8-sig",
    )


    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    imagen = ax.imshow(
        matriz_normalizada,
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        imagen,
        ax=ax,
        label=(
            "Proporción dentro "
            "de la clase real"
        ),
    )

    ax.set_xticks(
        np.arange(
            len(CLASES)
        )
    )

    ax.set_yticks(
        np.arange(
            len(CLASES)
        )
    )

    ax.set_xticklabels(
        CLASES,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        CLASES
    )

    ax.set_xlabel(
        "Clase predicha"
    )

    ax.set_ylabel(
        "Clase real"
    )

    ax.set_title(
        (
            "Matriz de confusión — "
            "GWO anidado\n"
            "Texto con proxies directos enmascarados"
        )
    )


    for i in range(
        len(CLASES)
    ):

        for j in range(
            len(CLASES)
        ):

            ax.text(
                j,
                i,
                (
                    f"{matriz[i, j]}\n"
                    f"{matriz_normalizada[i, j] * 100:.1f}%"
                ),
                ha="center",
                va="center",
                fontweight="bold",
            )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_MATRIZ,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 12. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 76
    )

    print(
        "VALIDACIÓN ROBUSTA — "
        "GWO ANIDADO + CONTROL DE FUGA LÉXICA"
    )

    print(
        "=" * 76
    )


    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True,
    )


    df = cargar_corpus()


    textos_originales = (
        df[
            "perfil_egreso"
        ]
        .astype(str)
        .to_numpy()
    )


    textos_enmascarados = (
        df[
            "perfil_egreso"
        ]
        .astype(str)
        .apply(
            enmascarar_texto
        )
        .to_numpy()
    )


    etiquetas = (
        df[
            "grado"
        ]
        .astype(str)
        .to_numpy()
    )


    print(
        f"\nCorpus: {len(df)} perfiles"
    )

    print(
        df[
            "grado"
        ].value_counts()
    )


    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


    resultados = []

    features_folds = []

    predicciones_robustas = np.empty(
        len(df),
        dtype=object,
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        "INICIANDO VALIDACIÓN EXTERNA"
    )

    print(
        "=" * 76
    )


    for fold, (
        idx_train,
        idx_test,
    ) in enumerate(
        outer_cv.split(
            textos_originales,
            etiquetas,
        ),
        start=1,
    ):


        print(
            f"\n{'-' * 76}"
        )

        print(
            f"OUTER FOLD {fold}/{OUTER_SPLITS}"
        )

        print(
            f"{'-' * 76}"
        )


        y_train = etiquetas[
            idx_train
        ]

        y_test = etiquetas[
            idx_test
        ]


        # ---------------------------------------------------------------------
        # A. ORIGINAL SIN GWO
        # ---------------------------------------------------------------------

        metricas_original, _, n_features_original = (
            evaluar_baseline_fold(
                textos_originales[
                    idx_train
                ],
                textos_originales[
                    idx_test
                ],
                y_train,
                y_test,
            )
        )


        resultados.append(
            {
                "Fold":
                    fold,

                "Condicion":
                    "Original_sin_GWO",

                "N_test":
                    int(
                        len(
                            idx_test
                        )
                    ),

                "N_features":
                    n_features_original,

                **metricas_original,
            }
        )


        print(
            "Original sin GWO      : "
            f"F1={metricas_original['F1_macro']:.4f}"
        )


        # ---------------------------------------------------------------------
        # B. ENMASCARADO SIN GWO
        # ---------------------------------------------------------------------

        metricas_mask, _, n_features_mask = (
            evaluar_baseline_fold(
                textos_enmascarados[
                    idx_train
                ],
                textos_enmascarados[
                    idx_test
                ],
                y_train,
                y_test,
            )
        )


        resultados.append(
            {
                "Fold":
                    fold,

                "Condicion":
                    "Enmascarado_sin_GWO",

                "N_test":
                    int(
                        len(
                            idx_test
                        )
                    ),

                "N_features":
                    n_features_mask,

                **metricas_mask,
            }
        )


        print(
            "Enmascarado sin GWO   : "
            f"F1={metricas_mask['F1_macro']:.4f}"
        )


        # ---------------------------------------------------------------------
        # C. ENMASCARADO + GWO ANIDADO
        # ---------------------------------------------------------------------

        print(
            "Ejecutando GWO dentro "
            "del entrenamiento externo..."
        )


        (
            metricas_gwo,
            pred_gwo,
            features_sel,
            fitness_inner,
            duracion,
            n_features_total,
        ) = evaluar_gwo_anidado_fold(
            textos_enmascarados[
                idx_train
            ],
            textos_enmascarados[
                idx_test
            ],
            y_train,
            y_test,
            fold,
        )


        predicciones_robustas[
            idx_test
        ] = pred_gwo


        resultados.append(
            {
                "Fold":
                    fold,

                "Condicion":
                    "Enmascarado_GWO_anidado",

                "N_test":
                    int(
                        len(
                            idx_test
                        )
                    ),

                "N_features":
                    int(
                        len(
                            features_sel
                        )
                    ),

                "Fitness_inner":
                    fitness_inner,

                "Tiempo_GWO_seg":
                    duracion,

                **metricas_gwo,
            }
        )


        for feature in features_sel:

            features_folds.append(
                {
                    "Fold":
                        fold,

                    "Feature":
                        str(
                            feature
                        ),
                }
            )


        print(
            "Enmascarado GWO       : "
            f"F1={metricas_gwo['F1_macro']:.4f}"
        )

        print(
            f"Features seleccionadas: "
            f"{len(features_sel)}/{n_features_total}"
        )

        print(
            f"Fitness inner GWO     : "
            f"{fitness_inner:.4f}"
        )

        print(
            f"Tiempo GWO            : "
            f"{duracion:.1f}s"
        )


    # =========================================================================
    # 13. GUARDAR RESULTADOS POR FOLD
    # =========================================================================

    df_resultados = pd.DataFrame(
        resultados
    )


    df_resultados.to_csv(
        RUTA_FOLDS,
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        features_folds
    ).to_csv(
        RUTA_FEATURES,
        index=False,
        encoding="utf-8-sig",
    )


    # =========================================================================
    # 14. MÉTRICAS OOF DEL MODELO ROBUSTO
    # =========================================================================

    reporte_clase = classification_report(
        etiquetas,
        predicciones_robustas,
        labels=CLASES,
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )


    df_reporte_clase = (
        pd.DataFrame(
            reporte_clase
        )
        .transpose()
    )


    df_reporte_clase.to_csv(
        RUTA_METRICAS_CLASE,
        encoding="utf-8-sig",
    )


    generar_matriz_confusion(
        etiquetas,
        predicciones_robustas,
    )


    generar_grafico_comparacion(
        df_resultados
    )


    # =========================================================================
    # 15. RESUMEN
    # =========================================================================

    resumen_condiciones = {}


    for condicion in [
        "Original_sin_GWO",
        "Enmascarado_sin_GWO",
        "Enmascarado_GWO_anidado",
    ]:

        datos = df_resultados[
            df_resultados[
                "Condicion"
            ] == condicion
        ]


        resumen_condiciones[
            condicion
        ] = {

            "f1_macro_media":
                round(
                    float(
                        datos[
                            "F1_macro"
                        ].mean()
                    ),
                    6,
                ),

            "f1_macro_std":
                round(
                    float(
                        datos[
                            "F1_macro"
                        ].std()
                    ),
                    6,
                ),

            "accuracy_media":
                round(
                    float(
                        datos[
                            "Accuracy"
                        ].mean()
                    ),
                    6,
                ),

            "f1_civil_media":
                round(
                    float(
                        datos[
                            "F1_civil"
                        ].mean()
                    ),
                    6,
                ),

            "f1_ejecucion_media":
                round(
                    float(
                        datos[
                            "F1_ejecucion"
                        ].mean()
                    ),
                    6,
                ),

            "f1_informatica_media":
                round(
                    float(
                        datos[
                            "F1_informatica"
                        ].mean()
                    ),
                    6,
                ),
        }


    f1_original = (
        resumen_condiciones[
            "Original_sin_GWO"
        ][
            "f1_macro_media"
        ]
    )


    f1_mask = (
        resumen_condiciones[
            "Enmascarado_sin_GWO"
        ][
            "f1_macro_media"
        ]
    )


    f1_robusto = (
        resumen_condiciones[
            "Enmascarado_GWO_anidado"
        ][
            "f1_macro_media"
        ]
    )


    datos_gwo = df_resultados[
        df_resultados[
            "Condicion"
        ] == "Enmascarado_GWO_anidado"
    ]


    resumen = {

        "version":
            "1.0",

        "objetivo":
            (
                "Estimar generalización con "
                "proxies directos del grado "
                "enmascarados y selección GWO "
                "realizada únicamente dentro "
                "del entrenamiento externo."
            ),

        "corpus": {

            "total":
                int(
                    len(df)
                ),

            "distribucion": {
                str(k):
                    int(v)

                for k, v
                in df[
                    "grado"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "configuracion": {

            "outer_folds":
                OUTER_SPLITS,

            "inner_folds_gwo":
                INNER_SPLITS,

            "gwo_epochs":
                GWO_EPOCHS,

            "gwo_population":
                GWO_POP_SIZE,

            "seed":
                SEED,

            "clasificador":
                "Complement Naive Bayes",

            "balanceo":
                (
                    "SMOTE dentro del "
                    "entrenamiento"
                ),

            "control_fuga":
                (
                    "Enmascaramiento de "
                    "denominaciones directas "
                    "del grado antes de TF-IDF."
                ),
        },

        "resultados":
            resumen_condiciones,

        "impacto_enmascaramiento": {

            "delta_f1":
                round(
                    f1_mask
                    - f1_original,
                    6,
                ),
        },

        "efecto_gwo_robusto": {

            "delta_vs_enmascarado_sin_gwo":
                round(
                    f1_robusto
                    - f1_mask,
                    6,
                ),

            "features_seleccionadas_media":
                round(
                    float(
                        datos_gwo[
                            "N_features"
                        ].mean()
                    ),
                    2,
                ),

            "features_seleccionadas_min":
                int(
                    datos_gwo[
                        "N_features"
                    ].min()
                ),

            "features_seleccionadas_max":
                int(
                    datos_gwo[
                        "N_features"
                    ].max()
                ),
        },

        "modelo_robusto_oof": {

            "accuracy":
                round(
                    float(
                        accuracy_score(
                            etiquetas,
                            predicciones_robustas,
                        )
                    ),
                    6,
                ),

            "f1_macro":
                round(
                    float(
                        f1_score(
                            etiquetas,
                            predicciones_robustas,
                            labels=CLASES,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    6,
                ),
        },

        "interpretacion": {

            "nota":
                (
                    "El resultado Enmascarado_GWO_anidado "
                    "es el candidato principal para una "
                    "estimación más conservadora de "
                    "generalización."
                ),

            "advertencia":
                (
                    "El tamaño reducido del corpus, "
                    "especialmente la clase Ejecución "
                    "con cinco observaciones, implica "
                    "alta incertidumbre y sensibilidad "
                    "a las particiones."
                ),
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


    # =========================================================================
    # 16. SALIDA FINAL
    # =========================================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "RESUMEN FINAL — VALIDACIÓN ROBUSTA"
    )

    print(
        "=" * 76
    )


    print(
        "\nOriginal sin GWO:"
    )

    print(
        f"  F1-macro = "
        f"{f1_original:.4f}"
    )


    print(
        "\nEnmascarado sin GWO:"
    )

    print(
        f"  F1-macro = "
        f"{f1_mask:.4f}"
    )


    print(
        "\nEnmascarado + GWO anidado:"
    )

    print(
        f"  F1-macro = "
        f"{f1_robusto:.4f}"
    )

    print(
        f"  Delta vs enmascarado sin GWO = "
        f"{f1_robusto - f1_mask:+.4f}"
    )


    print(
        "\nFeatures GWO por fold:"
    )

    print(
        f"  Media = "
        f"{datos_gwo['N_features'].mean():.1f}"
    )

    print(
        f"  Rango = "
        f"{int(datos_gwo['N_features'].min())}"
        f" - "
        f"{int(datos_gwo['N_features'].max())}"
    )


    print(
        "\nMétricas OOF robustas:"
    )

    print(
        classification_report(
            etiquetas,
            predicciones_robustas,
            labels=CLASES,
            target_names=CLASES,
            zero_division=0,
        )
    )


    print(
        "\nArchivos generados:"
    )

    print(
        f"  {RUTA_FOLDS}"
    )

    print(
        f"  {RUTA_FEATURES}"
    )

    print(
        f"  {RUTA_METRICAS_CLASE}"
    )

    print(
        f"  {RUTA_MATRIZ}"
    )

    print(
        f"  {RUTA_RESUMEN}"
    )

    print(
        f"  {RUTA_GRAFICO_COMPARACION}"
    )

    print(
        f"  {RUTA_GRAFICO_MATRIZ}"
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        "VALIDACIÓN ROBUSTA COMPLETADA CORRECTAMENTE"
    )

    print(
        "=" * 76
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )