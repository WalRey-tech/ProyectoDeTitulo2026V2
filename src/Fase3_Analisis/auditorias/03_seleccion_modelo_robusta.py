# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter

import ftfy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC


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
    "seleccion_modelo_robusta",
)

RUTA_MASCARAS = os.path.join(
    RESULTADOS_DIR,
    "comparacion_niveles_enmascaramiento.csv",
)

RUTA_OUTER = os.path.join(
    RESULTADOS_DIR,
    "validacion_modelo_seleccionado_outer.csv",
)

RUTA_INNER = os.path.join(
    RESULTADOS_DIR,
    "ranking_modelos_inner.csv",
)

RUTA_PREDICCIONES = os.path.join(
    RESULTADOS_DIR,
    "predicciones_oof_modelo_seleccionado.csv",
)

RUTA_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_modelo_seleccionado.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_seleccion_modelo_robusta.json",
)

RUTA_GRAFICO = os.path.join(
    RESULTADOS_DIR,
    "comparacion_modelos_robustos.png",
)

RUTA_GRAFICO_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_modelo_seleccionado.png",
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

SEED = 42

OUTER_SPLITS = 5
INNER_SPLITS = 3

CLASES = [
    "Civil",
    "Ejecución",
    "Informática",
]


# =============================================================================
# 3. STOPWORDS
# =============================================================================
#
# Se mantiene la misma base utilizada por la metodología GWO.
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
# 4. NIVELES DE ENMASCARAMIENTO
# =============================================================================
#
# NIVEL 1:
# Solo expresiones que identifican explícitamente la denominación del grado.
#
# NIVEL 2:
# Además elimina tokens aislados. Este es el escenario conservador/agresivo
# utilizado en la auditoría anterior.
# =============================================================================

PATRONES_TITULO = [

    r"\bingenier[ií]a\s+civil"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[ií]a\s+civil\s+inform[aá]tica\b",

    r"\bingenier[ií]a\s+de\s+ejecuci[oó]n"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[ií]a\s+en\s+inform[aá]tica\b",

    r"\bingenier[ií]a\s+inform[aá]tica\b",

    r"\bingenier[oa]\s+civil"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[oa]\s+de\s+ejecuci[oó]n"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[oa]\s+en\s+inform[aá]tica\b",
]


PATRONES_AGRESIVOS = PATRONES_TITULO + [

    r"\bcivil\b",

    r"\bejecuci[oó]n\b",

    r"\binform[aá]tica\b",

    r"\binform[aá]tico\b",

    r"\binform[aá]ticos\b",

    r"\binform[aá]ticas\b",
]


# =============================================================================
# 5. UTILIDADES DE TEXTO
# =============================================================================

def aplicar_patrones(
    texto: str,
    patrones: list[str],
) -> str:

    resultado = ftfy.fix_text(
        str(texto)
    )

    for patron in patrones:

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


def enmascarar_titulos(
    texto: str,
) -> str:

    return aplicar_patrones(
        texto,
        PATRONES_TITULO,
    )


def enmascarar_agresivo(
    texto: str,
) -> str:

    return aplicar_patrones(
        texto,
        PATRONES_AGRESIVOS,
    )


# =============================================================================
# 6. CORPUS
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
# 7. SMOTE DINÁMICO
# =============================================================================

def aplicar_smote(
    X,
    y: np.ndarray,
):

    conteos = pd.Series(
        y
    ).value_counts()

    minimo = int(
        conteos.min()
    )

    if minimo < 2:

        return X, y

    k = min(
        2,
        minimo - 1,
    )

    smote = SMOTE(
        k_neighbors=k,
        random_state=SEED,
    )

    return smote.fit_resample(
        X,
        y,
    )


# =============================================================================
# 8. MÉTRICAS
# =============================================================================

def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {

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

        "F1_Civil":
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Civil"
                ],
                average="macro",
                zero_division=0,
            ),

        "F1_Ejecucion":
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Ejecución"
                ],
                average="macro",
                zero_division=0,
            ),

        "F1_Informatica":
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Informática"
                ],
                average="macro",
                zero_division=0,
            ),
    }


# =============================================================================
# 9. CANDIDATOS DE MODELO
# =============================================================================

def crear_candidatos() -> list[dict]:

    candidatos = []


    # -------------------------------------------------------------------------
    # ComplementNB + SMOTE
    # -------------------------------------------------------------------------

    for alpha in [
        0.1,
        0.5,
        1.0,
    ]:

        candidatos.append(
            {
                "nombre":
                    (
                        "ComplementNB_SMOTE_"
                        f"alpha={alpha}"
                    ),

                "usar_smote":
                    True,

                "factory":
                    lambda a=alpha:
                        ComplementNB(
                            alpha=a
                        ),
            }
        )


    # -------------------------------------------------------------------------
    # ComplementNB sin SMOTE
    # -------------------------------------------------------------------------

    for alpha in [
        0.1,
        0.5,
        1.0,
    ]:

        candidatos.append(
            {
                "nombre":
                    (
                        "ComplementNB_sinSMOTE_"
                        f"alpha={alpha}"
                    ),

                "usar_smote":
                    False,

                "factory":
                    lambda a=alpha:
                        ComplementNB(
                            alpha=a
                        ),
            }
        )


    # -------------------------------------------------------------------------
    # Regresión logística con pesos de clase
    # -------------------------------------------------------------------------

    for c in [
        0.5,
        1.0,
        5.0,
    ]:

        candidatos.append(
            {
                "nombre":
                    (
                        "LogisticRegression_balanced_"
                        f"C={c}"
                    ),

                "usar_smote":
                    False,

                "factory":
                    lambda valor_c=c:
                        LogisticRegression(
                            C=valor_c,
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=SEED,
                        ),
            }
        )


    # -------------------------------------------------------------------------
    # Linear SVM
    # -------------------------------------------------------------------------

    for c in [
        0.1,
        0.5,
        1.0,
    ]:

        candidatos.append(
            {
                "nombre":
                    (
                        "LinearSVC_balanced_"
                        f"C={c}"
                    ),

                "usar_smote":
                    False,

                "factory":
                    lambda valor_c=c:
                        LinearSVC(
                            C=valor_c,
                            class_weight="balanced",
                            random_state=SEED,
                        ),
            }
        )


    # -------------------------------------------------------------------------
    # Ridge
    # -------------------------------------------------------------------------

    for alpha in [
        0.5,
        1.0,
        2.0,
    ]:

        candidatos.append(
            {
                "nombre":
                    (
                        "Ridge_balanced_"
                        f"alpha={alpha}"
                    ),

                "usar_smote":
                    False,

                "factory":
                    lambda a=alpha:
                        RidgeClassifier(
                            alpha=a,
                            class_weight="balanced",
                        ),
            }
        )


    return candidatos


# =============================================================================
# 10. ENTRENAR / PREDECIR UN CANDIDATO
# =============================================================================

def entrenar_predecir(
    candidato: dict,
    textos_train: np.ndarray,
    textos_test: np.ndarray,
    y_train: np.ndarray,
):

    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X_train = vectorizador.fit_transform(
        textos_train
    )

    X_test = vectorizador.transform(
        textos_test
    )


    if candidato[
        "usar_smote"
    ]:

        X_train_final, y_train_final = (
            aplicar_smote(
                X_train,
                y_train,
            )
        )

    else:

        X_train_final = X_train
        y_train_final = y_train


    modelo = candidato[
        "factory"
    ]()


    modelo.fit(
        X_train_final,
        y_train_final,
    )


    pred = modelo.predict(
        X_test
    )


    return (
        pred,
        int(
            X_train.shape[
                1
            ]
        ),
    )


# =============================================================================
# 11. COMPARACIÓN DE NIVELES DE ENMASCARAMIENTO
# =============================================================================

def evaluar_condicion_cnb(
    textos: np.ndarray,
    etiquetas: np.ndarray,
    condicion: str,
) -> pd.DataFrame:

    outer = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


    candidato = {
        "nombre":
            "ComplementNB_SMOTE_alpha=1.0",

        "usar_smote":
            True,

        "factory":
            lambda:
                ComplementNB(
                    alpha=1.0
                ),
    }


    resultados = []


    for fold, (
        idx_train,
        idx_test,
    ) in enumerate(
        outer.split(
            textos,
            etiquetas,
        ),
        start=1,
    ):


        pred, n_features = entrenar_predecir(
            candidato,
            textos[
                idx_train
            ],
            textos[
                idx_test
            ],
            etiquetas[
                idx_train
            ],
        )


        metricas = calcular_metricas(
            etiquetas[
                idx_test
            ],
            pred,
        )


        resultados.append(
            {
                "Condicion":
                    condicion,

                "Fold":
                    fold,

                "N_features":
                    n_features,

                **metricas,
            }
        )


    return pd.DataFrame(
        resultados
    )


# =============================================================================
# 12. SELECCIÓN INTERNA DE MODELO
# =============================================================================

def seleccionar_modelo_inner(
    textos_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    outer_fold: int,
):

    candidatos = crear_candidatos()


    inner = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


    ranking = []


    for candidato in candidatos:

        f1s = []


        for (
            idx_train_inner,
            idx_valid_inner,
        ) in inner.split(
            textos_train_outer,
            y_train_outer,
        ):


            try:

                pred, _ = entrenar_predecir(
                    candidato,
                    textos_train_outer[
                        idx_train_inner
                    ],
                    textos_train_outer[
                        idx_valid_inner
                    ],
                    y_train_outer[
                        idx_train_inner
                    ],
                )


                f1s.append(
                    f1_score(
                        y_train_outer[
                            idx_valid_inner
                        ],
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


        ranking.append(
            {
                "Outer_Fold":
                    outer_fold,

                "Modelo":
                    candidato[
                        "nombre"
                    ],

                "F1_inner_media":
                    float(
                        np.mean(
                            f1s
                        )
                    ),

                "F1_inner_std":
                    float(
                        np.std(
                            f1s
                        )
                    ),

                "Scores_inner":
                    "|".join(
                        f"{valor:.6f}"
                        for valor
                        in f1s
                    ),
            }
        )


    ranking_df = pd.DataFrame(
        ranking
    )


    ranking_df = ranking_df.sort_values(
        by=[
            "F1_inner_media",
            "F1_inner_std",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


    nombre_ganador = ranking_df.iloc[
        0
    ][
        "Modelo"
    ]


    ganador = next(
        candidato
        for candidato
        in candidatos
        if candidato[
            "nombre"
        ] == nombre_ganador
    )


    return (
        ganador,
        ranking_df,
    )


# =============================================================================
# 13. VALIDACIÓN ANIDADA DE SELECCIÓN DE MODELO
# =============================================================================

def ejecutar_seleccion_modelo(
    textos: np.ndarray,
    etiquetas: np.ndarray,
):

    outer = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


    resultados_outer = []

    rankings_inner = []

    predicciones_oof = np.empty(
        len(etiquetas),
        dtype=object,
    )


    for fold, (
        idx_train,
        idx_test,
    ) in enumerate(
        outer.split(
            textos,
            etiquetas,
        ),
        start=1,
    ):


        print(
            "\n"
            + "-" * 76
        )

        print(
            f"OUTER FOLD {fold}/{OUTER_SPLITS}"
        )

        print(
            "-" * 76
        )


        textos_train = textos[
            idx_train
        ]

        textos_test = textos[
            idx_test
        ]

        y_train = etiquetas[
            idx_train
        ]

        y_test = etiquetas[
            idx_test
        ]


        ganador, ranking = seleccionar_modelo_inner(
            textos_train,
            y_train,
            fold,
        )


        rankings_inner.append(
            ranking
        )


        print(
            "Modelo seleccionado por inner CV:"
        )

        print(
            f"  {ganador['nombre']}"
        )

        print(
            "F1 inner:"
            f" {ranking.iloc[0]['F1_inner_media']:.4f}"
        )


        pred, n_features = entrenar_predecir(
            ganador,
            textos_train,
            textos_test,
            y_train,
        )


        predicciones_oof[
            idx_test
        ] = pred


        metricas = calcular_metricas(
            y_test,
            pred,
        )


        resultados_outer.append(
            {
                "Fold":
                    fold,

                "Modelo_seleccionado":
                    ganador[
                        "nombre"
                    ],

                "F1_inner_seleccion":
                    float(
                        ranking.iloc[
                            0
                        ][
                            "F1_inner_media"
                        ]
                    ),

                "N_features":
                    n_features,

                "N_test":
                    int(
                        len(
                            idx_test
                        )
                    ),

                **metricas,
            }
        )


        print(
            "Resultado OUTER:"
        )

        print(
            f"  F1-macro = "
            f"{metricas['F1_macro']:.4f}"
        )


    return (
        pd.DataFrame(
            resultados_outer
        ),

        pd.concat(
            rankings_inner,
            ignore_index=True,
        ),

        predicciones_oof,
    )


# =============================================================================
# 14. MATRIZ DE CONFUSIÓN
# =============================================================================

def generar_matriz(
    etiquetas: np.ndarray,
    predicciones: np.ndarray,
) -> None:

    matriz = confusion_matrix(
        etiquetas,
        predicciones,
        labels=CLASES,
    )

    normalizada = confusion_matrix(
        etiquetas,
        predicciones,
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
        normalizada,
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
            "selección anidada de modelo\n"
            "Títulos explícitos del grado enmascarados"
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
                    f"{normalizada[i, j] * 100:.1f}%"
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
# 15. GRÁFICO COMPARATIVO
# =============================================================================

def generar_grafico(
    comparacion_mascaras: pd.DataFrame,
    resultados_outer: pd.DataFrame,
) -> None:

    condiciones = [
        "Original",
        "Solo_titulos_enmascarados",
        "Enmascaramiento_agresivo",
    ]


    medias = []

    desvios = []


    for condicion in condiciones:

        datos = comparacion_mascaras[
            comparacion_mascaras[
                "Condicion"
            ] == condicion
        ][
            "F1_macro"
        ]


        medias.append(
            float(
                datos.mean()
            )
        )

        desvios.append(
            float(
                datos.std()
            )
        )


    medias.append(
        float(
            resultados_outer[
                "F1_macro"
            ].mean()
        )
    )

    desvios.append(
        float(
            resultados_outer[
                "F1_macro"
            ].std()
        )
    )


    etiquetas = [
        "Original\nCNB",
        "Solo títulos\nCNB",
        "Agresivo\nCNB",
        "Solo títulos\nmodelo seleccionado",
    ]


    fig, ax = plt.subplots(
        figsize=(10, 6)
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
        "F1-macro medio — Outer 5-fold"
    )


    ax.set_title(
        (
            "Evaluación robusta de modelos\n"
            "Impacto del enmascaramiento y selección anidada"
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
            media + 0.035,
            f"{media:.3f}",
            ha="center",
            fontweight="bold",
        )


    fig.tight_layout()


    fig.savefig(
        RUTA_GRAFICO,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# =============================================================================
# 16. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 76
    )

    print(
        "SELECCIÓN ROBUSTA DE MODELO"
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


    textos_titulos = (
        df[
            "perfil_egreso"
        ]
        .astype(str)
        .apply(
            enmascarar_titulos
        )
        .to_numpy()
    )


    textos_agresivos = (
        df[
            "perfil_egreso"
        ]
        .astype(str)
        .apply(
            enmascarar_agresivo
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


    # =========================================================================
    # A. SENSIBILIDAD AL ENMASCARAMIENTO
    # =========================================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "A. SENSIBILIDAD AL ENMASCARAMIENTO"
    )

    print(
        "=" * 76
    )


    resultados_mascaras = []


    for nombre, textos in [

        (
            "Original",
            textos_originales,
        ),

        (
            "Solo_titulos_enmascarados",
            textos_titulos,
        ),

        (
            "Enmascaramiento_agresivo",
            textos_agresivos,
        ),
    ]:


        resultado = evaluar_condicion_cnb(
            textos,
            etiquetas,
            nombre,
        )


        resultados_mascaras.append(
            resultado
        )


        print(
            f"{nombre:<30} "
            f"F1={resultado['F1_macro'].mean():.4f}"
        )


    comparacion_mascaras = pd.concat(
        resultados_mascaras,
        ignore_index=True,
    )


    comparacion_mascaras.to_csv(
        RUTA_MASCARAS,
        index=False,
        encoding="utf-8-sig",
    )


    # =========================================================================
    # B. SELECCIÓN ANIDADA DE MODELO
    # =========================================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "B. SELECCIÓN ANIDADA DE MODELO"
    )

    print(
        "=" * 76
    )


    (
        resultados_outer,
        ranking_inner,
        predicciones_oof,
    ) = ejecutar_seleccion_modelo(
        textos_titulos,
        etiquetas,
    )


    resultados_outer.to_csv(
        RUTA_OUTER,
        index=False,
        encoding="utf-8-sig",
    )


    ranking_inner.to_csv(
        RUTA_INNER,
        index=False,
        encoding="utf-8-sig",
    )


    pd.DataFrame(
        {
            "Real":
                etiquetas,

            "Predicho":
                predicciones_oof,
        }
    ).to_csv(
        RUTA_PREDICCIONES,
        index=False,
        encoding="utf-8-sig",
    )


    generar_matriz(
        etiquetas,
        predicciones_oof,
    )


    generar_grafico(
        comparacion_mascaras,
        resultados_outer,
    )


    # =========================================================================
    # C. RESUMEN
    # =========================================================================

    f1_original = float(
        comparacion_mascaras[
            comparacion_mascaras[
                "Condicion"
            ] == "Original"
        ][
            "F1_macro"
        ].mean()
    )


    f1_titulos = float(
        comparacion_mascaras[
            comparacion_mascaras[
                "Condicion"
            ]
            == "Solo_titulos_enmascarados"
        ][
            "F1_macro"
        ].mean()
    )


    f1_agresivo = float(
        comparacion_mascaras[
            comparacion_mascaras[
                "Condicion"
            ]
            == "Enmascaramiento_agresivo"
        ][
            "F1_macro"
        ].mean()
    )


    f1_modelo = float(
        resultados_outer[
            "F1_macro"
        ].mean()
    )


    std_modelo = float(
        resultados_outer[
            "F1_macro"
        ].std()
    )


    modelos_seleccionados = Counter(
        resultados_outer[
            "Modelo_seleccionado"
        ].tolist()
    )


    f1_oof = float(
        f1_score(
            etiquetas,
            predicciones_oof,
            labels=CLASES,
            average="macro",
            zero_division=0,
        )
    )


    accuracy_oof = float(
        accuracy_score(
            etiquetas,
            predicciones_oof,
        )
    )


    reporte = classification_report(
        etiquetas,
        predicciones_oof,
        labels=CLASES,
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )


    resumen = {

        "version":
            "1.0",

        "objetivo":
            (
                "Evaluar la sensibilidad a distintos "
                "niveles de enmascaramiento y seleccionar "
                "el clasificador mediante validación "
                "anidada sin utilizar el conjunto externo "
                "de prueba para elegir el modelo."
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

        "sensibilidad_enmascaramiento": {

            "original_cnb_f1_macro":
                round(
                    f1_original,
                    6,
                ),

            "solo_titulos_cnb_f1_macro":
                round(
                    f1_titulos,
                    6,
                ),

            "agresivo_cnb_f1_macro":
                round(
                    f1_agresivo,
                    6,
                ),

            "delta_solo_titulos_vs_original":
                round(
                    f1_titulos
                    - f1_original,
                    6,
                ),

            "delta_agresivo_vs_original":
                round(
                    f1_agresivo
                    - f1_original,
                    6,
                ),
        },

        "seleccion_anidada": {

            "outer_folds":
                OUTER_SPLITS,

            "inner_folds":
                INNER_SPLITS,

            "f1_macro_media_outer":
                round(
                    f1_modelo,
                    6,
                ),

            "f1_macro_std_outer":
                round(
                    std_modelo,
                    6,
                ),

            "f1_macro_oof":
                round(
                    f1_oof,
                    6,
                ),

            "accuracy_oof":
                round(
                    accuracy_oof,
                    6,
                ),

            "modelos_seleccionados":
                dict(
                    modelos_seleccionados
                ),
        },

        "reporte_oof":
            reporte,

        "interpretacion": {

            "nota":
                (
                    "La métrica principal para comparación "
                    "es el F1-macro medio de los folds "
                    "externos."
                ),

            "advertencia":
                (
                    "El corpus es reducido y la clase "
                    "Ejecución contiene únicamente cinco "
                    "observaciones, por lo que se espera "
                    "variabilidad entre folds."
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
    # D. CONSOLA
    # =========================================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "RESUMEN FINAL"
    )

    print(
        "=" * 76
    )


    print(
        "\nSensibilidad al enmascaramiento — "
        "ComplementNB fijo:"
    )

    print(
        f"  Original                 : "
        f"{f1_original:.4f}"
    )

    print(
        f"  Solo títulos enmascarados: "
        f"{f1_titulos:.4f}"
    )

    print(
        f"  Enmascaramiento agresivo : "
        f"{f1_agresivo:.4f}"
    )


    print(
        "\nSelección anidada de modelo "
        "(solo títulos enmascarados):"
    )

    print(
        f"  F1-macro outer media : "
        f"{f1_modelo:.4f}"
    )

    print(
        f"  Std outer            : "
        f"{std_modelo:.4f}"
    )

    print(
        f"  F1-macro OOF         : "
        f"{f1_oof:.4f}"
    )

    print(
        f"  Accuracy OOF         : "
        f"{accuracy_oof:.4f}"
    )


    print(
        "\nModelos seleccionados por fold:"
    )

    for modelo, cantidad in (
        modelos_seleccionados.items()
    ):

        print(
            f"  {modelo}: "
            f"{cantidad} fold(s)"
        )


    print(
        "\nReporte OOF:"
    )

    print(
        classification_report(
            etiquetas,
            predicciones_oof,
            labels=CLASES,
            target_names=CLASES,
            zero_division=0,
        )
    )


    print(
        "\nArchivos generados:"
    )

    print(
        f"  {RUTA_MASCARAS}"
    )

    print(
        f"  {RUTA_OUTER}"
    )

    print(
        f"  {RUTA_INNER}"
    )

    print(
        f"  {RUTA_PREDICCIONES}"
    )

    print(
        f"  {RUTA_MATRIZ}"
    )

    print(
        f"  {RUTA_RESUMEN}"
    )

    print(
        f"  {RUTA_GRAFICO}"
    )

    print(
        f"  {RUTA_GRAFICO_MATRIZ}"
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        "SELECCIÓN ROBUSTA DE MODELO COMPLETADA"
    )

    print(
        "=" * 76
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )