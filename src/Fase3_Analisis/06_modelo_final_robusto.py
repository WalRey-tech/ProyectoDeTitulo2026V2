# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone

import ftfy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    "modelo_final_robusto",
)

RUTA_FOLDS = os.path.join(
    RESULTADOS_DIR,
    "metricas_folds_modelo_final.csv",
)

RUTA_PREDICCIONES = os.path.join(
    RESULTADOS_DIR,
    "predicciones_oof_modelo_final.csv",
)

RUTA_REPORTE_CLASES = os.path.join(
    RESULTADOS_DIR,
    "metricas_por_clase_modelo_final.csv",
)

RUTA_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_modelo_final.csv",
)

RUTA_AUDITORIA_MASCARA = os.path.join(
    RESULTADOS_DIR,
    "auditoria_enmascaramiento_modelo_final.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_modelo_final_robusto.json",
)

RUTA_GRAFICO_FOLDS = os.path.join(
    RESULTADOS_DIR,
    "f1_por_fold_modelo_final.png",
)

RUTA_GRAFICO_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_modelo_final.png",
)


# =============================================================================
# 2. CONFIGURACIÓN CONGELADA
# =============================================================================

SEED = 42
N_SPLITS = 5

CLASES = [
    "Civil",
    "Ejecución",
    "Informática",
]


# =============================================================================
# 3. STOPWORDS
# =============================================================================
#
# Se conserva la misma configuración utilizada en los experimentos
# robustos previos.
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
# 4. DENOMINACIONES EXPLÍCITAS DEL GRADO
# =============================================================================
#
# IMPORTANTE:
# Se eliminan expresiones que revelan directamente el nombre de la carrera.
#
# NO se eliminan de manera general palabras como:
#
#   civil
#   ejecución
#   informática
#
# cuando aparecen de forma aislada.
#
# De esta forma evitamos el enmascaramiento agresivo de la prueba de estrés.
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


# =============================================================================
# 5. UTILIDADES
# =============================================================================

def enmascarar_y_contar(
    texto: str,
) -> tuple[str, int]:

    resultado = ftfy.fix_text(
        str(texto)
    )

    total_reemplazos = 0

    for patron in PATRONES_TITULO:

        resultado, reemplazos = re.subn(
            patron,
            " ",
            resultado,
            flags=re.IGNORECASE,
        )

        total_reemplazos += (
            reemplazos
        )

    resultado = re.sub(
        r"\s+",
        " ",
        resultado,
    )

    return (
        resultado.strip(),
        int(total_reemplazos),
    )


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

        return X, y, None

    k = min(
        2,
        minimo - 1,
    )

    smote = SMOTE(
        k_neighbors=k,
        random_state=SEED,
    )

    X_balanceado, y_balanceado = (
        smote.fit_resample(
            X,
            y,
        )
    )

    return (
        X_balanceado,
        y_balanceado,
        k,
    )


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
            "El corpus V2 no contiene "
            f"las columnas: {sorted(faltantes)}"
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
        lambda x:
            ftfy.fix_text(
                str(x)
            )
    )

    return df


# =============================================================================
# 7. PREPARACIÓN AUDITADA
# =============================================================================

def preparar_textos(
    df: pd.DataFrame,
):

    textos_enmascarados = []

    auditoria = []

    for indice, fila in df.iterrows():

        texto_original = str(
            fila[
                "perfil_egreso"
            ]
        )

        (
            texto_enmascarado,
            reemplazos,
        ) = enmascarar_y_contar(
            texto_original
        )

        textos_enmascarados.append(
            texto_enmascarado
        )

        auditoria.append(
            {
                "indice":
                    int(indice),

                "grado":
                    str(
                        fila[
                            "grado"
                        ]
                    ),

                "denominaciones_enmascaradas":
                    int(
                        reemplazos
                    ),

                "texto_modificado":
                    bool(
                        reemplazos > 0
                    ),
            }
        )

    return (
        np.asarray(
            textos_enmascarados,
            dtype=object,
        ),
        pd.DataFrame(
            auditoria
        ),
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

        "Precision_macro":
            precision_score(
                y_true,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0,
            ),

        "Recall_macro":
            recall_score(
                y_true,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0,
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
# 9. GRÁFICO F1 POR FOLD
# =============================================================================

def generar_grafico_folds(
    df_folds: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.bar(
        df_folds[
            "Fold"
        ].astype(str),
        df_folds[
            "F1_macro"
        ],
    )

    ax.axhline(
        df_folds[
            "F1_macro"
        ].mean(),
        linestyle="--",
        label=(
            "Media "
            f"{df_folds['F1_macro'].mean():.4f}"
        ),
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel(
        "Fold"
    )

    ax.set_ylabel(
        "F1-macro"
    )

    ax.set_title(
        (
            "Modelo final robusto — "
            "F1-macro por fold\n"
            "ComplementNB + SMOTE"
        )
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_FOLDS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 10. MATRIZ DE CONFUSIÓN
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

    normalizada = confusion_matrix(
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
            "Matriz de confusión — modelo final robusto\n"
            "Denominaciones explícitas del grado enmascaradas"
        )
    )

    for fila in range(
        len(CLASES)
    ):

        for columna in range(
            len(CLASES)
        ):

            ax.text(
                columna,
                fila,
                (
                    f"{matriz[fila, columna]}\n"
                    f"{normalizada[fila, columna] * 100:.1f}%"
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
# 11. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 76
    )

    print(
        "MODELO FINAL ROBUSTO — "
        "COMPLEMENTNB + SMOTE"
    )

    print(
        "=" * 76
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
        df[
            "grado"
        ].value_counts()
    )


    # =========================================================================
    # ENMASCARAMIENTO
    # =========================================================================

    (
        textos,
        auditoria_mascara,
    ) = preparar_textos(
        df
    )


    auditoria_mascara.to_csv(
        RUTA_AUDITORIA_MASCARA,
        index=False,
        encoding="utf-8-sig",
    )


    total_reemplazos = int(
        auditoria_mascara[
            "denominaciones_enmascaradas"
        ].sum()
    )


    documentos_modificados = int(
        auditoria_mascara[
            "texto_modificado"
        ].sum()
    )


    etiquetas = (
        df[
            "grado"
        ]
        .astype(str)
        .to_numpy()
    )


    print(
        "\nControl de fuga léxica:"
    )

    print(
        f"  Documentos modificados      : "
        f"{documentos_modificados}/{len(df)}"
    )

    print(
        f"  Denominaciones enmascaradas : "
        f"{total_reemplazos}"
    )


    # =========================================================================
    # VALIDACIÓN CRUZADA
    # =========================================================================

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )


    resultados_folds = []

    predicciones_oof = np.empty(
        len(df),
        dtype=object,
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        "VALIDACIÓN CRUZADA ESTRATIFICADA 5-FOLD"
    )

    print(
        "=" * 76
    )


    for fold, (
        idx_train,
        idx_test,
    ) in enumerate(
        cv.split(
            textos,
            etiquetas,
        ),
        start=1,
    ):


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


        # ---------------------------------------------------------------------
        # TF-IDF se ajusta SOLO con entrenamiento.
        # ---------------------------------------------------------------------

        vectorizador = TfidfVectorizer(
            **TFIDF_CONFIG
        )


        X_train = vectorizador.fit_transform(
            textos_train
        )


        X_test = vectorizador.transform(
            textos_test
        )


        n_features = int(
            X_train.shape[
                1
            ]
        )


        # ---------------------------------------------------------------------
        # SMOTE SOLO sobre entrenamiento.
        # ---------------------------------------------------------------------

        (
            X_train_balanceado,
            y_train_balanceado,
            k_smote,
        ) = aplicar_smote(
            X_train,
            y_train,
        )


        # ---------------------------------------------------------------------
        # Modelo CONGELADO.
        #
        # No se buscan hiperparámetros.
        # ---------------------------------------------------------------------

        modelo = ComplementNB(
            alpha=1.0
        )


        modelo.fit(
            X_train_balanceado,
            y_train_balanceado,
        )


        predicciones = modelo.predict(
            X_test
        )


        predicciones_oof[
            idx_test
        ] = predicciones


        metricas = calcular_metricas(
            y_test,
            predicciones,
        )


        resultados_folds.append(
            {
                "Fold":
                    fold,

                "N_train":
                    int(
                        len(
                            idx_train
                        )
                    ),

                "N_test":
                    int(
                        len(
                            idx_test
                        )
                    ),

                "N_features":
                    n_features,

                "SMOTE_k":
                    (
                        int(k_smote)
                        if k_smote
                        is not None
                        else None
                    ),

                **metricas,
            }
        )


        print(
            f"Fold {fold}: "
            f"F1-macro="
            f"{metricas['F1_macro']:.4f} | "
            f"Accuracy="
            f"{metricas['Accuracy']:.4f} | "
            f"Features="
            f"{n_features}"
        )


    # =========================================================================
    # RESULTADOS POR FOLD
    # =========================================================================

    df_folds = pd.DataFrame(
        resultados_folds
    )


    df_folds.to_csv(
        RUTA_FOLDS,
        index=False,
        encoding="utf-8-sig",
    )


    # =========================================================================
    # PREDICCIONES OOF
    # =========================================================================

    df_predicciones = pd.DataFrame(
        {
            "grado_real":
                etiquetas,

            "grado_predicho":
                predicciones_oof,

            "correcto":
                (
                    etiquetas
                    == predicciones_oof
                ),
        }
    )


    df_predicciones.to_csv(
        RUTA_PREDICCIONES,
        index=False,
        encoding="utf-8-sig",
    )


    # =========================================================================
    # MÉTRICAS AGREGADAS
    # =========================================================================

    reporte_dict = classification_report(
        etiquetas,
        predicciones_oof,
        labels=CLASES,
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )


    reporte_df = pd.DataFrame(
        reporte_dict
    ).transpose()


    reporte_df.to_csv(
        RUTA_REPORTE_CLASES,
        encoding="utf-8-sig",
    )


    f1_media = float(
        df_folds[
            "F1_macro"
        ].mean()
    )


    f1_std = float(
        df_folds[
            "F1_macro"
        ].std()
    )


    accuracy_media = float(
        df_folds[
            "Accuracy"
        ].mean()
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


    # =========================================================================
    # GRÁFICOS
    # =========================================================================

    generar_grafico_folds(
        df_folds
    )


    generar_matriz_confusion(
        etiquetas,
        predicciones_oof,
    )


    # =========================================================================
    # RESUMEN OFICIAL
    # =========================================================================

    resumen = {

        "version":
            "1.0",

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "estado":
            "modelo_robusto_congelado",

        "objetivo":
            (
                "Estimar la capacidad predictiva "
                "conservadora de los perfiles de egreso "
                "evitando utilizar denominaciones "
                "explícitas de los grados como pistas "
                "directas de clasificación."
            ),

        "corpus": {

            "total_perfiles":
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

            "archivo":
                (
                    "perfiles_egreso_"
                    "etiquetado_v2.csv"
                ),
        },

        "control_fuga_lexica": {

            "metodo":
                (
                    "Enmascaramiento de "
                    "denominaciones explícitas "
                    "del grado."
                ),

            "documentos_modificados":
                documentos_modificados,

            "denominaciones_enmascaradas":
                total_reemplazos,

            "nota":
                (
                    "No se eliminan de forma "
                    "general términos aislados "
                    "como civil, ejecución o "
                    "informática."
                ),
        },

        "representacion": {

            "metodo":
                "TF-IDF",

            "max_features":
                400,

            "ngram_range":
                [
                    1,
                    2,
                ],

            "min_df":
                2,

            "max_df":
                0.90,

            "sublinear_tf":
                True,

            "ajuste":
                (
                    "El vectorizador se ajusta "
                    "exclusivamente con el fold "
                    "de entrenamiento."
                ),
        },

        "modelo": {

            "clasificador":
                "Complement Naive Bayes",

            "alpha":
                1.0,

            "balanceo":
                "SMOTE",

            "smote_k_maximo":
                2,

            "semilla":
                SEED,
        },

        "validacion": {

            "metodo":
                (
                    "StratifiedKFold "
                    "5-fold"
                ),

            "n_splits":
                N_SPLITS,

            "shuffle":
                True,

            "random_state":
                SEED,

            "metrica_principal":
                (
                    "Promedio de F1-macro "
                    "de los cinco folds."
                ),
        },

        "resultados_principales": {

            "f1_macro_media_folds":
                round(
                    f1_media,
                    6,
                ),

            "f1_macro_std_folds":
                round(
                    f1_std,
                    6,
                ),

            "accuracy_media_folds":
                round(
                    accuracy_media,
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
        },

        "metricas_oof_por_clase": {

            clase: {

                "precision":
                    round(
                        float(
                            reporte_dict[
                                clase
                            ][
                                "precision"
                            ]
                        ),
                        6,
                    ),

                "recall":
                    round(
                        float(
                            reporte_dict[
                                clase
                            ][
                                "recall"
                            ]
                        ),
                        6,
                    ),

                "f1_score":
                    round(
                        float(
                            reporte_dict[
                                clase
                            ][
                                "f1-score"
                            ]
                        ),
                        6,
                    ),

                "support":
                    int(
                        reporte_dict[
                            clase
                        ][
                            "support"
                        ]
                    ),
            }

            for clase
            in CLASES
        },

        "interpretacion": {

            "resultado_principal":
                (
                    "Este resultado constituye "
                    "la estimación conservadora "
                    "principal del rendimiento "
                    "predictivo."
                ),

            "relacion_con_gwo":
                (
                    "No reemplaza los resultados "
                    "exploratorios GWO. Los resultados "
                    "GWO se mantienen como evidencia "
                    "del potencial discriminativo y "
                    "de reducción de características, "
                    "pero se reportan separadamente."
                ),

            "limitacion":
                (
                    "La clase Ejecución contiene "
                    "únicamente cinco perfiles, "
                    "por lo que las métricas presentan "
                    "alta sensibilidad al particionado."
                ),
        },

        "artefactos": {

            "folds":
                "metricas_folds_modelo_final.csv",

            "predicciones_oof":
                "predicciones_oof_modelo_final.csv",

            "metricas_por_clase":
                "metricas_por_clase_modelo_final.csv",

            "matriz_confusion_csv":
                "matriz_confusion_modelo_final.csv",

            "auditoria_enmascaramiento":
                (
                    "auditoria_enmascaramiento_"
                    "modelo_final.csv"
                ),

            "grafico_folds":
                "f1_por_fold_modelo_final.png",

            "grafico_matriz":
                "matriz_confusion_modelo_final.png",
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
    # SALIDA FINAL
    # =========================================================================

    print(
        "\n"
        + "=" * 76
    )

    print(
        "RESULTADO FINAL ROBUSTO"
    )

    print(
        "=" * 76
    )


    print(
        "\nMétrica principal:"
    )

    print(
        f"  F1-macro medio 5-fold : "
        f"{f1_media:.4f}"
    )

    print(
        f"  Desviación estándar   : "
        f"{f1_std:.4f}"
    )

    print(
        f"  Accuracy media         : "
        f"{accuracy_media:.4f}"
    )


    print(
        "\nMétricas OOF secundarias:"
    )

    print(
        f"  F1-macro OOF : "
        f"{f1_oof:.4f}"
    )

    print(
        f"  Accuracy OOF : "
        f"{accuracy_oof:.4f}"
    )


    print(
        "\nReporte OOF por clase:"
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
        "\nArchivos oficiales generados:"
    )

    print(
        f"  {RUTA_FOLDS}"
    )

    print(
        f"  {RUTA_PREDICCIONES}"
    )

    print(
        f"  {RUTA_REPORTE_CLASES}"
    )

    print(
        f"  {RUTA_MATRIZ}"
    )

    print(
        f"  {RUTA_AUDITORIA_MASCARA}"
    )

    print(
        f"  {RUTA_RESUMEN}"
    )

    print(
        f"  {RUTA_GRAFICO_FOLDS}"
    )

    print(
        f"  {RUTA_GRAFICO_MATRIZ}"
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        "MODELO FINAL ROBUSTO GENERADO CORRECTAMENTE"
    )

    print(
        "=" * 76
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )