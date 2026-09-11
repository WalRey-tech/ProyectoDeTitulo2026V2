# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB


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

RUTA_FEATURES_GWO = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "gwo",
    "gwo_features_seleccionadas.csv",
)

RESULTADOS_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "auditoria_fuga_lexica",
)

RUTA_COMPARACION = os.path.join(
    RESULTADOS_DIR,
    "comparacion_original_vs_enmascarado.csv",
)

RUTA_FEATURES_PROXY = os.path.join(
    RESULTADOS_DIR,
    "auditoria_features_gwo_proxy.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_auditoria_fuga_lexica.json",
)

RUTA_GRAFICO = os.path.join(
    RESULTADOS_DIR,
    "comparacion_f1_fuga_lexica.png",
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

SEED = 42
N_SPLITS = 5

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
}

CLASES = [
    "Civil",
    "Ejecución",
    "Informática",
]


# =============================================================================
# 3. PROXIES DIRECTOS
# =============================================================================

TOKENS_PROXY = {
    "civil",
    "ejecucion",
    "informatica",
    "informatico",
    "informaticos",
    "informaticas",
}


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
# 4. UTILIDADES
# =============================================================================

def normalizar(
    texto: str,
) -> str:

    texto = str(texto).lower()

    texto = unicodedata.normalize(
        "NFD",
        texto,
    )

    return "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )


def es_proxy_feature(
    feature: str,
) -> bool:

    tokens = set(
        normalizar(feature).split()
    )

    return bool(
        tokens.intersection(
            TOKENS_PROXY
        )
    )


def enmascarar_texto(
    texto: str,
) -> str:

    resultado = str(texto)

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


# =============================================================================
# 5. CARGA
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
            f"Faltan columnas: {sorted(faltantes)}"
        )

    df = df.dropna(
        subset=[
            "perfil_egreso",
            "grado",
        ]
    ).copy()

    return df


# =============================================================================
# 6. AUDITORÍA DE FEATURES GWO
# =============================================================================

def auditar_features_gwo() -> dict:

    if not os.path.exists(
        RUTA_FEATURES_GWO
    ):
        raise FileNotFoundError(
            "No se encontró el archivo "
            "de features GWO."
        )

    df = pd.read_csv(
        RUTA_FEATURES_GWO,
        encoding="utf-8-sig",
    )

    if "feature" not in df.columns:
        raise ValueError(
            "El archivo GWO no contiene "
            "la columna 'feature'."
        )

    df[
        "es_proxy_directo"
    ] = df[
        "feature"
    ].astype(str).apply(
        es_proxy_feature
    )

    df_proxy = df[
        df[
            "es_proxy_directo"
        ]
    ].copy()

    df_proxy.to_csv(
        RUTA_FEATURES_PROXY,
        index=False,
        encoding="utf-8-sig",
    )

    total = int(
        len(df)
    )

    n_proxy = int(
        len(df_proxy)
    )

    porcentaje = (
        n_proxy / total * 100
        if total
        else 0.0
    )

    return {
        "features_gwo_total":
            total,

        "features_proxy_directo":
            n_proxy,

        "porcentaje_proxy":
            round(
                porcentaje,
                2,
            ),

        "features_proxy": (
            df_proxy[
                "feature"
            ]
            .astype(str)
            .tolist()
        ),
    }


# =============================================================================
# 7. EVALUACIÓN CV
# =============================================================================

def evaluar_condicion(
    textos: np.ndarray,
    etiquetas: np.ndarray,
    condicion: str,
) -> pd.DataFrame:

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    resultados = []

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

        X_train_text = textos[
            idx_train
        ]

        X_test_text = textos[
            idx_test
        ]

        y_train = etiquetas[
            idx_train
        ]

        y_test = etiquetas[
            idx_test
        ]

        vectorizador = TfidfVectorizer(
            **TFIDF_CONFIG
        )

        X_train = vectorizador.fit_transform(
            X_train_text
        )

        X_test = vectorizador.transform(
            X_test_text
        )

        smote = SMOTE(
            k_neighbors=2,
            random_state=SEED,
        )

        X_train_balanceado, y_train_balanceado = (
            smote.fit_resample(
                X_train,
                y_train,
            )
        )

        modelo = ComplementNB()

        modelo.fit(
            X_train_balanceado,
            y_train_balanceado,
        )

        pred = modelo.predict(
            X_test
        )

        resultado = {
            "condicion":
                condicion,

            "fold":
                fold,

            "n_test":
                int(
                    len(y_test)
                ),

            "f1_macro":
                f1_score(
                    y_test,
                    pred,
                    labels=CLASES,
                    average="macro",
                    zero_division=0,
                ),

            "accuracy":
                accuracy_score(
                    y_test,
                    pred,
                ),
        }

        for clase in CLASES:

            resultado[
                f"f1_{normalizar(clase)}"
            ] = f1_score(
                y_test,
                pred,
                labels=[
                    clase
                ],
                average="macro",
                zero_division=0,
            )

        resultados.append(
            resultado
        )

    return pd.DataFrame(
        resultados
    )


# =============================================================================
# 8. GRÁFICO
# =============================================================================

def generar_grafico(
    comparacion: pd.DataFrame,
) -> None:

    resumen = (
        comparacion
        .groupby(
            "condicion"
        )[
            "f1_macro"
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        resumen[
            "condicion"
        ],
        resumen[
            "mean"
        ],
        yerr=resumen[
            "std"
        ],
        capsize=6,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "F1-macro medio (5-fold)"
    )

    ax.set_title(
        (
            "Auditoría de fuga léxica\n"
            "Texto original vs. proxies enmascarados"
        )
    )

    ax.grid(
        axis="y",
        alpha=0.25,
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
# 9. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 72
    )

    print(
        "AUDITORÍA DE FUGA LÉXICA"
    )

    print(
        "=" * 72
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

    auditoria_features = (
        auditar_features_gwo()
    )

    original = evaluar_condicion(
        textos_originales,
        etiquetas,
        "Original",
    )

    enmascarado = evaluar_condicion(
        textos_enmascarados,
        etiquetas,
        "Enmascarado",
    )

    comparacion = pd.concat(
        [
            original,
            enmascarado,
        ],
        ignore_index=True,
    )

    comparacion.to_csv(
        RUTA_COMPARACION,
        index=False,
        encoding="utf-8-sig",
    )

    generar_grafico(
        comparacion
    )

    resumen_cv = (
        comparacion
        .groupby(
            "condicion"
        )
        .agg(
            f1_macro_media=(
                "f1_macro",
                "mean",
            ),
            f1_macro_std=(
                "f1_macro",
                "std",
            ),
            accuracy_media=(
                "accuracy",
                "mean",
            ),
        )
        .reset_index()
    )

    original_f1 = float(
        resumen_cv.loc[
            resumen_cv[
                "condicion"
            ] == "Original",
            "f1_macro_media",
        ].iloc[0]
    )

    masked_f1 = float(
        resumen_cv.loc[
            resumen_cv[
                "condicion"
            ] == "Enmascarado",
            "f1_macro_media",
        ].iloc[0]
    )

    delta = (
        masked_f1
        - original_f1
    )

    resumen = {

        "corpus": {
            "total":
                int(
                    len(df)
                ),

            "distribucion": {
                str(k): int(v)
                for k, v
                in df[
                    "grado"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "features_gwo":
            auditoria_features,

        "evaluacion_controlada": {

            "metodologia":
                (
                    "TF-IDF ajustado exclusivamente "
                    "en cada fold de entrenamiento + "
                    "SMOTE(k=2) + ComplementNB."
                ),

            "folds":
                N_SPLITS,

            "original_f1_macro":
                round(
                    original_f1,
                    6,
                ),

            "enmascarado_f1_macro":
                round(
                    masked_f1,
                    6,
                ),

            "delta_enmascarado_menos_original":
                round(
                    delta,
                    6,
                ),

            "caida_absoluta":
                round(
                    max(
                        0.0,
                        original_f1
                        - masked_f1,
                    ),
                    6,
                ),
        },

        "interpretacion": {

            "nota":
                (
                    "Esta comparación no sustituye "
                    "la evaluación GWO de referencia. "
                    "Su objetivo es estimar cuánto "
                    "cambia el rendimiento al retirar "
                    "pistas léxicas directas del grado."
                ),

            "criterio":
                (
                    "Una caída importante del F1 tras "
                    "el enmascaramiento sugiere que los "
                    "términos directos del grado estaban "
                    "aportando señal predictiva."
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

    print(
        f"\nCorpus: {len(df)} perfiles"
    )

    print(
        "\nFeatures GWO:"
    )

    print(
        f"  Total seleccionadas : "
        f"{auditoria_features['features_gwo_total']}"
    )

    print(
        f"  Proxies directos    : "
        f"{auditoria_features['features_proxy_directo']}"
    )

    print(
        f"  Porcentaje proxy    : "
        f"{auditoria_features['porcentaje_proxy']:.2f}%"
    )

    if auditoria_features[
        "features_proxy"
    ]:

        print(
            "  Detectadas          : "
            + ", ".join(
                auditoria_features[
                    "features_proxy"
                ]
            )
        )

    print(
        "\nComparación controla 5-fold:"
        "(TF-IDF + SMOTE + ComplementNB, sin GWO)"
    )

    print(
        f"  Original    F1-macro: "
        f"{original_f1:.4f}"
    )

    print(
        f"  Enmascarado F1-macro: "
        f"{masked_f1:.4f}"
    )

    print(
        f"  Delta               : "
        f"{delta:+.4f}"
    )

    print(
        "\nResultados:"
    )

    print(
        f"  {RUTA_COMPARACION}"
    )

    print(
        f"  {RUTA_FEATURES_PROXY}"
    )

    print(
        f"  {RUTA_RESUMEN}"
    )

    print(
        f"  {RUTA_GRAFICO}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "AUDITORÍA COMPLETADA CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )