# -*- coding: utf-8 -*-
"""
Reproducción exploratoria GWO y diagnóstico de sensibilidad
============================================================

Modelo:
    TF-IDF word (1,2)-grams + ComplementNB + SMOTE

Features:
    Subconjunto GWO cargado desde CSV.
    Ejecución de referencia: 249 de 400 características.

IMPORTANTE
----------
Este script reproduce el protocolo exploratorio asociado a la selección
GWO del paquete de referencia.

La selección GWO y el vocabulario TF-IDF fueron obtenidos utilizando
el corpus disponible completo antes de esta comparación. Por tanto,
los resultados obtenidos aquí NO constituyen una estimación final
insesgada de generalización.

El protocolo robusto final se evalúa separadamente en:
    06_modelo_final_robusto.py

La evaluación de 10 folds se conserva exclusivamente como diagnóstico
de sensibilidad y para reproducir el resultado de referencia. Debido a
que la clase Ejecución contiene solo 5 documentos, no es posible que
los 10 folds contengan las tres clases.
"""

from __future__ import annotations

import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer,
    encoding="utf-8",
    errors="replace",
)

sys.stderr = io.TextIOWrapper(
    sys.stderr.buffer,
    encoding="utf-8",
    errors="replace",
)

import ftfy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# ADVERTENCIA CONTROLADA
# =============================================================================
#
# sklearn emite automáticamente una advertencia porque la clase minoritaria
# posee 5 muestras y se ejecuta un diagnóstico de 10 folds.
#
# La advertencia NO se oculta metodológicamente: el script la explica de forma
# explícita antes de ejecutar el bloque 10-fold. Aquí solamente evitamos que
# sklearn imprima el mismo mensaje fuera de orden en stderr.
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message=(
        r"The least populated class in y has only 5 members, "
        r"which is less than n_splits=10\."
    ),
    category=UserWarning,
    module=r"sklearn\.model_selection\._split",
)

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# 1. RUTAS
# =============================================================================

BASE = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_ROOT = os.path.abspath(
    os.path.join(
        BASE,
        "..",
    )
)

CSV_V2 = os.path.join(
    SRC_ROOT,
    "data",
    "processed",
    "perfiles_egreso_etiquetado_v2.csv",
)

OUT_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "gwo",
)

GWO_CSV = os.path.join(
    OUT_DIR,
    "gwo_features_seleccionadas.csv",
)

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

SEED = 42
np.random.seed(SEED)

CLASES_ESPERADAS = [
    "Civil",
    "Ejecución",
    "Informática",
]


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


# =============================================================================
# 3. CARGA DEL CORPUS
# =============================================================================

print("=" * 76)
print("REPRODUCCIÓN EXPLORATORIA GWO Y DIAGNÓSTICO DE SENSIBILIDAD")
print("=" * 76)


if not os.path.exists(CSV_V2):

    raise FileNotFoundError(
        f"No se encontró el corpus V2:\n{CSV_V2}"
    )


if not os.path.exists(GWO_CSV):

    raise FileNotFoundError(
        f"No se encontró el archivo de features GWO:\n{GWO_CSV}"
    )


df = pd.read_csv(
    CSV_V2,
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


df["perfil_egreso"] = df[
    "perfil_egreso"
].apply(
    lambda x:
        ftfy.fix_text(
            str(x)
        )
)

df["grado"] = df[
    "grado"
].apply(
    lambda x:
        ftfy.fix_text(
            str(x)
        )
)


textos = df[
    "perfil_egreso"
].tolist()


le = LabelEncoder()

y = le.fit_transform(
    df[
        "grado"
    ].tolist()
)

clases_enc = list(
    le.classes_
)


if clases_enc != CLASES_ESPERADAS:

    raise ValueError(
        "Las clases detectadas no coinciden con "
        "el orden esperado.\n"
        f"Detectadas: {clases_enc}\n"
        f"Esperadas : {CLASES_ESPERADAS}"
    )


conteos_clase = (
    df[
        "grado"
    ]
    .value_counts()
    .to_dict()
)


print(
    f"\nDataset: {len(df)} documentos"
)

print(
    f"Distribución: {conteos_clase}"
)


# =============================================================================
# 4. RECONSTRUCCIÓN DEL ESPACIO GWO DE REFERENCIA
# =============================================================================
#
# IMPORTANTE:
#
# Se reconstruye deliberadamente el mismo espacio TF-IDF empleado por el
# experimento GWO de referencia. El vectorizador se ajusta sobre el corpus
# completo porque esta sección tiene como objetivo REPRODUCIR ese experimento.
#
# Por esta razón este bloque es exploratorio y NO una evaluación final
# libre de fuga.
# =============================================================================

df_gwo = pd.read_csv(
    GWO_CSV,
    encoding="utf-8-sig",
)

if "feature" not in df_gwo.columns:

    raise ValueError(
        "gwo_features_seleccionadas.csv "
        "no contiene la columna 'feature'."
    )


gwo_feats = set(
    df_gwo[
        "feature"
    ].astype(str)
)


VEC = TfidfVectorizer(
    max_features=400,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    stop_words=STOPWORDS_ES,
)


X_full = VEC.fit_transform(
    textos
).toarray()


feature_names = np.array(
    VEC.get_feature_names_out()
)


mask_gwo = np.array(
    [
        feature in gwo_feats
        for feature
        in feature_names
    ]
)


X_gwo = X_full[
    :,
    mask_gwo,
]


n_feat = int(
    X_gwo.shape[1]
)


print(
    f"Features GWO: {n_feat} "
    f"(de {X_full.shape[1]} totales)"
)


if n_feat == 0:

    raise ValueError(
        "Ninguna característica del CSV coincide "
        "con el vocabulario TF-IDF."
    )


if n_feat != len(gwo_feats):

    print()
    print(
        "ADVERTENCIA:"
    )

    print(
        f"El CSV contiene {len(gwo_feats)} features, "
        f"pero {n_feat} coinciden con el vocabulario "
        "TF-IDF reconstruido."
    )


print()
print(
    "NOTA METODOLÓGICA:"
)

print(
    "  Este bloque reproduce el espacio GWO de referencia."
)

print(
    "  La selección de features y el TF-IDF preceden a esta CV."
)

print(
    "  Por ello sus métricas se interpretan como exploratorias."
)


# =============================================================================
# 5. SMOTE
# =============================================================================

def aplicar_smote(
    X_train,
    y_train,
):

    conteos = np.bincount(
        y_train
    )

    conteos = conteos[
        conteos > 0
    ]

    minimo = int(
        conteos.min()
    )

    if minimo < 2:

        return (
            X_train,
            y_train,
            None,
        )


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
            X_train,
            y_train,
        )
    )


    return (
        X_balanceado,
        y_balanceado,
        k,
    )


# =============================================================================
# 6. EVALUACIÓN
# =============================================================================

def evaluar_cv(
    X,
    y,
    cv,
    etiqueta,
):

    resultados_fold = []

    y_true_all = []
    y_pred_all = []


    print()
    print("─" * 76)

    print(
        etiqueta
    )

    print("─" * 76)

    print(
        "  Fold  n_test  F1-ref   F1-3cl  Accuracy  "
        "F1-Civil  F1-Ejec  F1-Info  clases_test"
    )

    print(
        "  ----  ------  -------  -------  --------  "
        "--------  -------  -------  -----------"
    )


    labels_fijas = np.arange(
        len(
            clases_enc
        )
    )


    for fold, (
        tr,
        te,
    ) in enumerate(
        cv.split(
            X,
            y,
        ),
        start=1,
    ):


        X_tr = X[
            tr
        ]

        X_te = X[
            te
        ]

        y_tr = y[
            tr
        ]

        y_te = y[
            te
        ]


        (
            X_r,
            y_r,
            smote_k,
        ) = aplicar_smote(
            X_tr,
            y_tr,
        )


        clf = ComplementNB()

        clf.fit(
            X_r,
            y_r,
        )


        y_pred = clf.predict(
            X_te
        )


        # ---------------------------------------------------------------------
        # F1-ref:
        #
        # Reproduce exactamente el comportamiento histórico de sklearn con
        # average='macro' sin fijar labels. Si una clase está ausente del fold
        # y tampoco es predicha, esa clase no participa del promedio.
        #
        # Se conserva para reproducibilidad del resultado 10-fold original.
        # ---------------------------------------------------------------------

        f1_referencia = f1_score(
            y_te,
            y_pred,
            average="macro",
            zero_division=0,
        )


        # ---------------------------------------------------------------------
        # F1-3cl:
        #
        # Fuerza explícitamente las tres clases. Esto permite observar la
        # consecuencia de ejecutar 10-fold cuando Ejecución solo posee
        # cinco documentos.
        # ---------------------------------------------------------------------

        f1_tres_clases = f1_score(
            y_te,
            y_pred,
            labels=labels_fijas,
            average="macro",
            zero_division=0,
        )


        acc = accuracy_score(
            y_te,
            y_pred,
        )


        f1_cls = f1_score(
            y_te,
            y_pred,
            labels=labels_fijas,
            average=None,
            zero_division=0,
        )


        clases_presentes_indices = (
            np.unique(
                y_te
            )
        )


        clases_presentes = [
            clases_enc[
                indice
            ]
            for indice
            in clases_presentes_indices
        ]


        clases_texto = (
            "/".join(
                clases_presentes
            )
        )


        contiene_tres_clases = (
            len(
                clases_presentes
            )
            == len(
                clases_enc
            )
        )


        print(
            f"  {fold:>4}  "
            f"{len(y_te):>6}  "
            f"{f1_referencia:>7.4f}  "
            f"{f1_tres_clases:>7.4f}  "
            f"{acc:>8.4f}  "
            f"{f1_cls[0]:>8.4f}  "
            f"{f1_cls[1]:>7.4f}  "
            f"{f1_cls[2]:>7.4f}  "
            f"{clases_texto}"
        )


        resultados_fold.append(
            {
                "fold":
                    fold,

                "n_test":
                    int(
                        len(
                            y_te
                        )
                    ),

                # Compatibilidad histórica con resultados anteriores.
                "F1_macro":
                    f1_referencia,

                "F1_macro_3clases":
                    f1_tres_clases,

                "Accuracy":
                    acc,

                "F1_Civil":
                    f1_cls[
                        0
                    ],

                "F1_Ejecucion":
                    f1_cls[
                        1
                    ],

                "F1_Informatica":
                    f1_cls[
                        2
                    ],

                "SMOTE_k":
                    smote_k,

                "clases_reales_test":
                    clases_texto,

                "contiene_3_clases":
                    contiene_tres_clases,
            }
        )


        y_true_all.extend(
            y_te
        )

        y_pred_all.extend(
            y_pred
        )


    df_r = pd.DataFrame(
        resultados_fold
    )


    y_ta = np.asarray(
        y_true_all
    )

    y_pa = np.asarray(
        y_pred_all
    )


    f1_oof = f1_score(
        y_ta,
        y_pa,
        labels=labels_fijas,
        average="macro",
        zero_division=0,
    )


    acc_oof = accuracy_score(
        y_ta,
        y_pa,
    )


    folds_completos = int(
        df_r[
            "contiene_3_clases"
        ].sum()
    )


    folds_incompletos = int(
        len(
            df_r
        )
        - folds_completos
    )


    print()
    print(
        "Resumen:"
    )

    print(
        f"  F1-ref media            : "
        f"{df_r['F1_macro'].mean():.4f}"
    )

    print(
        f"  F1-ref std              : "
        f"{df_r['F1_macro'].std():.4f}"
    )

    print(
        f"  F1-3cl media            : "
        f"{df_r['F1_macro_3clases'].mean():.4f}"
    )

    print(
        f"  F1-3cl std              : "
        f"{df_r['F1_macro_3clases'].std():.4f}"
    )

    print(
        f"  Accuracy media          : "
        f"{df_r['Accuracy'].mean():.4f}"
    )

    print(
        f"  F1-macro OOF 3 clases  : "
        f"{f1_oof:.4f}"
    )

    print(
        f"  Accuracy OOF            : "
        f"{acc_oof:.4f}"
    )

    print(
        f"  Folds con las 3 clases  : "
        f"{folds_completos}/{len(df_r)}"
    )

    print(
        f"  Folds sin alguna clase  : "
        f"{folds_incompletos}/{len(df_r)}"
    )


    print()
    print(
        "Reporte OOF agregado:"
    )

    print(
        classification_report(
            y_ta,
            y_pa,
            labels=labels_fijas,
            target_names=clases_enc,
            zero_division=0,
        )
    )


    return (
        df_r,
        y_ta,
        y_pa,
        f1_oof,
        acc_oof,
    )


# =============================================================================
# 7. 5-FOLD — REPRODUCCIÓN EXPLORATORIA PRINCIPAL
# =============================================================================

CV5 = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


(
    df_5,
    yt5,
    yp5,
    f1_oof_5,
    acc_oof_5,
) = evaluar_cv(
    X_gwo,
    y,
    CV5,
    (
        "5-FOLD — REPRODUCCIÓN EXPLORATORIA "
        "GWO DE REFERENCIA"
    ),
)


# =============================================================================
# 8. 10-FOLD — DIAGNÓSTICO DE SENSIBILIDAD
# =============================================================================

n_minoritario = int(
    df[
        "grado"
    ]
    .value_counts()
    .min()
)


print()
print("=" * 76)

print(
    "ADVERTENCIA METODOLÓGICA — 10-FOLD"
)

print("=" * 76)

print(
    f"La clase minoritaria contiene "
    f"{n_minoritario} documentos."
)

print(
    "Con 10 folds no es posible incluir "
    "las tres clases en todos los folds de test."
)

print(
    "La ejecución 10-fold se conserva exclusivamente "
    "para reproducir el diagnóstico"
)

print(
    "de sensibilidad del paquete GWO de referencia."
)

print(
    "El valor histórico F1-ref no se utilizará como "
    "métrica robusta final de generalización."
)


CV10 = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=SEED,
)


(
    df_10,
    yt10,
    yp10,
    f1_oof_10,
    acc_oof_10,
) = evaluar_cv(
    X_gwo,
    y,
    CV10,
    (
        "10-FOLD — DIAGNÓSTICO DE "
        "SENSIBILIDAD GWO"
    ),
)


# =============================================================================
# 9. GUARDAR CSV
# =============================================================================

RUTA_CV5 = os.path.join(
    OUT_DIR,
    "cv5_gwo_resultados.csv",
)

RUTA_CV10 = os.path.join(
    OUT_DIR,
    "cv10_gwo_resultados.csv",
)


df_5.to_csv(
    RUTA_CV5,
    index=False,
    encoding="utf-8-sig",
)

df_10.to_csv(
    RUTA_CV10,
    index=False,
    encoding="utf-8-sig",
)


print()
print(
    "CSVs guardados:"
)

print(
    f"  {RUTA_CV5}"
)

print(
    f"  {RUTA_CV10}"
)


# =============================================================================
# 10. VISUALIZACIÓN PRINCIPAL
# =============================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5),
)


# -----------------------------------------------------------------------------
# A. 5-fold
# -----------------------------------------------------------------------------

ax = axes[
    0
]

ax.plot(
    df_5[
        "fold"
    ],
    df_5[
        "F1_macro_3clases"
    ],
    marker="o",
    linewidth=2,
)

ax.axhline(
    df_5[
        "F1_macro_3clases"
    ].mean(),
    linestyle="--",
    label=(
        "Media = "
        f"{df_5['F1_macro_3clases'].mean():.3f}"
    ),
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xlabel(
    "Fold"
)

ax.set_ylabel(
    "F1-macro"
)

ax.set_title(
    "5-fold exploratorio\n"
    "todos los folds contienen 3 clases"
)

ax.set_xticks(
    range(
        1,
        6,
    )
)

ax.grid(
    alpha=0.3,
)

ax.legend()


# -----------------------------------------------------------------------------
# B. 10-fold: diferencia entre cálculo histórico y tres clases fijas
# -----------------------------------------------------------------------------

ax = axes[
    1
]

ax.plot(
    df_10[
        "fold"
    ],
    df_10[
        "F1_macro"
    ],
    marker="o",
    linewidth=2,
    label="F1-ref histórico",
)

ax.plot(
    df_10[
        "fold"
    ],
    df_10[
        "F1_macro_3clases"
    ],
    marker="s",
    linewidth=2,
    label="F1 con 3 clases fijas",
)

ax.set_ylim(
    0,
    1.05,
)

ax.set_xlabel(
    "Fold"
)

ax.set_ylabel(
    "F1-macro"
)

ax.set_title(
    "10-fold diagnóstico\n"
    "Ejecución n=5"
)

ax.set_xticks(
    range(
        1,
        11,
    )
)

ax.grid(
    alpha=0.3,
)

ax.legend(
    fontsize=8,
)


# -----------------------------------------------------------------------------
# C. Matriz de confusión OOF 10-fold
# -----------------------------------------------------------------------------

ax = axes[
    2
]

cm = confusion_matrix(
    yt10,
    yp10,
    labels=np.arange(
        len(
            clases_enc
        )
    ),
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=clases_enc,
    yticklabels=clases_enc,
    ax=ax,
    cbar=False,
    linewidths=0.5,
)

ax.set_xlabel(
    "Predicción"
)

ax.set_ylabel(
    "Real"
)

ax.set_title(
    "Matriz de confusión OOF\n"
    "10-fold diagnóstico"
)


plt.tight_layout()


RUTA_GRAFICO = os.path.join(
    OUT_DIR,
    "cv10_gwo_resultados.png",
)


plt.savefig(
    RUTA_GRAFICO,
    dpi=150,
)

plt.close()


print(
    f"Gráfica guardada: {RUTA_GRAFICO}"
)


# =============================================================================
# 11. F1 POR CLASE — 10-FOLD
# =============================================================================

fig, ax = plt.subplots(
    figsize=(12, 4.5)
)


x = df_10[
    "fold"
]


ax.plot(
    x,
    df_10[
        "F1_Civil"
    ],
    marker="o",
    linewidth=2,
    label="Civil",
)

ax.plot(
    x,
    df_10[
        "F1_Ejecucion"
    ],
    marker="s",
    linewidth=2,
    label="Ejecución",
)

ax.plot(
    x,
    df_10[
        "F1_Informatica"
    ],
    marker="^",
    linewidth=2,
    label="Informática",
)


ax.set_xlabel(
    "Fold"
)

ax.set_ylabel(
    "F1 por clase"
)

ax.set_title(
    "F1 por clase — diagnóstico 10-fold GWO\n"
    "Los folds sin soporte para Ejecución se muestran con F1=0"
)

ax.legend(
    loc="lower left"
)

ax.grid(
    alpha=0.3
)

ax.set_ylim(
    -0.05,
    1.05,
)

ax.set_xticks(
    range(
        1,
        11,
    )
)


plt.tight_layout()


RUTA_GRAFICO_CLASE = os.path.join(
    OUT_DIR,
    "cv10_gwo_f1_clase.png",
)


plt.savefig(
    RUTA_GRAFICO_CLASE,
    dpi=150,
)

plt.close()


print(
    f"Gráfica guardada: {RUTA_GRAFICO_CLASE}"
)


# =============================================================================
# 12. RESUMEN FINAL
# =============================================================================

m5 = float(
    df_5[
        "F1_macro_3clases"
    ].mean()
)

s5 = float(
    df_5[
        "F1_macro_3clases"
    ].std()
)


m10_ref = float(
    df_10[
        "F1_macro"
    ].mean()
)

s10_ref = float(
    df_10[
        "F1_macro"
    ].std()
)


m10_3cl = float(
    df_10[
        "F1_macro_3clases"
    ].mean()
)

s10_3cl = float(
    df_10[
        "F1_macro_3clases"
    ].std()
)


folds_10_completos = int(
    df_10[
        "contiene_3_clases"
    ].sum()
)


print()
print("=" * 76)

print(
    "RESUMEN GWO — REPRODUCCIÓN Y DIAGNÓSTICO"
)

print("=" * 76)


print()
print(
    "5-fold exploratorio:"
)

print(
    f"  F1-macro 3 clases : "
    f"{m5:.4f}"
)

print(
    f"  Std               : "
    f"{s5:.4f}"
)

print(
    f"  F1-macro OOF      : "
    f"{f1_oof_5:.4f}"
)


print()
print(
    "10-fold diagnóstico:"
)

print(
    f"  F1-ref histórico       : "
    f"{m10_ref:.4f}"
)

print(
    f"  Std F1-ref             : "
    f"{s10_ref:.4f}"
)

print(
    f"  F1 con 3 clases fijas  : "
    f"{m10_3cl:.4f}"
)

print(
    f"  Std F1 3 clases        : "
    f"{s10_3cl:.4f}"
)

print(
    f"  F1-macro OOF           : "
    f"{f1_oof_10:.4f}"
)

print(
    f"  Folds con 3 clases     : "
    f"{folds_10_completos}/10"
)


print()
print(
    "INTERPRETACIÓN:"
)

print(
    "  - El 5-fold reproduce el resultado exploratorio GWO de referencia."
)

print(
    "  - El 10-fold se conserva como diagnóstico histórico de sensibilidad."
)

print(
    "  - Con Ejecución n=5, 10-fold no contiene las tres clases en cada fold."
)

print(
    "  - El F1-ref 10-fold no debe tratarse como estimación robusta final."
)

print(
    "  - La generalización conservadora se reporta desde "
    "06_modelo_final_robusto.py."
)


print()
print("=" * 76)

print(
    "VALIDACIÓN EXPLORATORIA GWO COMPLETADA"
)

print("=" * 76)