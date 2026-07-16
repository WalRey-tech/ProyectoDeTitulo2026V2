import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# =============================================================================

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

DIRECTORIO_DATOS = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
    )
)

RUTA_DATOS = os.path.join(
    DIRECTORIO_DATOS,
    "perfiles_egreso_limpio_v1.csv",
)

RUTA_SALIDA_IMAGEN = os.path.join(
    DIRECTORIO_DATOS,
    "matriz_confusion.png",
)

RUTA_SALIDA_CSV = os.path.join(
    DIRECTORIO_DATOS,
    "matriz_confusion.csv",
)

RUTA_REPORTE_CLASES = os.path.join(
    DIRECTORIO_DATOS,
    "reporte_clasificacion_por_clase.csv",
)

RUTA_METRICAS_JSON = os.path.join(
    DIRECTORIO_DATOS,
    "metricas_matriz_confusion_v1.json",
)

RUTA_METRICAS_FOLDS = os.path.join(
    DIRECTORIO_DATOS,
    "metricas_por_fold_matriz_confusion.csv",
)

SEMILLA = 42
N_SPLITS = 5

ORDEN_CLASES = [
    "Civil",
    "Ejecución",
    "Informática",
]


# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def validar_dataset(df):
    """Valida las columnas y registros necesarios para la evaluación."""
    columnas_requeridas = {
        "perfil_limpio",
        "grado",
    }

    faltantes = columnas_requeridas - set(df.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas obligatorias en el dataset: "
            f"{sorted(faltantes)}"
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

    conteos = df["grado"].value_counts()

    clases_insuficientes = conteos[
        conteos < N_SPLITS
    ]

    if not clases_insuficientes.empty:
        raise ValueError(
            "Cada clase debe contener al menos "
            f"{N_SPLITS} perfiles para aplicar "
            "StratifiedKFold de 5 pliegues. "
            f"Clases insuficientes: "
            f"{clases_insuficientes.to_dict()}"
        )

    return df


def crear_pipeline():
    """
    Crea el mismo pipeline utilizado para evaluar el modelo ganador.

    TF-IDF y SMOTE se ajustan exclusivamente con los datos de entrenamiento
    de cada fold, evitando fuga de información hacia el fold de validación.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=400,
                    ngram_range=(1, 2),
                    max_df=0.85,
                    min_df=2,
                ),
            ),
            (
                "smote",
                SMOTE(
                    sampling_strategy="auto",
                    k_neighbors=3,
                    random_state=SEMILLA,
                ),
            ),
            (
                "clasificador",
                LogisticRegression(
                    C=5,
                    max_iter=2000,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )


def construir_anotaciones(matriz, matriz_normalizada):
    """Combina conteos absolutos y porcentajes por clase real."""
    anotaciones = np.empty(
        matriz.shape,
        dtype=object,
    )

    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            conteo = int(matriz[fila, columna])
            porcentaje = (
                matriz_normalizada[fila, columna]
                * 100
            )

            anotaciones[fila, columna] = (
                f"{conteo}\n"
                f"{porcentaje:.1f}%"
            )

    return anotaciones


def guardar_grafico(
    matriz,
    matriz_normalizada,
    clases,
):
    """
    Guarda una matriz con conteos y porcentajes normalizados por clase real.
    """
    anotaciones = construir_anotaciones(
        matriz,
        matriz_normalizada,
    )

    figura, eje = plt.subplots(
        figsize=(8.5, 6.8)
    )

    imagen = eje.imshow(
        matriz_normalizada,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    figura.colorbar(
        imagen,
        ax=eje,
        label="Proporción dentro de la clase real",
    )

    eje.set(
        xticks=np.arange(len(clases)),
        yticks=np.arange(len(clases)),
        xticklabels=clases,
        yticklabels=clases,
        xlabel="Clase predicha",
        ylabel="Clase real",
        title=(
            "Matriz de confusión — Regresión Logística\n"
            "Predicciones fuera de muestra con validación cruzada 5-fold"
        ),
    )

    plt.setp(
        eje.get_xticklabels(),
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )

    umbral = matriz_normalizada.max() / 2.0

    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            eje.text(
                columna,
                fila,
                anotaciones[fila, columna],
                ha="center",
                va="center",
                color=(
                    "white"
                    if matriz_normalizada[
                        fila,
                        columna
                    ] > umbral
                    else "black"
                ),
                fontsize=11,
                fontweight="bold",
            )

    figura.tight_layout()

    figura.savefig(
        RUTA_SALIDA_IMAGEN,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figura)


# =============================================================================
# 3. FLUJO PRINCIPAL
# =============================================================================

def main():
    print("=" * 78)
    print(
        "MATRIZ DE CONFUSIÓN DEL MODELO GANADOR "
        "(REGRESIÓN LOGÍSTICA)"
    )
    print("=" * 78)

    try:
        df = pd.read_csv(
            RUTA_DATOS,
            encoding="utf-8-sig",
        )
    except FileNotFoundError:
        print(
            "Error: no se encontró el dataset limpio:\n"
            f"{RUTA_DATOS}"
        )
        return
    except Exception as error:
        print(
            "Error al leer el dataset: "
            f"{error}"
        )
        return

    try:
        df = validar_dataset(df)
    except ValueError as error:
        print(
            "Error de validación: "
            f"{error}"
        )
        return

    textos = (
        df["perfil_limpio"]
        .astype(str)
        .to_numpy()
    )

    etiquetas = (
        df["grado"]
        .astype(str)
        .to_numpy()
    )

    clases_presentes = [
        clase
        for clase in ORDEN_CLASES
        if clase in set(etiquetas)
    ]

    clases_adicionales = sorted(
        set(etiquetas) - set(clases_presentes)
    )

    clases = (
        clases_presentes
        + clases_adicionales
    )

    print(
        f"Total de perfiles reales: "
        f"{len(df)}"
    )

    print("\nDistribución por grado:")
    print(df["grado"].value_counts())

    validacion = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEMILLA,
    )

    predicciones_oof = np.empty(
        len(etiquetas),
        dtype=object,
    )

    metricas_folds = []

    print(
        "\nGenerando predicciones fuera de muestra "
        "para cada fold..."
    )

    for numero_fold, (
        indices_entrenamiento,
        indices_validacion,
    ) in enumerate(
        validacion.split(textos, etiquetas),
        start=1,
    ):
        pipeline = crear_pipeline()

        X_train = textos[indices_entrenamiento]
        X_valid = textos[indices_validacion]

        y_train = etiquetas[indices_entrenamiento]
        y_valid = etiquetas[indices_validacion]

        pipeline.fit(
            X_train,
            y_train,
        )

        y_pred_fold = pipeline.predict(
            X_valid
        )

        predicciones_oof[
            indices_validacion
        ] = y_pred_fold

        metricas_fold = {
            "Fold": numero_fold,
            "N_Validacion": int(
                len(indices_validacion)
            ),
            "Accuracy": accuracy_score(
                y_valid,
                y_pred_fold,
            ),
            "Balanced_Accuracy":
                balanced_accuracy_score(
                    y_valid,
                    y_pred_fold,
                ),
            "F1_Macro": f1_score(
                y_valid,
                y_pred_fold,
                average="macro",
                zero_division=0,
            ),
            "Precision_Macro": precision_score(
                y_valid,
                y_pred_fold,
                average="macro",
                zero_division=0,
            ),
            "Recall_Macro": recall_score(
                y_valid,
                y_pred_fold,
                average="macro",
                zero_division=0,
            ),
        }

        metricas_folds.append(
            metricas_fold
        )

        print(
            f"Fold {numero_fold}: "
            f"Accuracy="
            f"{metricas_fold['Accuracy'] * 100:.2f}% | "
            f"Balanced Accuracy="
            f"{metricas_fold['Balanced_Accuracy'] * 100:.2f}% | "
            f"F1 Macro="
            f"{metricas_fold['F1_Macro'] * 100:.2f}%"
        )

    # =========================================================================
    # 4. MATRIZ DE CONFUSIÓN FUERA DE MUESTRA
    # =========================================================================

    matriz = confusion_matrix(
        etiquetas,
        predicciones_oof,
        labels=clases,
    )

    matriz_normalizada = confusion_matrix(
        etiquetas,
        predicciones_oof,
        labels=clases,
        normalize="true",
    )

    guardar_grafico(
        matriz,
        matriz_normalizada,
        clases,
    )

    # =========================================================================
    # 5. MÉTRICAS AGREGADAS Y POR CLASE
    # =========================================================================

    reporte_dict = classification_report(
        etiquetas,
        predicciones_oof,
        labels=clases,
        target_names=clases,
        output_dict=True,
        zero_division=0,
    )

    reporte_clases = pd.DataFrame(
        reporte_dict
    ).transpose()

    metricas_folds_df = pd.DataFrame(
        metricas_folds
    )

    promedio_folds = {
        "accuracy": float(
            metricas_folds_df[
                "Accuracy"
            ].mean()
        ),
        "accuracy_std": float(
            metricas_folds_df[
                "Accuracy"
            ].std(ddof=0)
        ),
        "balanced_accuracy": float(
            metricas_folds_df[
                "Balanced_Accuracy"
            ].mean()
        ),
        "balanced_accuracy_std": float(
            metricas_folds_df[
                "Balanced_Accuracy"
            ].std(ddof=0)
        ),
        "f1_macro": float(
            metricas_folds_df[
                "F1_Macro"
            ].mean()
        ),
        "f1_macro_std": float(
            metricas_folds_df[
                "F1_Macro"
            ].std(ddof=0)
        ),
        "precision_macro": float(
            metricas_folds_df[
                "Precision_Macro"
            ].mean()
        ),
        "recall_macro": float(
            metricas_folds_df[
                "Recall_Macro"
            ].mean()
        ),
    }

    metricas_oof = {
        "accuracy": float(
            accuracy_score(
                etiquetas,
                predicciones_oof,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                etiquetas,
                predicciones_oof,
            )
        ),
        "f1_macro": float(
            f1_score(
                etiquetas,
                predicciones_oof,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_macro": float(
            precision_score(
                etiquetas,
                predicciones_oof,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                etiquetas,
                predicciones_oof,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # =========================================================================
    # 6. EXPORTACIÓN
    # =========================================================================

    os.makedirs(
        DIRECTORIO_DATOS,
        exist_ok=True,
    )

    pd.DataFrame(
        matriz,
        index=[
            f"Real_{clase}"
            for clase in clases
        ],
        columns=[
            f"Predicha_{clase}"
            for clase in clases
        ],
    ).to_csv(
        RUTA_SALIDA_CSV,
        sep=";",
        encoding="utf-8-sig",
    )

    reporte_clases.to_csv(
        RUTA_REPORTE_CLASES,
        sep=";",
        encoding="utf-8-sig",
    )

    metricas_folds_df.to_csv(
        RUTA_METRICAS_FOLDS,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        RUTA_METRICAS_JSON,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            {
                "modelo": "Regresión Logística",
                "metodologia": (
                    "Predicciones out-of-fold obtenidas "
                    "mediante StratifiedKFold de 5 pliegues. "
                    "TF-IDF y SMOTE se ajustan únicamente "
                    "sobre el fold de entrenamiento."
                ),
                "configuracion": {
                    "tfidf": {
                        "max_features": 400,
                        "ngram_range": [1, 2],
                        "max_df": 0.85,
                        "min_df": 2,
                    },
                    "smote": {
                        "k_neighbors": 3,
                        "random_state": SEMILLA,
                    },
                    "regresion_logistica": {
                        "C": 5,
                        "max_iter": 2000,
                        "random_state": SEMILLA,
                    },
                    "validacion": {
                        "n_splits": N_SPLITS,
                        "shuffle": True,
                        "random_state": SEMILLA,
                    },
                },
                "clases": clases,
                "total_perfiles": int(
                    len(df)
                ),
                "distribucion_clases": {
                    str(clase): int(cantidad)
                    for clase, cantidad
                    in df["grado"]
                    .value_counts()
                    .to_dict()
                    .items()
                },
                "promedio_metricas_por_fold": {
                    clave: round(valor, 6)
                    for clave, valor
                    in promedio_folds.items()
                },
                "metricas_globales_oof": {
                    clave: round(valor, 6)
                    for clave, valor
                    in metricas_oof.items()
                },
                "matriz_confusion": (
                    matriz.astype(int).tolist()
                ),
                "matriz_confusion_normalizada": (
                    np.round(
                        matriz_normalizada,
                        6,
                    ).tolist()
                ),
                "reporte_por_clase": reporte_dict,
                "nota_interpretacion": (
                    "Las métricas promedio por fold son "
                    "comparables con el benchmark. Las métricas "
                    "globales OOF se calculan acumulando todas las "
                    "predicciones fuera de muestra y pueden variar "
                    "ligeramente respecto del promedio de folds."
                ),
            },
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    # =========================================================================
    # 7. SALIDA EN CONSOLA
    # =========================================================================

    print("\n" + "=" * 78)
    print("MATRIZ DE CONFUSIÓN — CONTEOS")
    print("=" * 78)

    matriz_df = pd.DataFrame(
        matriz,
        index=clases,
        columns=clases,
    )

    print(matriz_df)

    print("\n" + "=" * 78)
    print("MÉTRICAS PROMEDIO DE LOS 5 FOLDS")
    print("=" * 78)

    print(
        "Accuracy: "
        f"{promedio_folds['accuracy'] * 100:.2f}% "
        f"± "
        f"{promedio_folds['accuracy_std'] * 100:.2f}"
    )

    print(
        "Balanced Accuracy: "
        f"{promedio_folds['balanced_accuracy'] * 100:.2f}% "
        f"± "
        f"{promedio_folds['balanced_accuracy_std'] * 100:.2f}"
    )

    print(
        "F1 Macro: "
        f"{promedio_folds['f1_macro'] * 100:.2f}% "
        f"± "
        f"{promedio_folds['f1_macro_std'] * 100:.2f}"
    )

    print(
        "Precision Macro: "
        f"{promedio_folds['precision_macro'] * 100:.2f}%"
    )

    print(
        "Recall Macro: "
        f"{promedio_folds['recall_macro'] * 100:.2f}%"
    )

    print("\n" + "=" * 78)
    print("REPORTE POR CLASE — PREDICCIONES OOF")
    print("=" * 78)

    print(
        classification_report(
            etiquetas,
            predicciones_oof,
            labels=clases,
            target_names=clases,
            zero_division=0,
        )
    )

    print("=" * 78)
    print("ARCHIVOS GENERADOS")
    print("=" * 78)
    print(f"Imagen:\n  {RUTA_SALIDA_IMAGEN}")
    print(f"Matriz CSV:\n  {RUTA_SALIDA_CSV}")
    print(f"Reporte por clase:\n  {RUTA_REPORTE_CLASES}")
    print(f"Métricas por fold:\n  {RUTA_METRICAS_FOLDS}")
    print(f"Resumen JSON:\n  {RUTA_METRICAS_JSON}")


if __name__ == "__main__":
    main()
