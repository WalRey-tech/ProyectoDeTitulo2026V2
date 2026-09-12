import os
import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# =============================================================================
# MODELOS
# =============================================================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# =============================================================================
# RUTAS
# =============================================================================

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

RUTA_ENTRADA = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "perfiles_egreso_limpio_v1.csv"
    )
)

RUTA_BENCHMARK = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "benchmark_modelos_v3.csv"
    )
)

RUTA_METRICAS_CLASE = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "metricas_por_clase_v3.csv"
    )
)

RUTA_MATRIZ_CSV = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "matriz_confusion_v3.csv"
    )
)

RUTA_MATRIZ_PNG = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "matriz_confusion_v3.png"
    )
)

RUTA_MODELO = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "modelo_ganador_pipeline_v3.joblib"
    )
)

RUTA_LABEL_ENCODER = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "label_encoder_v3.joblib"
    )
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 5

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "max_df": 0.85,
    "min_df": 2
}


# =============================================================================
# TRANSFORMADOR A MATRIZ DENSA
# =============================================================================

class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Convierte la matriz dispersa de TF-IDF a una matriz densa.

    El corpus es pequeño y se limita a 400 características, por lo que esta
    conversión permite utilizar modelos que no admiten matrices dispersas,
    como LDA y Gradient Boosting.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()

        return np.asarray(X)


# =============================================================================
# CONSTRUCCIÓN DEL PIPELINE
# =============================================================================

def construir_pipeline(modelo):
    """
    TF-IDF y SMOTE se ajustan dentro de cada fold de entrenamiento.

    El fold de validación no participa en:
    - la construcción del vocabulario;
    - el cálculo de los valores IDF;
    - la generación de muestras sintéticas.
    """

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(**TFIDF_CONFIG)
            ),
            (
                "to_dense",
                DenseTransformer()
            ),
            (
                "smote",
                SMOTE(
                    sampling_strategy="auto",
                    k_neighbors=3,
                    random_state=RANDOM_STATE
                )
            ),
            (
                "clasificador",
                modelo
            )
        ]
    )


# =============================================================================
# MODELOS DEL BENCHMARK
# =============================================================================

def crear_modelos(numero_clases):
    modelos = {
        "LDA": LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto"
        ),

        "SVM Kernel Linear": SVC(
            kernel="linear",
            C=10,
            random_state=RANDOM_STATE
        ),

        "SVM Kernel RBF": SVC(
            kernel="rbf",
            C=5,
            random_state=RANDOM_STATE
        ),

        "Regresión Logística": LogisticRegression(
            C=5,
            max_iter=2000,
            random_state=RANDOM_STATE
        ),

        "MLP Red Neuronal": MLPClassifier(
            hidden_layer_sizes=(150, 100),
            max_iter=1500,
            random_state=RANDOM_STATE
        ),

        "Multinomial Naive Bayes": MultinomialNB(),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=3
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        )
    }

    if HAS_XGB:
        modelos["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=numero_clases,
            eval_metric="mlogloss",
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

    return modelos


# =============================================================================
# FLUJO PRINCIPAL
# =============================================================================

def main():
    print("=" * 76)
    print("BENCHMARK SUPERVISADO SIN FUGA DE DATOS")
    print("=" * 76)

    # -------------------------------------------------------------------------
    # 1. Carga y validación
    # -------------------------------------------------------------------------

    try:
        df = pd.read_csv(
            RUTA_ENTRADA,
            encoding="utf-8-sig"
        )
    except FileNotFoundError:
        print(f"Error: no existe el archivo:\n{RUTA_ENTRADA}")
        return
    except Exception as error:
        print(f"Error al leer el dataset: {error}")
        return

    columnas_obligatorias = {
        "perfil_limpio",
        "grado"
    }

    columnas_faltantes = columnas_obligatorias - set(df.columns)

    if columnas_faltantes:
        print(
            "Error: faltan columnas obligatorias: "
            f"{sorted(columnas_faltantes)}"
        )
        return

    df = df.dropna(
        subset=["perfil_limpio", "grado"]
    ).copy()

    df = df[
        df["perfil_limpio"].astype(str).str.strip() != ""
    ].copy()

    X_text = df["perfil_limpio"].astype(str).values
    y_text = df["grado"].astype(str).values

    print(f"Total de perfiles reales: {len(df)}")
    print("\nDistribución original:")
    print(df["grado"].value_counts())

    # -------------------------------------------------------------------------
    # 2. Codificación de etiquetas
    # -------------------------------------------------------------------------

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    nombres_clases = label_encoder.classes_

    print("\nClases codificadas:")
    for indice, clase in enumerate(nombres_clases):
        print(f"  {indice}: {clase}")

    # -------------------------------------------------------------------------
    # 3. Validación cruzada
    # -------------------------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro"
    }

    modelos = crear_modelos(
        numero_clases=len(nombres_clases)
    )

    resultados = []

    print("\nEjecutando benchmark...")
    print("-" * 76)

    for nombre, modelo in modelos.items():
        print(f"Evaluando: {nombre}")

        pipeline = construir_pipeline(
            clone(modelo)
        )

        try:
            scores = cross_validate(
                pipeline,
                X_text,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
                error_score="raise"
            )

            resultado = {
                "Modelo": nombre,

                "Accuracy (%)":
                    scores["test_accuracy"].mean() * 100,

                "Accuracy std (%)":
                    scores["test_accuracy"].std() * 100,

                "Balanced Accuracy (%)":
                    scores["test_balanced_accuracy"].mean() * 100,

                "Balanced Accuracy std (%)":
                    scores["test_balanced_accuracy"].std() * 100,

                "F1 Macro (%)":
                    scores["test_f1_macro"].mean() * 100,

                "F1 Macro std (%)":
                    scores["test_f1_macro"].std() * 100,

                "Precision Macro (%)":
                    scores["test_precision_macro"].mean() * 100,

                "Recall Macro (%)":
                    scores["test_recall_macro"].mean() * 100
            }

            resultados.append(resultado)

            print(
                f"  Accuracy: "
                f"{resultado['Accuracy (%)']:.2f}%"
            )

            print(
                f"  Balanced Accuracy: "
                f"{resultado['Balanced Accuracy (%)']:.2f}%"
            )

            print(
                f"  F1 Macro: "
                f"{resultado['F1 Macro (%)']:.2f}%"
            )

        except Exception as error:
            print(f"  Error durante evaluación: {error}")

        print("-" * 76)

    if not resultados:
        print("No fue posible evaluar ningún modelo.")
        return

    # -------------------------------------------------------------------------
    # 4. Selección del modelo ganador
    # -------------------------------------------------------------------------

    df_resultados = pd.DataFrame(resultados)

    df_resultados = df_resultados.sort_values(
        by=[
            "F1 Macro (%)",
            "Balanced Accuracy (%)",
            "Accuracy (%)"
        ],
        ascending=False
    ).reset_index(drop=True)

    columnas_numericas = df_resultados.select_dtypes(
        include=[np.number]
    ).columns

    df_resultados[columnas_numericas] = (
        df_resultados[columnas_numericas].round(2)
    )

    os.makedirs(
        os.path.dirname(RUTA_BENCHMARK),
        exist_ok=True
    )

    df_resultados.to_csv(
        RUTA_BENCHMARK,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    print("\n" + "=" * 76)
    print("RANKING FINAL")
    print("=" * 76)

    print(
        df_resultados[
            [
                "Modelo",
                "Accuracy (%)",
                "Balanced Accuracy (%)",
                "F1 Macro (%)",
                "F1 Macro std (%)"
            ]
        ].to_string(index=False)
    )

    nombre_ganador = df_resultados.iloc[0]["Modelo"]
    modelo_ganador = modelos[nombre_ganador]

    print("\nModelo ganador por F1 Macro:")
    print(f"  {nombre_ganador}")

    # -------------------------------------------------------------------------
    # 5. Predicciones fuera de muestra
    # -------------------------------------------------------------------------

    pipeline_ganador = construir_pipeline(
        clone(modelo_ganador)
    )

    print("\nGenerando predicciones fuera de muestra...")

    y_pred_oof = cross_val_predict(
        pipeline_ganador,
        X_text,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1
    )

    # -------------------------------------------------------------------------
    # 6. Métricas por clase
    # -------------------------------------------------------------------------

    reporte = classification_report(
        y,
        y_pred_oof,
        target_names=nombres_clases,
        output_dict=True,
        zero_division=0
    )

    df_reporte = pd.DataFrame(
        reporte
    ).transpose()

    df_reporte = df_reporte.reset_index().rename(
        columns={"index": "Clase"}
    )

    columnas_metricas = [
        "precision",
        "recall",
        "f1-score",
        "support"
    ]

    df_reporte[columnas_metricas] = (
        df_reporte[columnas_metricas].round(4)
    )

    df_reporte.to_csv(
        RUTA_METRICAS_CLASE,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    print("\nMétricas por clase:")
    print(df_reporte.to_string(index=False))

    # -------------------------------------------------------------------------
    # 7. Matriz de confusión
    # -------------------------------------------------------------------------

    matriz = confusion_matrix(
        y,
        y_pred_oof
    )

    df_matriz = pd.DataFrame(
        matriz,
        index=nombres_clases,
        columns=nombres_clases
    )

    df_matriz.index.name = "Real"
    df_matriz.columns.name = "Predicho"

    df_matriz.to_csv(
        RUTA_MATRIZ_CSV,
        sep=";",
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL
    )

    print("\nMatriz de confusión:")
    print(df_matriz)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nombres_clases
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        f"Matriz de confusión — {nombre_ganador}\n"
        "Predicciones fuera de muestra"
    )

    plt.tight_layout()
    plt.savefig(
        RUTA_MATRIZ_PNG,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # -------------------------------------------------------------------------
    # 8. Entrenamiento final
    # -------------------------------------------------------------------------

    print("\nEntrenando pipeline ganador con todo el corpus...")

    pipeline_ganador.fit(
        X_text,
        y
    )

    joblib.dump(
        pipeline_ganador,
        RUTA_MODELO
    )

    joblib.dump(
        label_encoder,
        RUTA_LABEL_ENCODER
    )

    print("\n" + "=" * 76)
    print("PROCESO COMPLETADO")
    print("=" * 76)

    print(f"Benchmark:\n  {RUTA_BENCHMARK}")
    print(f"Métricas por clase:\n  {RUTA_METRICAS_CLASE}")
    print(f"Matriz CSV:\n  {RUTA_MATRIZ_CSV}")
    print(f"Matriz PNG:\n  {RUTA_MATRIZ_PNG}")
    print(f"Pipeline ganador:\n  {RUTA_MODELO}")
    print(f"Codificador:\n  {RUTA_LABEL_ENCODER}")


if __name__ == "__main__":
    main()