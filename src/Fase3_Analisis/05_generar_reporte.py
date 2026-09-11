# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd


# =============================================================================
# 1. RUTAS DEL PROYECTO
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

DATA_DIR = os.path.join(
    SRC_ROOT,
    "data",
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed",
)

RESULTADOS_DIR = os.path.join(
    DATA_DIR,
    "resultados_cientificos",
)

GWO_DIR = os.path.join(
    RESULTADOS_DIR,
    "gwo",
)


# =============================================================================
# 2. ARCHIVOS OFICIALES
# =============================================================================

RUTA_CORPUS = os.path.join(
    PROCESSED_DIR,
    "perfiles_egreso_etiquetado_v2.csv",
)

RUTA_GWO_RESULTADOS = os.path.join(
    GWO_DIR,
    "gwo_resultados.csv",
)

RUTA_CV5 = os.path.join(
    GWO_DIR,
    "cv5_gwo_resultados.csv",
)

RUTA_CV10 = os.path.join(
    GWO_DIR,
    "cv10_gwo_resultados.csv",
)

RUTA_FEATURES = os.path.join(
    GWO_DIR,
    "gwo_features_seleccionadas.csv",
)

RUTA_SALIDA = os.path.join(
    RESULTADOS_DIR,
    "resultados_finales.json",
)


# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================

def verificar_archivo(
    ruta: str,
    descripcion: str,
) -> None:

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró {descripcion}:\n{ruta}"
        )


def cargar_csv(
    ruta: str,
) -> pd.DataFrame:

    return pd.read_csv(
        ruta,
        encoding="utf-8-sig",
    )


def redondear(
    valor,
    decimales: int = 4,
):

    return round(
        float(valor),
        decimales,
    )


# =============================================================================
# 4. CORPUS
# =============================================================================

def obtener_datos_corpus() -> dict:

    verificar_archivo(
        RUTA_CORPUS,
        "el corpus V2",
    )

    df = cargar_csv(
        RUTA_CORPUS,
    )

    columnas_requeridas = {
        "grado",
        "perfil_egreso",
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

    conteos = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_perfiles": int(len(df)),
        "distribucion": {
            str(clase): int(cantidad)
            for clase, cantidad
            in conteos.items()
        },
        "archivo": (
            "perfiles_egreso_etiquetado_v2.csv"
        ),
    }


# =============================================================================
# 5. RESULTADOS GWO
# =============================================================================

def obtener_resultados_gwo() -> dict:

    verificar_archivo(
        RUTA_GWO_RESULTADOS,
        "los resultados de GWO",
    )

    df = cargar_csv(
        RUTA_GWO_RESULTADOS,
    )

    baseline = df[
        df["Modelo"]
        .astype(str)
        .str.contains(
            "Baseline",
            case=False,
            na=False,
        )
    ]

    gwo = df[
        df["Modelo"]
        .astype(str)
        .str.contains(
            "GWO",
            case=False,
            na=False,
        )
    ]

    if baseline.empty:
        raise ValueError(
            "No se encontró la fila Baseline "
            "en gwo_resultados.csv."
        )

    if gwo.empty:
        raise ValueError(
            "No se encontró la fila GWO "
            "en gwo_resultados.csv."
        )

    baseline = baseline.iloc[0]
    gwo = gwo.iloc[0]

    features_iniciales = int(
        baseline["n_features"]
    )

    features_seleccionadas = int(
        gwo["n_features"]
    )

    reduccion = (
        1
        - (
            features_seleccionadas
            / features_iniciales
        )
    ) * 100

    return {
        "features_iniciales":
            features_iniciales,

        "features_seleccionadas":
            features_seleccionadas,

        "features_descartadas":
            (
                features_iniciales
                - features_seleccionadas
            ),

        "reduccion_porcentual":
            redondear(
                reduccion,
                2,
            ),

        "baseline_f1_macro":
            redondear(
                baseline["F1_media"],
            ),

        "baseline_f1_std":
            redondear(
                baseline["F1_std"],
            ),

        "gwo_f1_macro_exploratorio":
            redondear(
                gwo["F1_media"],
            ),

        "gwo_f1_std":
            redondear(
                gwo["F1_std"],
            ),

        "delta_f1":
            redondear(
                gwo["F1_media"]
                - baseline["F1_media"],
            ),
    }


# =============================================================================
# 6. VALIDACIÓN CRUZADA
# =============================================================================

def resumir_validacion(
    ruta: str,
    n_folds: int,
) -> dict:

    verificar_archivo(
        ruta,
        f"los resultados CV-{n_folds}",
    )

    df = cargar_csv(
        ruta,
    )

    columnas = [
        "F1_macro",
        "Accuracy",
        "F1_Civil",
        "F1_Ejecucion",
        "F1_Informatica",
    ]

    faltantes = [
        columna
        for columna in columnas
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"El archivo CV-{n_folds} no contiene "
            f"las columnas: {faltantes}"
        )

    f1_media = df[
        "F1_macro"
    ].mean()

    f1_std = df[
        "F1_macro"
    ].std()

    ic95 = (
        1.96
        * f1_std
        / (n_folds ** 0.5)
    )

    return {
        "folds": n_folds,

        "f1_macro_media":
            redondear(
                f1_media,
            ),

        "f1_macro_std":
            redondear(
                f1_std,
            ),

        "f1_macro_ic95":
            redondear(
                ic95,
            ),

        "accuracy_media":
            redondear(
                df["Accuracy"].mean(),
            ),

        "f1_civil_media":
            redondear(
                df["F1_Civil"].mean(),
            ),

        "f1_ejecucion_media":
            redondear(
                df["F1_Ejecucion"].mean(),
            ),

        "f1_informatica_media":
            redondear(
                df["F1_Informatica"].mean(),
            ),
    }


# =============================================================================
# 7. FEATURES SELECCIONADAS
# =============================================================================

def obtener_features() -> dict:

    verificar_archivo(
        RUTA_FEATURES,
        "las características seleccionadas",
    )

    df = cargar_csv(
        RUTA_FEATURES,
    )

    if "feature" not in df.columns:
        raise ValueError(
            "gwo_features_seleccionadas.csv "
            "no contiene la columna 'feature'."
        )

    top_features = (
        df
        .head(30)["feature"]
        .astype(str)
        .tolist()
    )

    return {
        "total": int(len(df)),
        "top_30": top_features,
    }


# =============================================================================
# 8. REPORTE FINAL
# =============================================================================

def main() -> int:

    print(
        "=" * 72
    )
    print(
        "GENERACIÓN DEL REPORTE CIENTÍFICO FINAL"
    )
    print(
        "=" * 72
    )

    corpus = obtener_datos_corpus()

    gwo = obtener_resultados_gwo()

    cv5 = resumir_validacion(
        RUTA_CV5,
        5,
    )

    cv10 = resumir_validacion(
        RUTA_CV10,
        10,
    )

    features = obtener_features()

    reporte = {

        "version_reporte": "2.0",

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuente_oficial_resultados":
            "src/data/resultados_cientificos/gwo",

        "corpus":
            corpus,

        "metodologia": {

            "representacion":
                "TF-IDF word (1,2)-grams",

            "max_features":
                400,

            "clasificador":
                "Complement Naive Bayes",

            "balanceo":
                "SMOTE",

            "seleccion_caracteristicas":
                "Grey Wolf Optimizer (GWO)",

            "semilla":
                42,

        },

        "seleccion_caracteristicas":
            gwo,

        "validacion_5_fold":
            cv5,

        "validacion_10_fold":
            cv10,

        "features_seleccionadas":
            features,

        "interpretacion": {

            "resultado_principal":
                (
                    "GWO redujo el espacio de características "
                    "de 400 a 249 y aumentó el F1-macro "
                    "exploratorio respecto del baseline."
                ),

            "sensibilidad_particionado":
                (
                    "La validación de 10 pliegues presentó "
                    "un F1-macro inferior al observado en "
                    "5 pliegues, evidenciando sensibilidad "
                    "al particionado del corpus."
                ),

        },

        "limitaciones": [

            (
                "El corpus contiene solo cinco perfiles "
                "de Ingeniería de Ejecución en Informática."
            ),

            (
                "El resultado de 5 pliegues posterior a GWO "
                "es exploratorio, debido a que la selección "
                "de características fue realizada antes de "
                "esa comparación externa."
            ),

            (
                "Una estimación estrictamente insesgada de "
                "generalización requiere validación cruzada "
                "anidada, ejecutando la selección GWO dentro "
                "de cada pliegue externo."
            ),

        ],

        "artefactos": {

            "gwo_resultados":
                "gwo/gwo_resultados.csv",

            "features":
                "gwo/gwo_features_seleccionadas.csv",

            "cv5":
                "gwo/cv5_gwo_resultados.csv",

            "cv10":
                "gwo/cv10_gwo_resultados.csv",

            "grafico_seleccion":
                "gwo/gwo_seleccion.png",

            "grafico_mapa_features":
                "gwo/gwo_mapa_features.png",

            "grafico_cv":
                "gwo/cv10_gwo_resultados.png",

            "grafico_f1_clase":
                "gwo/cv10_gwo_f1_clase.png",

        },

    }

    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True,
    )

    with open(
        RUTA_SALIDA,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            reporte,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\nCorpus: "
        f"{corpus['total_perfiles']} perfiles"
    )

    print(
        "Distribución: "
        + ", ".join(
            f"{clase}={cantidad}"
            for clase, cantidad
            in corpus["distribucion"].items()
        )
    )

    print(
        "\nSelección de características:"
    )

    print(
        f"400 -> "
        f"{gwo['features_seleccionadas']} "
        f"({gwo['reduccion_porcentual']}% menos)"
    )

    print(
        f"\nBaseline F1-macro: "
        f"{gwo['baseline_f1_macro']:.4f}"
    )

    print(
        f"GWO 5-fold F1-macro: "
        f"{cv5['f1_macro_media']:.4f}"
    )

    print(
        f"GWO 10-fold F1-macro: "
        f"{cv10['f1_macro_media']:.4f}"
    )

    print(
        f"\nReporte generado:"
        f"\n{RUTA_SALIDA}"
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REPORTE FINAL GENERADO CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )