# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd


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

PCA_DIR = os.path.join(
    RESULTADOS_DIR,
    "visualizaciones_exploratorias",
)

HOMOGENEIDAD_DIR = os.path.join(
    RESULTADOS_DIR,
    "homogeneidad_semantica",
)

LEXICO_DIR = os.path.join(
    RESULTADOS_DIR,
    "diferenciacion_lexica",
)


# =============================================================================
# 2. ARCHIVOS DE ENTRADA
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

RUTA_PCA_RESUMEN = os.path.join(
    PCA_DIR,
    "resumen_pca_lda_v2.json",
)

RUTA_HOMOGENEIDAD_RESUMEN = os.path.join(
    HOMOGENEIDAD_DIR,
    "resumen_homogeneidad_v2.json",
)

RUTA_LEXICO_RESUMEN = os.path.join(
    LEXICO_DIR,
    "resumen_diferenciacion_lexica_v2.json",
)


# =============================================================================
# 3. SALIDA OFICIAL
# =============================================================================

RUTA_SALIDA = os.path.join(
    RESULTADOS_DIR,
    "resultados_finales.json",
)


# =============================================================================
# 4. UTILIDADES
# =============================================================================

def verificar_archivo(
    ruta: str,
    descripcion: str,
) -> None:

    if not os.path.exists(
        ruta
    ):
        raise FileNotFoundError(
            f"No se encontró {descripcion}:\n"
            f"{ruta}"
        )


def cargar_csv(
    ruta: str,
) -> pd.DataFrame:

    return pd.read_csv(
        ruta,
        encoding="utf-8-sig",
    )


def cargar_json(
    ruta: str,
) -> dict:

    with open(
        ruta,
        "r",
        encoding="utf-8",
    ) as archivo:

        return json.load(
            archivo
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
# 5. CORPUS
# =============================================================================

def obtener_corpus() -> dict:

    verificar_archivo(
        RUTA_CORPUS,
        "el corpus V2",
    )

    df = cargar_csv(
        RUTA_CORPUS
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
            "El corpus V2 no contiene "
            f"las columnas: {sorted(faltantes)}"
        )

    df = df.dropna(
        subset=[
            "perfil_egreso",
            "grado",
        ]
    ).copy()

    conteos = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_perfiles":
            int(
                len(df)
            ),

        "distribucion": {
            str(clase):
                int(cantidad)

            for clase, cantidad
            in conteos.items()
        },

        "archivo":
            "perfiles_egreso_etiquetado_v2.csv",
    }


# =============================================================================
# 6. RESULTADOS GWO
# =============================================================================

def obtener_gwo() -> dict:

    verificar_archivo(
        RUTA_GWO_RESULTADOS,
        "gwo_resultados.csv",
    )

    verificar_archivo(
        RUTA_FEATURES,
        "gwo_features_seleccionadas.csv",
    )

    resultados = cargar_csv(
        RUTA_GWO_RESULTADOS
    )

    features = cargar_csv(
        RUTA_FEATURES
    )

    baseline = resultados[
        resultados["Modelo"]
        .astype(str)
        .str.contains(
            "Baseline",
            case=False,
            na=False,
        )
    ]

    gwo = resultados[
        resultados["Modelo"]
        .astype(str)
        .str.contains(
            "GWO",
            case=False,
            na=False,
        )
    ]

    if baseline.empty:
        raise ValueError(
            "No existe la fila Baseline "
            "en gwo_resultados.csv."
        )

    if gwo.empty:
        raise ValueError(
            "No existe la fila GWO "
            "en gwo_resultados.csv."
        )

    baseline = baseline.iloc[0]
    gwo = gwo.iloc[0]

    n_inicial = int(
        baseline[
            "n_features"
        ]
    )

    n_seleccionadas = int(
        gwo[
            "n_features"
        ]
    )

    reduccion = (
        1
        - n_seleccionadas
        / n_inicial
    ) * 100

    return {

        "features_iniciales":
            n_inicial,

        "features_seleccionadas":
            n_seleccionadas,

        "features_descartadas":
            n_inicial
            - n_seleccionadas,

        "reduccion_porcentual":
            redondear(
                reduccion,
                2,
            ),

        "baseline_f1_macro":
            redondear(
                baseline[
                    "F1_media"
                ]
            ),

        "gwo_f1_macro_exploratorio":
            redondear(
                gwo[
                    "F1_media"
                ]
            ),

        "delta_f1":
            redondear(
                gwo[
                    "F1_media"
                ]
                - baseline[
                    "F1_media"
                ]
            ),

        "total_features_csv":
            int(
                len(features)
            ),
    }


# =============================================================================
# 7. VALIDACIÓN CRUZADA
# =============================================================================

def resumir_validacion(
    ruta: str,
    folds: int,
) -> dict:

    verificar_archivo(
        ruta,
        f"resultados CV-{folds}",
    )

    df = cargar_csv(
        ruta
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
            f"CV-{folds} no contiene "
            f"las columnas: {faltantes}"
        )

    f1_media = float(
        df[
            "F1_macro"
        ].mean()
    )

    f1_std = float(
        df[
            "F1_macro"
        ].std()
    )

    ic95 = (
        1.96
        * f1_std
        / (folds ** 0.5)
    )

    return {

        "folds":
            folds,

        "f1_macro_media":
            redondear(
                f1_media
            ),

        "f1_macro_std":
            redondear(
                f1_std
            ),

        "f1_macro_ic95":
            redondear(
                ic95
            ),

        "accuracy_media":
            redondear(
                df[
                    "Accuracy"
                ].mean()
            ),

        "f1_por_clase": {

            "Civil":
                redondear(
                    df[
                        "F1_Civil"
                    ].mean()
                ),

            "Ejecución":
                redondear(
                    df[
                        "F1_Ejecucion"
                    ].mean()
                ),

            "Informática":
                redondear(
                    df[
                        "F1_Informatica"
                    ].mean()
                ),
        },
    }


# =============================================================================
# 8. ANÁLISIS EXPLORATORIOS
# =============================================================================

def obtener_pca_lda() -> dict:

    verificar_archivo(
        RUTA_PCA_RESUMEN,
        "el resumen PCA/LDA",
    )

    return cargar_json(
        RUTA_PCA_RESUMEN
    )


def obtener_homogeneidad() -> dict:

    verificar_archivo(
        RUTA_HOMOGENEIDAD_RESUMEN,
        "el resumen de homogeneidad",
    )

    return cargar_json(
        RUTA_HOMOGENEIDAD_RESUMEN
    )


def obtener_diferenciacion_lexica() -> dict:

    verificar_archivo(
        RUTA_LEXICO_RESUMEN,
        "el resumen de diferenciación léxica",
    )

    return cargar_json(
        RUTA_LEXICO_RESUMEN
    )


# =============================================================================
# 9. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 72
    )

    print(
        "GENERACIÓN DEL REPORTE CIENTÍFICO CONSOLIDADO"
    )

    print(
        "=" * 72
    )

    corpus = obtener_corpus()

    pca_lda = obtener_pca_lda()

    homogeneidad = obtener_homogeneidad()

    lexico = obtener_diferenciacion_lexica()

    gwo = obtener_gwo()

    cv5 = resumir_validacion(
        RUTA_CV5,
        5,
    )

    cv10 = resumir_validacion(
        RUTA_CV10,
        10,
    )


    reporte = {

        "version_reporte":
            "3.0",

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "corpus":
            corpus,

        "analisis_exploratorio": {

            "pca_lda":
                pca_lda,

            "homogeneidad_semantica":
                homogeneidad,

            "diferenciacion_lexica":
                lexico,
        },

        "modelo_predictivo": {

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

            "gwo":
                gwo,

            "validacion_5_fold":
                cv5,

            "validacion_10_fold":
                cv10,
        },

        "hallazgos_principales": {

            "estructura_semantica":
                (
                    "La similitud intra-grado fue "
                    "superior a la similitud inter-grado "
                    "y la diferencia resultó significativa "
                    "en el test de permutación."
                ),

            "proximidad_grados":
                (
                    "Los centroides de Civil e Informática "
                    "presentaron la mayor similitud entre "
                    "grados en el análisis semántico."
                ),

            "reduccion_gwo":
                (
                    "GWO redujo el espacio de 400 a 249 "
                    "características."
                ),

            "rendimiento":
                (
                    "El F1-macro exploratorio de 5 folds "
                    "fue superior al baseline, mientras "
                    "que la comprobación de 10 folds mostró "
                    "sensibilidad al particionado."
                ),
        },

        "advertencias_metodologicas": [

            (
                "La proyección LDA utiliza las etiquetas "
                "de clase y se interpreta únicamente como "
                "visualización exploratoria."
            ),

            (
                "Los términos distintivos representan "
                "diferencias léxicas del corpus y no "
                "competencias exclusivas de cada grado."
            ),

            (
                "La clase Ejecución contiene únicamente "
                "cinco perfiles."
            ),

            (
                "El resultado GWO de 5 folds es "
                "exploratorio debido a que la selección "
                "de características fue realizada antes "
                "de la comparación externa."
            ),

            (
                "Una evaluación estrictamente insesgada "
                "del modelo requiere validación cruzada "
                "anidada, realizando GWO dentro de cada "
                "fold externo."
            ),

            (
                "El corpus original puede contener "
                "términos asociados directamente a la "
                "denominación del programa o grado; "
                "este aspecto debe ser considerado en "
                "la auditoría de fuga léxica."
            ),
        ],

        "artefactos_oficiales": {

            "pca_lda":
                (
                    "visualizaciones_exploratorias/"
                    "proyeccion_pca_vs_lda_v2.png"
                ),

            "similitud_centroides":
                (
                    "homogeneidad_semantica/"
                    "similitud_centroides_v2.png"
                ),

            "test_permutacion":
                (
                    "homogeneidad_semantica/"
                    "test_permutacion_homogeneidad_v2.png"
                ),

            "terminos_distintivos":
                (
                    "diferenciacion_lexica/"
                    "terminos_distintivos_v2.png"
                ),

            "gwo_seleccion":
                "gwo/gwo_seleccion.png",

            "gwo_mapa_features":
                "gwo/gwo_mapa_features.png",

            "validacion_gwo":
                "gwo/cv10_gwo_resultados.png",

            "f1_por_clase":
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


    pca_total = (
        pca_lda[
            "pca"
        ][
            "varianza_explicada_total_2d"
        ]
        * 100
    )

    hom = homogeneidad[
        "similitud_global"
    ]

    test_perm = homogeneidad[
        "test_permutacion"
    ]


    print(
        f"\nCorpus: "
        f"{corpus['total_perfiles']} perfiles"
    )

    print(
        "Distribución: "
        + ", ".join(
            f"{clase}={cantidad}"
            for clase, cantidad
            in corpus[
                "distribucion"
            ].items()
        )
    )


    print(
        "\nANÁLISIS EXPLORATORIO"
    )

    print(
        f"  PCA — varianza explicada 2D: "
        f"{pca_total:.2f}%"
    )

    print(
        f"  Similitud intra-clase: "
        f"{hom['intra_clase_media']:.4f}"
    )

    print(
        f"  Similitud inter-clase: "
        f"{hom['inter_clase_media']:.4f}"
    )

    print(
        f"  Delta intra-inter: "
        f"{hom['diferencia_intra_inter']:.4f}"
    )

    print(
        f"  Test permutación p: "
        f"{test_perm['p_valor']:.6f}"
    )


    print(
        "\nMODELO PREDICTIVO"
    )

    print(
        f"  Features: "
        f"{gwo['features_iniciales']} -> "
        f"{gwo['features_seleccionadas']} "
        f"({gwo['reduccion_porcentual']}% menos)"
    )

    print(
        f"  Baseline F1-macro: "
        f"{gwo['baseline_f1_macro']:.4f}"
    )

    print(
        f"  GWO 5-fold F1-macro: "
        f"{cv5['f1_macro_media']:.4f}"
    )

    print(
        f"  GWO 10-fold F1-macro: "
        f"{cv10['f1_macro_media']:.4f}"
    )


    print(
        "\nReporte maestro generado:"
    )

    print(
        RUTA_SALIDA
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "REPORTE CONSOLIDADO GENERADO CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )