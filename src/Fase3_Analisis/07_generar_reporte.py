# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# 1. RUTAS
# =============================================================================

BASE = Path(__file__).resolve().parent
SRC_ROOT = BASE.parent

DATA_DIR = SRC_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTADOS_DIR = DATA_DIR / "resultados_cientificos"

CORPUS = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)

SALIDA_FINAL = (
    RESULTADOS_DIR
    / "resultados_finales.json"
)


# -----------------------------------------------------------------------------
# Análisis semántico
# -----------------------------------------------------------------------------

PCA_RESUMEN = (
    RESULTADOS_DIR
    / "visualizaciones_exploratorias"
    / "resumen_pca_lda_v2.json"
)

HOMOGENEIDAD_RESUMEN = (
    RESULTADOS_DIR
    / "homogeneidad_semantica"
    / "resumen_homogeneidad_v2.json"
)

LEXICO_RESUMEN = (
    RESULTADOS_DIR
    / "diferenciacion_lexica"
    / "resumen_diferenciacion_lexica_v2.json"
)


# -----------------------------------------------------------------------------
# GWO de referencia / profesor
# -----------------------------------------------------------------------------

GWO_DIR = (
    RESULTADOS_DIR
    / "gwo"
)

GWO_RESULTADOS = (
    GWO_DIR
    / "gwo_resultados.csv"
)

GWO_FEATURES = (
    GWO_DIR
    / "gwo_features_seleccionadas.csv"
)

GWO_CV5 = (
    GWO_DIR
    / "cv5_gwo_resultados.csv"
)

GWO_CV10 = (
    GWO_DIR
    / "cv10_gwo_resultados.csv"
)


# -----------------------------------------------------------------------------
# Auditoría metodológica
# -----------------------------------------------------------------------------

AUDITORIA_DIR = (
    RESULTADOS_DIR
    / "auditoria_fuga_lexica"
)

AUDITORIA_RESUMEN = (
    AUDITORIA_DIR
    / "resumen_auditoria_fuga_lexica.json"
)

AUDITORIA_PROXIES = (
    AUDITORIA_DIR
    / "auditoria_features_gwo_proxy.csv"
)


VALIDACION_ANIDADA_RESUMEN = (
    RESULTADOS_DIR
    / "validacion_robusta"
    / "resumen_validacion_anidada.json"
)


SELECCION_MODELO_RESUMEN = (
    RESULTADOS_DIR
    / "seleccion_modelo_robusta"
    / "resumen_seleccion_modelo_robusta.json"
)


# -----------------------------------------------------------------------------
# Modelo robusto final
# -----------------------------------------------------------------------------

MODELO_FINAL_DIR = (
    RESULTADOS_DIR
    / "modelo_final_robusto"
)

MODELO_FINAL_RESUMEN = (
    MODELO_FINAL_DIR
    / "resumen_modelo_final_robusto.json"
)

MODELO_FINAL_FOLDS = (
    MODELO_FINAL_DIR
    / "metricas_folds_modelo_final.csv"
)

MODELO_FINAL_CLASES = (
    MODELO_FINAL_DIR
    / "metricas_por_clase_modelo_final.csv"
)


# =============================================================================
# 2. UTILIDADES
# =============================================================================

def exigir_archivo(
    ruta: Path,
    descripcion: str,
) -> None:

    if not ruta.exists():

        raise FileNotFoundError(
            f"No se encontró {descripcion}:\n"
            f"{ruta}"
        )


def leer_json(
    ruta: Path,
) -> dict:

    exigir_archivo(
        ruta,
        ruta.name,
    )

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:

        return json.load(
            archivo
        )


def leer_json_opcional(
    ruta: Path,
):

    if not ruta.exists():

        return None

    with ruta.open(
        "r",
        encoding="utf-8",
    ) as archivo:

        return json.load(
            archivo
        )


def leer_csv(
    ruta: Path,
) -> pd.DataFrame:

    exigir_archivo(
        ruta,
        ruta.name,
    )

    return pd.read_csv(
        ruta,
        encoding="utf-8-sig",
    )


def leer_csv_opcional(
    ruta: Path,
):

    if not ruta.exists():

        return None

    return pd.read_csv(
        ruta,
        encoding="utf-8-sig",
    )


def redondear(
    valor,
    decimales: int = 6,
):

    if pd.isna(
        valor
    ):
        return None

    return round(
        float(valor),
        decimales,
    )


# =============================================================================
# 3. VALIDACIÓN DE ARCHIVOS ESENCIALES
# =============================================================================

def verificar_archivos_esenciales() -> None:

    archivos = {

        "corpus V2":
            CORPUS,

        "resumen PCA/LDA":
            PCA_RESUMEN,

        "resumen homogeneidad":
            HOMOGENEIDAD_RESUMEN,

        "resumen diferenciación léxica":
            LEXICO_RESUMEN,

        "resultados GWO":
            GWO_RESULTADOS,

        "features GWO":
            GWO_FEATURES,

        "validación GWO 5-fold":
            GWO_CV5,

        "validación GWO 10-fold":
            GWO_CV10,

        "modelo robusto final":
            MODELO_FINAL_RESUMEN,

        "folds modelo robusto":
            MODELO_FINAL_FOLDS,

        "métricas por clase modelo robusto":
            MODELO_FINAL_CLASES,
    }

    for descripcion, ruta in (
        archivos.items()
    ):

        exigir_archivo(
            ruta,
            descripcion,
        )


# =============================================================================
# 4. CORPUS
# =============================================================================

def construir_bloque_corpus() -> dict:

    df = leer_csv(
        CORPUS
    )

    requeridas = {
        "perfil_egreso",
        "grado",
    }

    faltantes = (
        requeridas
        - set(
            df.columns
        )
    )

    if faltantes:

        raise ValueError(
            "El corpus V2 no contiene "
            f"las columnas: {sorted(faltantes)}"
        )

    distribucion = (
        df[
            "grado"
        ]
        .value_counts()
        .to_dict()
    )

    return {

        "version":
            "V2",

        "archivo":
            CORPUS.name,

        "total_perfiles":
            int(
                len(df)
            ),

        "distribucion": {
            str(clase):
                int(cantidad)

            for clase, cantidad
            in distribucion.items()
        },

        "uso":
            (
                "Corpus científico congelado "
                "y versionado utilizado para "
                "los resultados oficiales."
            ),
    }


# =============================================================================
# 5. ANÁLISIS SEMÁNTICO
# =============================================================================

def construir_bloque_semantico() -> dict:

    return {

        "pca_lda":
            leer_json(
                PCA_RESUMEN
            ),

        "homogeneidad_semantica":
            leer_json(
                HOMOGENEIDAD_RESUMEN
            ),

        "diferenciacion_lexica":
            leer_json(
                LEXICO_RESUMEN
            ),
    }


# =============================================================================
# 6. GWO EXPLORATORIO
# =============================================================================

def construir_bloque_gwo() -> dict:

    df_gwo = leer_csv(
        GWO_RESULTADOS
    )

    df_features = leer_csv(
        GWO_FEATURES
    )

    df_cv5 = leer_csv(
        GWO_CV5
    )

    df_cv10 = leer_csv(
        GWO_CV10
    )


    columnas_gwo = {
        "Modelo",
        "n_features",
        "F1_media",
        "F1_std",
    }

    if not columnas_gwo.issubset(
        df_gwo.columns
    ):

        raise ValueError(
            "gwo_resultados.csv no contiene "
            "las columnas esperadas."
        )


    if "F1_macro" not in (
        df_cv5.columns
    ):

        raise ValueError(
            "cv5_gwo_resultados.csv no contiene "
            "la columna F1_macro."
        )


    if "F1_macro" not in (
        df_cv10.columns
    ):

        raise ValueError(
            "cv10_gwo_resultados.csv no contiene "
            "la columna F1_macro."
        )


    baseline = df_gwo[
        df_gwo[
            "Modelo"
        ]
        .astype(str)
        .str.contains(
            "Baseline",
            case=False,
            na=False,
        )
    ]


    gwo = df_gwo[
        df_gwo[
            "Modelo"
        ]
        .astype(str)
        .str.contains(
            "GWO",
            case=False,
            na=False,
        )
    ]


    if baseline.empty:

        baseline = df_gwo.iloc[
            [
                0
            ]
        ]


    if gwo.empty:

        gwo = df_gwo.iloc[
            [
                len(df_gwo) - 1
            ]
        ]


    baseline_row = (
        baseline.iloc[
            0
        ]
    )

    gwo_row = (
        gwo.iloc[
            0
        ]
    )


    total_features = int(
        baseline_row[
            "n_features"
        ]
    )


    seleccionadas = int(
        len(
            df_features
        )
    )


    reduccion = (
        100.0
        * (
            1.0
            - (
                seleccionadas
                / total_features
            )
        )
    )


    return {

        "estado":
            "exploratorio",

        "metodo":
            (
                "Grey Wolf Optimizer "
                "sobre representación TF-IDF."
            ),

        "features_originales":
            total_features,

        "features_seleccionadas":
            seleccionadas,

        "reduccion_porcentual":
            redondear(
                reduccion,
                2,
            ),

        "baseline_exploratorio": {

            "f1_macro":
                redondear(
                    baseline_row[
                        "F1_media"
                    ]
                ),

            "f1_std":
                redondear(
                    baseline_row[
                        "F1_std"
                    ]
                ),
        },

        "gwo_5fold": {

            "f1_macro":
                redondear(
                    df_cv5[
                        "F1_macro"
                    ].mean()
                ),

            "f1_std":
                redondear(
                    df_cv5[
                        "F1_macro"
                    ].std()
                ),

            "numero_folds":
                int(
                    len(
                        df_cv5
                    )
                ),
        },

        "gwo_10fold": {

            "f1_macro":
                redondear(
                    df_cv10[
                        "F1_macro"
                    ].mean()
                ),

            "f1_std":
                redondear(
                    df_cv10[
                        "F1_macro"
                    ].std()
                ),

            "numero_folds":
                int(
                    len(
                        df_cv10
                    )
                ),
        },

        "advertencia_metodologica":
            (
                "La selección GWO se realizó antes "
                "de la comparación externa sobre el "
                "corpus disponible. Por esta razón, "
                "el resultado de 5-fold se interpreta "
                "como exploratorio y no como una "
                "estimación definitiva de generalización."
            ),

        "interpretacion":
            (
                "GWO identifica un subconjunto "
                "altamente discriminativo y reduce "
                "el espacio de representación, pero "
                "su rendimiento debe analizarse junto "
                "con las auditorías de fuga léxica y "
                "el modelo robusto final."
            ),
    }


# =============================================================================
# 7. AUDITORÍA METODOLÓGICA
# =============================================================================

def construir_bloque_auditoria() -> dict:

    resumen_fuga = leer_json_opcional(
        AUDITORIA_RESUMEN
    )

    df_proxies = leer_csv_opcional(
        AUDITORIA_PROXIES
    )

    validacion_anidada = leer_json_opcional(
        VALIDACION_ANIDADA_RESUMEN
    )

    seleccion_modelo = leer_json_opcional(
        SELECCION_MODELO_RESUMEN
    )


    proxies_directos = None

    if df_proxies is not None:

        proxies_directos = int(
            len(
                df_proxies
            )
        )


    porcentaje_proxy = None

    if (
        proxies_directos
        is not None
        and GWO_FEATURES.exists()
    ):

        total_gwo = len(
            leer_csv(
                GWO_FEATURES
            )
        )

        if total_gwo > 0:

            porcentaje_proxy = (
                100.0
                * proxies_directos
                / total_gwo
            )


    return {

        "estado":
            "auditoria_posterior",

        "proxies_directos_gwo": {

            "cantidad":
                proxies_directos,

            "porcentaje":
                (
                    redondear(
                        porcentaje_proxy,
                        2,
                    )
                    if porcentaje_proxy
                    is not None
                    else None
                ),
        },

        "auditoria_fuga_lexica":
            resumen_fuga,

        "validacion_gwo_anidada":
            validacion_anidada,

        "seleccion_modelos_robusta":
            seleccion_modelo,

        "interpretacion":
            (
                "Las auditorías se conservaron "
                "como trazabilidad metodológica. "
                "Su función es cuantificar cuánto "
                "del rendimiento aparente puede "
                "estar asociado a pistas léxicas "
                "directas y evaluar si las mejoras "
                "se mantienen bajo protocolos más "
                "estrictos."
            ),
    }


# =============================================================================
# 8. MODELO ROBUSTO FINAL
# =============================================================================

def construir_bloque_modelo_final() -> dict:

    resumen = leer_json(
        MODELO_FINAL_RESUMEN
    )

    df_folds = leer_csv(
        MODELO_FINAL_FOLDS
    )

    df_clases = leer_csv(
        MODELO_FINAL_CLASES
    )


    if "F1_macro" not in (
        df_folds.columns
    ):

        raise ValueError(
            "metricas_folds_modelo_final.csv "
            "no contiene F1_macro."
        )


    if "Accuracy" not in (
        df_folds.columns
    ):

        raise ValueError(
            "metricas_folds_modelo_final.csv "
            "no contiene Accuracy."
        )


    resultados = resumen.get(
        "resultados_principales",
        {}
    )


    f1_media = resultados.get(
        "f1_macro_media_folds"
    )

    if f1_media is None:

        f1_media = float(
            df_folds[
                "F1_macro"
            ].mean()
        )


    f1_std = resultados.get(
        "f1_macro_std_folds"
    )

    if f1_std is None:

        f1_std = float(
            df_folds[
                "F1_macro"
            ].std()
        )


    accuracy_media = resultados.get(
        "accuracy_media_folds"
    )

    if accuracy_media is None:

        accuracy_media = float(
            df_folds[
                "Accuracy"
            ].mean()
        )


    return {

        "estado":
            "resultado_robusto_principal",

        "modelo":
            "Complement Naive Bayes",

        "representacion":
            "TF-IDF word 1-2 grams",

        "balanceo":
            (
                "SMOTE únicamente sobre "
                "el entrenamiento."
            ),

        "control_fuga":
            (
                "Enmascaramiento de "
                "denominaciones explícitas "
                "del grado antes de vectorizar."
            ),

        "validacion":
            (
                "StratifiedKFold 5-fold "
                "con TF-IDF ajustado "
                "exclusivamente en train."
            ),

        "f1_macro_media":
            redondear(
                f1_media
            ),

        "f1_macro_std":
            redondear(
                f1_std
            ),

        "accuracy_media":
            redondear(
                accuracy_media
            ),

        "f1_macro_oof":
            resultados.get(
                "f1_macro_oof"
            ),

        "accuracy_oof":
            resultados.get(
                "accuracy_oof"
            ),

        "metricas_por_clase":
            resumen.get(
                "metricas_oof_por_clase",
                {}
            ),

        "resultado_completo":
            resumen,

        "archivo_metricas_clase":
            MODELO_FINAL_CLASES.name,

        "numero_folds":
            int(
                len(
                    df_folds
                )
            ),

        "interpretacion":
            (
                "Este es el resultado conservador "
                "principal utilizado para estimar "
                "capacidad de generalización después "
                "de controlar denominaciones "
                "explícitas asociadas a la etiqueta."
            ),
    }


# =============================================================================
# 9. CONCLUSIÓN INTEGRADA
# =============================================================================

def construir_conclusion_integrada(
    gwo: dict,
    modelo_final: dict,
    auditoria: dict,
) -> dict:

    return {

        "hallazgo_central":
            (
                "Los perfiles de egreso presentan "
                "estructura semántica asociada al tipo "
                "de grado, pero también un solapamiento "
                "considerable entre las clases."
            ),

        "gwo":
            (
                "La selección GWO muestra un elevado "
                "potencial discriminativo y una "
                "reducción importante del espacio de "
                "características; su F1-macro de "
                "5-fold debe interpretarse como "
                "exploratorio."
            ),

        "auditoria":
            (
                "La auditoría posterior detectó "
                "características directamente asociadas "
                "a la denominación de los grados, "
                "confirmando que parte de la capacidad "
                "predictiva puede proceder de pistas "
                "léxicas directas."
            ),

        "resultado_robusto":
            (
                "Tras controlar las denominaciones "
                "explícitas y separar correctamente "
                "entrenamiento y evaluación, la "
                "capacidad predictiva se mantiene, "
                "aunque en un nivel moderado."
            ),

        "limitacion_principal":
            (
                "El corpus contiene únicamente cinco "
                "perfiles de la clase Ejecución, lo "
                "que genera alta sensibilidad al "
                "particionado y limita la precisión "
                "de las estimaciones."
            ),

        "metricas_clave": {

            "gwo_5fold_exploratorio":
                gwo[
                    "gwo_5fold"
                ][
                    "f1_macro"
                ],

            "gwo_10fold_comprobacion":
                gwo[
                    "gwo_10fold"
                ][
                    "f1_macro"
                ],

            "proxies_directos_gwo":
                auditoria[
                    "proxies_directos_gwo"
                ][
                    "cantidad"
                ],

            "modelo_robusto_f1_macro":
                modelo_final[
                    "f1_macro_media"
                ],

            "modelo_robusto_f1_std":
                modelo_final[
                    "f1_macro_std"
                ],
        },
    }


# =============================================================================
# 10. ARTEFACTOS
# =============================================================================

def construir_bloque_artefactos() -> dict:

    return {

        "analisis_semantico": {

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

            "diferenciacion_lexica":
                (
                    "diferenciacion_lexica/"
                    "terminos_distintivos_v2.png"
                ),
        },

        "gwo": {

            "seleccion":
                "gwo/gwo_seleccion.png",

            "mapa_features":
                "gwo/gwo_mapa_features.png",

            "validacion_10fold":
                "gwo/cv10_gwo_resultados.png",

            "f1_por_clase":
                "gwo/cv10_gwo_f1_clase.png",
        },

        "auditoria": {

            "comparacion_fuga_lexica":
                (
                    "auditoria_fuga_lexica/"
                    "comparacion_f1_fuga_lexica.png"
                ),

            "validacion_gwo_anidada":
                (
                    "validacion_robusta/"
                    "comparacion_validacion_robusta.png"
                ),
        },

        "modelo_final_robusto": {

            "f1_por_fold":
                (
                    "modelo_final_robusto/"
                    "f1_por_fold_modelo_final.png"
                ),

            "matriz_confusion":
                (
                    "modelo_final_robusto/"
                    "matriz_confusion_modelo_final.png"
                ),
        },
    }


# =============================================================================
# 11. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    print(
        "=" * 76
    )

    print(
        "GENERACIÓN DEL REPORTE CIENTÍFICO FINAL"
    )

    print(
        "=" * 76
    )


    verificar_archivos_esenciales()


    corpus = construir_bloque_corpus()

    analisis_semantico = (
        construir_bloque_semantico()
    )

    gwo = construir_bloque_gwo()

    auditoria = (
        construir_bloque_auditoria()
    )

    modelo_final = (
        construir_bloque_modelo_final()
    )

    conclusion = (
        construir_conclusion_integrada(
            gwo,
            modelo_final,
            auditoria,
        )
    )


    reporte = {

        "metadata": {

            "version":
                "4.0",

            "fecha_generacion_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "estado":
                "resultados_cientificos_consolidados",

            "proyecto":
                (
                    "Análisis de perfiles de egreso "
                    "de carreras de informática en Chile"
                ),
        },

        "corpus":
            corpus,

        "analisis_semantico":
            analisis_semantico,

        "gwo_exploratorio":
            gwo,

        "auditoria_metodologica":
            auditoria,

        "resultado_robusto_final":
            modelo_final,

        "conclusion_integrada":
            conclusion,

        "advertencias_metodologicas": [

            (
                "El F1-macro GWO de 5-fold "
                "es un resultado exploratorio y "
                "no debe describirse como una "
                "estimación definitiva de "
                "generalización."
            ),

            (
                "La auditoría detectó proxies "
                "léxicos asociados directamente "
                "a la denominación del grado."
            ),

            (
                "El resultado robusto final debe "
                "reportarse separadamente del "
                "resultado exploratorio GWO."
            ),

            (
                "La clase Ejecución contiene "
                "solamente cinco observaciones."
            ),

            (
                "No debe utilizarse el término "
                "'precisión' como sinónimo de "
                "F1-macro."
            ),
        ],

        "artefactos_oficiales":
            construir_bloque_artefactos(),
    }


    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with SALIDA_FINAL.open(
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            reporte,
            archivo,
            indent=4,
            ensure_ascii=False,
        )


    print()
    print(
        f"Corpus: "
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


    print()
    print(
        "GWO exploratorio:"
    )

    print(
        f"  Features: "
        f"{gwo['features_originales']} "
        f"-> "
        f"{gwo['features_seleccionadas']}"
    )

    print(
        f"  Reducción: "
        f"{gwo['reduccion_porcentual']:.2f}%"
    )

    print(
        f"  F1-macro 5-fold : "
        f"{gwo['gwo_5fold']['f1_macro']:.4f}"
    )

    print(
        f"  F1-macro 10-fold: "
        f"{gwo['gwo_10fold']['f1_macro']:.4f}"
    )


    print()
    print(
        "Auditoría metodológica:"
    )

    proxies = auditoria[
        "proxies_directos_gwo"
    ][
        "cantidad"
    ]

    porcentaje = auditoria[
        "proxies_directos_gwo"
    ][
        "porcentaje"
    ]


    if proxies is not None:

        print(
            f"  Proxies directos GWO: "
            f"{proxies} "
            f"({porcentaje:.2f}%)"
        )

    else:

        print(
            "  Auditoría de proxies: "
            "no disponible."
        )


    print()
    print(
        "Resultado robusto final:"
    )

    print(
        f"  Modelo        : "
        f"{modelo_final['modelo']}"
    )

    print(
        f"  F1-macro medio: "
        f"{modelo_final['f1_macro_media']:.4f}"
    )

    print(
        f"  F1 std        : "
        f"{modelo_final['f1_macro_std']:.4f}"
    )

    print(
        f"  Accuracy media: "
        f"{modelo_final['accuracy_media']:.4f}"
    )


    print()
    print(
        "Reporte generado:"
    )

    print(
        SALIDA_FINAL
    )


    print()
    print(
        "=" * 76
    )

    print(
        "REPORTE CONSOLIDADO GENERADO CORRECTAMENTE"
    )

    print(
        "=" * 76
    )


    return 0


if __name__ == "__main__":

    raise SystemExit(
        main()
    )