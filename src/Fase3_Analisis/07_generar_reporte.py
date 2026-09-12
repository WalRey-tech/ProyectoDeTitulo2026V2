# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =============================================================================
# 1. RUTAS GENERALES
# =============================================================================

BASE = Path(__file__).resolve().parent
SRC_ROOT = BASE.parent

DATA_DIR = (
    SRC_ROOT
    / "data"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

RESULTADOS_DIR = (
    DATA_DIR
    / "resultados_cientificos"
)

CORPUS = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)

SALIDA_FINAL = (
    RESULTADOS_DIR
    / "resultados_finales.json"
)


# =============================================================================
# 2. ANÁLISIS SEMÁNTICO
# =============================================================================

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


# =============================================================================
# 3. GWO — RESULTADOS DE REFERENCIA
# =============================================================================

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


# =============================================================================
# 4. AUDITORÍA METODOLÓGICA
# =============================================================================

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


# =============================================================================
# 5. MODELO ROBUSTO FINAL
# =============================================================================

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
# 6. UTILIDADES
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

    if valor is None:

        return None

    if pd.isna(
        valor
    ):

        return None

    return round(
        float(valor),
        decimales,
    )


def contar_booleanos_verdaderos(
    serie: pd.Series,
) -> int:

    """
    Permite leer correctamente columnas booleanas
    tanto si pandas las interpreta como bool como
    si vienen almacenadas como texto.
    """

    if pd.api.types.is_bool_dtype(
        serie
    ):

        return int(
            serie.sum()
        )

    normalizada = (
        serie
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return int(
        normalizada.isin(
            [
                "true",
                "1",
                "sí",
                "si",
                "yes",
            ]
        ).sum()
    )


# =============================================================================
# 7. VALIDACIÓN DE ARCHIVOS ESENCIALES
# =============================================================================

def verificar_archivos_esenciales() -> None:

    archivos = {

        "corpus científico V2":
            CORPUS,

        "resumen PCA/LDA":
            PCA_RESUMEN,

        "resumen de homogeneidad":
            HOMOGENEIDAD_RESUMEN,

        "resumen de diferenciación léxica":
            LEXICO_RESUMEN,

        "resultados GWO":
            GWO_RESULTADOS,

        "features seleccionadas por GWO":
            GWO_FEATURES,

        "validación GWO 5-fold":
            GWO_CV5,

        "diagnóstico GWO 10-fold":
            GWO_CV10,

        "resumen del modelo robusto final":
            MODELO_FINAL_RESUMEN,

        "métricas por fold del modelo robusto":
            MODELO_FINAL_FOLDS,

        "métricas por clase del modelo robusto":
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
# 8. CORPUS
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
            f"las columnas requeridas: "
            f"{sorted(faltantes)}"
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
                "la obtención de los resultados "
                "oficiales del estudio."
            ),
    }


# =============================================================================
# 9. ANÁLISIS SEMÁNTICO
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
# 10. GWO EXPLORATORIO Y DIAGNÓSTICO 10-FOLD
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


    # -------------------------------------------------------------------------
    # Validación de estructura GWO
    # -------------------------------------------------------------------------

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


    columnas_cv_actualizadas = {
        "F1_macro",
        "F1_macro_3clases",
        "contiene_3_clases",
    }


    if not columnas_cv_actualizadas.issubset(
        df_cv5.columns
    ):

        raise ValueError(
            "cv5_gwo_resultados.csv no corresponde "
            "a la versión metodológica actualizada "
            "de 05_validacion_gwo.py."
        )


    if not columnas_cv_actualizadas.issubset(
        df_cv10.columns
    ):

        raise ValueError(
            "cv10_gwo_resultados.csv no corresponde "
            "a la versión metodológica actualizada "
            "de 05_validacion_gwo.py."
        )


    # -------------------------------------------------------------------------
    # Identificar baseline y GWO
    # -------------------------------------------------------------------------

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
            [0]
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


    # -------------------------------------------------------------------------
    # Reducción de características
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Folds que realmente contienen las tres clases
    # -------------------------------------------------------------------------

    folds_5_completos = (
        contar_booleanos_verdaderos(
            df_cv5[
                "contiene_3_clases"
            ]
        )
    )


    folds_10_completos = (
        contar_booleanos_verdaderos(
            df_cv10[
                "contiene_3_clases"
            ]
        )
    )


    folds_10_incompletos = int(
        len(
            df_cv10
        )
        - folds_10_completos
    )


    # -------------------------------------------------------------------------
    # Métricas
    # -------------------------------------------------------------------------

    f1_5 = float(
        df_cv5[
            "F1_macro_3clases"
        ].mean()
    )

    std_5 = float(
        df_cv5[
            "F1_macro_3clases"
        ].std()
    )


    f1_10_historico = float(
        df_cv10[
            "F1_macro"
        ].mean()
    )

    std_10_historico = float(
        df_cv10[
            "F1_macro"
        ].std()
    )


    f1_10_3clases = float(
        df_cv10[
            "F1_macro_3clases"
        ].mean()
    )

    std_10_3clases = float(
        df_cv10[
            "F1_macro_3clases"
        ].std()
    )


    # -------------------------------------------------------------------------
    # Resultado
    # -------------------------------------------------------------------------

    return {

        "estado":
            "exploratorio",

        "metodo":
            (
                "Grey Wolf Optimizer (GWO) "
                "aplicado sobre representación "
                "TF-IDF de hasta 400 características."
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

        "resultado_archivo_gwo_referencia": {

            "f1_macro_baseline":
                redondear(
                    baseline_row[
                        "F1_media"
                    ]
                ),

            "f1_std_baseline":
                redondear(
                    baseline_row[
                        "F1_std"
                    ]
                ),

            "f1_macro_gwo":
                redondear(
                    gwo_row[
                        "F1_media"
                    ]
                ),

            "f1_std_gwo":
                redondear(
                    gwo_row[
                        "F1_std"
                    ]
                ),
        },

        "gwo_5fold_exploratorio": {

            "f1_macro":
                redondear(
                    f1_5
                ),

            "f1_std":
                redondear(
                    std_5
                ),

            "numero_folds":
                int(
                    len(
                        df_cv5
                    )
                ),

            "folds_con_3_clases":
                folds_5_completos,

            "folds_sin_alguna_clase":
                int(
                    len(
                        df_cv5
                    )
                    - folds_5_completos
                ),

            "todas_las_clases_en_cada_fold":
                bool(
                    folds_5_completos
                    == len(
                        df_cv5
                    )
                ),

            "interpretacion":
                (
                    "Reproduce el resultado "
                    "exploratorio GWO de referencia. "
                    "Los cinco folds contienen las "
                    "tres clases. No obstante, "
                    "la selección de características "
                    "GWO y la construcción del "
                    "vocabulario TF-IDF se realizaron "
                    "sobre el corpus disponible antes "
                    "de esta comparación, por lo que "
                    "el resultado no constituye una "
                    "estimación final insesgada de "
                    "generalización."
                ),
        },

        "gwo_10fold_diagnostico": {

            "f1_referencia_historico":
                redondear(
                    f1_10_historico
                ),

            "f1_referencia_std":
                redondear(
                    std_10_historico
                ),

            "f1_macro_3clases_fijas":
                redondear(
                    f1_10_3clases
                ),

            "f1_macro_3clases_std":
                redondear(
                    std_10_3clases
                ),

            "numero_folds":
                int(
                    len(
                        df_cv10
                    )
                ),

            "folds_con_3_clases":
                folds_10_completos,

            "folds_sin_alguna_clase":
                folds_10_incompletos,

            "valido_como_resultado_final":
                False,

            "motivo_limitacion":
                (
                    "La clase Ejecución contiene "
                    "solo cinco observaciones. "
                    "Con StratifiedKFold de diez "
                    "pliegues no es posible incluir "
                    "las tres clases en todos los "
                    "conjuntos de prueba."
                ),

            "interpretacion_f1_historico":
                (
                    "El F1 histórico reproduce el "
                    "comportamiento del protocolo "
                    "original, donde el promedio macro "
                    "de sklearn considera únicamente "
                    "las clases presentes o predichas "
                    "en cada fold."
                ),

            "interpretacion_f1_3clases":
                (
                    "El cálculo con tres clases fijas "
                    "incluye explícitamente Civil, "
                    "Ejecución e Informática en el "
                    "promedio de cada fold. Se utiliza "
                    "como diagnóstico para mostrar la "
                    "sensibilidad del resultado 10-fold "
                    "ante la ausencia de Ejecución en "
                    "cinco de los diez folds."
                ),

            "uso":
                (
                    "Se conserva para reproducibilidad "
                    "y análisis de sensibilidad. "
                    "No se emplea como estimación "
                    "robusta final de generalización."
                ),
        },

        "advertencia_metodologica_general":
            (
                "La selección GWO y el espacio TF-IDF "
                "se construyeron utilizando el corpus "
                "disponible completo antes de la "
                "comparación mediante validación "
                "cruzada. Por esta razón los "
                "resultados GWO se clasifican como "
                "exploratorios."
            ),

        "resultado_robusto_relacionado":
            (
                "La estimación conservadora principal "
                "de generalización se obtiene mediante "
                "06_modelo_final_robusto.py, donde "
                "el TF-IDF se ajusta exclusivamente "
                "sobre entrenamiento y se controlan "
                "las denominaciones explícitas del "
                "grado."
            ),
    }


# =============================================================================
# 11. AUDITORÍA METODOLÓGICA
# =============================================================================

def construir_bloque_auditoria() -> dict:

    resumen_fuga = (
        leer_json_opcional(
            AUDITORIA_RESUMEN
        )
    )


    df_proxies = (
        leer_csv_opcional(
            AUDITORIA_PROXIES
        )
    )


    validacion_anidada = (
        leer_json_opcional(
            VALIDACION_ANIDADA_RESUMEN
        )
    )


    seleccion_modelo = (
        leer_json_opcional(
            SELECCION_MODELO_RESUMEN
        )
    )


    proxies_directos = None
    porcentaje_proxy = None


    if df_proxies is not None:

        proxies_directos = int(
            len(
                df_proxies
            )
        )


    if (
        proxies_directos
        is not None
        and GWO_FEATURES.exists()
    ):

        total_gwo = int(
            len(
                leer_csv(
                    GWO_FEATURES
                )
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
            "auditoria_metodologica_posterior",

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
                "Las auditorías se conservan como "
                "trazabilidad metodológica del estudio. "
                "Permiten cuantificar cuánto del "
                "rendimiento exploratorio puede estar "
                "asociado a pistas léxicas directas "
                "del grado y comprobar si las mejoras "
                "se sostienen bajo protocolos de "
                "evaluación más estrictos."
            ),
    }


# =============================================================================
# 12. MODELO ROBUSTO FINAL
# =============================================================================

def construir_bloque_modelo_final() -> dict:

    resumen = leer_json(
        MODELO_FINAL_RESUMEN
    )

    df_folds = leer_csv(
        MODELO_FINAL_FOLDS
    )

    leer_csv(
        MODELO_FINAL_CLASES
    )


    columnas_requeridas = {
        "F1_macro",
        "Accuracy",
    }


    if not columnas_requeridas.issubset(
        df_folds.columns
    ):

        raise ValueError(
            "metricas_folds_modelo_final.csv "
            "no contiene las columnas requeridas."
        )


    resultados = resumen.get(
        "resultados_principales",
        {},
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
            (
                "TF-IDF de palabras y bigramas, "
                "máximo 400 características."
            ),

        "balanceo":
            (
                "SMOTE aplicado exclusivamente "
                "sobre el conjunto de entrenamiento "
                "de cada fold."
            ),

        "control_fuga":
            (
                "Enmascaramiento de las "
                "denominaciones explícitas del "
                "grado antes de la vectorización."
            ),

        "validacion":
            (
                "StratifiedKFold de cinco pliegues "
                "con TF-IDF ajustado exclusivamente "
                "sobre los documentos de entrenamiento."
            ),

        "metrica_principal":
            "F1-macro medio entre folds",

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

        "f1_macro_oof_secundario":
            resultados.get(
                "f1_macro_oof"
            ),

        "accuracy_oof_secundaria":
            resultados.get(
                "accuracy_oof"
            ),

        "metricas_oof_por_clase":
            resumen.get(
                "metricas_oof_por_clase",
                {},
            ),

        "numero_folds":
            int(
                len(
                    df_folds
                )
            ),

        "resultado_completo":
            resumen,

        "interpretacion":
            (
                "Este resultado constituye la "
                "estimación conservadora principal "
                "de capacidad predictiva del estudio "
                "después de controlar las "
                "denominaciones explícitas asociadas "
                "directamente con la etiqueta."
            ),

        "limitacion_principal":
            (
                "La clase Ejecución contiene "
                "únicamente cinco perfiles, por lo "
                "que las métricas presentan una "
                "sensibilidad elevada al particionado."
            ),
    }


# =============================================================================
# 13. CONCLUSIÓN INTEGRADA
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
                "estructura semántica asociada al "
                "tipo de grado, aunque existe un "
                "solapamiento considerable entre "
                "las clases analizadas."
            ),

        "gwo":
            (
                "La selección mediante GWO redujo "
                "el espacio de 400 a 249 "
                "características y reprodujo un "
                "F1-macro exploratorio de 0.8412 "
                "en validación de cinco pliegues. "
                "Este resultado demuestra potencial "
                "discriminativo, pero no se interpreta "
                "como estimación final insesgada de "
                "generalización."
            ),

        "diagnostico_10fold":
            (
                "La comprobación histórica de diez "
                "pliegues reproduce un F1-macro de "
                "0.7316. Sin embargo, debido a que "
                "Ejecución posee únicamente cinco "
                "observaciones, solo cinco de los "
                "diez folds contienen las tres clases. "
                "Al fijar explícitamente las tres "
                "clases en el cálculo por fold, "
                "el promedio diagnóstico disminuye "
                "a 0.6337. Por ello el análisis "
                "10-fold se conserva únicamente "
                "como evidencia de sensibilidad."
            ),

        "auditoria":
            (
                "La auditoría léxica identificó "
                "características seleccionadas por "
                "GWO directamente relacionadas con "
                "la denominación de los grados. "
                "Esto demuestra que parte de la "
                "capacidad discriminativa exploratoria "
                "depende de pistas léxicas directas."
            ),

        "resultado_robusto":
            (
                "Al controlar las denominaciones "
                "explícitas del grado y ajustar la "
                "representación exclusivamente sobre "
                "los datos de entrenamiento, el modelo "
                "Complement Naive Bayes obtuvo un "
                "F1-macro medio de 0.5742. Este valor "
                "se adopta como estimación conservadora "
                "principal de generalización."
            ),

        "limitacion_principal":
            (
                "La principal limitación estadística "
                "es el reducido número de perfiles "
                "de Ingeniería de Ejecución, con "
                "cinco observaciones, lo que genera "
                "alta variabilidad entre folds."
            ),

        "metricas_clave": {

            "gwo_5fold_exploratorio":
                gwo[
                    "gwo_5fold_exploratorio"
                ][
                    "f1_macro"
                ],

            "gwo_5fold_std":
                gwo[
                    "gwo_5fold_exploratorio"
                ][
                    "f1_std"
                ],

            "gwo_10fold_historico":
                gwo[
                    "gwo_10fold_diagnostico"
                ][
                    "f1_referencia_historico"
                ],

            "gwo_10fold_3clases_diagnostico":
                gwo[
                    "gwo_10fold_diagnostico"
                ][
                    "f1_macro_3clases_fijas"
                ],

            "gwo_10fold_folds_con_3_clases":
                gwo[
                    "gwo_10fold_diagnostico"
                ][
                    "folds_con_3_clases"
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

            "modelo_robusto_accuracy":
                modelo_final[
                    "accuracy_media"
                ],
        },
    }


# =============================================================================
# 14. ARTEFACTOS OFICIALES
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

        "gwo_exploratorio": {

            "seleccion":
                (
                    "gwo/"
                    "gwo_seleccion.png"
                ),

            "mapa_features":
                (
                    "gwo/"
                    "gwo_mapa_features.png"
                ),

            "diagnostico_validacion":
                (
                    "gwo/"
                    "cv10_gwo_resultados.png"
                ),

            "f1_por_clase":
                (
                    "gwo/"
                    "cv10_gwo_f1_clase.png"
                ),

            "resultados_5fold":
                (
                    "gwo/"
                    "cv5_gwo_resultados.csv"
                ),

            "resultados_10fold":
                (
                    "gwo/"
                    "cv10_gwo_resultados.csv"
                ),
        },

        "auditoria_metodologica": {

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

            "seleccion_modelo_robusta":
                (
                    "seleccion_modelo_robusta/"
                    "comparacion_modelos_robustos.png"
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

            "metricas_folds":
                (
                    "modelo_final_robusto/"
                    "metricas_folds_modelo_final.csv"
                ),

            "metricas_por_clase":
                (
                    "modelo_final_robusto/"
                    "metricas_por_clase_modelo_final.csv"
                ),

            "predicciones_oof":
                (
                    "modelo_final_robusto/"
                    "predicciones_oof_modelo_final.csv"
                ),
        },
    }


# =============================================================================
# 15. FLUJO PRINCIPAL
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


    corpus = (
        construir_bloque_corpus()
    )


    analisis_semantico = (
        construir_bloque_semantico()
    )


    gwo = (
        construir_bloque_gwo()
    )


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
                "5.0",

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
                "El F1-macro GWO 5-fold de 0.8412 "
                "es reproducible dentro del protocolo "
                "de referencia, pero se considera "
                "exploratorio porque la selección "
                "GWO y la construcción del espacio "
                "TF-IDF preceden a la validación."
            ),

            (
                "El diagnóstico GWO de 10-fold se "
                "conserva para reproducibilidad. "
                "Debido a que Ejecución contiene "
                "solo cinco perfiles, cinco de los "
                "diez folds no contienen las tres "
                "clases."
            ),

            (
                "El valor histórico 10-fold de "
                "0.7316 y el cálculo diagnóstico "
                "con tres clases fijas de 0.6337 "
                "no se utilizan como estimación "
                "final robusta de generalización."
            ),

            (
                "La auditoría metodológica detectó "
                "proxies léxicos asociados "
                "directamente a la denominación "
                "de los grados."
            ),

            (
                "El resultado robusto final debe "
                "reportarse de manera separada de "
                "los resultados exploratorios GWO."
            ),

            (
                "La métrica F1-macro no debe "
                "denominarse 'precisión'."
            ),

            (
                "No se reportan intervalos normales "
                "de confianza calculados directamente "
                "sobre los folds como evidencia "
                "principal, debido al número reducido "
                "de particiones y a la naturaleza "
                "acotada de la métrica."
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


    # =========================================================================
    # SALIDA DE CONSOLA
    # =========================================================================

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
        f"  F1 5-fold exploratorio      : "
        f"{gwo['gwo_5fold_exploratorio']['f1_macro']:.4f}"
    )


    print(
        f"  Std 5-fold                  : "
        f"{gwo['gwo_5fold_exploratorio']['f1_std']:.4f}"
    )


    print(
        f"  Folds 5-fold con 3 clases   : "
        f"{gwo['gwo_5fold_exploratorio']['folds_con_3_clases']}"
        f"/"
        f"{gwo['gwo_5fold_exploratorio']['numero_folds']}"
    )


    print()

    print(
        "Diagnóstico GWO 10-fold:"
    )


    print(
        f"  F1 histórico                : "
        f"{gwo['gwo_10fold_diagnostico']['f1_referencia_historico']:.4f}"
    )


    print(
        f"  F1 con 3 clases fijas       : "
        f"{gwo['gwo_10fold_diagnostico']['f1_macro_3clases_fijas']:.4f}"
    )


    print(
        f"  Folds con las 3 clases      : "
        f"{gwo['gwo_10fold_diagnostico']['folds_con_3_clases']}"
        f"/"
        f"{gwo['gwo_10fold_diagnostico']['numero_folds']}"
    )


    print(
        f"  Folds sin alguna clase      : "
        f"{gwo['gwo_10fold_diagnostico']['folds_sin_alguna_clase']}"
        f"/"
        f"{gwo['gwo_10fold_diagnostico']['numero_folds']}"
    )


    print()

    print(
        "Auditoría metodológica:"
    )


    proxies = (
        auditoria[
            "proxies_directos_gwo"
        ][
            "cantidad"
        ]
    )


    porcentaje = (
        auditoria[
            "proxies_directos_gwo"
        ][
            "porcentaje"
        ]
    )


    if (
        proxies is not None
        and porcentaje is not None
    ):

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
        "Interpretación oficial:"
    )


    print(
        "  0.8412 -> resultado GWO exploratorio reproducido."
    )

    print(
        "  0.7316 -> diagnóstico histórico 10-fold reproducido."
    )

    print(
        "  0.6337 -> sensibilidad 10-fold con tres clases fijas."
    )

    print(
        "  0.5742 -> estimación robusta principal de generalización."
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