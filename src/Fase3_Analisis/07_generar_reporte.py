# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import hashlib
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
    resumen = leer_json(MODELO_FINAL_RESUMEN)
    if resumen.get('validacion', {}).get('protocolo_id') != 'gwo_nested_v2':
        raise ValueError('El resultado pertenece al modelo anterior. Ejecuta primero el nuevo 06_modelo_final_robusto.py.')
    df_folds = leer_csv(MODELO_FINAL_FOLDS)
    baseline = leer_csv(MODELO_FINAL_DIR / 'metricas_folds_baseline.csv')
    requeridas = {'Fold', 'F1_macro', 'Accuracy', 'run_id'}
    for tabla in [df_folds, baseline]:
        if not requeridas.issubset(tabla.columns):
            raise ValueError('Las métricas no contienen todas las columnas del protocolo nuevo.')
        if len(tabla) != 5 or sorted(tabla['Fold'].tolist()) != [1, 2, 3, 4, 5]:
            raise ValueError('Se requieren exactamente cinco folds externos completos.')
        if set(tabla['run_id'].astype(str)) != {resumen.get('run_id')}:
            raise ValueError('CSV y JSON pertenecen a ejecuciones diferentes. Vuelve a ejecutar 06.')
        if tabla[['F1_macro', 'Accuracy']].isna().any().any():
            raise ValueError('Hay métricas incompletas.')
    hash_actual = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if resumen.get('corpus', {}).get('sha256') != hash_actual:
        raise ValueError('El corpus cambió después de la evaluación. Vuelve a ejecutar 06.')
    resultados = resumen['resultados_principales']
    f1_media = float(df_folds['F1_macro'].mean())
    f1_std = float(df_folds['F1_macro'].std(ddof=1))
    accuracy_media = float(df_folds['Accuracy'].mean())
    valores = {'f1_macro_media_folds': f1_media, 'f1_macro_std_folds': f1_std,
               'accuracy_media_folds': accuracy_media}
    for clave, observado in valores.items():
        if abs(float(resultados[clave]) - observado) > 1e-9:
            raise ValueError(f'CSV y JSON discrepan en {clave}. Vuelve a ejecutar 06.')
    f1_base = float(baseline['F1_macro'].mean())
    if abs(f1_base - float(resumen['baseline_sin_gwo']['f1_macro_media_folds'])) > 1e-9:
        raise ValueError('CSV y JSON discrepan en el baseline.')
    return {
        'estado': resumen['estado'], 'modelo': 'TF-IDF + GWO + SMOTE + Complement Naive Bayes',
        'representacion': 'TF-IDF de palabras y bigramas; selección GWO en cada entrenamiento externo.',
        'balanceo': 'SMOTE exclusivamente en entrenamiento interno o externo, nunca en prueba.',
        'control_fuga': 'Enmascaramiento de denominaciones y aislamiento de la prueba externa.',
        'validacion': resumen['validacion']['metodo'],
        'metrica_principal': 'F1-macro medio en los cinco tests externos',
        'f1_macro_media': f1_media, 'f1_macro_std': f1_std, 'accuracy_media': accuracy_media,
        'f1_macro_oof_secundario': resultados['f1_macro_oof'],
        'accuracy_oof_secundaria': resultados['accuracy_oof'],
        'metricas_oof_por_clase': resumen['metricas_oof_por_clase'],
        'numero_folds': len(df_folds), 'presupuesto_referencia': resumen['presupuesto_referencia'],
        'baseline_sin_gwo': resumen['baseline_sin_gwo'],
        'seleccion_gwo': resumen['seleccion_gwo'], 'resultado_completo': resumen,
        'interpretacion': 'Estimación externa del procedimiento con selección GWO interna. La comparación con baseline es descriptiva y no garantiza una mejora.',
        'limitacion_principal': resumen['interpretacion']['limitacion'],
    }



# =============================================================================
# 13. CONCLUSIÓN INTEGRADA
# =============================================================================

def construir_conclusion_integrada(gwo: dict, modelo_final: dict, auditoria: dict) -> dict:
    cv5 = gwo['gwo_5fold_exploratorio']
    cv10 = gwo['gwo_10fold_diagnostico']
    base = modelo_final['baseline_sin_gwo']
    seleccion = modelo_final['seleccion_gwo']
    cantidad_proxies = auditoria['proxies_directos_gwo']['cantidad']
    return {
        'hallazgo_central': 'La similitud y la diferenciación deben interpretarse junto con los resultados semánticos del reporte; la clasificación evalúa separabilidad textual, no calidad educativa.',
        'gwo': (f"La ejecución exploratoria registrada seleccionó {gwo['features_seleccionadas']} "
                f"de {gwo['features_originales']} características y obtuvo "
                f"F1-macro={cv5['f1_macro']:.4f}. La selección previa a los folds impide tratarlo como evaluación externa independiente."),
        'diagnostico_10fold': (
            f"F1 de referencia={cv10['f1_referencia_historico']:.4f}; "
            f"con tres clases fijas={cv10['f1_macro_3clases_fijas']:.4f}. "
            f"{cv10['folds_con_3_clases']}/{cv10['numero_folds']} folds contienen todas las clases. Uso diagnóstico."),
        'auditoria': ('No se adjuntó una auditoría de proxies; no se presume su resultado.'
                      if cantidad_proxies is None else
                      f'La auditoría disponible registra {cantidad_proxies} características proxy.'),
        'resultado_robusto': (
            f"GWO dentro del entrenamiento externo obtuvo F1-macro medio={modelo_final['f1_macro_media']:.4f} "
            f"(desviación={modelo_final['f1_macro_std']:.4f}). "
            f"Subconjuntos por fold: {seleccion['features_por_fold']}."),
        'comparacion_baseline': (
            f"Sin GWO, en las mismas particiones, F1-macro={base['f1_macro_media_folds']:.4f}. "
            f"Diferencia GWO menos baseline={base['delta_f1_gwo_menos_baseline']:+.4f}. "
            'Esta diferencia descriptiva no demuestra significancia estadística ni justifica elegir y reevaluar un ganador con los mismos tests.'),
        'limitacion_principal': modelo_final['limitacion_principal'],
        'metricas_clave': {
            'gwo_5fold_exploratorio': cv5['f1_macro'], 'gwo_5fold_std': cv5['f1_std'],
            'gwo_10fold_historico': cv10['f1_referencia_historico'],
            'gwo_10fold_3clases_diagnostico': cv10['f1_macro_3clases_fijas'],
            'gwo_10fold_folds_con_3_clases': cv10['folds_con_3_clases'],
            'proxies_directos_gwo': cantidad_proxies,
            'modelo_robusto_f1_macro': modelo_final['f1_macro_media'],
            'modelo_robusto_f1_std': modelo_final['f1_macro_std'],
            'modelo_robusto_accuracy': modelo_final['accuracy_media'],
            'baseline_f1_macro': base['f1_macro_media_folds'],
            'delta_gwo_baseline': base['delta_f1_gwo_menos_baseline'],
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
    print('GENERACIÓN DEL REPORTE: GWO EXPLORATORIO Y GWO CON VALIDACIÓN EXTERNA')
    verificar_archivos_esenciales()
    corpus = construir_bloque_corpus()
    analisis_semantico = construir_bloque_semantico()
    gwo = construir_bloque_gwo()
    auditoria = construir_bloque_auditoria()
    modelo_final = construir_bloque_modelo_final()
    conclusion = construir_conclusion_integrada(gwo, modelo_final, auditoria)
    artefactos = construir_bloque_artefactos()
    artefactos['modelo_final_robusto'].update({
        'baseline': 'modelo_final_robusto/metricas_folds_baseline.csv',
        'comparacion': 'modelo_final_robusto/comparacion_gwo_baseline.csv',
        'features_por_fold': 'modelo_final_robusto/features_gwo_por_fold.csv',
        'auditoria_cv': 'modelo_final_robusto/auditoria_cv_gwo.json',
    })
    reporte = {
        'metadata': {'version': '6.0',
            'fecha_generacion_utc': datetime.now(timezone.utc).isoformat(),
            'estado': 'resultados_consolidados_gwo_anidado',
            'proyecto': 'Análisis de perfiles de egreso de carreras de informática en Chile',
            'run_id_modelo_final': modelo_final['resultado_completo']['run_id'],
            'presupuesto_referencia': modelo_final['presupuesto_referencia']},
        'corpus': corpus, 'analisis_semantico': analisis_semantico,
        'gwo_exploratorio': gwo, 'auditoria_metodologica': auditoria,
        'resultado_robusto_final': modelo_final, 'conclusion_integrada': conclusion,
        'advertencias_metodologicas': [
            'Los pasos 04/05 siguen siendo exploratorios. El paso 06 realiza una selección nueva en cada entrenamiento externo.',
            'Los resultados se leen de archivos; no se reutiliza un F1 histórico como resultado de esta ejecución.',
            'Los subconjuntos GWO pueden cambiar entre folds y no tienen que contener 249 términos.',
            'El F1 interno es un criterio de búsqueda; el rendimiento se estima en los tests externos.',
            'PCA y LDA se mantienen como visualizaciones independientes.',
            'F1-macro no equivale a precisión ni al porcentaje de aciertos.',
            'La media de F1 por fold puede diferir del F1 de las predicciones OOF reunidas.',
            'La comparación con baseline es descriptiva: no garantiza mejora ni significancia.',
            'Los resúmenes semánticos y exploratorios previos carecen de hash obligatorio; verifica que correspondan al mismo V2 antes de consolidarlos.',
            'No se ajusta ni serializa aquí un modelo de despliegue entrenado sobre todo el corpus.',
        ],
        'artefactos_oficiales': artefactos,
    }
    if not modelo_final['presupuesto_referencia']:
        reporte['advertencias_metodologicas'].append(
            'El presupuesto GWO es distinto de 100 iteraciones y 30 lobos; identifícalo al comparar resultados.')
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    with SALIDA_FINAL.open('w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=4)
    print(conclusion['resultado_robusto'])
    print(conclusion['comparacion_baseline'])
    print(f'Reporte generado: {SALIDA_FINAL}')
    return 0



if __name__ == "__main__":

    raise SystemExit(
        main()
    )