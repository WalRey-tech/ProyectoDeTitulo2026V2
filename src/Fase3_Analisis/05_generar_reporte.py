import json
import os
import re
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_DATOS = os.path.normpath(
    os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed")
)

RUTA_DATOS_LIMPIOS = os.path.join(
    DIRECTORIO_DATOS,
    "perfiles_egreso_limpio_v1.csv",
)

RUTA_BENCHMARK = os.path.join(
    DIRECTORIO_DATOS,
    "benchmark_modelos_v3.csv",
)

RUTA_TOP_WORDS = os.path.join(
    DIRECTORIO_DATOS,
    "top15_palabras_clave.csv",
)

RUTA_SALIDA_JSON = os.path.join(
    DIRECTORIO_DATOS,
    "resultados.json",
)

N_PERMUTACIONES = 1000
SEMILLA = 42

VECTORIZADOR_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "max_df": 0.85,
    "min_df": 2,
}


# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def leer_csv_flexible(ruta):
    """
    Lee archivos CSV separados por coma o punto y coma.
    El pipeline actual exporta algunos archivos con ';'.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo requerido: {ruta}"
        )

    return pd.read_csv(
        ruta,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


def normalizar_nombre_columna(nombre):
    """Normaliza nombres para detectar columnas aunque cambie su formato."""
    nombre = str(nombre).strip().lower()

    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }

    for origen, destino in reemplazos.items():
        nombre = nombre.replace(origen, destino)

    return re.sub(r"[^a-z0-9]+", " ", nombre).strip()


def buscar_columna(df, candidatos, obligatoria=False):
    """
    Busca una columna por nombre exacto normalizado o por coincidencia parcial.
    """
    mapa = {
        normalizar_nombre_columna(columna): columna
        for columna in df.columns
    }

    candidatos_normalizados = [
        normalizar_nombre_columna(candidato)
        for candidato in candidatos
    ]

    for candidato in candidatos_normalizados:
        if candidato in mapa:
            return mapa[candidato]

    for candidato in candidatos_normalizados:
        for nombre_normalizado, nombre_original in mapa.items():
            if candidato in nombre_normalizado:
                return nombre_original

    if obligatoria:
        raise ValueError(
            "No se encontró ninguna de las columnas esperadas: "
            f"{candidatos}. Columnas disponibles: {list(df.columns)}"
        )

    return None


def convertir_numero(valor):
    """Convierte valores numéricos con coma decimal, porcentaje o texto."""
    if pd.isna(valor):
        return None

    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)

    texto = str(valor).strip()
    texto = texto.replace("%", "")
    texto = texto.replace(",", ".")

    coincidencia = re.search(r"-?\d+(?:\.\d+)?", texto)

    if coincidencia is None:
        return None

    return float(coincidencia.group())


def convertir_porcentaje_a_decimal(valor):
    """
    Convierte 67.86 en 0.6786, pero conserva 0.6786 si ya está normalizado.
    """
    numero = convertir_numero(valor)

    if numero is None:
        return None

    if abs(numero) > 1:
        return numero / 100.0

    return numero


def redondear_o_none(valor, decimales=6):
    if valor is None or pd.isna(valor):
        return None

    return round(float(valor), decimales)


def calcular_intra_inter(matriz_similitud, etiquetas):
    """Calcula similitud promedio intragrupo, intergrupo y su diferencia."""
    etiquetas = np.asarray(etiquetas)
    n = matriz_similitud.shape[0]

    indices = np.triu_indices(n, k=1)
    similitudes = matriz_similitud[indices]

    misma_clase = (
        etiquetas[:, None] == etiquetas[None, :]
    )[indices]

    intra = float(similitudes[misma_clase].mean())
    inter = float(similitudes[~misma_clase].mean())

    return intra, inter, intra - inter


# =============================================================================
# 3. CARGA DEL CORPUS
# =============================================================================

def cargar_corpus():
    df = leer_csv_flexible(RUTA_DATOS_LIMPIOS)

    columnas_requeridas = {"grado", "perfil_limpio"}
    faltantes = columnas_requeridas - set(df.columns)

    if faltantes:
        raise ValueError(
            f"El corpus no contiene las columnas requeridas: {sorted(faltantes)}"
        )

    df = df.dropna(
        subset=["grado", "perfil_limpio"]
    ).copy()

    df = df[
        df["perfil_limpio"].astype(str).str.strip() != ""
    ].copy()

    if df.empty:
        raise ValueError("El corpus limpio no contiene perfiles válidos.")

    conteos = {
        str(grado): int(cantidad)
        for grado, cantidad in df["grado"]
        .value_counts()
        .to_dict()
        .items()
    }

    conteos["Total"] = int(len(df))

    return df, conteos


# =============================================================================
# 4. BENCHMARK SUPERVISADO
# =============================================================================

def cargar_benchmark():
    df = leer_csv_flexible(RUTA_BENCHMARK)

    columna_modelo = buscar_columna(
        df,
        ["Modelo", "Algoritmo"],
        obligatoria=True,
    )

    columna_accuracy = buscar_columna(
        df,
        [
            "Accuracy (%)",
            "Accuracy Media (%)",
            "Accuracy Medio",
            "Accuracy",
        ],
    )

    columna_balanced = buscar_columna(
        df,
        [
            "Balanced Accuracy (%)",
            "Balanced Accuracy Media (%)",
            "Balanced Accuracy",
        ],
    )

    columna_f1 = buscar_columna(
        df,
        [
            "F1 Macro (%)",
            "F1 Macro Media (%)",
            "F1 Macro Medio",
            "F1 Macro",
        ],
    )

    columna_f1_std = buscar_columna(
        df,
        [
            "F1 Macro Desviacion (%)",
            "F1 Macro Desv (%)",
            "F1 Macro Std (%)",
            "Desviacion F1 Macro (%)",
            "F1 Std (%)",
        ],
    )

    columna_accuracy_std = buscar_columna(
        df,
        [
            "Desviacion Accuracy (%)",
            "Accuracy Std (%)",
            "Desviacion (%)",
        ],
    )

    if columna_f1 is not None:
        criterio = "F1 Macro"
        columna_criterio = columna_f1
    elif columna_balanced is not None:
        criterio = "Balanced Accuracy"
        columna_criterio = columna_balanced
    elif columna_accuracy is not None:
        criterio = "Accuracy"
        columna_criterio = columna_accuracy
    else:
        raise ValueError(
            "El benchmark no contiene F1 Macro, Balanced Accuracy ni Accuracy."
        )

    valores_criterio = df[columna_criterio].apply(convertir_numero)

    if valores_criterio.isna().all():
        raise ValueError(
            f"La columna usada como criterio ({columna_criterio}) "
            "no contiene valores numéricos."
        )

    indice_ganador = valores_criterio.idxmax()
    fila_ganadora = df.loc[indice_ganador]

    def valor_decimal(columna):
        if columna is None:
            return None
        return convertir_porcentaje_a_decimal(fila_ganadora[columna])

    benchmark_completo = []

    for _, fila in df.iterrows():
        registro = {
            "modelo": str(fila[columna_modelo]),
            "accuracy": (
                convertir_porcentaje_a_decimal(fila[columna_accuracy])
                if columna_accuracy is not None
                else None
            ),
            "balanced_accuracy": (
                convertir_porcentaje_a_decimal(fila[columna_balanced])
                if columna_balanced is not None
                else None
            ),
            "f1_macro": (
                convertir_porcentaje_a_decimal(fila[columna_f1])
                if columna_f1 is not None
                else None
            ),
            "f1_macro_desviacion": (
                convertir_porcentaje_a_decimal(fila[columna_f1_std])
                if columna_f1_std is not None
                else None
            ),
            "accuracy_desviacion": (
                convertir_porcentaje_a_decimal(fila[columna_accuracy_std])
                if columna_accuracy_std is not None
                else None
            ),
        }

        benchmark_completo.append(
            {
                clave: redondear_o_none(valor)
                if clave != "modelo"
                else valor
                for clave, valor in registro.items()
            }
        )

    metricas = {
        "modelo_ganador": str(fila_ganadora[columna_modelo]),
        "criterio_seleccion": criterio,
        "accuracy": redondear_o_none(valor_decimal(columna_accuracy)),
        "balanced_accuracy": redondear_o_none(
            valor_decimal(columna_balanced)
        ),
        "f1_macro": redondear_o_none(valor_decimal(columna_f1)),
        "f1_macro_desviacion": redondear_o_none(
            valor_decimal(columna_f1_std)
        ),
        "accuracy_desviacion": redondear_o_none(
            valor_decimal(columna_accuracy_std)
        ),
        "validacion": "Validación cruzada estratificada de 5 pliegues",
        "pipeline": (
            "TF-IDF y SMOTE ajustados exclusivamente dentro de cada "
            "fold de entrenamiento mediante imblearn.Pipeline"
        ),
        "tecnica_balanceo": (
            "SMOTE aplicado solo en los folds de entrenamiento"
        ),
        "nota": (
            "La selección del modelo se realiza por F1 Macro cuando "
            "la métrica está disponible. LDA se utiliza también como "
            "proyección supervisada descriptiva, no como validación "
            "independiente de separabilidad."
        ),
        "benchmark_completo": benchmark_completo,
    }

    return metricas


# =============================================================================
# 5. HOMOGENEIDAD Y TEST DE PERMUTACIÓN
# =============================================================================

def calcular_homogeneidad(df):
    textos = df["perfil_limpio"].astype(str).values
    etiquetas = df["grado"].astype(str).values

    vectorizador = TfidfVectorizer(
        **VECTORIZADOR_CONFIG
    )

    X_tfidf = vectorizador.fit_transform(textos)
    grados = sorted(np.unique(etiquetas))

    centroides = []

    for grado in grados:
        centroides.append(
            np.asarray(
                X_tfidf[etiquetas == grado].mean(axis=0)
            ).ravel()
        )

    matriz_centroides = cosine_similarity(
        np.vstack(centroides)
    )

    matriz_documentos = cosine_similarity(X_tfidf)

    intra, inter, diferencia_observada = calcular_intra_inter(
        matriz_documentos,
        etiquetas,
    )

    rng = np.random.default_rng(SEMILLA)
    diferencias_permutadas = np.zeros(
        N_PERMUTACIONES,
        dtype=float,
    )

    for indice in range(N_PERMUTACIONES):
        etiquetas_permutadas = rng.permutation(etiquetas)

        _, _, diferencia = calcular_intra_inter(
            matriz_documentos,
            etiquetas_permutadas,
        )

        diferencias_permutadas[indice] = diferencia

    casos_extremos = int(
        np.sum(
            diferencias_permutadas >= diferencia_observada
        )
    )

    p_valor = (
        casos_extremos + 1
    ) / (
        N_PERMUTACIONES + 1
    )

    matriz_dict = {}

    for i, grado_fila in enumerate(grados):
        matriz_dict[grado_fila] = {
            grado_columna: round(
                float(matriz_centroides[i, j]),
                6,
            )
            for j, grado_columna in enumerate(grados)
        }

    similitud_civil_informatica = None

    if "Civil" in grados and "Informática" in grados:
        i_civil = grados.index("Civil")
        i_info = grados.index("Informática")

        similitud_civil_informatica = float(
            matriz_centroides[i_civil, i_info]
        )

    return {
        "metodo": (
            "Similitud coseno sobre centroides TF-IDF del corpus real, "
            "sin datos sintéticos"
        ),
        "configuracion_tfidf": {
            "max_features": VECTORIZADOR_CONFIG["max_features"],
            "ngram_range": list(
                VECTORIZADOR_CONFIG["ngram_range"]
            ),
            "max_df": VECTORIZADOR_CONFIG["max_df"],
            "min_df": VECTORIZADOR_CONFIG["min_df"],
        },
        "matriz_similitud_centroides": matriz_dict,
        "similitud_civil_informatica": redondear_o_none(
            similitud_civil_informatica
        ),
        "similitud_promedio_intragrupo": round(intra, 6),
        "similitud_promedio_intergrupo": round(inter, 6),
        "diferencia_intra_inter": round(
            diferencia_observada,
            6,
        ),
        "test_permutacion_p_valor": round(p_valor, 6),
        "casos_extremos": casos_extremos,
        "numero_permutaciones": N_PERMUTACIONES,
        "significancia": (
            "Estadísticamente significativo (p < 0.05)"
            if p_valor < 0.05
            else "No estadísticamente significativo (p >= 0.05)"
        ),
        "interpretacion": (
            "El test evalúa la estructura global de similitudes "
            "intragrupo frente a las intergrupo; no valida de forma "
            "aislada una única comparación entre dos grados."
        ),
        "grafico": "similitud_centroides.png",
    }


# =============================================================================
# 6. TÉRMINOS DISTINTIVOS
# =============================================================================

def cargar_terminos_distintivos():
    df = leer_csv_flexible(RUTA_TOP_WORDS)

    columna_grado = buscar_columna(
        df,
        ["Grado"],
        obligatoria=True,
    )

    columna_termino = buscar_columna(
        df,
        ["Termino", "Término"],
        obligatoria=True,
    )

    columna_ranking = buscar_columna(
        df,
        ["Ranking"],
    )

    columna_z = buscar_columna(
        df,
        ["Z Score", "Z_Score"],
    )

    columna_frecuencia = buscar_columna(
        df,
        ["Frecuencia Absoluta Grado"],
    )

    columna_documentos = buscar_columna(
        df,
        ["Documentos Grado"],
    )

    columna_total_documentos = buscar_columna(
        df,
        ["Total Documentos Grado"],
    )

    columna_soporte = buscar_columna(
        df,
        ["Soporte Documental Grado"],
    )

    terminos_compatibles = {}
    detalle_terminos = {}

    for grado in df[columna_grado].dropna().unique():
        grupo = df[
            df[columna_grado] == grado
        ].copy()

        if columna_ranking is not None:
            grupo = grupo.sort_values(
                by=columna_ranking
            )

        terminos_compatibles[str(grado)] = (
            grupo[columna_termino]
            .astype(str)
            .tolist()
        )

        detalle = []

        for _, fila in grupo.iterrows():
            registro = {
                "ranking": (
                    int(convertir_numero(fila[columna_ranking]))
                    if columna_ranking is not None
                    and convertir_numero(fila[columna_ranking]) is not None
                    else None
                ),
                "termino": str(fila[columna_termino]),
                "z_score": (
                    redondear_o_none(
                        convertir_numero(fila[columna_z])
                    )
                    if columna_z is not None
                    else None
                ),
                "frecuencia_absoluta_grado": (
                    int(convertir_numero(fila[columna_frecuencia]))
                    if columna_frecuencia is not None
                    and convertir_numero(fila[columna_frecuencia]) is not None
                    else None
                ),
                "documentos_grado": (
                    int(convertir_numero(fila[columna_documentos]))
                    if columna_documentos is not None
                    and convertir_numero(fila[columna_documentos]) is not None
                    else None
                ),
                "total_documentos_grado": (
                    int(convertir_numero(fila[columna_total_documentos]))
                    if columna_total_documentos is not None
                    and convertir_numero(
                        fila[columna_total_documentos]
                    ) is not None
                    else None
                ),
                "soporte_documental_grado": (
                    redondear_o_none(
                        convertir_numero(fila[columna_soporte])
                    )
                    if columna_soporte is not None
                    else None
                ),
            }

            detalle.append(registro)

        detalle_terminos[str(grado)] = detalle

    return terminos_compatibles, detalle_terminos


# =============================================================================
# 7. CONSOLIDACIÓN FINAL
# =============================================================================

def main():
    print("=" * 76)
    print("GENERACIÓN DEL REPORTE CONSOLIDADO")
    print("=" * 76)

    try:
        df_corpus, conteos = cargar_corpus()
        print(
            f"Corpus cargado correctamente: "
            f"{conteos['Total']} perfiles."
        )

        metricas_supervisadas = cargar_benchmark()
        print(
            "Modelo ganador por "
            f"{metricas_supervisadas['criterio_seleccion']}: "
            f"{metricas_supervisadas['modelo_ganador']}"
        )

        analisis_homogeneidad = calcular_homogeneidad(
            df_corpus
        )

        print(
            "Similitud Civil–Informática: "
            f"{analisis_homogeneidad['similitud_civil_informatica']}"
        )

        print(
            "p-valor corregido: "
            f"{analisis_homogeneidad['test_permutacion_p_valor']}"
        )

        terminos, detalle_terminos = (
            cargar_terminos_distintivos()
        )

        reporte = {
            "version_reporte": "1.0",
            "fecha_generacion_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "n_por_grado": conteos,
            "metricas_supervisadas": metricas_supervisadas,
            "analisis_homogeneidad": analisis_homogeneidad,

            # Se conserva esta estructura simple para no romper
            # la integración existente con la landing page.
            "terminos_distintivos": terminos,

            # Estructura ampliada para auditoría y trazabilidad.
            "detalle_terminos_distintivos": detalle_terminos,

            "limitaciones": [
                (
                    "El corpus contiene una clase minoritaria de "
                    "Ingeniería de Ejecución con cinco perfiles."
                ),
                (
                    "Las métricas presentan variabilidad elevada y deben "
                    "interpretarse como evidencia exploratoria."
                ),
                (
                    "La proyección LDA utiliza las etiquetas conocidas y "
                    "no constituye validación predictiva independiente."
                ),
            ],
            "artefactos": {
                "dataset_limpio":
                    "perfiles_egreso_limpio_v1.csv",
                "benchmark":
                    "benchmark_modelos.csv",
                "matriz_confusion":
                    "matriz_confusion.png",
                "proyeccion":
                    "proyeccion_pca_vs_lda.png",
                "similitud":
                    "similitud_centroides.png",
                "terminos_csv":
                    "top15_palabras_clave.csv",
                "terminos_json":
                    "terminos_distintivos_v1.json",
            },
        }

        os.makedirs(
            os.path.dirname(RUTA_SALIDA_JSON),
            exist_ok=True,
        )

        with open(
            RUTA_SALIDA_JSON,
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(
                reporte,
                archivo,
                indent=4,
                ensure_ascii=False,
            )

        print("\n" + "=" * 76)
        print("CONSOLIDACIÓN COMPLETADA")
        print("=" * 76)
        print(f"Salida: {RUTA_SALIDA_JSON}")
        print(
            "Distribución: "
            + ", ".join(
                f"{grado}={cantidad}"
                for grado, cantidad in conteos.items()
            )
        )

    except Exception as error:
        print("\nERROR CRÍTICO")
        print(str(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
