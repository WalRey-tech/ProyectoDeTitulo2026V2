import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))

RUTA_ENTRADA = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "perfiles_egreso_limpio_v1.csv",
    )
)

RUTA_GRAFICO_SALIDA = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "similitud_centroides.png",
    )
)

RUTA_MATRIZ_CSV = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "matriz_similitud_centroides_v1.csv",
    )
)

RUTA_METRICAS_JSON = os.path.normpath(
    os.path.join(
        DIRECTORIO_ACTUAL,
        "..",
        "data",
        "processed",
        "metricas_homogeneidad_v1.json",
    )
)

RANDOM_STATE = 42
N_PERMUTACIONES = 1000

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "max_df": 0.85,
    "min_df": 2,
}


def calcular_metricas_intra_inter(sim_matrix, labels):
    labels_arr = np.asarray(labels)
    n = sim_matrix.shape[0]

    indices_superiores = np.triu_indices(n, k=1)
    similitudes = sim_matrix[indices_superiores]

    misma_etiqueta = (
        labels_arr[:, None] == labels_arr[None, :]
    )[indices_superiores]

    similitudes_intra = similitudes[misma_etiqueta]
    similitudes_inter = similitudes[~misma_etiqueta]

    promedio_intra = float(similitudes_intra.mean())
    promedio_inter = float(similitudes_inter.mean())
    diferencia = promedio_intra - promedio_inter

    return promedio_intra, promedio_inter, diferencia


def ejecutar_test_permutacion(
    sim_matrix,
    labels,
    diferencia_observada,
    n_permutaciones=N_PERMUTACIONES,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    labels_arr = np.asarray(labels)

    diferencias_permutadas = np.empty(
        n_permutaciones,
        dtype=float,
    )

    for i in range(n_permutaciones):
        etiquetas_permutadas = rng.permutation(labels_arr)

        _, _, diferencia_permutada = calcular_metricas_intra_inter(
            sim_matrix,
            etiquetas_permutadas,
        )

        diferencias_permutadas[i] = diferencia_permutada

    casos_extremos = int(
        np.sum(diferencias_permutadas >= diferencia_observada)
    )

    p_valor = (
        casos_extremos + 1
    ) / (
        n_permutaciones + 1
    )

    return p_valor, casos_extremos, diferencias_permutadas


def main():
    print("=" * 76)
    print("ANÁLISIS DE HOMOGENEIDAD LÉXICA Y TEST DE PERMUTACIÓN")
    print("=" * 76)

    try:
        df = pd.read_csv(
            RUTA_ENTRADA,
            encoding="utf-8-sig",
        )
    except FileNotFoundError:
        print(
            "Error: no se encontró el archivo de entrada:\n"
            f"{RUTA_ENTRADA}"
        )
        return
    except Exception as error:
        print(f"Error al leer el dataset: {error}")
        return

    columnas_obligatorias = {
        "perfil_limpio",
        "grado",
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

    X_text = df["perfil_limpio"].astype(str).to_numpy()
    y = df["grado"].astype(str).to_numpy()

    print(f"Total de perfiles reales: {len(df)}")
    print("\nDistribución por grado:")
    print(df["grado"].value_counts())

    print("\nEjecutando vectorización TF-IDF sobre perfiles reales...")
    print(f"Configuración: {TFIDF_CONFIG}")

    vectorizer = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X_tfidf_dense = vectorizer.fit_transform(
        X_text
    ).toarray()

    print(
        "Dimensiones de la matriz TF-IDF: "
        f"{X_tfidf_dense.shape}"
    )

    grados_unicos = sorted(np.unique(y))
    centroides = []

    print("\nCalculando centroides con documentos reales...")

    for grado in grados_unicos:
        vectores_grado = X_tfidf_dense[y == grado]
        centroides.append(
            vectores_grado.mean(axis=0)
        )

    matriz_centroides = np.vstack(centroides)
    similitud_centroides = cosine_similarity(
        matriz_centroides
    )

    df_similitud = pd.DataFrame(
        similitud_centroides,
        index=grados_unicos,
        columns=grados_unicos,
    )

    df_similitud.index.name = "Grado"
    df_similitud.columns.name = "Grado comparado"

    print("\n" + "=" * 76)
    print("SIMILITUD COSENO ENTRE CENTROIDES REALES")
    print("=" * 76)
    print(df_similitud.round(4).to_string())

    os.makedirs(
        os.path.dirname(RUTA_MATRIZ_CSV),
        exist_ok=True,
    )

    df_similitud.to_csv(
        RUTA_MATRIZ_CSV,
        sep=";",
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        df_similitud,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={
            "label": "Similitud coseno",
        },
    )

    plt.title(
        "Similitud coseno entre centroides léxicos reales",
        fontsize=13,
        fontweight="bold",
    )

    plt.xlabel("Grado comparado")
    plt.ylabel("Grado")
    plt.tight_layout()

    plt.savefig(
        RUTA_GRAFICO_SALIDA,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nMapa de calor guardado en:\n"
        f"{RUTA_GRAFICO_SALIDA}"
    )

    similitud_documentos = cosine_similarity(
        X_tfidf_dense
    )

    (
        promedio_intra,
        promedio_inter,
        diferencia_observada,
    ) = calcular_metricas_intra_inter(
        similitud_documentos,
        y,
    )

    print("\n" + "=" * 76)
    print("DIFERENCIA GLOBAL INTRAGRUPO VS INTERGRUPO")
    print("=" * 76)

    print(
        f"Similitud promedio intragrupo: "
        f"{promedio_intra:.4f}"
    )

    print(
        f"Similitud promedio intergrupo: "
        f"{promedio_inter:.4f}"
    )

    print(
        f"Diferencia observada (intra - inter): "
        f"{diferencia_observada:.4f}"
    )

    print("\n" + "=" * 76)
    print("TEST DE PERMUTACIÓN")
    print("=" * 76)

    print(
        f"Ejecutando {N_PERMUTACIONES} permutaciones "
        "de las etiquetas reales..."
    )

    (
        p_valor,
        casos_extremos,
        diferencias_permutadas,
    ) = ejecutar_test_permutacion(
        similitud_documentos,
        y,
        diferencia_observada,
    )

    print(
        "Casos permutados iguales o superiores a la "
        f"diferencia observada: {casos_extremos}"
    )

    print(f"p-valor corregido: {p_valor:.6f}")

    if p_valor < 0.05:
        conclusion = (
            "Se rechaza la hipótesis nula. La estructura global "
            "de similitudes intra e intergrado difiere de una "
            "asignación aleatoria de etiquetas."
        )
    else:
        conclusion = (
            "No existe evidencia suficiente para rechazar la "
            "hipótesis nula. La estructura observada podría ser "
            "compatible con una asignación aleatoria."
        )

    print(f"\nConclusión: {conclusion}")

    pares_centroides = {}

    for i, grado_a in enumerate(grados_unicos):
        for j in range(i + 1, len(grados_unicos)):
            grado_b = grados_unicos[j]
            clave = f"{grado_a}__{grado_b}"
            pares_centroides[clave] = float(
                similitud_centroides[i, j]
            )

    metricas = {
        "total_perfiles": int(len(df)),
        "distribucion_grados": {
            str(grado): int(cantidad)
            for grado, cantidad in df["grado"]
            .value_counts()
            .to_dict()
            .items()
        },
        "configuracion_tfidf": {
            "max_features": TFIDF_CONFIG["max_features"],
            "ngram_range": list(
                TFIDF_CONFIG["ngram_range"]
            ),
            "max_df": TFIDF_CONFIG["max_df"],
            "min_df": TFIDF_CONFIG["min_df"],
        },
        "usa_smote": False,
        "similitud_centroides": pares_centroides,
        "similitud_promedio_intra": promedio_intra,
        "similitud_promedio_inter": promedio_inter,
        "diferencia_intra_inter": diferencia_observada,
        "test_permutacion": {
            "n_permutaciones": N_PERMUTACIONES,
            "casos_extremos": casos_extremos,
            "p_valor": p_valor,
            "media_distribucion_nula": float(
                diferencias_permutadas.mean()
            ),
            "desviacion_distribucion_nula": float(
                diferencias_permutadas.std()
            ),
            "significativo_005": bool(p_valor < 0.05),
            "conclusion": conclusion,
        },
    }

    with open(
        RUTA_METRICAS_JSON,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            metricas,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\nMétricas exportadas en:\n"
        f"{RUTA_METRICAS_JSON}"
    )

    print("\nProceso completado correctamente.")


if __name__ == "__main__":
    main()
