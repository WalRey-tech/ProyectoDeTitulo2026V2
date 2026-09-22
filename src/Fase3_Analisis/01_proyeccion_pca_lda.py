# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer


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

CORPUS_SELECCIONADO = "actual"
RUTA_ENTRADA = ""
RESULTADOS_DIR = ""
RUTA_GRAFICO = ""
RUTA_COORDENADAS = ""
RUTA_RESUMEN = ""


def configurar_corpus(corpus: str) -> None:
    """Selecciona una entrada explícita; no busca otro corpus como respaldo."""
    global CORPUS_SELECCIONADO, RUTA_ENTRADA, RESULTADOS_DIR
    global RUTA_GRAFICO, RUTA_COORDENADAS, RUTA_RESUMEN
    if corpus not in {"actual", "v2"}:
        raise ValueError("Corpus no válido: usa actual o v2.")
    CORPUS_SELECCIONADO = corpus
    nombre = ("perfiles_egreso_etiquetado_actual_corregido.csv"
              if corpus == "actual" else "perfiles_egreso_etiquetado_v2.csv")
    RUTA_ENTRADA = os.path.join(SRC_ROOT, "data", "processed", nombre)
    RESULTADOS_DIR = os.path.join(
        SRC_ROOT, "data", "resultados_cientificos", "visualizaciones_exploratorias"
    )
    # Mantiene los nombres históricos V2 y separa los resultados actuales.
    RUTA_GRAFICO = os.path.join(RESULTADOS_DIR, f"proyeccion_pca_vs_lda_{corpus}.png")
    RUTA_COORDENADAS = os.path.join(RESULTADOS_DIR, f"coordenadas_pca_lda_{corpus}.csv")
    RUTA_RESUMEN = os.path.join(RESULTADOS_DIR, f"resumen_pca_lda_{corpus}.json")


def solicitar_corpus() -> str:
    print("\nSelecciona el corpus:")
    print("1. Actual — salida corregida del encoding")
    print("2. V2 — corpus histórico")
    opciones = {"1": "actual", "actual": "actual", "2": "v2", "v2": "v2"}
    while True:
        try:
            respuesta = input("Opción [1/2]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("Selección cancelada. Usa --corpus actual o --corpus v2.") from None
        if respuesta in opciones:
            return opciones[respuesta]
        print("Opción no válida. Escribe 1 o 2.")


def sha256_archivo(ruta: str) -> str:
    with open(ruta, "rb") as archivo:
        return hashlib.sha256(archivo.read()).hexdigest()


configurar_corpus("actual")


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

RANDOM_STATE = 42

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
}


# =============================================================================
# 3. COLORES PARA VISUALIZACIÓN
# =============================================================================

COLORES = {
    "Civil": "#1f77b4",
    "Informática": "#ff7f0e",
    "Ejecución": "#2ca02c",
}


# =============================================================================
# 4. CARGA Y VALIDACIÓN
# =============================================================================

def cargar_corpus() -> pd.DataFrame:
    if not os.path.isfile(RUTA_ENTRADA):
        raise FileNotFoundError(
            f"No se encontró el corpus seleccionado ({CORPUS_SELECCIONADO}):\n"
            f"{RUTA_ENTRADA}\n"
            "Para actual, ejecuta primero el encoding con --corpus actual."
        )
    huella = sha256_archivo(RUTA_ENTRADA)
    # Admite los CSV históricos con comas y los actuales con punto y coma.
    df = pd.read_csv(RUTA_ENTRADA, sep=None, engine="python",
                     encoding="utf-8-sig", keep_default_na=False)
    if sha256_archivo(RUTA_ENTRADA) != huella:
        raise ValueError("El CSV cambió durante la lectura. Repite la ejecución.")
    faltantes = {"perfil_egreso", "grado"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    if df.empty:
        raise ValueError("El corpus está vacío.")
    for columna in ["perfil_egreso", "grado"]:
        vacias = df[columna].astype(str).str.strip().eq("")
        if vacias.any():
            raise ValueError(
                f"Hay {int(vacias.sum())} filas sin {columna}; "
                "corrige el corpus en fase 2. No se eliminaron filas."
            )
    desconocidas = set(df["grado"]) - set(COLORES)
    if desconocidas:
        raise ValueError(f"Grados fuera del catálogo: {sorted(desconocidas)}")
    for columna in ["estado_registro", "estado_etiquetado"]:
        if columna in df:
            pendientes = df[columna].astype(str).str.strip().str.upper().isin(["REVISAR", "ERROR"])
            if pendientes.any():
                raise ValueError(f"El CSV contiene filas REVISAR/ERROR en {columna}.")
    conteos = df["grado"].value_counts()
    if len(conteos) != 3 or conteos.min() < 2:
        raise ValueError("PCA/LDA 2D requiere las tres clases, con al menos dos perfiles cada una.")
    df.attrs["sha256_entrada"] = huella
    return df


# =============================================================================
# 5. VECTORIZACIÓN
# =============================================================================

def vectorizar(
    textos: pd.Series,
):

    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X_tfidf = vectorizador.fit_transform(
        textos.astype(str)
    )

    if X_tfidf.shape[1] < 2:
        raise ValueError("TF-IDF generó menos de dos características; no permite la proyección 2D.")
    return vectorizador, X_tfidf.toarray()


# =============================================================================
# 6. PROYECCIONES
# =============================================================================

def calcular_proyecciones(
    X: np.ndarray,
    y: pd.Series,
):

    # PCA:
    # proyección no supervisada.
    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    X_pca = pca.fit_transform(
        X
    )

    # LDA:
    # proyección supervisada.
    #
    # IMPORTANTE:
    # usa las etiquetas de grado y, por tanto,
    # solo se interpreta como visualización
    # descriptiva/exploratoria.
    lda = LinearDiscriminantAnalysis(
        n_components=2
    )

    X_lda = lda.fit_transform(
        X,
        y,
    )

    if X_lda.shape[1] < 2 or not np.isfinite(X_lda).all() or not np.isfinite(X_pca).all():
        raise ValueError("El corpus no permite dos ejes finitos de PCA/LDA. Revisa su variación textual.")
    return pca, X_pca, lda, X_lda


# =============================================================================
# 7. GUARDAR COORDENADAS
# =============================================================================

def guardar_coordenadas(
    df: pd.DataFrame,
    X_pca: np.ndarray,
    X_lda: np.ndarray,
) -> None:

    coordenadas = pd.DataFrame(
        {
            "grado":
                df["grado"].astype(str).values,

            "PCA_1":
                X_pca[:, 0],

            "PCA_2":
                X_pca[:, 1],

            "LDA_1":
                X_lda[:, 0],

            "LDA_2":
                X_lda[:, 1],
        }
    )

    # Identidad de cada punto y grupos para su uso en pasos posteriores.
    coordenadas.insert(0, "fila_datos", np.arange(1, len(df) + 1))
    for columna in ["indice_fuente", "universidad", "carrera", "url",
                    "modalidad", "id_programa", "grupo_perfil"]:
        if columna in df:
            coordenadas[columna] = df[columna].values

    coordenadas.to_csv(
        RUTA_COORDENADAS,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 8. GENERACIÓN DEL GRÁFICO
# =============================================================================

def generar_grafico(
    y: pd.Series,
    X_pca: np.ndarray,
    X_lda: np.ndarray,
    varianza_pca: np.ndarray,
) -> None:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
    )

    fig.suptitle(
        (
            f"Perfiles de egreso {CORPUS_SELECCIONADO.upper()} (n={len(y)}) — "
            "Proyección PCA vs. LDA"
        ),
        fontsize=16,
        fontweight="bold",
    )

    clases = [
        "Civil",
        "Informática",
        "Ejecución",
    ]

    # -------------------------------------------------------------------------
    # PCA
    # -------------------------------------------------------------------------

    for clase in clases:

        mascara = (
            y.astype(str).values
            == clase
        )

        axes[0].scatter(
            X_pca[mascara, 0],
            X_pca[mascara, 1],
            label=clase,
            alpha=0.75,
            s=70,
            edgecolors="black",
            linewidths=0.4,
            color=COLORES.get(
                clase
            ),
        )

    axes[0].set_title(
        "PCA — Proyección no supervisada"
    )

    axes[0].set_xlabel(
        (
            "Componente principal 1 "
            f"({varianza_pca[0] * 100:.1f}% var.)"
        )
    )

    axes[0].set_ylabel(
        (
            "Componente principal 2 "
            f"({varianza_pca[1] * 100:.1f}% var.)"
        )
    )

    axes[0].legend(
        title="Grado"
    )

    axes[0].grid(
        alpha=0.25
    )

    # -------------------------------------------------------------------------
    # LDA
    # -------------------------------------------------------------------------

    for clase in clases:

        mascara = (
            y.astype(str).values
            == clase
        )

        axes[1].scatter(
            X_lda[mascara, 0],
            X_lda[mascara, 1],
            label=clase,
            alpha=0.75,
            s=70,
            edgecolors="black",
            linewidths=0.4,
            color=COLORES.get(
                clase
            ),
        )

    axes[1].set_title(
        (
            "LDA — Proyección supervisada\n"
            "(visualización exploratoria)"
        )
    )

    axes[1].set_xlabel(
        "Función discriminante 1"
    )

    axes[1].set_ylabel(
        "Función discriminante 2"
    )

    axes[1].legend(
        title="Grado"
    )

    axes[1].grid(
        alpha=0.25
    )

    fig.text(
        0.5,
        0.01,
        (
            "Nota: la proyección LDA utiliza las etiquetas "
            "de clase y no representa una estimación "
            "de rendimiento predictivo."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            0.94,
        ]
    )

    plt.savefig(
        RUTA_GRAFICO,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 9. RESUMEN DE RESULTADOS
# =============================================================================

def guardar_resumen(
    df: pd.DataFrame,
    vectorizador: TfidfVectorizer,
    pca: PCA,
    lda: LinearDiscriminantAnalysis,
) -> dict:

    distribucion = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    varianza_pca = (
        pca.explained_variance_ratio_
    )

    if hasattr(
        lda,
        "explained_variance_ratio_",
    ):
        varianza_lda = (
            lda.explained_variance_ratio_
        )
    else:
        varianza_lda = np.array(
            []
        )

    resumen = {

        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "corpus": {
            "seleccion": CORPUS_SELECCIONADO,
            "archivo_entrada": RUTA_ENTRADA,
            "sha256": df.attrs["sha256_entrada"],
            "filas_descartadas": 0,
            "total": int(len(df)),
            "distribucion": {
                str(clase): int(cantidad)
                for clase, cantidad
                in distribucion.items()
            },
        },

        "representacion": {
            "metodo": "TF-IDF",
            "max_features": 400,
            "ngram_range": [
                1,
                2,
            ],
            "min_df": 2,
            "max_df": 0.90,
            "sublinear_tf": True,
            "features_generadas":
                int(
                    len(
                        vectorizador
                        .get_feature_names_out()
                    )
                ),
        },

        "pca": {
            "tipo":
                "no supervisado",

            "varianza_explicada_componente_1":
                round(
                    float(
                        varianza_pca[0]
                    ),
                    6,
                ),

            "varianza_explicada_componente_2":
                round(
                    float(
                        varianza_pca[1]
                    ),
                    6,
                ),

            "varianza_explicada_total_2d":
                round(
                    float(
                        varianza_pca.sum()
                    ),
                    6,
                ),
        },

        "lda": {
            "tipo":
                "supervisado",

            "uso":
                (
                    "Visualización exploratoria "
                    "de separabilidad entre clases."
                ),

            "advertencia":
                (
                    "LDA utiliza las etiquetas reales; "
                    "no debe interpretarse como "
                    "rendimiento predictivo."
                ),

            "varianza_discriminante":
                [
                    round(
                        float(valor),
                        6,
                    )
                    for valor
                    in varianza_lda
                ],
        },
    }

    resumen["grupos_perfil_repetidos"] = {}
    if "grupo_perfil" in df:
        grupos = df["grupo_perfil"].astype(str).str.strip()
        conteos = grupos[grupos.ne("")].value_counts()
        resumen["grupos_perfil_repetidos"] = {
            str(grupo): int(n) for grupo, n in conteos.items() if n > 1
        }
    resumen["alcance"] = (
        "PCA y LDA ajustados al corpus completo para exploración. "
        "Este script no realiza particiones ni validación predictiva. "
        "Los grupos de perfiles deben tratarse en la validación posterior."
    )
    if sha256_archivo(RUTA_ENTRADA) != df.attrs["sha256_entrada"]:
        raise ValueError("El corpus cambió durante el análisis. Repite la ejecución.")

    with open(
        RUTA_RESUMEN,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    return resumen


# =============================================================================
# 10. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Proyección exploratoria PCA/LDA del corpus seleccionado.")
    parser.add_argument("--corpus", choices=["actual", "v2"],
                        help="Si se omite, muestra el menú de selección.")
    args = parser.parse_args()
    configurar_corpus(args.corpus or solicitar_corpus())

    print(
        "=" * 72
    )

    print(
        f"VISUALIZACIÓN EXPLORATORIA PCA vs. LDA — {CORPUS_SELECCIONADO.upper()}"
    )

    print(
        "=" * 72
    )

    os.makedirs(
        RESULTADOS_DIR,
        exist_ok=True,
    )

    print(f"Entrada seleccionada: {RUTA_ENTRADA}")
    df = cargar_corpus()
    print(f"SHA-256: {df.attrs['sha256_entrada']}")

    print(
        f"\nCorpus: {len(df)} perfiles"
    )

    print(
        "Distribución:"
    )

    print(
        df["grado"].value_counts()
    )

    vectorizador, X = vectorizar(
        df["perfil_egreso"]
    )

    print(
        f"\nTF-IDF generado: "
        f"{X.shape[1]} características"
    )

    (
        pca,
        X_pca,
        lda,
        X_lda,
    ) = calcular_proyecciones(
        X,
        df["grado"],
    )

    guardar_coordenadas(
        df,
        X_pca,
        X_lda,
    )

    generar_grafico(
        df["grado"],
        X_pca,
        X_lda,
        pca.explained_variance_ratio_,
    )

    resumen = guardar_resumen(
        df,
        vectorizador,
        pca,
        lda,
    )

    if resumen["grupos_perfil_repetidos"]:
        print("Grupos compartidos conservados:", resumen["grupos_perfil_repetidos"])
        print("La validación posterior debe mantener juntos los registros de cada grupo.")

    print(
        "\nPCA:"
    )

    print(
        "  Varianza PC1: "
        f"{resumen['pca']['varianza_explicada_componente_1'] * 100:.2f}%"
    )

    print(
        "  Varianza PC2: "
        f"{resumen['pca']['varianza_explicada_componente_2'] * 100:.2f}%"
    )

    print(
        "  Varianza total 2D: "
        f"{resumen['pca']['varianza_explicada_total_2d'] * 100:.2f}%"
    )

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Gráfico:\n  {RUTA_GRAFICO}"
    )

    print(
        f"  Coordenadas:\n  {RUTA_COORDENADAS}"
    )

    print(
        f"  Resumen:\n  {RUTA_RESUMEN}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "La proyección LDA es supervisada y se utiliza "
        "únicamente como visualización exploratoria."
    )

    print(
        "No representa una estimación de generalización "
        "del modelo."
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PROYECCIÓN PCA/LDA GENERADA CORRECTAMENTE"
    )

    print(
        "=" * 72
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None