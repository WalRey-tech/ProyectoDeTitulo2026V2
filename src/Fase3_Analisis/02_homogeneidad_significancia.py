# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def configurar_corpus(corpus: str) -> None:
    global CORPUS_SELECCIONADO, RUTA_ENTRADA, RESULTADOS_DIR
    global RUTA_MATRIZ_CENTROIDES, RUTA_HOMOGENEIDAD_CLASE, RUTA_RESUMEN
    global RUTA_GRAFICO_CENTROIDES, RUTA_GRAFICO_PERMUTACION, RUTA_GRUPOS, RUTA_PERMUTACIONES
    if corpus not in {"actual", "v2"}:
        raise ValueError("Corpus no válido: usa actual o v2.")
    CORPUS_SELECCIONADO = corpus
    nombre = ("perfiles_egreso_etiquetado_actual_corregido.csv"
              if corpus == "actual" else "perfiles_egreso_etiquetado_v2.csv")
    RUTA_ENTRADA = os.path.join(SRC_ROOT, "data", "processed", nombre)
    RESULTADOS_DIR = os.path.join(SRC_ROOT, "data", "resultados_cientificos", "homogeneidad_semantica")
    def ruta(nombre, extension):
        return os.path.join(RESULTADOS_DIR, f"{nombre}_{corpus}.{extension}")
    RUTA_MATRIZ_CENTROIDES = ruta("similitud_centroides", "csv")
    RUTA_HOMOGENEIDAD_CLASE = ruta("homogeneidad_por_clase", "csv")
    RUTA_RESUMEN = ruta("resumen_homogeneidad", "json")
    RUTA_GRAFICO_CENTROIDES = ruta("similitud_centroides", "png")
    RUTA_GRAFICO_PERMUTACION = ruta("test_permutacion_homogeneidad", "png")
    RUTA_GRUPOS = ruta("unidades_test_homogeneidad", "csv")
    RUTA_PERMUTACIONES = ruta("distribucion_nula_homogeneidad", "csv")


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

N_PERMUTACIONES = 5000

TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
}


# =============================================================================
# 3. CARGA Y VALIDACIÓN
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
    desconocidas = set(df["grado"]) - {"Civil", "Informática", "Ejecución"}
    if desconocidas:
        raise ValueError(f"Grados fuera del catálogo: {sorted(desconocidas)}")
    for columna in ["estado_registro", "estado_etiquetado"]:
        if columna in df:
            pendientes = df[columna].astype(str).str.strip().str.upper().isin(["REVISAR", "ERROR"])
            if pendientes.any():
                raise ValueError(f"El CSV contiene filas REVISAR/ERROR en {columna}.")
    conteos = df["grado"].value_counts()
    if len(conteos) != 3 or conteos.min() < 2:
        raise ValueError("El análisis requiere las tres clases, con al menos dos perfiles cada una.")
    df.attrs["sha256_entrada"] = huella
    return df


# =============================================================================
# 4. VECTORIZACIÓN
# =============================================================================

def vectorizar(
    textos: pd.Series,
):

    vectorizador = TfidfVectorizer(
        **TFIDF_CONFIG
    )

    X = vectorizador.fit_transform(
        textos.astype(str)
    )

    if np.any(np.asarray(X.getnnz(axis=1)) == 0):
        raise ValueError("TF-IDF dejó perfiles sin términos; revisa el corpus o su configuración.")
    return vectorizador, X.toarray()


# =============================================================================
# 5. SIMILITUD INTRA / INTER
# =============================================================================

def obtener_similitudes_intra_inter(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
):

    n = len(etiquetas)

    indices = np.triu_indices(
        n,
        k=1,
    )

    similitudes = matriz_similitud[
        indices
    ]

    mismas_clases = (
        etiquetas[:, None]
        == etiquetas[None, :]
    )[indices]

    intra = similitudes[
        mismas_clases
    ]

    inter = similitudes[
        ~mismas_clases
    ]

    if not len(intra) or not len(inter):
        raise ValueError("No hay suficientes pares intra/inter para calcular el estadístico.")
    return intra, inter


def calcular_estadistico(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
) -> float:

    intra, inter = obtener_similitudes_intra_inter(
        matriz_similitud,
        etiquetas,
    )

    return float(
        intra.mean()
        - inter.mean()
    )


# =============================================================================
# 6. HOMOGENEIDAD POR CLASE
# =============================================================================

def calcular_homogeneidad_por_clase(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
) -> pd.DataFrame:

    resultados = []

    clases = sorted(
        np.unique(etiquetas)
    )

    for clase in clases:

        indices = np.where(
            etiquetas == clase
        )[0]

        n = len(indices)

        submatriz = matriz_similitud[
            np.ix_(
                indices,
                indices,
            )
        ]

        pares = np.triu_indices(
            n,
            k=1,
        )

        if len(pares[0]) > 0:
            valores = submatriz[
                pares
            ]

            media = float(
                valores.mean()
            )

            desviacion = float(
                valores.std(
                    ddof=1
                )
            ) if len(valores) > 1 else 0.0

            n_pares = int(
                len(valores)
            )

        else:
            media = float(
                "nan"
            )

            desviacion = float(
                "nan"
            )

            n_pares = 0

        resultados.append(
            {
                "grado": clase,
                "n_perfiles": int(n),
                "n_pares": n_pares,
                "similitud_intra_media":
                    media,
                "similitud_intra_std":
                    desviacion,
            }
        )

    return pd.DataFrame(
        resultados
    )


# =============================================================================
# 7. CENTROIDES
# =============================================================================

def calcular_centroides(
    X: np.ndarray,
    etiquetas: np.ndarray,
):

    clases = sorted(
        np.unique(etiquetas)
    )

    centroides = []

    for clase in clases:

        vectores = X[
            etiquetas == clase
        ]

        centroides.append(
            vectores.mean(
                axis=0
            )
        )

    centroides = np.asarray(
        centroides
    )

    matriz = cosine_similarity(
        centroides
    )

    return (
        clases,
        centroides,
        matriz,
    )


def preparar_unidades_test(df: pd.DataFrame, X: np.ndarray):
    """Actual: un vector promedio por grupo; V2: esquema histórico por perfil.

    La agrupación se define sin usar el resultado del test. Solo se comprueba
    que cada grupo tenga una etiqueta coherente. TF-IDF se ajusta al corpus
    completo (descriptivo), después se promedian los vectores de cada grupo.
    La permutación asume intercambiabilidad entre las unidades resultantes;
    agrupar dependencias conocidas no demuestra independencia entre todas ellas.
    """
    n = len(df)
    padres = list(range(n))
    def raiz(i):
        while padres[i] != i:
            padres[i] = padres[padres[i]]
            i = padres[i]
        return i
    def unir(i, j):
        padres[raiz(j)] = raiz(i)
    motivos = [[] for _ in range(n)]
    if CORPUS_SELECCIONADO == "actual":
        vistos_grupo, vistos_texto = {}, {}
        for i, (_, fila) in enumerate(df.iterrows()):
            grupo = str(fila.get("grupo_perfil", "")).strip()
            url = urlparse(str(fila.get("url", "")))
            # Recupera el vínculo conocido si el CSV no trae grupo_perfil.
            ucsc = (url.hostname in {"it.ucsc.cl", "advance.ucsc.cl"}
                    and url.path.rstrip("/") == "/carreras/ingenieria-de-ejecucion-en-informatica")
            claves = ([grupo] if grupo else [])
            if ucsc:
                claves.append("ucsc_ejecucion_informatica")
                motivos[i].append("modalidades UCSC documentadas")
            for clave in claves:
                if clave in vistos_grupo:
                    unir(vistos_grupo[clave], i)
                else:
                    vistos_grupo[clave] = i
                motivos[i].append("grupo:" + clave)
            texto = " ".join(str(fila["perfil_egreso"]).casefold().split())
            if texto in vistos_texto:
                unir(vistos_texto[texto], i)
                motivos[i].append("texto duplicado normalizado")
            else:
                vistos_texto[texto] = i
    miembros = {}
    for i in range(n):
        miembros.setdefault(raiz(i), []).append(i)
    vectores, etiquetas, filas = [], [], []
    for numero, indices in enumerate(miembros.values(), 1):
        clases = df.iloc[indices]["grado"].unique()
        if len(clases) != 1:
            raise ValueError(f"El grupo con filas {[i+1 for i in indices]} mezcla grados; revisa las etiquetas.")
        vectores.append(X[indices].mean(axis=0))
        etiquetas.append(clases[0])
        for i in indices:
            fila = df.iloc[i]
            filas.append({"unidad_test": f"unidad_{numero:03d}", "fila_datos": i+1,
                          "n_perfiles_unidad": len(indices), "grado": clases[0],
                          "universidad": fila.get("universidad", ""),
                          "carrera": fila.get("carrera", ""), "url": fila.get("url", ""),
                          "grupo_perfil_original": fila.get("grupo_perfil", ""),
                          "criterio_agrupacion": "; ".join(motivos[i]) or "perfil individual"})
    y = np.asarray(etiquetas)
    conteos = pd.Series(y).value_counts()
    if len(conteos) != 3 or conteos.min() < 2:
        raise ValueError("El test requiere al menos dos unidades por grado después de agrupar.")
    auditoria = pd.DataFrame(filas).sort_values("fila_datos")
    meta = {"unidad": "grupo de perfiles" if CORPUS_SELECCIONADO == "actual" else "perfil (histórico)",
            "n_unidades": len(y), "distribucion_unidades": {str(k): int(v) for k,v in conteos.items()},
            "grupos_con_varios_perfiles": sum(len(v)>1 for v in miembros.values()),
            "vector_por_unidad": "media de los vectores TF-IDF de sus perfiles",
            "tfidf_ajustado_sobre": "todos los perfiles de entrada, sin etiquetas",
            "supuesto": "Intercambiabilidad de etiquetas entre unidades bajo H0; no garantiza independencia institucional."}
    return np.asarray(vectores), y, auditoria, meta


# =============================================================================
# 8. TEST DE PERMUTACIÓN
# =============================================================================

def ejecutar_test_permutacion(
    matriz_similitud: np.ndarray,
    etiquetas: np.ndarray,
):

    if N_PERMUTACIONES < 1:
        raise ValueError("El número de permutaciones debe ser positivo.")
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    observado = calcular_estadistico(
        matriz_similitud,
        etiquetas,
    )

    permutados = np.empty(
        N_PERMUTACIONES,
        dtype=float,
    )

    for i in range(
        N_PERMUTACIONES
    ):

        etiquetas_permutadas = (
            rng.permutation(
                etiquetas
            )
        )

        permutados[i] = calcular_estadistico(
            matriz_similitud,
            etiquetas_permutadas,
        )

    # Test unilateral:
    # H1: similitud intra-clase > similitud inter-clase.
    #
    # La corrección +1 evita reportar un p-valor
    # exactamente igual a cero.
    extremos = int(
        np.sum(
            permutados
            >= observado - 100 * np.finfo(float).eps * abs(observado)
        )
    )

    p_valor = (
        extremos + 1
    ) / (
        N_PERMUTACIONES + 1
    )

    return (
        observado,
        permutados,
        extremos,
        float(p_valor),
    )


# =============================================================================
# 9. GRÁFICO DE CENTROIDES
# =============================================================================

def generar_grafico_centroides(
    matriz: np.ndarray,
    clases: list[str],
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    imagen = ax.imshow(
        matriz,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )

    ax.set_xticks(
        np.arange(
            len(clases)
        )
    )

    ax.set_yticks(
        np.arange(
            len(clases)
        )
    )

    ax.set_xticklabels(
        clases,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        clases
    )

    for i in range(
        len(clases)
    ):

        for j in range(
            len(clases)
        ):

            ax.text(
                j,
                i,
                f"{matriz[i, j]:.3f}",
                ha="center",
                va="center",
            )

    ax.set_title(
        (
            "Similitud coseno entre centroides\n"
            f"Perfiles de egreso — Corpus {CORPUS_SELECCIONADO.upper()}"
        ),
        fontweight="bold",
    )

    barra = fig.colorbar(
        imagen,
        ax=ax,
    )

    barra.set_label(
        "Similitud coseno"
    )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_CENTROIDES,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 10. GRÁFICO DEL TEST DE PERMUTACIÓN
# =============================================================================

def generar_grafico_permutacion(
    observado: float,
    permutados: np.ndarray,
    p_valor: float,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        permutados,
        bins=40,
        alpha=0.8,
        edgecolor="black",
    )

    ax.axvline(
        observado,
        linestyle="--",
        linewidth=2,
        label=(
            "Δ observado = "
            f"{observado:.4f}"
        ),
    )

    ax.set_title(
        (
            "Test de permutación — "
            f"Homogeneidad ({CORPUS_SELECCIONADO.upper()}; unidades del test)"
        ),
        fontweight="bold",
    )

    ax.set_xlabel(
        (
            "Δ similitud "
            "(intra-clase − inter-clase)"
        )
    )

    ax.set_ylabel(
        "Frecuencia"
    )

    ax.legend()

    ax.text(
        0.98,
        0.95,
        f"p = {p_valor:.5f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.15,
        },
    )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_PERMUTACION,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 11. RESUMEN JSON
# =============================================================================

def guardar_resumen(
    df: pd.DataFrame,
    vectorizador: TfidfVectorizer,
    matriz_centroides: np.ndarray,
    clases: list[str],
    intra: np.ndarray,
    inter: np.ndarray,
    observado: float,
    extremos: int,
    p_valor: float,
    homogeneidad_clase: pd.DataFrame,
    observado_test: float,
    metadatos_test: dict,
) -> dict:

    distribucion = (
        df["grado"]
        .value_counts()
        .to_dict()
    )

    matriz_dict = {}

    for i, clase_i in enumerate(
        clases
    ):

        matriz_dict[
            clase_i
        ] = {}

        for j, clase_j in enumerate(
            clases
        ):

            matriz_dict[
                clase_i
            ][
                clase_j
            ] = round(
                float(
                    matriz_centroides[
                        i,
                        j,
                    ]
                ),
                6,
            )

    macro_intra = float(
        homogeneidad_clase[
            "similitud_intra_media"
        ].mean()
    )

    significativo = bool(
        p_valor < 0.05
    )

    resumen = {

        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "seleccion": CORPUS_SELECCIONADO,
            "archivo_entrada": RUTA_ENTRADA,
            "sha256": df.attrs["sha256_entrada"],
            "filas_descartadas": 0,
            "total": int(
                len(df)
            ),

            "distribucion": {
                str(clase):
                    int(cantidad)

                for clase, cantidad
                in distribucion.items()
            },
        },

        "representacion": {
            "metodo":
                "TF-IDF",

            "max_features":
                400,

            "features_generadas":
                int(
                    len(
                        vectorizador
                        .get_feature_names_out()
                    )
                ),

            "ngram_range": [
                1,
                2,
            ],

            "min_df":
                2,

            "max_df":
                0.90,

            "sublinear_tf":
                True,
        },

        "similitud_global": {

            "intra_clase_media":
                round(
                    float(
                        intra.mean()
                    ),
                    6,
                ),

            "inter_clase_media":
                round(
                    float(
                        inter.mean()
                    ),
                    6,
                ),

            "diferencia_intra_inter":
                round(
                    float(
                        observado
                    ),
                    6,
                ),

            "homogeneidad_intra_macro":
                round(
                    macro_intra,
                    6,
                ),
        },

        "test_permutacion": {
            "agrupacion": metadatos_test,
            "estadistico_observado_unidades": float(observado_test),
            "estadistico": "media de similitudes intra menos media inter entre unidades",
            "recuentos_preservados": "número de unidades por grado",
            "correccion_p_valor": "(extremas + 1) / (permutaciones + 1)",
            "referencia_calculo_p": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html",
            "archivo_asignacion_unidades": RUTA_GRUPOS,
            "archivo_distribucion_nula": RUTA_PERMUTACIONES,

            "hipotesis_nula":
                (
                    "La asociación entre unidades del test y "
                    "grados no produce una diferencia "
                    "intra/inter mayor a la esperada "
                    "por azar."
                ),

            "hipotesis_alternativa":
                (
                    "Entre unidades del test, la similitud intra-grado es mayor "
                    "que la similitud inter-grado."
                ),

            "tipo":
                "unilateral",

            "n_permutaciones":
                N_PERMUTACIONES,

            "semilla":
                RANDOM_STATE,

            "permutaciones_extremas":
                extremos,

            "p_valor":
                round(
                    p_valor,
                    8,
                ),

            "alpha":
                0.05,

            "significativo":
                significativo,
        },

        "similitud_centroides":
            matriz_dict,

        "interpretacion": {

            "resultado":
                (
                    "Los perfiles del mismo grado "
                    "presentan una similitud promedio "
                    "mayor que los perfiles de grados "
                    "diferentes."
                    if observado > 0
                    else
                    "No se observó una similitud "
                    "intra-grado superior a la "
                    "similitud inter-grado."
                ),

            "significancia":
                (
                    "La diferencia entre unidades del test es "
                    "estadísticamente significativa "
                    "bajo el test de permutación."
                    if significativo
                    else
                    "La diferencia entre unidades del test no "
                    "alcanza significancia estadística "
                    "bajo el test de permutación."
                ),

            "advertencia":
                (
                    "Este es un análisis descriptivo "
                    "y de asociación del corpus. "
                    "No constituye una estimación "
                    "de rendimiento predictivo."
                ),
        },

        "limitaciones": [

            (
                f"La clase Ejecución contiene {distribucion.get('Ejecución', 0)} perfiles "
                f"y {metadatos_test['distribucion_unidades'].get('Ejecución', 0)} unidades del test; sus "
                "estimaciones presentan mayor "
                "incertidumbre."
            ),

            (
                "El corpus utiliza el texto original "
                "de los perfiles de egreso. Algunos "
                "documentos pueden contener términos "
                "directamente asociados al nombre "
                "del programa o grado."
            ),

            (
                "La similitud global intra-clase está "
                "influida por el número de pares de "
                "cada clase; por ello también se "
                "reporta homogeneidad por clase."
            ),
        ],
    }

    resumen["limitaciones"].extend([
        "Las medias descriptivas usan todos los perfiles; el p-valor corresponde al estadístico entre unidades agrupadas.",
        "Los pares de similitudes comparten perfiles: no son observaciones independientes. Se permutan etiquetas de unidades, no pares.",
        "Solo se agrupan vínculos declarados, modalidades UCSC conocidas y duplicados textuales normalizados. Pueden existir otras dependencias.",
        "TF-IDF mide coincidencia léxica ponderada, no equivalencia semántica completa.",
    ])
    if sha256_archivo(RUTA_ENTRADA) != df.attrs["sha256_entrada"]:
        raise ValueError("El archivo de entrada cambió durante el análisis; repite la ejecución.")
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
# 12. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:
    global N_PERMUTACIONES
    parser = argparse.ArgumentParser(description="Homogeneidad del corpus y permutación por unidades.")
    parser.add_argument("--corpus", choices=["actual", "v2"], help="Si se omite, muestra un menú.")
    parser.add_argument("--permutaciones", type=int, default=5000)
    args = parser.parse_args()
    if args.permutaciones < 1:
        parser.error("--permutaciones debe ser positivo")
    N_PERMUTACIONES = args.permutaciones
    configurar_corpus(args.corpus or solicitar_corpus())

    print(
        "=" * 72
    )

    print(
        f"HOMOGENEIDAD Y SIGNIFICANCIA — {CORPUS_SELECCIONADO.upper()}"
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

    etiquetas = (
        df["grado"]
        .astype(str)
        .to_numpy()
    )

    print(
        f"\nCorpus: {len(df)} perfiles"
    )

    print(
        "Distribución:"
    )

    print(
        df["grado"]
        .value_counts()
    )

    vectorizador, X = vectorizar(
        df["perfil_egreso"]
    )

    print(
        f"\nTF-IDF generado: "
        f"{X.shape[1]} características"
    )

    X_test, etiquetas_test, auditoria_grupos, metadatos_test = preparar_unidades_test(df, X)
    auditoria_grupos.to_csv(RUTA_GRUPOS, sep=";", index=False, encoding="utf-8-sig")
    matriz_test = cosine_similarity(X_test)
    print(f"Unidades del test: {len(etiquetas_test)}; distribución: {metadatos_test['distribucion_unidades']}")
    print("Las descripciones usan todos los perfiles; el test usa un vector por unidad.")

    matriz_similitud = cosine_similarity(
        X
    )

    intra, inter = obtener_similitudes_intra_inter(
        matriz_similitud,
        etiquetas,
    )

    clases, _, matriz_centroides = calcular_centroides(
        X,
        etiquetas,
    )

    df_centroides = pd.DataFrame(
        matriz_centroides,
        index=clases,
        columns=clases,
    )

    df_centroides.index.name = (
        "grado"
    )

    df_centroides.to_csv(
        RUTA_MATRIZ_CENTROIDES,
        encoding="utf-8-sig",
    )

    homogeneidad_clase = calcular_homogeneidad_por_clase(
        matriz_similitud,
        etiquetas,
    )

    homogeneidad_clase.to_csv(
        RUTA_HOMOGENEIDAD_CLASE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nSimilitud entre centroides:"
    )

    print(
        df_centroides.round(
            4
        )
    )

    print(
        "\nHomogeneidad por clase:"
    )

    print(
        homogeneidad_clase.round(
            4
        ).to_string(
            index=False
        )
    )

    print(
        "\nEjecutando test de permutación..."
    )

    observado = calcular_estadistico(matriz_similitud, etiquetas)
    observado_test, permutados, extremos, p_valor = ejecutar_test_permutacion(matriz_test, etiquetas_test)
    pd.DataFrame({"permutacion": np.arange(1, len(permutados)+1),
                  "delta_unidades": permutados}).to_csv(RUTA_PERMUTACIONES, index=False, encoding="utf-8-sig")

    generar_grafico_centroides(
        matriz_centroides,
        clases,
    )

    generar_grafico_permutacion(
        observado_test,
        permutados,
        p_valor,
    )

    resumen = guardar_resumen(
        df,
        vectorizador,
        matriz_centroides,
        clases,
        intra,
        inter,
        observado,
        extremos,
        p_valor,
        homogeneidad_clase,
        observado_test,
        metadatos_test,
    )

    print(
        "\nResultados globales:"
    )

    print(
        "  Similitud intra-clase media : "
        f"{intra.mean():.4f}"
    )

    print(
        "  Similitud inter-clase media : "
        f"{inter.mean():.4f}"
    )

    print(
        "  Δ intra - inter             : "
        f"{observado:.4f}"
    )

    print(
        "  Homogeneidad intra macro    : "
        f"{resumen['similitud_global']['homogeneidad_intra_macro']:.4f}"
    )

    print(f"\nΔ entre unidades del test: {observado_test:.6f}")
    print(
        "\nTest de permutación:"
    )

    print(
        f"  Permutaciones : "
        f"{N_PERMUTACIONES}"
    )

    print(
        f"  Extremas      : "
        f"{extremos}"
    )

    print(
        f"  p-valor       : "
        f"{p_valor:.6f}"
    )

    print(
        "  Significativo : "
        f"{'Sí' if p_valor < 0.05 else 'No'}"
    )

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Matriz centroides:\n  "
        f"{RUTA_MATRIZ_CENTROIDES}"
    )

    print(
        f"  Homogeneidad por clase:\n  "
        f"{RUTA_HOMOGENEIDAD_CLASE}"
    )

    print(
        f"  Resumen:\n  "
        f"{RUTA_RESUMEN}"
    )

    print(
        f"  Gráfico centroides:\n  "
        f"{RUTA_GRAFICO_CENTROIDES}"
    )

    print(
        f"  Gráfico permutación:\n  "
        f"{RUTA_GRAFICO_PERMUTACION}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "Este análisis describe estructura semántica "
        "del corpus; no mide rendimiento predictivo."
    )

    print(
        "El texto original puede contener términos "
        "asociados directamente al nombre del grado."
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ANÁLISIS DE HOMOGENEIDAD COMPLETADO CORRECTAMENTE"
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