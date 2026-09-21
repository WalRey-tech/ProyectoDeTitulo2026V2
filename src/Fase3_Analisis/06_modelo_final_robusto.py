# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import argparse
import hashlib
import importlib.metadata
import time
from pathlib import Path
import os
import re
from datetime import datetime, timezone

import ftfy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB




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

RUTA_CORPUS = os.path.join(
    SRC_ROOT,
    "data",
    "processed",
    "perfiles_egreso_etiquetado_v2.csv",
)

RESULTADOS_DIR = os.path.join(
    SRC_ROOT,
    "data",
    "resultados_cientificos",
    "modelo_final_robusto",
)

RUTA_FOLDS = os.path.join(
    RESULTADOS_DIR,
    "metricas_folds_modelo_final.csv",
)

RUTA_PREDICCIONES = os.path.join(
    RESULTADOS_DIR,
    "predicciones_oof_modelo_final.csv",
)

RUTA_REPORTE_CLASES = os.path.join(
    RESULTADOS_DIR,
    "metricas_por_clase_modelo_final.csv",
)

RUTA_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_modelo_final.csv",
)

RUTA_AUDITORIA_MASCARA = os.path.join(
    RESULTADOS_DIR,
    "auditoria_enmascaramiento_modelo_final.csv",
)

RUTA_RESUMEN = os.path.join(
    RESULTADOS_DIR,
    "resumen_modelo_final_robusto.json",
)

RUTA_GRAFICO_FOLDS = os.path.join(
    RESULTADOS_DIR,
    "f1_por_fold_modelo_final.png",
)

RUTA_GRAFICO_MATRIZ = os.path.join(
    RESULTADOS_DIR,
    "matriz_confusion_modelo_final.png",
)


# =============================================================================
# 2. CONFIGURACIÓN CONGELADA
# =============================================================================

SEED = 42
N_SPLITS = 5

CLASES = [
    "Civil",
    "Ejecución",
    "Informática",
]


# =============================================================================
# 3. STOPWORDS
# =============================================================================
#
# Se conserva la misma configuración utilizada en los experimentos
# robustos previos.
# =============================================================================

STOPWORDS_ES = [
    "a", "al", "algo", "algunas", "algunos", "ante", "antes",
    "como", "con", "contra", "cual", "cuando", "de", "del",
    "desde", "donde", "durante", "e", "el", "ella", "ellas",
    "ellos", "en", "entre", "era", "erais", "eran", "eras",
    "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
    "estaba", "estaban", "estado", "estar", "estas", "este",
    "esto", "estos", "estoy", "fue", "fueron", "fui", "ha",
    "han", "has", "hasta", "hay", "he", "hun", "la", "las",
    "le", "les", "lo", "los", "mas", "me", "mi", "mia",
    "mias", "mientras", "mis", "mo", "mucho", "muchos", "muy",
    "más", "mí", "nada", "ni", "no", "nos", "nosotras",
    "nosotros", "nuestra", "nuestras", "nuestro", "nuestros",
    "o", "os", "otra", "otras", "otro", "otros", "para",
    "pero", "poco", "por", "porque", "que", "quien", "quienes",
    "qué", "se", "sea", "seais", "sean", "seas", "ser", "si",
    "sin", "sobre", "sois", "somos", "son", "su", "sus",
    "también", "tanto", "te", "tenemos", "tengo", "ti",
    "tiene", "tienen", "toda", "todas", "todo", "todos",
    "tu", "tus", "un", "una", "unas", "uno", "unos",
    "vosotras", "vosotros", "vuestra", "vuestras", "vuestro",
    "vuestros", "y", "ya", "yo", "él", "ésta", "éstas",
    "éste", "éstos", "última", "últimas", "último", "últimos",
]


TFIDF_CONFIG = {
    "max_features": 400,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.90,
    "sublinear_tf": True,
    "stop_words": STOPWORDS_ES,
}


# =============================================================================
# 4. DENOMINACIONES EXPLÍCITAS DEL GRADO
# =============================================================================
#
# IMPORTANTE:
# Se eliminan expresiones que revelan directamente el nombre de la carrera.
#
# NO se eliminan de manera general palabras como:
#
#   civil
#   ejecución
#   informática
#
# cuando aparecen de forma aislada.
#
# De esta forma evitamos el enmascaramiento agresivo de la prueba de estrés.
# =============================================================================

PATRONES_TITULO = [

    r"\bingenier[ií]a\s+civil"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[ií]a\s+civil\s+inform[aá]tica\b",

    r"\bingenier[ií]a\s+de\s+ejecuci[oó]n"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[ií]a\s+en\s+inform[aá]tica\b",

    r"\bingenier[ií]a\s+inform[aá]tica\b",

    r"\bingenier[oa]\s+civil"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[oa]\s+de\s+ejecuci[oó]n"
    r"(?:\s+en\s+inform[aá]tica)?\b",

    r"\bingenier[oa]\s+en\s+inform[aá]tica\b",
]


# =============================================================================
# 5. UTILIDADES
# =============================================================================

def enmascarar_y_contar(
    texto: str,
) -> tuple[str, int]:

    resultado = ftfy.fix_text(
        str(texto)
    )

    total_reemplazos = 0

    for patron in PATRONES_TITULO:

        resultado, reemplazos = re.subn(
            patron,
            " ",
            resultado,
            flags=re.IGNORECASE,
        )

        total_reemplazos += (
            reemplazos
        )

    resultado = re.sub(
        r"\s+",
        " ",
        resultado,
    )

    return (
        resultado.strip(),
        int(total_reemplazos),
    )


def aplicar_smote(
    X,
    y: np.ndarray,
):

    conteos = pd.Series(
        y
    ).value_counts()

    minimo = int(
        conteos.min()
    )

    if minimo < 2:

        return X, y, None

    k = min(
        2,
        minimo - 1,
    )

    smote = SMOTE(
        k_neighbors=k,
        random_state=SEED,
    )

    X_balanceado, y_balanceado = (
        smote.fit_resample(
            X,
            y,
        )
    )

    return (
        X_balanceado,
        y_balanceado,
        k,
    )


# =============================================================================
# 6. CARGA DEL CORPUS
# =============================================================================

def cargar_corpus() -> pd.DataFrame:
    if not os.path.exists(RUTA_CORPUS):
        raise FileNotFoundError(f"No se encontró el corpus: {RUTA_CORPUS}")
    df = pd.read_csv(RUTA_CORPUS, encoding="utf-8-sig")
    faltantes = {"perfil_egreso", "grado"} - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    if df[["perfil_egreso", "grado"]].isna().any().any():
        raise ValueError("Hay valores vacíos. Auditar el corpus antes de evaluar.")
    for col in ["perfil_egreso", "grado"]:
        df[col] = df[col].map(lambda x: ftfy.fix_text(str(x)).strip())
    if df["perfil_egreso"].eq("").any() or set(df["grado"]) != set(CLASES):
        raise ValueError("Se requieren textos no vacíos y las tres clases oficiales.")
    if int(df["grado"].value_counts().min()) < N_SPLITS:
        raise ValueError("Cada clase necesita al menos 5 documentos para los cinco folds externos.")
    return df.reset_index(drop=True)


# =============================================================================
# 7. PREPARACIÓN AUDITADA
# =============================================================================

def preparar_textos(
    df: pd.DataFrame,
):

    textos_enmascarados = []

    auditoria = []

    for indice, fila in df.iterrows():

        texto_original = str(
            fila[
                "perfil_egreso"
            ]
        )

        (
            texto_enmascarado,
            reemplazos,
        ) = enmascarar_y_contar(
            texto_original
        )

        textos_enmascarados.append(
            texto_enmascarado
        )

        auditoria.append(
            {
                "indice":
                    int(indice),

                "grado":
                    str(
                        fila[
                            "grado"
                        ]
                    ),

                "denominaciones_enmascaradas":
                    int(
                        reemplazos
                    ),

                "texto_modificado":
                    bool(
                        reemplazos > 0
                    ),
            }
        )

    return (
        np.asarray(
            textos_enmascarados,
            dtype=object,
        ),
        pd.DataFrame(
            auditoria
        ),
    )


# =============================================================================
# 8. MÉTRICAS
# =============================================================================

def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    return {

        "F1_macro":
            f1_score(
                y_true,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0,
            ),

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Precision_macro":
            precision_score(
                y_true,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0,
            ),

        "Recall_macro":
            recall_score(
                y_true,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0,
            ),

        "F1_Civil":
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Civil"
                ],
                average="macro",
                zero_division=0,
            ),

        "F1_Ejecucion":
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Ejecución"
                ],
                average="macro",
                zero_division=0,
            ),

        "F1_Informatica":
            f1_score(
                y_true,
                y_pred,
                labels=[
                    "Informática"
                ],
                average="macro",
                zero_division=0,
            ),
    }


# =============================================================================
# 9. GRÁFICO F1 POR FOLD
# =============================================================================

def generar_grafico_folds(
    df_folds: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.bar(
        df_folds[
            "Fold"
        ].astype(str),
        df_folds[
            "F1_macro"
        ],
    )

    ax.axhline(
        df_folds[
            "F1_macro"
        ].mean(),
        linestyle="--",
        label=(
            "Media "
            f"{df_folds['F1_macro'].mean():.4f}"
        ),
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel(
        "Fold"
    )

    ax.set_ylabel(
        "F1-macro"
    )

    ax.set_title(
        (
            "Modelo final robusto — "
            "F1-macro por fold\n"
            "TF-IDF + GWO + SMOTE + ComplementNB"
        )
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_FOLDS,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 10. MATRIZ DE CONFUSIÓN
# =============================================================================

def generar_matriz_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    matriz = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASES,
    )

    normalizada = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASES,
        normalize="true",
    )

    df_matriz = pd.DataFrame(
        matriz,
        index=CLASES,
        columns=CLASES,
    )

    df_matriz.index.name = (
        "Real"
    )

    df_matriz.columns.name = (
        "Predicho"
    )

    df_matriz.to_csv(
        RUTA_MATRIZ,
        encoding="utf-8-sig",
    )


    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    imagen = ax.imshow(
        normalizada,
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        imagen,
        ax=ax,
        label=(
            "Proporción dentro "
            "de la clase real"
        ),
    )

    ax.set_xticks(
        np.arange(
            len(CLASES)
        )
    )

    ax.set_yticks(
        np.arange(
            len(CLASES)
        )
    )

    ax.set_xticklabels(
        CLASES,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        CLASES
    )

    ax.set_xlabel(
        "Clase predicha"
    )

    ax.set_ylabel(
        "Clase real"
    )

    ax.set_title(
        (
            "Matriz de confusión — modelo final robusto\n"
            "Denominaciones explícitas del grado enmascaradas"
        )
    )

    for fila in range(
        len(CLASES)
    ):

        for columna in range(
            len(CLASES)
        ):

            ax.text(
                columna,
                fila,
                (
                    f"{matriz[fila, columna]}\n"
                    f"{normalizada[fila, columna] * 100:.1f}%"
                ),
                ha="center",
                va="center",
                fontweight="bold",
            )

    fig.tight_layout()

    fig.savefig(
        RUTA_GRAFICO_MATRIZ,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 11. FLUJO PRINCIPAL
# =============================================================================

def preparar_validacion_interna(textos, etiquetas, nombres_candidatos):
    """Precalcula TF-IDF por fold interno, sin recibir la prueba externa.

    El espacio de candidatos procede SOLO del entrenamiento externo. Cada
    vectorizador interno aprende su vocabulario e IDF en su propio train;
    la máscara se alinea por nombre de término, nunca por posición de columna.
    El fitness interno sirve para buscar, no para estimar generalización.
    """
    minimo = int(pd.Series(etiquetas).value_counts().min())
    n_splits = min(3, minimo)
    if n_splits < 2:
        raise ValueError('No hay suficientes ejemplos para la validación interna.')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    posicion = {nombre: i for i, nombre in enumerate(nombres_candidatos)}
    particiones = []
    for tr, va in cv.split(textos, etiquetas):
        vec = TfidfVectorizer(**TFIDF_CONFIG)
        X_tr = vec.fit_transform(textos[tr])
        X_va = vec.transform(textos[va])
        indices = np.array([posicion.get(t, -1) for t in vec.get_feature_names_out()])
        particiones.append((X_tr, X_va, etiquetas[tr], etiquetas[va], indices))
    return particiones


def seleccionar_gwo(textos_train, y_train, nombres_candidatos, epochs, poblacion, seed):
    """Busca un subconjunto usando únicamente el entrenamiento externo.

    No lee gwo_features_seleccionadas.csv: se repite la búsqueda en cada fold.
    Un subconjunto vacío se invalida; el completo es válido, porque no debemos
    imponer una reducción si la evaluación interna no la respalda.
    """
    from mealpy import GWO, FloatVar, Problem

    particiones = preparar_validacion_interna(textos_train, y_train, nombres_candidatos)
    n_features = len(nombres_candidatos)
    cache = {}

    def binarizar(solucion):
        return np.abs(2 / np.pi * np.arctan(np.pi / 2 * solucion)) > 0.5

    def fitness(mask):
        if not mask.any():
            return -1.0
        clave = np.packbits(mask).tobytes()
        if clave in cache:
            return cache[clave]
        f1s = []
        for X_tr, X_va, y_tr, y_va, indices in particiones:
            presentes = indices >= 0
            columnas = np.zeros(len(indices), dtype=bool)
            columnas[presentes] = mask[indices[presentes]]
            if not columnas.any():
                # Este candidato no tiene ninguna variable aprendible en ese train.
                f1s.append(0.0)
                continue
            X_r, y_r, _ = aplicar_smote(X_tr[:, columnas], y_tr)
            modelo = ComplementNB(alpha=1.0)
            modelo.fit(X_r, y_r)
            pred = modelo.predict(X_va[:, columnas])
            f1s.append(f1_score(y_va, pred, labels=CLASES,
                               average='macro', zero_division=0))
        valor = float(np.mean(f1s))
        cache[clave] = valor
        return valor

    class ProblemaGWO(Problem):
        def obj_func(self, solution):
            return fitness(binarizar(solution))

    problema = ProblemaGWO(
        bounds=FloatVar(lb=(-6.0,) * n_features, ub=(6.0,) * n_features,
                        name='caracteristicas'),
        minmax='max', log_to=None)
    optimizador = GWO.OriginalGWO(epoch=epochs, pop_size=poblacion)
    inicio = time.perf_counter()
    optimizador.solve(problema, seed=seed)
    mask = binarizar(optimizador.g_best.solution)
    if not mask.any():
        raise RuntimeError('GWO no encontró un subconjunto no vacío.')
    return mask, {
        'fitness_interno': fitness(mask),
        'n_splits_internos': len(particiones),
        'subconjuntos_evaluados': len(cache),
        'segundos': round(time.perf_counter() - inicio, 3),
        'seed_gwo': seed,
        'historial_fitness': [float(v) for v in optimizador.history.list_global_best_fit],
    }


def evaluar_modelos(textos, etiquetas, epochs=100, poblacion=30):
    """CV externa común a GWO y baseline; cada documento se prueba una vez."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.empty(len(etiquetas), dtype=object)
    oof_base = np.empty(len(etiquetas), dtype=object)
    fold_oof = np.zeros(len(etiquetas), dtype=int)
    folds, folds_base, features, auditoria = [], [], [], []
    for fold, (tr, te) in enumerate(cv.split(textos, etiquetas), start=1):
        print(f'Fold externo {fold}/{N_SPLITS}: búsqueda GWO en entrenamiento...', flush=True)
        y_tr, y_te = etiquetas[tr], etiquetas[te]
        vec = TfidfVectorizer(**TFIDF_CONFIG)
        X_tr = vec.fit_transform(textos[tr])
        nombres = vec.get_feature_names_out()
        mask, detalle = seleccionar_gwo(
            textos[tr], y_tr, nombres, epochs, poblacion, SEED + fold)
        # La prueba externa se transforma después de cerrar la selección.
        X_te = vec.transform(textos[te])
        X_r, y_r, k = aplicar_smote(X_tr[:, mask], y_tr)
        modelo = ComplementNB(alpha=1.0)
        modelo.fit(X_r, y_r)
        pred = modelo.predict(X_te[:, mask])
        oof[te] = pred
        fold_oof[te] = fold
        metricas = calcular_metricas(y_te, pred)
        folds.append({
            'Fold': fold, 'N_train': len(tr), 'N_test': len(te),
            'N_features': int(mask.sum()), 'N_features_antes_gwo': len(nombres),
            'Reduccion_pct': 100.0 * (1 - mask.mean()), 'SMOTE_k': k,
            'F1_interno_busqueda': detalle['fitness_interno'], **metricas,
        })
        # Comparador predefinido: mismos textos, vectorizador, folds y clasificador.
        X_b, y_b, k_b = aplicar_smote(X_tr, y_tr)
        baseline = ComplementNB(alpha=1.0)
        baseline.fit(X_b, y_b)
        pred_b = baseline.predict(X_te)
        oof_base[te] = pred_b
        folds_base.append({'Fold': fold, 'N_train': len(tr), 'N_test': len(te),
                           'N_features': len(nombres), 'SMOTE_k': k_b,
                           **calcular_metricas(y_te, pred_b)})
        features.extend({'Fold': fold, 'feature': str(t)} for t in nombres[mask])
        auditoria.append({
            'fold': fold, 'indices_train': tr.tolist(), 'indices_test': te.tolist(),
            'distribucion_train': {str(c): int(n) for c, n in pd.Series(y_tr).value_counts().items()},
            'distribucion_test': {str(c): int(n) for c, n in pd.Series(y_te).value_counts().items()},
            'n_candidatos_train': len(nombres), 'n_seleccionados': int(mask.sum()), **detalle,
        })
        print(f"  GWO F1={metricas['F1_macro']:.4f}; "
              f"baseline F1={folds_base[-1]['F1_macro']:.4f}; "
              f'variables {len(nombres)} -> {int(mask.sum())}', flush=True)
    if not np.all(fold_oof > 0):
        raise RuntimeError('Hay documentos sin predicción externa.')
    return (pd.DataFrame(folds), pd.DataFrame(folds_base), oof, oof_base,
            fold_oof, pd.DataFrame(features), auditoria)


def construir_parser():
    parser = argparse.ArgumentParser(description='GWO anidado + SMOTE + ComplementNB, con baseline.')
    parser.add_argument('--corpus', default=RUTA_CORPUS, help='CSV de perfiles etiquetados.')
    parser.add_argument('--output-dir', default=RESULTADOS_DIR, help='Directorio de resultados.')
    parser.add_argument('--epochs', type=int, default=100, help='Iteraciones GWO (100 por defecto).')
    parser.add_argument('--poblacion', type=int, default=30, help='Lobos GWO (30 por defecto, mínimo 5).')
    return parser


def main(argv=None) -> int:
    global RUTA_CORPUS, RESULTADOS_DIR
    args = construir_parser().parse_args(argv)
    if not 1 <= args.epochs <= 100000 or not 5 <= args.poblacion <= 10000:
        raise ValueError('epochs debe estar entre 1 y 100000; poblacion entre 5 y 10000.')
    # Comprobar la dependencia antes de producir salidas de una ejecución incompleta.
    try:
        import mealpy
    except ImportError as error:
        raise SystemExit('Falta mealpy. Instala las dependencias de requirements_gwo.txt.') from error
    RUTA_CORPUS = os.path.abspath(args.corpus)
    RESULTADOS_DIR = os.path.abspath(args.output_dir)
    # Mantener nombres de salida compatibles con 07_generar_reporte.py.
    for nombre, valor in list(globals().items()):
        if nombre.startswith('RUTA_') and nombre != 'RUTA_CORPUS' and isinstance(valor, str):
            globals()[nombre] = os.path.join(RESULTADOS_DIR, os.path.basename(valor))
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    df = cargar_corpus()
    textos, auditoria_mascara = preparar_textos(df)
    if any(not str(t).strip() for t in textos):
        raise ValueError('Hay textos vacíos después del enmascaramiento. Auditar antes de evaluar.')
    etiquetas = df['grado'].to_numpy()
    print('TF-IDF + GWO + SMOTE + ComplementNB: validación externa de 5 folds', flush=True)
    folds, baseline, oof, oof_base, fold_oof, features, auditoria = evaluar_modelos(
        textos, etiquetas, epochs=args.epochs, poblacion=args.poblacion)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folds['run_id'] = run_id
    baseline['run_id'] = run_id
    folds.to_csv(RUTA_FOLDS, index=False, encoding='utf-8-sig')
    baseline.to_csv(os.path.join(RESULTADOS_DIR, 'metricas_folds_baseline.csv'),
                    index=False, encoding='utf-8-sig')
    features.to_csv(os.path.join(RESULTADOS_DIR, 'features_gwo_por_fold.csv'),
                    index=False, encoding='utf-8-sig')
    auditoria_mascara.to_csv(RUTA_AUDITORIA_MASCARA, index=False, encoding='utf-8-sig')
    pd.DataFrame({'indice_corpus': np.arange(len(df)), 'fold': fold_oof,
                  'grado_real': etiquetas, 'grado_predicho': oof,
                  'grado_predicho_baseline': oof_base, 'correcto': etiquetas == oof,
                  'run_id': run_id}).to_csv(RUTA_PREDICCIONES, index=False, encoding='utf-8-sig')
    reporte = classification_report(etiquetas, oof, labels=CLASES, target_names=CLASES,
                                    output_dict=True, zero_division=0)
    pd.DataFrame(reporte).transpose().to_csv(RUTA_REPORTE_CLASES, encoding='utf-8-sig')
    generar_grafico_folds(folds)
    generar_matriz_confusion(etiquetas, oof)
    f1_media = float(folds['F1_macro'].mean())
    f1_base = float(baseline['F1_macro'].mean())
    delta = f1_media - f1_base
    comparacion = pd.DataFrame({
        'Fold': folds['Fold'], 'F1_GWO': folds['F1_macro'],
        'F1_baseline': baseline['F1_macro'],
        'Delta_GWO_menos_baseline': folds['F1_macro'] - baseline['F1_macro'],
        'N_features_GWO': folds['N_features'], 'N_features_baseline': baseline['N_features'],
        'run_id': run_id})
    comparacion.to_csv(os.path.join(RESULTADOS_DIR, 'comparacion_gwo_baseline.csv'),
                       index=False, encoding='utf-8-sig')
    presupuesto_referencia = args.epochs == 100 and args.poblacion == 30
    resumen = {
        'version': '2.0', 'run_id': run_id,
        'fecha_generacion_utc': datetime.now(timezone.utc).isoformat(),
        'estado': 'evaluado_gwo_anidado',
        'presupuesto_referencia': presupuesto_referencia,
        'objetivo': 'Evaluar GWO dentro del entrenamiento externo y compararlo con el baseline.',
        'corpus': {'total_perfiles': len(df),
                   'distribucion': {str(c): int(n) for c, n in df['grado'].value_counts().items()},
                   'archivo': os.path.basename(RUTA_CORPUS),
                   'sha256': hashlib.sha256(Path(RUTA_CORPUS).read_bytes()).hexdigest()},
        'control_fuga_lexica': {
            'metodo': 'Enmascaramiento de denominaciones explícitas del grado.',
            'documentos_modificados': int(auditoria_mascara['texto_modificado'].sum()),
            'denominaciones_enmascaradas': int(auditoria_mascara['denominaciones_enmascaradas'].sum()),
            'nota': 'No elimina todas las palabras aisladas ni todas las pistas institucionales.'},
        'representacion': {**{k: v for k, v in TFIDF_CONFIG.items() if k != 'stop_words'},
            'metodo': 'TF-IDF', 'stopwords': 'Lista española conservada del modelo anterior.',
            'ajuste': 'TF-IDF externo solo en train externo; cada TF-IDF interno solo en su train.'},
        'modelo': {'clasificador': 'Complement Naive Bayes', 'alpha': 1.0,
                   'seleccion': 'GWO anidado', 'balanceo': 'SMOTE solo en entrenamiento',
                   'smote_k_maximo': 2, 'semilla': SEED},
        'validacion': {'metodo': 'CV externa estratificada 5-fold con búsqueda GWO interna',
            'protocolo_id': 'gwo_nested_v2', 'n_splits': N_SPLITS,
            'n_splits_internos_maximo': 3, 'shuffle': True, 'random_state': SEED,
            'metrica_principal': 'Promedio del F1-macro en los cinco tests externos.',
            'nota_interna': 'El universo de candidatos se define con train externo; los vocabularios e IDF internos se ajustan por train interno. El fitness es de optimización, no una estimación de generalización.',
            'pca_lda': 'Visualizaciones separadas; no intervienen en este clasificador.'},
        'seleccion_gwo': {'algoritmo': 'mealpy.GWO.OriginalGWO',
            'epochs': args.epochs, 'poblacion': args.poblacion,
            'binarizacion': 'V-shape, umbral 0.5',
            'fitness': 'F1-macro interno; vacío inválido y subconjunto completo permitido.',
            'features_por_fold': folds['N_features'].astype(int).tolist(),
            'reduccion_media_pct': float(folds['Reduccion_pct'].mean()),
            'nota': 'No existe un subconjunto global de 249 variables. Cada train produce su selección.'},
        'resultados_principales': {
            'f1_macro_media_folds': f1_media,
            'f1_macro_std_folds': float(folds['F1_macro'].std(ddof=1)),
            'accuracy_media_folds': float(folds['Accuracy'].mean()),
            'f1_macro_oof': float(f1_score(etiquetas, oof, labels=CLASES, average='macro', zero_division=0)),
            'accuracy_oof': float(accuracy_score(etiquetas, oof))},
        'baseline_sin_gwo': {
            'f1_macro_media_folds': f1_base,
            'f1_macro_std_folds': float(baseline['F1_macro'].std(ddof=1)),
            'accuracy_media_folds': float(baseline['Accuracy'].mean()),
            'f1_macro_oof': float(f1_score(etiquetas, oof_base, labels=CLASES, average='macro', zero_division=0)),
            'accuracy_oof': float(accuracy_score(etiquetas, oof_base)),
            'delta_f1_gwo_menos_baseline': delta,
            'comparacion': 'Mismos folds externos y preprocesamiento; comparación descriptiva, sin elegir ganador para volver a estimar sobre estos mismos tests.'},
        'metricas_oof_por_clase': {c: {'precision': reporte[c]['precision'],
            'recall': reporte[c]['recall'], 'f1_score': reporte[c]['f1-score'],
            'support': int(reporte[c]['support'])} for c in CLASES},
        'interpretacion': {
            'resultado_principal': 'Evaluación externa del procedimiento TF-IDF + GWO + SMOTE + ComplementNB.',
            'relacion_con_gwo': 'Los pasos 04/05 mantienen su carácter exploratorio; esta selección se vuelve a ejecutar dentro de cada train externo.',
            'comparacion_baseline': f'Diferencia media GWO menos baseline: {delta:+.4f}; no implica significancia estadística.',
            'limitacion': 'Corpus pequeño y desbalanceado; cinco particiones y enmascaramiento parcial no garantizan generalización a nuevas instituciones.'},
        'versiones_librerias': {p: importlib.metadata.version(p) for p in
            ['numpy', 'pandas', 'scikit-learn', 'imbalanced-learn', 'mealpy', 'ftfy']},
        'artefactos': {'folds': os.path.basename(RUTA_FOLDS),
            'predicciones_oof': os.path.basename(RUTA_PREDICCIONES),
            'metricas_por_clase': os.path.basename(RUTA_REPORTE_CLASES),
            'matriz_confusion_csv': os.path.basename(RUTA_MATRIZ),
            'auditoria_enmascaramiento': os.path.basename(RUTA_AUDITORIA_MASCARA),
            'grafico_folds': os.path.basename(RUTA_GRAFICO_FOLDS),
            'grafico_matriz': os.path.basename(RUTA_GRAFICO_MATRIZ),
            'baseline': 'metricas_folds_baseline.csv', 'features_por_fold': 'features_gwo_por_fold.csv',
            'comparacion': 'comparacion_gwo_baseline.csv', 'auditoria_cv': 'auditoria_cv_gwo.json'},
    }
    with open(os.path.join(RESULTADOS_DIR, 'auditoria_cv_gwo.json'), 'w', encoding='utf-8') as f:
        json.dump({'run_id': run_id, 'folds': auditoria}, f, ensure_ascii=False, indent=2)
    # Escribir el resumen al final: el reporte verifica run_id y hash del corpus.
    with open(RUTA_RESUMEN, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f'F1-macro externo GWO: {f1_media:.4f}; baseline: {f1_base:.4f}; delta: {delta:+.4f}')
    print(f'Resumen: {RUTA_RESUMEN}')
    if not presupuesto_referencia:
        print('Se utilizó un presupuesto de búsqueda distinto de 100 iteraciones y 30 lobos.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
