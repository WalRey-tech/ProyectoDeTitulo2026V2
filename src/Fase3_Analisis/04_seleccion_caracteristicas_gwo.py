# -*- coding: utf-8 -*-
"""GWO exploratorio: selección TF-IDF con SMOTE y ComplementNB.

Instalación: python -m pip install mealpy imbalanced-learn scikit-learn pandas matplotlib
Entorno probado: Python 3.12, numpy 1.26.0, scikit-learn 1.7.2,
imbalanced-learn 0.14.0, mealpy 3.0.3. El resumen registra las versiones usadas.
Ejecutar desde src/Fase3_Analisis:
    python 04_seleccion_caracteristicas_gwo.py --corpus actual
Prueba rápida del funcionamiento (no resultados finales): --epochs 3 --poblacion 6

TF-IDF se ajusta a todo el corpus para mantener un vocabulario común de selección.
Los F1 del optimizador y de la comparación posterior son EXPLORATORIOS:
la selección ya utilizó las etiquetas y el vocabulario contiene información global.
Este script no hace validación anidada ni produce una estimación independiente.

GWO continuo (mealpy.OriginalGWO) + transformación V y umbral determinista 0.5.
No se implementa una transición binaria probabilística ni inversión de bits.
Las particiones estratifican GRUPOS homogéneos en grado y luego expanden a filas.
SMOTE se aplica únicamente al entrenamiento; k=min(2, mínimo por clase - 1).
Los términos de grado/institución se conservan, como en el método de entrada.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB

SRC_ROOT = str(Path(__file__).resolve().parent.parent)
SEED = 42
CLASES = ["Civil", "Ejecución", "Informática"]
TFIDF_CONFIG = dict(max_features=400, ngram_range=(1, 2), min_df=2,
                    max_df=0.9, sublinear_tf=True)
CORPUS_SELECCIONADO = "actual"


def configurar_corpus(corpus: str) -> None:
    global CORPUS_SELECCIONADO, RUTA_ENTRADA, OUT_DIR
    if corpus not in {"actual", "v2"}:
        raise ValueError("Corpus no válido: usa actual o v2.")
    CORPUS_SELECCIONADO = corpus
    archivo = ("perfiles_egreso_etiquetado_actual_corregido.csv"
               if corpus == "actual" else "perfiles_egreso_etiquetado_v2.csv")
    RUTA_ENTRADA = os.path.join(SRC_ROOT, "data", "processed", archivo)
    OUT_DIR = Path(SRC_ROOT) / "data" / "resultados_cientificos" / "gwo" / corpus


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


STOPWORDS_ES = [
    'a','al','algo','algunas','algunos','ante','antes','como','con','contra',
    'cual','cuando','de','del','desde','donde','durante','e','el','ella',
    'ellas','ellos','en','entre','era','erais','eran','eras','eres','es',
    'esa','esas','ese','eso','esos','esta','estaba','estaban','estado',
    'estar','estas','este','esto','estos','estoy','fue','fueron','fui',
    'ha','han','has','hasta','hay','he','hun','la','las','le','les','lo',
    'los','mas','me','mi','mia','mias','mientras','mis','mo','mucho',
    'muchos','muy','más','mí','nada','ni','no','nos','nosotras','nosotros',
    'nuestra','nuestras','nuestro','nuestros','o','os','otra','otras','otro',
    'otros','para','pero','poco','por','porque','que','quien','quienes',
    'qué','se','sea','seais','sean','seas','ser','si','sin','sobre','sois',
    'somos','son','su','sus','también','tanto','te','tenemos','tengo','ti',
    'tiene','tienen','toda','todas','todo','todos','tu','tus','un','una',
    'unas','uno','unos','vosotras','vosotros','vuestra','vuestras','vuestro',
    'vuestros','y','ya','yo','él','ésta','éstas','éste','éstos','última',
    'últimas','último','últimos',
]


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
    desconocidas = set(df["grado"]) - set(CLASES)
    if desconocidas:
        raise ValueError(f"Grados fuera del catálogo: {sorted(desconocidas)}")
    for columna in ["estado_registro", "estado_etiquetado"]:
        if columna in df:
            pendientes = df[columna].astype(str).str.strip().str.upper().isin(["REVISAR", "ERROR"])
            if pendientes.any():
                raise ValueError(f"El CSV contiene filas REVISAR/ERROR en {columna}.")
    conteos = df["grado"].value_counts()
    if len(conteos) != 3 or conteos.min() < 2:
        raise ValueError("GWO requiere las tres clases, con al menos dos perfiles cada una.")
    df.attrs["sha256_entrada"] = huella
    return df




def construir_grupos(df: pd.DataFrame):
    """Une grupos declarados, el par UCSC conocido y duplicados textuales.

    No usa similitud ajustada a etiquetas ni resultados del optimizador.
    Conserva cada perfil; solo impide separar sus grupos entre train y test.
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
    vistos_grupos, vistos_textos = {}, {}
    criterios = [[] for _ in range(n)]
    for i, (_, fila) in enumerate(df.iterrows()):
        grupo = str(fila.get("grupo_perfil", "")).strip()
        claves = [grupo] if grupo else []
        url = urlparse(str(fila.get("url", "")))
        if (url.hostname in {"it.ucsc.cl", "advance.ucsc.cl"}
                and url.path.rstrip("/") == "/carreras/ingenieria-de-ejecucion-en-informatica"):
            claves.append("ucsc_ejecucion_informatica")
        for clave in claves:
            if clave in vistos_grupos:
                unir(vistos_grupos[clave], i)
            else:
                vistos_grupos[clave] = i
            criterios[i].append("grupo:" + clave)
        texto = " ".join(str(fila["perfil_egreso"]).casefold().split())
        if texto in vistos_textos:
            unir(vistos_textos[texto], i)
            criterios[i].append("duplicado textual normalizado")
        else:
            vistos_textos[texto] = i
    nombres = {}
    grupos = np.asarray([nombres.setdefault(raiz(i), f"grupo_{len(nombres)+1:03d}")
                         for i in range(n)])
    auditoria = pd.DataFrame({"fila_datos": np.arange(1, n+1), "grupo_cv": grupos,
                             "grado": df.grado.to_numpy(),
                             "criterio": ["; ".join(c) or "perfil individual" for c in criterios]})
    for columna in ["indice_fuente", "universidad", "carrera", "url", "modalidad", "grupo_perfil"]:
        if columna in df:
            auditoria[columna] = df[columna].to_numpy()
    if auditoria.groupby("grupo_cv")["grado"].nunique().gt(1).any():
        raise ValueError("Un grupo reúne perfiles con distintos grados. Revisa su etiquetado.")
    return grupos, auditoria


def crear_particiones(y: np.ndarray, grupos: np.ndarray, solicitadas: int, semilla=SEED):
    """Estratifica IDs únicos por grado y expande a documentos sin separarlos."""
    tabla = pd.DataFrame({"grupo": grupos, "grado": y})
    if tabla.groupby("grupo").grado.nunique().gt(1).any():
        raise ValueError("Cada grupo debe tener un único grado.")
    tabla = tabla.drop_duplicates("grupo").reset_index(drop=True)
    cantidades = tabla.grado.value_counts()
    if set(cantidades.index) != set(CLASES):
        raise ValueError("Falta alguna clase en las unidades de validación.")
    n_splits = min(solicitadas, int(cantidades.min()))
    if n_splits < 2:
        raise ValueError("No hay grupos suficientes para validación estratificada.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    particiones, detalles = [], []
    for fold, (trg, teg) in enumerate(cv.split(tabla.grupo, tabla.grado), 1):
        tr = np.flatnonzero(np.isin(grupos, tabla.grupo.iloc[trg]))
        te = np.flatnonzero(np.isin(grupos, tabla.grupo.iloc[teg]))
        if set(grupos[tr]) & set(grupos[te]):
            raise ValueError("Hay grupos compartidos entre entrenamiento y prueba.")
        if set(y[tr]) != set(CLASES) or set(y[te]) != set(CLASES):
            raise ValueError("Una partición no contiene todas las clases.")
        conteos = pd.Series(y[tr]).value_counts()
        k = min(2, int(conteos.min())-1)
        if k < 1:
            raise ValueError("Un entrenamiento tiene menos de dos perfiles por clase; SMOTE no es viable.")
        particiones.append((tr, te))
        detalles.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                         "grados_train": {str(a):int(b) for a,b in conteos.items()},
                         "grados_test": {str(a):int(b) for a,b in pd.Series(y[te]).value_counts().items()},
                         "grupos_train": sorted(set(grupos[tr])), "grupos_test": sorted(set(grupos[te])),
                         "smote_k": k})
    return particiones, detalles


def evaluar_mascara(X, y, mascara, particiones):
    from imblearn.over_sampling import SMOTE
    mascara = np.asarray(mascara, dtype=bool)
    if mascara.shape != (X.shape[1],) or not mascara.any():
        raise ValueError("La máscara debe seleccionar al menos una característica válida.")
    X_sel = X[:, mascara]
    puntuaciones = []
    for tr, te in particiones:
        minimo = int(pd.Series(y[tr]).value_counts().min())
        if minimo < 2:
            raise ValueError("SMOTE necesita al menos dos ejemplos de cada clase en train.")
        smote = SMOTE(k_neighbors=min(2, minimo-1), random_state=SEED)
        X_r, y_r = smote.fit_resample(X_sel[tr], y[tr])
        clasificador = ComplementNB()
        clasificador.fit(X_r, y_r)
        puntuaciones.append(f1_score(y[te], clasificador.predict(X_sel[te]),
                                     labels=CLASES, average="macro", zero_division=0))
    return np.asarray(puntuaciones)


def binarizar(posicion):
    """Umbral V determinista compartido entre fitness y solución guardada."""
    return np.abs((2 / np.pi) * np.arctan((np.pi / 2) * np.asarray(posicion))) > 0.5


def optimizar(X, y, particiones, epochs, poblacion):
    from mealpy import GWO, FloatVar, Problem
    n_features = X.shape[1]
    cache = {}
    class SeleccionGWO(Problem):
        def obj_func(self, solution):
            mascara = binarizar(solution)
            n = int(mascara.sum())
            if n == 0 or n == n_features:
                return 0.0
            clave = mascara.tobytes()
            if clave not in cache:
                cache[clave] = float(evaluar_mascara(X, y, mascara, particiones).mean())
            return cache[clave]
    problema = SeleccionGWO(bounds=FloatVar(lb=(-6.,)*n_features, ub=(6.,)*n_features,
                                          name="features"), minmax="max", log_to=None)
    optimizador = GWO.OriginalGWO(epoch=epochs, pop_size=poblacion)
    inicio = time.perf_counter()
    optimizador.solve(problema, seed=SEED)
    mascara = binarizar(optimizador.g_best.solution)
    if not 0 < mascara.sum() < n_features:
        raise ValueError("GWO no encontró una selección reducida válida. Revisa datos o aumenta epochs/población.")
    fitness = float(optimizador.g_best.target.fitness)
    historial = np.asarray(optimizador.history.list_global_best_fit, dtype=float)
    if not np.isfinite(fitness) or not np.isfinite(historial).all():
        raise ValueError("El optimizador devolvió resultados no finitos.")
    return mascara, fitness, historial, time.perf_counter()-inicio, len(cache)


def graficar(historial, baseline_inner, resultados, features, mascara):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    axes[0].plot(np.arange(1, len(historial)+1), historial)
    axes[0].axhline(baseline_inner, linestyle="--", color="gray", label="Baseline búsqueda")
    axes[0].set(xlabel="Iteración", ylabel="F1-macro de búsqueda", title="Convergencia GWO (exploratoria)")
    axes[0].legend()
    valores = resultados.F1_media.to_numpy()
    axes[1].bar([0,1], valores, color=["gray", "#1F497D"])
    axes[1].errorbar([0,1], valores, yerr=resultados.F1_std.to_numpy(), fmt="none", capsize=5, color="black")
    axes[1].set_xticks([0,1], resultados.Modelo)
    axes[1].set(ylabel="F1-macro", title="Comparación exploratoria\nMedia ± desviación entre folds", ylim=(0,1.15))
    top = features.head(20).iloc[::-1]
    axes[2].barh(top.feature, top.tfidf_mean, color="#1E6823")
    axes[2].tick_params(axis="y", labelsize=8)
    axes[2].set(xlabel="TF-IDF medio", title="Características seleccionadas (top 20)")
    fig.suptitle(f"GWO — {CORPUS_SELECCIONADO.upper()} — Selección previa; no validación anidada", fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(OUT_DIR / "gwo_seleccion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(14,3))
    ax.imshow(mascara.reshape(1,-1), cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set(xlabel=f"Índice de característica (0–{len(mascara)-1})", yticks=[],
           title=f"{CORPUS_SELECCIONADO.upper()}: {int(mascara.sum())} de {len(mascara)} características seleccionadas")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gwo_mapa_features.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="GWO exploratorio con particiones por grupos.")
    parser.add_argument("--corpus", choices=["actual", "v2"], help="Si se omite, muestra el menú.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--poblacion", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100000 or not 5 <= args.poblacion <= 10000:
        parser.error("epochs debe estar entre 1 y 100000; población entre 5 y 10000.")
    # Error de dependencias antes de iniciar una búsqueda costosa.
    from imblearn.over_sampling import SMOTE
    from mealpy import GWO
    configurar_corpus(args.corpus or solicitar_corpus())
    print(f"GWO EXPLORATORIO — {CORPUS_SELECCIONADO.upper()}", flush=True)
    print(f"Entrada: {RUTA_ENTRADA}", flush=True)
    df = cargar_corpus()
    print(f"SHA-256: {df.attrs['sha256_entrada']}")
    print(f"Corpus: {len(df)} perfiles; {df.grado.value_counts().to_dict()}")
    y = df.grado.to_numpy()
    grupos, auditoria = construir_grupos(df)
    inner, detalle_inner = crear_particiones(y, grupos, 3)
    comparacion, detalle_comparacion = crear_particiones(y, grupos, 5)
    print(f"Grupos: {len(set(grupos))}; folds búsqueda: {len(inner)}; comparación: {len(comparacion)}")
    print("TF-IDF global y selección previa: los F1 son exploratorios, no rendimiento independiente.", flush=True)
    vectorizador = TfidfVectorizer(**TFIDF_CONFIG, stop_words=STOPWORDS_ES)
    X = vectorizador.fit_transform(df.perfil_egreso).toarray()
    nombres = vectorizador.get_feature_names_out()
    if X.shape[1] < 2 or not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1)==0):
        raise ValueError("La matriz TF-IDF necesita al menos dos características y perfiles con términos.")
    todas = np.ones(X.shape[1], dtype=bool)
    baseline_inner = evaluar_mascara(X, y, todas, inner)
    print(f"Baseline ({X.shape[1]} características), F1 búsqueda={baseline_inner.mean():.4f}")
    print(f"Iniciando GWO: {args.epochs} iteraciones, {args.poblacion} lobos...", flush=True)
    mascara, fitness, historial, segundos, evaluaciones = optimizar(X, y, inner, args.epochs, args.poblacion)
    base_cmp = evaluar_mascara(X, y, todas, comparacion)
    gwo_cmp = evaluar_mascara(X, y, mascara, comparacion)
    n_sel = int(mascara.sum())
    resultados = pd.DataFrame({"Modelo":[f"Baseline ({X.shape[1]} feat)", f"GWO ({n_sel} feat)"],
                               "n_features":[X.shape[1],n_sel], "F1_media":[base_cmp.mean(),gwo_cmp.mean()],
                               "F1_std":[base_cmp.std(ddof=1),gwo_cmp.std(ddof=1)],
                               "n_folds":[len(comparacion)]*2, "alcance":["exploratorio"]*2})
    features = pd.DataFrame({"feature":nombres[mascara], "tfidf_mean":X[:,mascara].mean(axis=0),
                             "tfidf_std":X[:,mascara].std(axis=0)}).sort_values("tfidf_mean",ascending=False)
    resumen = {
        "version_analisis":"gwo_exploratorio_grupos_v1", "fecha_utc":datetime.now(timezone.utc).isoformat(),
        "corpus":{"seleccion":CORPUS_SELECCIONADO, "archivo":RUTA_ENTRADA, "sha256":df.attrs["sha256_entrada"],
                  "total":len(df), "distribucion":{str(k):int(v) for k,v in df.grado.value_counts().items()},
                  "grupos":len(set(grupos)), "filas_descartadas":0},
        "tfidf":{**TFIDF_CONFIG, "stop_words":STOPWORDS_ES, "ajuste":"corpus completo"},
        "optimizador":{"algoritmo":"mealpy.GWO.OriginalGWO", "semilla":SEED, "epochs":args.epochs,
                       "poblacion":args.poblacion, "transformacion":"abs(2/pi*arctan(pi/2*x)) > 0.5; determinista",
                       "penalizacion":"fitness 0 para máscara vacía o completa", "segundos":segundos,
                       "mascaras_validas_evaluadas":evaluaciones},
        "seleccion":{"features_entrada":X.shape[1], "features_seleccionadas":n_sel,
                     "reduccion_porcentaje":100*(1-n_sel/X.shape[1]), "fitness_busqueda":fitness,
                     "baseline_busqueda":float(baseline_inner.mean())},
        "cv":{"metodo":"StratifiedKFold sobre grupos homogéneos, expandido a documentos",
              "busqueda":detalle_inner, "comparacion_exploratoria":detalle_comparacion,
              "f1_baseline_comparacion":base_cmp.tolist(), "f1_gwo_comparacion":gwo_cmp.tolist(),
              "smote":"Solo train; k=min(2, mínimo de perfiles por clase en train - 1)",
              "barras_error":"desviación estándar entre folds; no intervalo de confianza"},
        "versiones":{p:version(p) for p in ["numpy","pandas","scikit-learn","imbalanced-learn","mealpy"]},
        "limitaciones":[
            "El vocabulario y los IDF usan el corpus completo: hay información global en las particiones.",
            "GWO selecciona antes de la comparación: esos folds no son una evaluación externa independiente.",
            "Agrupar UCSC y duplicados conocidos evita separarlos; no elimina otras dependencias institucionales.",
            "Las métricas ponderan perfiles individuales; los grupos no se promedian ni eliminan.",
            "Se conservan términos que nombran el grado o la institución, potencialmente informativos de la etiqueta.",
            "Para estimar generalización se requiere validación anidada con TF-IDF y GWO ajustados solo al entrenamiento."]}
    if sha256_archivo(RUTA_ENTRADA) != df.attrs["sha256_entrada"]:
        raise ValueError("El corpus cambió durante la búsqueda. No se guardaron resultados.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(OUT_DIR/"gwo_resultados.csv", index=False, encoding="utf-8-sig")
    features.to_csv(OUT_DIR/"gwo_features_seleccionadas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"indice":np.arange(len(nombres)), "feature":nombres, "seleccionada":mascara.astype(int)}).to_csv(
        OUT_DIR/"gwo_mascara_vocabulario.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"epoch":np.arange(1,len(historial)+1),"fitness":historial}).to_csv(
        OUT_DIR/"gwo_convergencia.csv", index=False, encoding="utf-8-sig")
    # Una fila por perfil: el fold indica en qué prueba aparece en cada esquema.
    for columna, splits in [("fold_busqueda",inner),("fold_comparacion",comparacion)]:
        asignacion=np.zeros(len(df),dtype=int)
        for fold, (_,te) in enumerate(splits,1): asignacion[te]=fold
        auditoria[columna]=asignacion
    auditoria.to_csv(OUT_DIR/"gwo_auditoria_particiones.csv", index=False, sep=";", encoding="utf-8-sig")
    pd.DataFrame({"fold":np.arange(1,len(comparacion)+1), "f1_baseline":base_cmp, "f1_gwo":gwo_cmp}).to_csv(
        OUT_DIR/"gwo_comparacion_folds.csv",index=False,encoding="utf-8-sig")
    (OUT_DIR/"resumen_gwo.json").write_text(json.dumps(resumen,ensure_ascii=False,indent=2),encoding="utf-8")
    graficar(historial,float(baseline_inner.mean()),resultados,features,mascara)
    print(f"GWO finalizado en {segundos:.1f}s: {n_sel}/{X.shape[1]} características")
    print(resultados.to_string(index=False))
    print(f"Resultados: {OUT_DIR}")
    print("Siguiente: adaptar 05_validacion_gwo.py; su versión anterior no consume esta carpeta por corpus.")
    return 0


configurar_corpus("actual")
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as error:
        raise SystemExit(f"Dependencia ausente o incompatible: {error}. Revisa las versiones del entorno. Instalación de paquetes: python -m pip install mealpy imbalanced-learn scikit-learn pandas matplotlib") from None
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from None
