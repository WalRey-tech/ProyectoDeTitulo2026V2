# -*- coding: utf-8 -*-
"""Auditoría 02: validación anidada de GWO con enmascaramiento agresivo.

Ubicación: src/Fase3_Analisis/auditorias/02_validacion_gwo_anidada.py
Ejecutar: python 02_validacion_gwo_anidada.py --corpus actual
Sin --corpus muestra menú actual/V2. Por defecto: 100 iteraciones, 30 lobos.
Prueba de funcionamiento: --epochs 3 --poblacion 6 (no resultados finales).
Dependencias: pandas, numpy, scikit-learn, imbalanced-learn, mealpy, matplotlib.

Compara: original sin GWO; enmascaramiento agresivo sin GWO; enmascaramiento
agresivo con GWO anidado. Elimina títulos y tokens aislados, incluidos términos
con contenido disciplinar válido. Una variación de F1 no prueba por sí sola fuga.
Mantiene las mismas reglas que auditoría 01. El paso 05 principal usa original.

Grupos construidos con los perfiles originales; hasta 5 folds externos y 3
internos según grupos por clase. Vocabulario candidato solo en train externo.
Cada TF-IDF interno se ajusta en su propio train y alinea términos por nombre.
La fitness interna solo guía búsqueda; el rendimiento procede de prueba externa.
SMOTE solo en train, k=min(2,n_min-1); ComplementNB(alpha=1). GWO OriginalGWO
con transformación V determinista, umbral 0.5; máscaras vacía/completa inválidas.

No usa features globales del paso 04. No ejecuta otros scripts ni modifica
sus resultados. El reporte 07 no incorpora automáticamente estas auditorías.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import re
import unicodedata
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
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.naive_bayes import ComplementNB

SRC_ROOT = str(Path(__file__).resolve().parents[2])
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
    OUT_DIR = Path(SRC_ROOT) / "data" / "resultados_cientificos" / "validacion_robusta" / corpus


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


def ajustar_tfidf(textos):
    vectorizador = TfidfVectorizer(stop_words=STOPWORDS_ES, **TFIDF_CONFIG)
    try:
        X = vectorizador.fit_transform(textos)
    except ValueError as exc:
        raise ValueError(f"TF-IDF no pudo ajustarse al entrenamiento: {exc}") from exc
    return vectorizador, X


def entrenar_modelo(X, y, semilla):
    from imblearn.over_sampling import SMOTE
    minimo = int(pd.Series(y).value_counts().min())
    if set(y) != set(CLASES) or minimo < 2:
        raise ValueError("SMOTE requiere las tres clases y dos perfiles por clase en train.")
    k = min(2, minimo - 1)
    X_r, y_r = SMOTE(k_neighbors=k, random_state=semilla).fit_resample(X, y)
    modelo = ComplementNB(alpha=1.0)
    modelo.fit(X_r, y_r)
    return modelo, k


def preparar_internas(textos, y, candidatos, particiones):
    """Solo recibe textos del entrenamiento externo; no recibe su prueba."""
    posicion = {str(t): i for i, t in enumerate(candidatos)}
    preparadas, vocabularios = [], []
    for fold, (tr, va) in enumerate(particiones, 1):
        vec, X_tr = ajustar_tfidf(textos[tr])
        X_va = vec.transform(textos[va])
        nombres = vec.get_feature_names_out()
        indices = np.asarray([posicion.get(str(t), -1) for t in nombres])
        if not (indices >= 0).any():
            raise ValueError("No hay términos comunes entre un train interno y los candidatos.")
        preparadas.append((X_tr, X_va, y[tr], y[va], indices))
        vocabularios.extend({"fold_interno": fold, "feature": str(t),
                             "idf": float(idf), "indice_candidato": int(i)}
                            for t, idf, i in zip(nombres, vec.idf_, indices))
    return preparadas, vocabularios


def binarizar(posicion):
    return np.abs((2 / np.pi) * np.arctan((np.pi / 2) * np.asarray(posicion))) > 0.5


def fitness_mascara(mascara, preparadas, semilla):
    puntuaciones = []
    for X_tr, X_va, y_tr, y_va, indices in preparadas:
        presentes = indices >= 0
        columnas = np.zeros(len(indices), dtype=bool)
        columnas[presentes] = mascara[indices[presentes]]
        if not columnas.any():
            # Máscara inviable; no se ocultan fallos de SMOTE o del clasificador.
            return -1.0
        modelo, _ = entrenar_modelo(X_tr[:, columnas], y_tr, semilla)
        pred = modelo.predict(X_va[:, columnas])
        puntuaciones.append(f1_score(y_va, pred, labels=CLASES,
                                     average="macro", zero_division=0))
    return float(np.mean(puntuaciones))


def seleccionar_gwo(preparadas, n_features, epochs, poblacion, semilla):
    from mealpy import GWO, FloatVar, Problem
    if n_features < 2:
        raise ValueError("Se requieren al menos dos términos candidatos para reducir el vocabulario.")
    cache = {}
    class ProblemaGWO(Problem):
        def obj_func(self, solution):
            mascara = binarizar(solution)
            if not 0 < mascara.sum() < n_features:
                return -1.0
            clave = mascara.tobytes()
            if clave not in cache:
                cache[clave] = fitness_mascara(mascara, preparadas, semilla)
            return cache[clave]
    problema = ProblemaGWO(bounds=FloatVar(lb=(-6.,)*n_features, ub=(6.,)*n_features,
                                         name="features"), minmax="max", log_to=None)
    optimizador = GWO.OriginalGWO(epoch=epochs, pop_size=poblacion)
    inicio = time.perf_counter()
    optimizador.solve(problema, seed=semilla)
    mascara = binarizar(optimizador.g_best.solution)
    fitness = float(optimizador.g_best.target.fitness)
    historial = np.asarray(optimizador.history.list_global_best_fit, dtype=float)
    if (not 0 < mascara.sum() < n_features or fitness < 0
            or not np.isfinite(fitness) or not np.isfinite(historial).all()):
        raise ValueError("GWO no encontró una selección válida; revisa datos o aumenta la búsqueda.")
    return mascara, {"fitness_interno_busqueda": fitness,
                     "historial_fitness": historial.tolist(), "semilla": semilla,
                     "subconjuntos_evaluados": len(cache),
                     "segundos": round(time.perf_counter() - inicio, 3)}


def metricas(y, pred):
    return {"F1_macro": float(f1_score(y, pred, labels=CLASES, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def evaluar_fold(textos_train, y_train, textos_test, internas, epochs, poblacion, semilla):
    """La selección no recibe textos_test; las etiquetas test no entran aquí."""
    vec, X_train = ajustar_tfidf(textos_train)
    nombres = vec.get_feature_names_out()
    preparadas, vocab_inner = preparar_internas(textos_train, y_train, nombres, internas)
    mascara, detalle = seleccionar_gwo(preparadas, len(nombres), epochs, poblacion, semilla)
    # Transformar no ajusta vocabulario ni IDF.
    X_test = vec.transform(textos_test)
    predicciones = {}
    for nombre, columnas in [("Enmascarado_sin_GWO", np.ones(len(nombres), dtype=bool)),
                             ("Enmascarado_GWO_anidado", mascara)]:
        modelo, k = entrenar_modelo(X_train[:, columnas], y_train, semilla)
        predicciones[nombre] = modelo.predict(X_test[:, columnas])
    vocab_outer = [{"indice": i, "feature": str(t), "idf": float(idf),
                    "seleccionada": bool(mascara[i])}
                   for i, (t, idf) in enumerate(zip(nombres, vec.idf_))]
    detalle.update(n_features=len(nombres), n_seleccionadas=int(mascara.sum()), smote_k=k)
    return predicciones, detalle, vocab_outer, vocab_inner


def evaluar_anidada(df, epochs=100, poblacion=30, textos_evaluacion=None):
    textos = df.perfil_egreso.to_numpy() if textos_evaluacion is None else textos_evaluacion
    y = df.grado.to_numpy()
    grupos, audit_grupos = construir_grupos(df)
    externas, detalles_ext = crear_particiones(y, grupos, 5)
    # Verificar todas las particiones antes de invertir tiempo en GWO.
    internas_por_fold = [crear_particiones(y[tr], grupos[tr], 3, SEED+fold)
                         for fold, (tr, _) in enumerate(externas, 1)]
    print(f"Perfiles: {len(df)} | Grupos: {len(set(grupos))} | "
          f"Folds externos: {len(externas)}", flush=True)
    oof = {nombre: np.empty(len(y), dtype=object) for nombre in ["Enmascarado_sin_GWO", "Enmascarado_GWO_anidado"]}
    visitas = np.zeros(len(y), dtype=int)
    folds_oof = np.zeros(len(y), dtype=int)
    resultados, vocab_ext, vocab_int, auditoria, detalles = [], [], [], [], []
    for fold, ((tr, te), detalle_ext, (internas, detalles_int)) in enumerate(
            zip(externas, detalles_ext, internas_por_fold), 1):
        print(f"Fold externo {fold}/{len(externas)}: train={len(tr)}, test={len(te)}, "
              f"búsqueda con {len(internas)} folds internos...", flush=True)
        pred, detalle, vocab, vocab_inner = evaluar_fold(
            textos[tr], y[tr], textos[te], internas, epochs, poblacion, SEED+fold)
        visitas[te] += 1
        folds_oof[te] = fold
        for nombre, valores in pred.items():
            oof[nombre][te] = valores
            resultados.append({"fold_externo": fold, "modelo": nombre,
                               "n_train": len(tr), "n_test": len(te),
                               "n_features": detalle["n_seleccionadas" if nombre == "Enmascarado_GWO_anidado" else "n_features"],
                               "smote_k": detalle["smote_k"], **metricas(y[te], valores)})
        vocab_ext.extend({"fold_externo": fold, **fila} for fila in vocab)
        vocab_int.extend({"fold_externo": fold, **fila} for fila in vocab_inner)
        for rol, indices in [("train", tr), ("test", te)]:
            auditoria.extend({"fold_externo": fold, "fold_interno": 0, "rol": rol,
                              "fila_datos": int(i)+1, "grupo_cv": grupos[i], "grado": y[i]}
                             for i in indices)
        for fi, ((itr, iva), di) in enumerate(zip(internas, detalles_int), 1):
            di["filas_train"] = (tr[itr]+1).tolist()
            di["filas_validacion"] = (tr[iva]+1).tolist()
            for rol, indices in [("train", tr[itr]), ("validacion", tr[iva])]:
                auditoria.extend({"fold_externo": fold, "fold_interno": fi, "rol": rol,
                                  "fila_datos": int(i)+1, "grupo_cv": grupos[i], "grado": y[i]}
                                 for i in indices)
        detalles.append({**detalle_ext, **detalle, "filas_train": (tr+1).tolist(),
                         "filas_test": (te+1).tolist(), "particiones_internas": detalles_int})
        print(f"  F1 externo: completo={resultados[-2]['F1_macro']:.4f}, "
              f"GWO={resultados[-1]['F1_macro']:.4f}; "
              f"variables {detalle['n_features']} -> {detalle['n_seleccionadas']}", flush=True)
    if not np.all(visitas == 1):
        raise RuntimeError("Cada perfil debe recibir exactamente una predicción externa por modelo.")
    audit_grupos["fold_prueba_externa"] = folds_oof
    for nombre in oof:
        audit_grupos[f"prediccion_{nombre}"] = oof[nombre]
    return {"folds": pd.DataFrame(resultados), "predicciones": audit_grupos,
            "vocabularios_externos": pd.DataFrame(vocab_ext),
            "vocabularios_internos": pd.DataFrame(vocab_int),
            "particiones": pd.DataFrame(auditoria), "detalles": detalles}


def guardar_resultados(resultado, df, epochs, poblacion):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nombre in ["folds", "predicciones", "vocabularios_externos", "vocabularios_internos", "particiones", "auditoria_enmascaramiento", "vocabulario_original"]:
        resultado[nombre].to_csv(OUT_DIR / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    resumen_modelos, informes = {}, []
    folds = resultado["folds"]
    y = df.grado.to_numpy()
    for nombre in ["Original_sin_GWO", "Enmascarado_sin_GWO", "Enmascarado_GWO_anidado"]:
        pred = resultado["predicciones"][f"prediccion_{nombre}"].to_numpy()
        seleccion = folds[folds.modelo == nombre]
        resumen_modelos[nombre] = {
            "metricas_predicciones_externas_conjuntas": metricas(y, pred),
            "media_folds": {m: float(seleccion[m].mean()) for m in metricas(y, pred)},
            "desviacion_folds_ddof1": {m: float(seleccion[m].std(ddof=1)) for m in metricas(y, pred)},
            "matriz_confusion_orden_clases": confusion_matrix(y, pred, labels=CLASES).tolist()}
        informe = classification_report(y, pred, labels=CLASES, output_dict=True, zero_division=0)
        informes.extend({"modelo": nombre, "grado": clase, **informe[clase]} for clase in CLASES)
    pd.DataFrame(informes).to_csv(OUT_DIR / "metricas_por_clase.csv", index=False, encoding="utf-8-sig")
    trazas = [{"fold_externo": d["fold"], "epoch": i, "fitness_interno": float(f)}
              for d in resultado["detalles"] for i, f in enumerate(d["historial_fitness"], 1)]
    pd.DataFrame(trazas).to_csv(OUT_DIR / "convergencia.csv", index=False, encoding="utf-8-sig")
    resumen = {
        "protocolo": "auditoria_gwo_anidado_agresivo_grupos_v1",
        "fecha_utc": datetime.now(timezone.utc).isoformat(), "corpus": CORPUS_SELECCIONADO,
        "entrada": str(Path(RUTA_ENTRADA).resolve()), "sha256_entrada": df.attrs["sha256_entrada"],
        "sha256_script": sha256_archivo(__file__), "n_perfiles": len(df),
        "n_grupos": int(resultado["predicciones"].grupo_cv.nunique()),
        "clases": CLASES, "distribucion_grados": {str(k): int(v) for k,v in df.grado.value_counts().items()},
        "enmascaramiento": {"patrones_titulo": PATRONES_TITULO, "patron_token": PATRON_TOKEN.pattern,
                            "titulos_eliminados": int(resultado["auditoria_enmascaramiento"].n_titulos.sum()),
                            "tokens_aislados_eliminados": int(resultado["auditoria_enmascaramiento"].n_tokens_aislados.sum())},
        "parametros": {"epochs": epochs, "poblacion": poblacion, "seed": SEED,
                       "tfidf": TFIDF_CONFIG, "stopwords": STOPWORDS_ES, "alpha_nb": 1.0,
                       "folds_externos_solicitados": 5, "folds_internos_solicitados": 3,
                       "folds_externos_efectivos": len(resultado["detalles"]),
                       "gwo": "OriginalGWO; V determinista >0.5; limites [-6,6]",
                       "mascaras_invalidas": "vacia, completa o sin columnas en algun train interno",
                       "smote": "solo train; k=min(2,minimo_clase_train-1); seed=42+fold_externo"},
        "versiones": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "mealpy"]},
        "modelos": resumen_modelos, "particiones": resultado["detalles"],
        "notas": [
            "Candidatos definidos en train externo; vocabulario e IDF internos ajustados en cada train interno.",
            "Fitness interno es un criterio de búsqueda; la evaluación procede de predicciones externas.",
            "F1 macro conjunto y media de F1 por fold son agregaciones distintas.",
            "La desviación entre folds describe variabilidad; no es un intervalo de confianza.",
            "Se retienen todos los perfiles; las métricas ponderan filas y los grupos solo gobiernan particiones.",
            "El enmascaramiento agresivo elimina títulos y tokens de grado, también información disciplinar legítima.",
            "No mide generalización a universidades desconocidas ni demuestra causalmente fuga léxica.",
            "Revisar los resultados para cambiar el protocolo requiere una evaluación futura independiente.",
            "No utiliza la selección global del paso 04 ni exporta un modelo entrenado con todo el corpus."]}
    (OUT_DIR / "resumen_validacion_anidada.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre in ["Original_sin_GWO", "Enmascarado_sin_GWO", "Enmascarado_GWO_anidado"]:
        datos = folds[folds.modelo == nombre]
        ax.plot(datos.fold_externo, datos.F1_macro, marker="o", label=nombre)
    ax.set(xlabel="Fold externo", ylabel="F1 macro", ylim=(0, 1.03),
           title=f"Validación anidada — corpus {CORPUS_SELECCIONADO}")
    ax.set_xticks(sorted(folds.fold_externo.unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparacion_externa.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, nombre in zip(axes, ["Original_sin_GWO", "Enmascarado_sin_GWO", "Enmascarado_GWO_anidado"]):
        matriz = resumen_modelos[nombre]["matriz_confusion_orden_clases"]
        ConfusionMatrixDisplay(np.asarray(matriz), display_labels=CLASES).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(nombre)
        ax.set_xlabel("Predicción externa")
        ax.set_ylabel("Grado real")
    fig.suptitle(f"Predicciones externas: {len(df)} perfiles — {CORPUS_SELECCIONADO}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matrices_confusion.png", dpi=160)
    plt.close(fig)
    return resumen


# Alternativas más largas primero: no dejar "Informática" como resto de
# "Ingeniería Civil Informática" por haber consumido solo "Ingeniería Civil".
DISCIPLINA = (r"(?:computaci[oó]n\s+(?:e|y)\s+inform[aá]tica|"
              r"inform[aá]tica\s+(?:e|y)\s+computaci[oó]n|"
              r"inform[aá]tica|computaci[oó]n)")
PATRONES_TITULO = [
    r"\bingenier(?:[ií]as?|[oa]s?)\s+"
    r"(?:civil(?:es)?|(?:de\s+)?ejecuci[oó]n)"
    r"(?:\s+(?:en\s+)?" + DISCIPLINA + r")?\b",
    r"\bingenier[ií]as?\s+(?:en\s+)?inform[aá]tica\b",
    r"\bingenier[oa]s?\s+(?:en\s+)?inform[aá]tic[oa]s?\b",
]
PATRON_TITULO = re.compile("|".join(f"(?:{p})" for p in PATRONES_TITULO), re.IGNORECASE)


def enmascarar_y_contar(texto):
    original = str(texto)
    coincidencias = [m.group(0) for m in PATRON_TITULO.finditer(original)]
    limpio = re.sub(r"\s+", " ", PATRON_TITULO.sub(" ", original)).strip()
    return limpio, coincidencias


TOKENS_PROXY = {'civil', 'civiles', 'ejecucion', 'informatica', 'informatico', 'informaticos', 'informaticas'}
PATRON_TOKEN = re.compile(r"\b(?:civil(?:es)?|ejecuci[oó]n|inform[aá]tic[oa]s?)\b", re.IGNORECASE)


def normalizar(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', str(texto).casefold())
                   if unicodedata.category(c) != 'Mn')


def tokens_proxy(feature):
    return sorted(set(re.findall(r'\b\w+\b', normalizar(feature))) & TOKENS_PROXY)


def es_proxy_feature(feature):
    return bool(tokens_proxy(feature))


def enmascarar_texto(texto):
    sin_titulo, titulos = enmascarar_y_contar(texto)
    tokens = [m.group(0) for m in PATRON_TOKEN.finditer(sin_titulo)]
    limpio = re.sub(r'\s+', ' ', PATRON_TOKEN.sub(' ', sin_titulo)).strip()
    return limpio, titulos, tokens


def preparar_textos(df):
    textos, registros = [], []
    for i, (_, fila) in enumerate(df.iterrows(), 1):
        original = str(fila.perfil_egreso)
        limpio, titulos, tokens = enmascarar_texto(original)
        if not re.search(r'\w', limpio):
            raise ValueError(f'La fila de datos {i} queda vacía tras enmascarar. No se eliminaron filas.')
        textos.append(limpio)
        registro = {'fila_datos': i, 'grado': str(fila.grado), 'n_titulos': len(titulos),
                    'n_tokens_aislados': len(tokens), 'tiene_eliminaciones': bool(titulos or tokens),
                    'titulos_eliminados': json.dumps(titulos, ensure_ascii=False),
                    'tokens_eliminados': json.dumps(tokens, ensure_ascii=False),
                    'texto_original': original, 'texto_enmascarado': limpio}
        for c in ['id_programa', 'universidad', 'carrera', 'url', 'modalidad']:
            if c in df:
                registro[c] = fila[c]
        registros.append(registro)
    return np.asarray(textos, dtype=object), pd.DataFrame(registros)



def agregar_baseline_original(df, resultado):
    filas, vocabulario = [], []
    textos, y = df.perfil_egreso.to_numpy(), df.grado.to_numpy()
    oof = np.empty(len(df), dtype=object)
    for d in resultado['detalles']:
        tr = np.asarray(d['filas_train'])-1
        te = np.asarray(d['filas_test'])-1
        vec, X_tr = ajustar_tfidf(textos[tr])
        modelo, k = entrenar_modelo(X_tr, y[tr], SEED+d['fold'])
        oof[te] = modelo.predict(vec.transform(textos[te]))
        filas.append({'fold_externo': d['fold'], 'modelo': 'Original_sin_GWO',
                      'n_train': len(tr), 'n_test': len(te), 'n_features': X_tr.shape[1],
                      'smote_k': k, **metricas(y[te], oof[te])})
        vocabulario.extend({'fold_externo': d['fold'], 'feature': str(t), 'idf': float(v)}
                           for t,v in zip(vec.get_feature_names_out(), vec.idf_))
    resultado['predicciones']['prediccion_Original_sin_GWO'] = oof
    resultado['folds'] = pd.concat([pd.DataFrame(filas), resultado['folds']], ignore_index=True)
    resultado['vocabulario_original'] = pd.DataFrame(vocabulario)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["actual", "v2"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--poblacion", type=int, default=30)
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100000 or not 5 <= args.poblacion <= 10000:
        parser.error("epochs debe estar entre 1 y 100000; poblacion entre 5 y 10000.")
    configurar_corpus(args.corpus or solicitar_corpus())
    print("\nAUDITORÍA 02 — GWO ANIDADO CON ENMASCARAMIENTO AGRESIVO")
    print(f"Entrada: {RUTA_ENTRADA}\nSalida: {OUT_DIR}")
    df = cargar_corpus()
    print(f"Registros leídos: {len(df)}; no se eliminan filas.")
    textos, auditoria = preparar_textos(df)
    resultado = evaluar_anidada(df, args.epochs, args.poblacion, textos)
    agregar_baseline_original(df, resultado)
    resultado["auditoria_enmascaramiento"] = auditoria
    if sha256_archivo(RUTA_ENTRADA) != df.attrs["sha256_entrada"]:
        raise ValueError("El corpus cambió durante la ejecución; no se guardaron resultados.")
    resumen = guardar_resultados(resultado, df, args.epochs, args.poblacion)
    for nombre, valores in resumen["modelos"].items():
        print(f"{nombre}: F1 macro externo conjunto="
              f"{valores['metricas_predicciones_externas_conjuntas']['F1_macro']:.4f}; "
              f"media por fold={valores['media_folds']['F1_macro']:.4f}")
    print(f"Resultados guardados en: {OUT_DIR}")
    print("La selección se repite en cada fold: no existe una única máscara final en este paso.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as exc:
        print(f"Dependencia ausente o incompatible: {exc}\n"
              "Revisa mealpy, imbalanced-learn y scikit-learn en tu entorno virtual.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
