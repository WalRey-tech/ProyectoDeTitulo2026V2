# -*- coding: utf-8 -*-
"""Evaluación robusta: títulos enmascarados + TF-IDF + SMOTE + ComplementNB.

Ejecutar desde src/Fase3_Analisis:
    python 06_modelo_final_robusto.py --corpus actual
Sin --corpus muestra el menú actual/V2. No necesita resultados del paso 05.
Instalación: python -m pip install imbalanced-learn scikit-learn pandas matplotlib
Entorno probado: Python 3.12, numpy 1.26.0, scikit-learn 1.7.2,
imbalanced-learn 0.14.0. Se registran las versiones efectivas en el resumen.

Se mantiene el método de este paso: ComplementNB(alpha=1) con hasta 400
características, SIN selección GWO ni búsqueda de hiperparámetros. El paso 05
valida GWO por separado; aquí se controla la presencia explícita del título.
Las reglas de enmascaramiento son fijas y no consultan el grado de la fila.
No se aplica ftfy: la corrección de encoding corresponde a la fase 2.

Los grupos se construyen con el corpus de entrada (igual que en 05), antes
del enmascaramiento. Se conservan todas las filas y los perfiles vinculados
no se separan. Hasta 5 folds estratificados por grupos, limitados por la clase
con menos grupos. TF-IDF y SMOTE se ajustan solo en cada entrenamiento.
Una sola predicción fuera de entrenamiento por perfil. La métrica principal
sigue siendo la media de F1 macro por fold; se informa también el F1 conjunto.
No se exporta un modelo para uso posterior: este archivo evalúa el protocolo.

Eliminar títulos no elimina todas las pistas de grado o institución. Tampoco
constituye validación en universidades desconocidas. Los folds no son muestras
independientes: su desviación no es un intervalo de confianza. Los valores
históricos (por ejemplo 0.5742) no se copian ni se fuerzan en los resultados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import re
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
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score)
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
    OUT_DIR = Path(SRC_ROOT) / "data" / "resultados_cientificos" / "modelo_final_robusto" / corpus


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
        raise ValueError("El modelo requiere las tres clases, con al menos dos perfiles cada una.")
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
    for columna in ["id_programa", "indice_fuente", "universidad", "carrera", "url", "modalidad", "grupo_perfil"]:
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


def preparar_textos(df):
    textos, filas = [], []
    for i, (_, fila) in enumerate(df.iterrows(), 1):
        original = str(fila.perfil_egreso)
        limpio, coincidencias = enmascarar_y_contar(original)
        # No eliminar silenciosamente perfiles que solo contuvieran el título.
        if not re.search(r"\w", limpio):
            raise ValueError(f"El perfil de la fila de datos {i} queda vacío tras enmascarar el título.")
        textos.append(limpio)
        registro = {"fila_datos": i, "grado": str(fila.grado),
                    "denominaciones_enmascaradas": len(coincidencias),
                    "titulo_enmascarado": bool(coincidencias),
                    "texto_modificado": original != limpio,
                    "expresiones_eliminadas": json.dumps(coincidencias, ensure_ascii=False),
                    "texto_original": original, "texto_enmascarado": limpio}
        for columna in ["id_programa", "universidad", "carrera", "url", "modalidad"]:
            if columna in df:
                registro[columna] = fila[columna]
        filas.append(registro)
    return np.asarray(textos, dtype=object), pd.DataFrame(filas)


def calcular_metricas(y, pred):
    resultado = {
        "F1_macro": float(f1_score(y, pred, labels=CLASES, average="macro", zero_division=0)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Precision_macro": float(precision_score(y, pred, labels=CLASES, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y, pred, labels=CLASES, average="macro", zero_division=0)),
        "Balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    por_clase = f1_score(y, pred, labels=CLASES, average=None, zero_division=0)
    resultado.update(zip(["F1_Civil", "F1_Ejecucion", "F1_Informatica"], map(float, por_clase)))
    return resultado


def evaluar_modelo(df):
    grupos, predicciones = construir_grupos(df)
    y = df.grado.to_numpy()
    particiones, detalles = crear_particiones(y, grupos, 5)
    textos, audit_mascara = preparar_textos(df)
    print(f"Perfiles: {len(df)} | Grupos: {len(set(grupos))} | Folds: {len(particiones)}")
    print(f"Títulos enmascarados: {audit_mascara.denominaciones_enmascaradas.sum()} "
          f"en {audit_mascara.titulo_enmascarado.sum()} perfiles.")
    oof = np.empty(len(y), dtype=object)
    visitas = np.zeros(len(y), dtype=int)
    fold_oof = np.zeros(len(y), dtype=int)
    resultados, vocabularios, auditoria = [], [], []
    for fold, ((tr, te), detalle) in enumerate(zip(particiones, detalles), 1):
        vectorizador, X_train = ajustar_tfidf(textos[tr])
        X_test = vectorizador.transform(textos[te])
        # Misma semilla por fold que el baseline del paso 05.
        modelo, k = entrenar_modelo(X_train, y[tr], SEED+fold)
        pred = modelo.predict(X_test)
        oof[te] = pred
        visitas[te] += 1
        fold_oof[te] = fold
        met = calcular_metricas(y[te], pred)
        resultados.append({"Fold": fold, "N_train": len(tr), "N_test": len(te),
                           "N_features": X_train.shape[1], "SMOTE_k": k, **met})
        vocabularios.extend({"Fold": fold, "feature": str(t), "idf": float(idf)}
                            for t, idf in zip(vectorizador.get_feature_names_out(), vectorizador.idf_))
        for rol, indices in [("train", tr), ("test", te)]:
            auditoria.extend({"Fold": fold, "rol": rol, "fila_datos": int(i)+1,
                              "grupo_cv": grupos[i], "grado": y[i]} for i in indices)
        detalle.update(filas_train=(tr+1).tolist(), filas_test=(te+1).tolist(),
                       n_features=X_train.shape[1], semilla_smote=SEED+fold,
                       perfiles_test_sin_terminos=int((X_test.getnnz(axis=1) == 0).sum()))
        print(f"Fold {fold}/{len(particiones)}: F1 macro={met['F1_macro']:.4f}; "
              f"train={len(tr)}, test={len(te)}, variables={X_train.shape[1]}", flush=True)
    if not np.all(visitas == 1):
        raise RuntimeError("Cada perfil debe tener exactamente una predicción de prueba.")
    predicciones["Fold"] = fold_oof
    predicciones["grado_real"] = y
    predicciones["grado_predicho"] = oof
    predicciones["correcto"] = y == oof
    return {"folds": pd.DataFrame(resultados), "predicciones": predicciones,
            "auditoria_mascara": audit_mascara, "particiones": pd.DataFrame(auditoria),
            "vocabularios": pd.DataFrame(vocabularios), "detalles": detalles}


def guardar_resultados(resultado, df):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artefactos = {
        "folds": "metricas_folds_modelo_final.csv",
        "predicciones": "predicciones_oof_modelo_final.csv",
        "auditoria_mascara": "auditoria_enmascaramiento_modelo_final.csv",
        "particiones": "auditoria_particiones_modelo_final.csv",
        "vocabularios": "vocabularios_modelo_final.csv",
    }
    for clave, nombre in artefactos.items():
        resultado[clave].to_csv(OUT_DIR / nombre, index=False, encoding="utf-8-sig")
    y = df.grado.to_numpy()
    pred = resultado["predicciones"].grado_predicho.to_numpy()
    folds = resultado["folds"]
    audit = resultado["auditoria_mascara"]
    met = calcular_metricas(y, pred)
    informe = classification_report(y, pred, labels=CLASES, output_dict=True, zero_division=0)
    por_clase = {clase: {"precision": float(informe[clase]["precision"]),
                         "recall": float(informe[clase]["recall"]),
                         "f1_score": float(informe[clase]["f1-score"]),
                         "support": int(informe[clase]["support"])} for clase in CLASES}
    pd.DataFrame([{"clase": c, **v} for c,v in por_clase.items()]).to_csv(
        OUT_DIR / "metricas_por_clase_modelo_final.csv", index=False, encoding="utf-8-sig")
    matriz = confusion_matrix(y, pred, labels=CLASES)
    pd.DataFrame(matriz, index=CLASES, columns=CLASES).rename_axis("grado_real").to_csv(
        OUT_DIR / "matriz_confusion_modelo_final.csv", encoding="utf-8-sig")
    resumen = {
        "version": "2.0", "protocolo": "complementnb_smote_titulos_enmascarados_grupos_v1",
        "fecha_generacion_utc": datetime.now(timezone.utc).isoformat(),
        "estado": "evaluacion_modelo_robusto_parametros_fijos",
        "sha256_entrada": df.attrs["sha256_entrada"], "sha256_script": sha256_archivo(__file__),
        "corpus": {"seleccion": CORPUS_SELECCIONADO, "archivo": str(Path(RUTA_ENTRADA).resolve()),
                   "total_perfiles": len(df), "total_grupos": int(resultado["predicciones"].grupo_cv.nunique()),
                   "distribucion": {str(k): int(v) for k,v in df.grado.value_counts().items()}},
        "control_fuga_lexica": {"metodo": "Reglas fijas de títulos, independientes de la etiqueta",
                               "patrones": PATRONES_TITULO,
                               "documentos_con_titulos_enmascarados": int(audit.titulo_enmascarado.sum()),
                               "documentos_modificados": int(audit.texto_modificado.sum()),
                               "denominaciones_enmascaradas": int(audit.denominaciones_enmascaradas.sum()),
                               "nota": "Se conservan términos aislados e instituciones; no es anonimización completa."},
        "representacion": {"metodo": "TF-IDF", **TFIDF_CONFIG, "stop_words": STOPWORDS_ES,
                           "ajuste": "Vocabulario e IDF exclusivamente en train de cada fold"},
        "modelo": {"clasificador": "ComplementNB", "alpha": 1.0, "balanceo": "SMOTE solo train",
                   "smote_k_maximo": 2, "semilla_smote": "42 + fold", "seleccion_gwo": False},
        "validacion": {"metodo": "StratifiedKFold sobre grupos homogéneos y expansión a filas",
                       "n_splits_solicitados": 5, "n_splits": len(folds), "random_state": SEED,
                       "metrica_principal": "Media de F1 macro de los folds",
                       "unidad_metricas": "perfil", "unidad_particionado": "grupo",
                       "grupos_definidos_antes_del_enmascaramiento": True,
                       "particiones": resultado["detalles"]},
        "resultados_principales": {"f1_macro_media_folds": float(folds.F1_macro.mean()),
                                   "f1_macro_std_folds": float(folds.F1_macro.std(ddof=1)),
                                   "accuracy_media_folds": float(folds.Accuracy.mean()),
                                   "f1_macro_oof": met["F1_macro"], "accuracy_oof": met["Accuracy"]},
        "metricas_oof": met, "metricas_oof_por_clase": por_clase,
        "matriz_confusion": {"orden_clases": CLASES, "valores": matriz.tolist()},
        "versiones": {**{p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn"]},
                      "python": sys.version.split()[0]},
        "interpretacion": {
            "resultado_principal": "Evaluación con títulos explícitos enmascarados y parámetros fijos.",
            "relacion_con_gwo": "Paso 04: exploración GWO; paso 05: validación anidada GWO; paso 06: modelo fijo enmascarado.",
            "limitaciones": [
                "El enmascaramiento reduce pistas directas, pero no garantiza eliminarlas todas.",
                "Los grupos se fijan con los textos originales; no es validación por universidad.",
                "Los parámetros fijos no constituyen prerregistro: cambios tras observar resultados necesitan evaluación independiente.",
                "La clase con menos grupos limita las particiones y vuelve sensibles sus métricas.",
                "La desviación entre folds no es un intervalo de confianza.",
                "F1 conjunto y media de F1 por fold son agregaciones distintas.",
                "Este paso evalúa modelos por fold; no produce un modelo único para despliegue."]},
        "artefactos": {**artefactos, "metricas_por_clase": "metricas_por_clase_modelo_final.csv",
                       "matriz_confusion_csv": "matriz_confusion_modelo_final.csv",
                       "grafico_folds": "f1_por_fold_modelo_final.png",
                       "grafico_matriz": "matriz_confusion_modelo_final.png"},
    }
    (OUT_DIR / "resumen_modelo_final_robusto.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(folds.Fold, folds.F1_macro, marker="o", label="F1 macro por fold")
    ax.axhline(folds.F1_macro.mean(), linestyle="--", color="gray", label="Media de folds")
    ax.set(xlabel="Fold", ylabel="F1 macro", ylim=(0, 1.03),
           title=f"ComplementNB + SMOTE con títulos enmascarados — {CORPUS_SELECCIONADO}")
    ax.set_xticks(folds.Fold)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f1_por_fold_modelo_final.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(matriz, display_labels=CLASES).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set(xlabel="Grado predicho", ylabel="Grado real",
           title=f"Predicciones de prueba: {len(df)} perfiles\nTítulos enmascarados — {CORPUS_SELECCIONADO}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "matriz_confusion_modelo_final.png", dpi=160)
    plt.close(fig)
    return resumen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["actual", "v2"])
    args = parser.parse_args(argv)
    configurar_corpus(args.corpus or solicitar_corpus())
    print("\nFASE 3 — MODELO ROBUSTO CON TÍTULOS ENMASCARADOS")
    print(f"Entrada: {RUTA_ENTRADA}\nSalida: {OUT_DIR}")
    df = cargar_corpus()
    print(f"Registros leídos: {len(df)}; no se eliminan filas.")
    resultado = evaluar_modelo(df)
    if sha256_archivo(RUTA_ENTRADA) != df.attrs["sha256_entrada"]:
        raise ValueError("El corpus cambió durante la ejecución; no se guardaron resultados.")
    resumen = guardar_resultados(resultado, df)
    principal = resumen["resultados_principales"]
    print(f"F1 macro medio por fold: {principal['f1_macro_media_folds']:.4f}")
    print(f"F1 macro conjunto (OOF): {principal['f1_macro_oof']:.4f}")
    print(f"Resultados guardados en: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as exc:
        print(f"Dependencia ausente o incompatible: {exc}\n"
              "Revisa imbalanced-learn y scikit-learn en tu entorno virtual.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
