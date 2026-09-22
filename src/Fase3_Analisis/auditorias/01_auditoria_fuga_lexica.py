# -*- coding: utf-8 -*-
"""Auditoría descriptiva de términos GWO y sensibilidad al enmascaramiento.

Ubicación: src/Fase3_Analisis/auditorias/01_auditoria_fuga_lexica.py
Desde esa carpeta:
    python 01_auditoria_fuga_lexica.py --corpus actual
Sin --corpus muestra actual/V2. Requiere la salida actualizada del paso 04
para el mismo corpus: resumen_gwo.json, features y máscara del vocabulario.
Dependencias: pandas, numpy, matplotlib, scikit-learn, imbalanced-learn.

1. Cuenta términos seleccionados por GWO que contienen tokens potencialmente
   asociados al grado. Esta inspección es descriptiva, no prueba una fuga.
2. Compara texto original y enmascaramiento AGRESIVO: títulos más tokens
   aislados (civil, ejecución, informática y variantes). También elimina
   contenido disciplinar válido. El paso 06 solo elimina títulos explícitos.
3. Evalúa TF-IDF completo + SMOTE + ComplementNB, SIN aplicar la máscara GWO.
   Usa las mismas particiones por grupos para ambas condiciones. Cada TF-IDF
   y cada SMOTE se ajustan solo al entrenamiento; k=min(2, mínimo_clase-1).
   Stopwords y semillas coinciden con los pasos 05/06 actualizados.

Una diferencia de F1 describe sensibilidad al procedimiento de eliminación;
no cuantifica causalmente una fuga ni constituye un test de significancia.
La auditoría de features corresponde al GWO exploratorio del paso 04, no a
las selecciones internas de la validación anidada del paso 05.
No se cambia el CSV de entrada ni se descartan perfiles silenciosamente.
No se exporta un modelo final. Los otros dos scripts de auditoría requieren
su propia revisión; este archivo no los ejecuta ni modifica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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
                             classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score)
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
    OUT_DIR = Path(SRC_ROOT) / "data" / "resultados_cientificos" / "auditoria_fuga_lexica" / corpus


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


def auditar_features_gwo(df):
    base = Path(SRC_ROOT) / 'data' / 'resultados_cientificos' / 'gwo' / CORPUS_SELECCIONADO
    rutas = {nombre: base/nombre for nombre in ['resumen_gwo.json', 'gwo_features_seleccionadas.csv',
                                               'gwo_mascara_vocabulario.csv']}
    instruccion = ('Repite el paso 04_seleccion_caracteristicas_gwo.py '
                   f'con --corpus {CORPUS_SELECCIONADO}.')
    for p in rutas.values():
        if not p.is_file():
            raise FileNotFoundError(f'Falta {p}\n{instruccion}')
    fuentes = {n: {'archivo': str(p), 'sha256': sha256_archivo(str(p))} for n,p in rutas.items()}
    r = json.loads(rutas['resumen_gwo.json'].read_text(encoding='utf-8-sig'))
    corpus = r.get('corpus', {})
    if (corpus.get('seleccion') != CORPUS_SELECCIONADO
            or corpus.get('sha256') != df.attrs['sha256_entrada']
            or corpus.get('total') != len(df)
            or corpus.get('distribucion') != df.grado.value_counts().to_dict()):
        raise ValueError(f'Las features GWO corresponden a otro corpus o a un resumen antiguo. {instruccion}')
    features = pd.read_csv(rutas['gwo_features_seleccionadas.csv'], encoding='utf-8-sig', keep_default_na=False)
    mascara = pd.read_csv(rutas['gwo_mascara_vocabulario.csv'], encoding='utf-8-sig', keep_default_na=False)
    if 'feature' not in features or not {'feature','seleccionada'}.issubset(mascara.columns):
        raise ValueError(f'CSV de features/máscara incompatible. {instruccion}')
    for tabla in [features, mascara]:
        if (tabla.empty or tabla.feature.duplicated().any()
                or tabla.feature.astype(str).str.strip().eq('').any()):
            raise ValueError(f'El vocabulario GWO está vacío, tiene duplicados o términos vacíos. {instruccion}')
    if not mascara.seleccionada.isin([0,1]).all():
        raise ValueError(f'Máscara GWO distinta de 0/1. {instruccion}')
    seleccionadas = set(mascara.loc[mascara.seleccionada.eq(1), 'feature'])
    seleccion = r.get('seleccion', {})
    if (seleccionadas != set(features.feature)
            or seleccion.get('features_seleccionadas') != len(features)
            or seleccion.get('features_entrada') != len(mascara)):
        raise ValueError(f'El resumen, la máscara y las features GWO no coinciden. {instruccion}')
    features['tokens_proxy_potenciales'] = features.feature.map(lambda t: '; '.join(tokens_proxy(t)))
    features['es_proxy_potencial'] = features.feature.map(es_proxy_feature)
    proxies = features[features.es_proxy_potencial].copy()
    resumen = {'alcance': 'Selección exploratoria del paso 04; sin aplicación a los clasificadores de esta auditoría',
               'features_gwo_total': len(features), 'features_proxy_potencial': len(proxies),
               'porcentaje_proxy': 100*len(proxies)/len(features),
               'features_proxy': proxies.feature.tolist(), 'tokens_buscados': sorted(TOKENS_PROXY),
               'optimizador_origen': r.get('optimizador', {}), 'fuentes': fuentes,
               'nota': 'Coincidencia por token completo normalizado; no identifica causalmente fuga de etiquetas.'}
    return features, proxies, resumen


def evaluar_comparacion(df):
    grupos, predicciones = construir_grupos(df)
    y = df.grado.to_numpy()
    particiones, detalles = crear_particiones(y, grupos, 5)
    enmascarados, auditoria_textos = preparar_textos(df)
    condiciones = {'Original': df.perfil_egreso.to_numpy(), 'Enmascarado': enmascarados}
    filas, vocabularios, auditoria_particiones = [], [], []
    print(f'Perfiles: {len(df)} | Grupos: {len(set(grupos))} | Folds por condición: {len(particiones)}')
    for fold, ((tr, te), detalle) in enumerate(zip(particiones, detalles), 1):
        detalle.update(filas_train=(tr+1).tolist(), filas_test=(te+1).tolist(), semilla_smote=SEED+fold)
        for rol, indices in [('train',tr), ('test',te)]:
            auditoria_particiones.extend({'fold': fold, 'rol': rol, 'fila_datos': int(i)+1,
                                          'grupo_cv': grupos[i], 'grado': y[i]} for i in indices)
    for condicion, textos in condiciones.items():
        oof = np.empty(len(y), dtype=object)
        visitas = np.zeros(len(y), dtype=int)
        folds_oof = np.zeros(len(y), dtype=int)
        for fold, (tr, te) in enumerate(particiones, 1):
            vec, X_train = ajustar_tfidf(textos[tr])
            X_test = vec.transform(textos[te])
            modelo, k = entrenar_modelo(X_train, y[tr], SEED+fold)
            pred = modelo.predict(X_test)
            oof[te] = pred
            visitas[te] += 1
            folds_oof[te] = fold
            met = {'condicion': condicion, 'fold': fold, 'n_train': len(tr), 'n_test': len(te),
                   'n_features': X_train.shape[1], 'smote_k': k,
                   'n_test_sin_terminos': int((X_test.getnnz(axis=1) == 0).sum()),
                   'f1_macro': float(f1_score(y[te], pred, labels=CLASES, average='macro', zero_division=0)),
                   'accuracy': float(accuracy_score(y[te], pred))}
            for clase, valor in zip(CLASES, f1_score(y[te], pred, labels=CLASES, average=None, zero_division=0)):
                met['f1_'+normalizar(clase)] = float(valor)
            filas.append(met)
            vocabularios.extend({'condicion': condicion, 'fold': fold, 'feature': str(t), 'idf': float(idf)}
                                for t,idf in zip(vec.get_feature_names_out(), vec.idf_))
            print(f"  {condicion}, fold {fold}: F1 macro={met['f1_macro']:.4f}", flush=True)
        if not np.all(visitas == 1):
            raise RuntimeError('Cada perfil debe recibir una predicción externa por condición.')
        predicciones['fold'] = folds_oof
        predicciones['prediccion_'+condicion] = oof
    return {'comparacion': pd.DataFrame(filas), 'predicciones': predicciones,
            'textos': auditoria_textos, 'particiones': pd.DataFrame(auditoria_particiones),
            'vocabularios': pd.DataFrame(vocabularios), 'detalles': detalles}


def guardar_resultados(df, resultado, features, proxies, resumen_features):
    comparacion = resultado['comparacion']
    pred = resultado['predicciones']
    metricas = {}
    for condicion in ['Original','Enmascarado']:
        filas = comparacion[comparacion.condicion == condicion]
        metricas[condicion] = {
            'f1_macro_media_folds': float(filas.f1_macro.mean()),
            'f1_macro_std_folds': float(filas.f1_macro.std(ddof=1)),
            'accuracy_media_folds': float(filas.accuracy.mean()),
            'f1_macro_oof': float(f1_score(pred.grado, pred['prediccion_'+condicion], labels=CLASES, average='macro', zero_division=0)),
            'accuracy_oof': float(accuracy_score(pred.grado, pred['prediccion_'+condicion])),
            'metricas_por_clase_oof': classification_report(pred.grado, pred['prediccion_'+condicion],
                                                          labels=CLASES, output_dict=True, zero_division=0)}
    diferencias = comparacion.pivot(index='fold', columns='condicion', values='f1_macro')
    diferencias['delta_enmascarado_menos_original'] = diferencias.Enmascarado - diferencias.Original
    original = metricas['Original']['f1_macro_media_folds']
    enmascarado = metricas['Enmascarado']['f1_macro_media_folds']
    resumen = {
        'version': '2.0', 'protocolo': 'sensibilidad_lexica_agresiva_por_grupos_v1',
        'fecha_utc': datetime.now(timezone.utc).isoformat(), 'sha256_script': sha256_archivo(__file__),
        'corpus': {'seleccion': CORPUS_SELECCIONADO, 'archivo': RUTA_ENTRADA,
                   'sha256': df.attrs['sha256_entrada'], 'total': len(df),
                   'grupos': int(pred.grupo_cv.nunique()), 'filas_descartadas': 0,
                   'distribucion': {str(k):int(v) for k,v in df.grado.value_counts().items()}},
        'features_gwo': resumen_features,
        'enmascaramiento': {'tipo': 'agresivo: títulos y palabras aisladas',
                           'patrones_titulo': PATRONES_TITULO, 'patron_token': PATRON_TOKEN.pattern,
                           'documentos_con_eliminaciones': int(resultado['textos'].tiene_eliminaciones.sum()),
                           'titulos_eliminados': int(resultado['textos'].n_titulos.sum()),
                           'tokens_aislados_eliminados': int(resultado['textos'].n_tokens_aislados.sum()),
                           'independiente_de_la_etiqueta': True},
        'evaluacion_controlada': {'metodologia': 'TF-IDF local + SMOTE solo train + ComplementNB; sin selección GWO',
                                 'tfidf': {**TFIDF_CONFIG, 'stop_words': STOPWORDS_ES}, 'alpha_nb': 1.0,
                                 'folds_solicitados': 5, 'folds': len(resultado['detalles']),
                                 'semilla_particiones': SEED, 'semilla_smote': '42+fold',
                                 'smote_k': 'min(2, mínimo_clase_train-1)',
                                 'original_f1_macro': original, 'enmascarado_f1_macro': enmascarado,
                                 'delta_enmascarado_menos_original': enmascarado-original,
                                 'metricas': metricas, 'particiones': resultado['detalles']},
        'interpretacion': {'nota': 'Prueba de sensibilidad al retiro de términos potencialmente asociados al grado.',
                          'limitaciones': [
                              'Una coincidencia léxica no demuestra fuga de datos ni uso causal de una característica.',
                              'El enmascaramiento agresivo también retira información disciplinar legítima; no equivale al paso 06.',
                              'La diferencia de F1 no constituye un test de significancia ni cuantifica causalmente la fuga.',
                              'Cada condición reajusta su vocabulario e IDF; el cambio corresponde al procedimiento completo.',
                              'No se usan las features GWO para predecir en esta comparación.',
                              'Grupos definidos con el texto original; no es validación por universidad.',
                              'Las métricas ponderan perfiles y la desviación entre folds no es un intervalo de confianza.',
                              'Estas salidas no se incorporan automáticamente al reporte 07.']},
        'versiones': {**{p:version(p) for p in ['numpy','pandas','scikit-learn','imbalanced-learn']},
                      'python': sys.version.split()[0]},
    }
    # Validar coherencia temporal de entrada y salidas GWO antes de escribir.
    if sha256_archivo(RUTA_ENTRADA) != df.attrs['sha256_entrada']:
        raise ValueError('El corpus cambió durante la auditoría. Repite la ejecución.')
    for fuente in resumen_features['fuentes'].values():
        if sha256_archivo(fuente['archivo']) != fuente['sha256']:
            raise ValueError('Las salidas GWO cambiaron durante la auditoría. Repite la ejecución.')
    contenido_json = json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nombres = {'comparacion': 'comparacion_original_vs_enmascarado.csv',
               'predicciones': 'predicciones_oof_auditoria.csv', 'textos': 'auditoria_enmascaramiento.csv',
               'particiones': 'auditoria_particiones.csv', 'vocabularios': 'vocabularios_por_condicion.csv'}
    for clave,nombre in nombres.items():
        resultado[clave].to_csv(OUT_DIR/nombre, index=False, encoding='utf-8-sig')
    features.to_csv(OUT_DIR/'auditoria_features_gwo_completa.csv', index=False, encoding='utf-8-sig')
    proxies.to_csv(OUT_DIR/'auditoria_features_gwo_proxy.csv', index=False, encoding='utf-8-sig')
    diferencias.to_csv(OUT_DIR/'diferencias_f1_por_fold.csv', encoding='utf-8-sig')
    (OUT_DIR/'resumen_auditoria_fuga_lexica.json').write_text(contenido_json, encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8,5))
    for condicion in ['Original','Enmascarado']:
        datos = comparacion[comparacion.condicion == condicion]
        ax.plot(datos.fold, datos.f1_macro, marker='o', label=condicion)
    ax.set(xlabel='Fold de prueba (compartido)', ylabel='F1 macro', ylim=(0,1.03),
           title=f'Sensibilidad al enmascaramiento agresivo — {CORPUS_SELECCIONADO}')
    ax.set_xticks(sorted(comparacion.fold.unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR/'comparacion_f1_fuga_lexica.png', dpi=160)
    plt.close(fig)
    return resumen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', choices=['actual','v2'])
    args = parser.parse_args(argv)
    configurar_corpus(args.corpus or solicitar_corpus())
    print('\nAUDITORÍA LÉXICA — PRUEBA DE SENSIBILIDAD')
    print(f'Entrada: {RUTA_ENTRADA}\nSalida: {OUT_DIR}')
    df = cargar_corpus()
    features, proxies, resumen_features = auditar_features_gwo(df)
    resultado = evaluar_comparacion(df)
    resumen = guardar_resultados(df, resultado, features, proxies, resumen_features)
    met = resumen['evaluacion_controlada']
    print(f'Términos GWO con proxies potenciales: {len(proxies)}/{len(features)}')
    print(f"F1 medio original: {met['original_f1_macro']:.4f}")
    print(f"F1 medio enmascarado: {met['enmascarado_f1_macro']:.4f}")
    print(f"Delta: {met['delta_enmascarado_menos_original']:+.4f}")
    print('El cambio describe sensibilidad; por sí solo no demuestra una fuga de datos.')
    print(f'Resultados: {OUT_DIR}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ImportError as exc:
        print(f'Dependencia ausente o incompatible: {exc}', file=sys.stderr)
        sys.exit(1)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
