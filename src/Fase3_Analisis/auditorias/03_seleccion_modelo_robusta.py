# -*- coding: utf-8 -*-
"""Auditoría 03: sensibilidad al enmascaramiento y selección anidada de modelo.

Ubicación: src/Fase3_Analisis/auditorias/03_seleccion_modelo_robusta.py
Ejecutar: python 03_seleccion_modelo_robusta.py --corpus actual
Sin --corpus muestra menú actual/V2. No requiere mealpy ni resultados previos.
Dependencias: pandas, numpy, matplotlib, scikit-learn, imbalanced-learn.

Compara ComplementNB(alpha=1)+SMOTE en original, títulos enmascarados y
máscara agresiva, con folds externos comunes. Esa comparación es descriptiva.
La selección de modelo usa SIEMPRE títulos enmascarados: el nivel de máscara
no se elige mirando los resultados externos. Se conserva la parrilla original
(15 candidatos): ComplementNB con/sin SMOTE; regresión logística, SVM lineal
y Ridge con pesos balanced. TF-IDF local en cada train interno/externo.

Hasta 5 folds externos y 3 internos, limitados por grupos/clase. Mantiene
vinculados los perfiles UCSC y duplicados conocidos. SMOTE solo en train;
k=min(2,n_min-1). Ranking interno: F1 medio mayor, desviación menor, nombre.
Las predicciones externas evalúan el PROCEDIMIENTO de selección, no un modelo
único entrenado con todo el corpus. No escoger ganador mirando el F1 externo.
No se silencian errores ni falta de convergencia como puntuaciones cero.
El reporte 07 no incorpora automáticamente estas auditorías.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import re
import warnings
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

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
    OUT_DIR = Path(SRC_ROOT) / "data" / "resultados_cientificos" / "seleccion_modelo_robusta" / corpus


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


def crear_candidatos():
    candidatos = []
    for smote in [True, False]:
        for alpha in [0.1, 0.5, 1.0]:
            candidatos.append({'nombre': f'ComplementNB_{"SMOTE" if smote else "sinSMOTE"}_alpha={alpha}',
                               'familia': 'ComplementNB', 'usar_smote': smote, 'parametros': {'alpha': alpha}})
    for familia, valores, parametro in [('LogisticRegression', [0.5,1.0,5.0], 'C'),
                                        ('LinearSVC', [0.1,0.5,1.0], 'C'),
                                        ('Ridge', [0.5,1.0,2.0], 'alpha')]:
        for valor in valores:
            candidatos.append({'nombre': f'{familia}_balanced_{parametro}={valor}',
                               'familia': familia, 'usar_smote': False,
                               'parametros': {parametro: valor, 'class_weight': 'balanced'}})
    return candidatos


def ajustar_candidato(candidato, X, y, semilla):
    from imblearn.over_sampling import SMOTE
    parametros = candidato['parametros']
    familia = candidato['familia']
    if familia == 'ComplementNB':
        modelo = ComplementNB(**parametros)
    elif familia == 'LogisticRegression':
        modelo = LogisticRegression(**parametros, max_iter=3000, random_state=semilla)
    elif familia == 'LinearSVC':
        modelo = LinearSVC(**parametros, dual='auto', max_iter=10000, random_state=semilla)
    elif familia == 'Ridge':
        modelo = RidgeClassifier(**parametros)
    else:
        raise ValueError(f'Familia desconocida: {familia}')
    k = None
    if candidato['usar_smote']:
        minimo = int(pd.Series(y).value_counts().min())
        if minimo < 2 or set(y) != set(CLASES):
            raise ValueError('SMOTE requiere al menos dos perfiles por clase en train.')
        k = min(2, minimo-1)
        X, y = SMOTE(k_neighbors=k, random_state=semilla).fit_resample(X,y)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        modelo.fit(X,y)
    return modelo, k


def seleccionar_modelo_inner(textos, y, particiones, candidatos, semilla):
    # Ajustar una sola vez por train interno: todos los candidatos comparten TF-IDF.
    preparadas, vocab = [], []
    for fi,(tr,va) in enumerate(particiones,1):
        vec, X_tr = ajustar_tfidf(textos[tr])
        preparadas.append((X_tr, vec.transform(textos[va]), y[tr], y[va]))
        vocab.extend({'fold_interno':fi, 'feature':str(t), 'idf':float(v)}
                     for t,v in zip(vec.get_feature_names_out(),vec.idf_))
    ranking, puntuaciones = [], []
    for candidato in candidatos:
        f1s = []
        for fi,(X_tr,X_va,y_tr,y_va) in enumerate(preparadas,1):
            try:
                modelo,k = ajustar_candidato(candidato,X_tr,y_tr,semilla)
            except (ValueError, ConvergenceWarning) as exc:
                raise ValueError(f"Falló {candidato['nombre']} en fold interno {fi}: {exc}") from exc
            f1 = float(f1_score(y_va,modelo.predict(X_va),labels=CLASES,average='macro',zero_division=0))
            f1s.append(f1)
            puntuaciones.append({'modelo':candidato['nombre'], 'fold_interno':fi, 'f1_macro':f1, 'smote_k':k})
        ranking.append({'modelo':candidato['nombre'], 'F1_inner_media':float(np.mean(f1s)),
                        'F1_inner_std':float(np.std(f1s,ddof=1)), 'n_folds_internos':len(f1s)})
    ranking = pd.DataFrame(ranking).sort_values(
        ['F1_inner_media','F1_inner_std','modelo'],ascending=[False,True,True]).reset_index(drop=True)
    ranking['ranking'] = np.arange(1,len(ranking)+1)
    ganador = next(c for c in candidatos if c['nombre']==ranking.iloc[0].modelo)
    return ganador, ranking, puntuaciones, vocab


def metricas(y,pred):
    return {'F1_macro':float(f1_score(y,pred,labels=CLASES,average='macro',zero_division=0)),
            'Accuracy':float(accuracy_score(y,pred))}


def evaluar_fold(textos_train, y_train, textos_test, internas, candidatos, semilla):
    # No recibe etiquetas de prueba. Seleccionar solo con títulos enmascarados.
    ganador,ranking,scores,vocab_inner = seleccionar_modelo_inner(
        textos_train['Solo_titulos'],y_train,internas,candidatos,semilla)
    predicciones, info, vocab_outer = {}, {}, []
    for condicion in ['Original','Solo_titulos','Agresivo']:
        vec,X_train = ajustar_tfidf(textos_train[condicion])
        X_test = vec.transform(textos_test[condicion])
        modelo,k = entrenar_modelo(X_train,y_train,semilla)
        predicciones[condicion] = modelo.predict(X_test)
        info[condicion] = {'n_features':X_train.shape[1], 'smote_k':k}
        vocab_outer.extend({'condicion':condicion,'feature':str(t),'idf':float(v)}
                           for t,v in zip(vec.get_feature_names_out(),vec.idf_))
        if condicion=='Solo_titulos':
            elegido,k_g = ajustar_candidato(ganador,X_train,y_train,semilla)
            predicciones['Seleccion_anidada'] = elegido.predict(X_test)
            info['Seleccion_anidada'] = {'n_features':X_train.shape[1],'smote_k':k_g}
    return predicciones,info,ganador,ranking,scores,vocab_inner,vocab_outer


def ejecutar(df):
    agresivo,audit = preparar_textos(df)
    titulos = np.asarray([enmascarar_y_contar(t)[0] for t in df.perfil_egreso],dtype=object)
    textos = {'Original':df.perfil_egreso.to_numpy(),'Solo_titulos':titulos,'Agresivo':agresivo}
    audit['texto_solo_titulos'] = titulos
    y = df.grado.to_numpy()
    grupos,predicciones = construir_grupos(df)
    externas,detalles = crear_particiones(y,grupos,5)
    internas = [crear_particiones(y[tr],grupos[tr],3,SEED+fold)
                for fold,(tr,_) in enumerate(externas,1)]
    candidatos = crear_candidatos()
    folds, rankings, scores, vocab_i, vocab_o, particiones = [],[],[],[],[],[]
    oof = {c:np.empty(len(df),dtype=object) for c in [*textos,'Seleccion_anidada']}
    visitas = np.zeros(len(df),dtype=int)
    fold_oof = np.zeros(len(df),dtype=int)
    elegidos = np.empty(len(df),dtype=object)
    print(f'Perfiles: {len(df)} | Grupos: {len(set(grupos))} | Folds externos: {len(externas)}')
    for fold,((tr,te),detalle,(splits,di)) in enumerate(zip(externas,detalles,internas),1):
        pred,info,ganador,ranking,ps,vi,vo = evaluar_fold(
            {c:t[tr] for c,t in textos.items()},y[tr],{c:t[te] for c,t in textos.items()},
            splits,candidatos,SEED+fold)
        visitas[te]+=1
        fold_oof[te]=fold
        elegidos[te]=ganador['nombre']
        for condicion,valores in pred.items():
            oof[condicion][te]=valores
            folds.append({'fold_externo':fold,'condicion':condicion,'n_train':len(tr),'n_test':len(te),
                          'modelo':ganador['nombre'] if condicion=='Seleccion_anidada' else 'ComplementNB_SMOTE_alpha=1.0',
                          **info[condicion],**metricas(y[te],valores)})
        ranking.insert(0,'fold_externo',fold)
        rankings.append(ranking)
        scores.extend({'fold_externo':fold,**r} for r in ps)
        vocab_i.extend({'fold_externo':fold,**r} for r in vi)
        vocab_o.extend({'fold_externo':fold,**r} for r in vo)
        for rol,indices in [('train',tr),('test',te)]:
            particiones.extend({'fold_externo':fold,'fold_interno':0,'rol':rol,
                                'fila_datos':int(i)+1,'grupo_cv':grupos[i],'grado':y[i]} for i in indices)
        for fi,((itr,iva),d) in enumerate(zip(splits,di),1):
            d.update(filas_train=(tr[itr]+1).tolist(),filas_validacion=(tr[iva]+1).tolist())
            for rol,indices in [('train',tr[itr]),('validacion',tr[iva])]:
                particiones.extend({'fold_externo':fold,'fold_interno':fi,'rol':rol,
                                    'fila_datos':int(i)+1,'grupo_cv':grupos[i],'grado':y[i]} for i in indices)
        detalle.update(filas_train=(tr+1).tolist(),filas_test=(te+1).tolist(),particiones_internas=di,
                       modelo_seleccionado=ganador,fitness_interno=float(ranking.iloc[0].F1_inner_media))
        print(f"Fold {fold}: {ganador['nombre']}; F1 externo={folds[-1]['F1_macro']:.4f}",flush=True)
    if not np.all(visitas==1):
        raise RuntimeError('Cada perfil debe tener una predicción externa por condición.')
    predicciones['fold_externo']=fold_oof
    predicciones['modelo_seleccionado']=elegidos
    for c,valores in oof.items():
        predicciones['prediccion_'+c]=valores
    return {'folds':pd.DataFrame(folds),'rankings':pd.concat(rankings,ignore_index=True),
            'scores':pd.DataFrame(scores),'predicciones':predicciones,'textos':audit,
            'particiones':pd.DataFrame(particiones),'vocabularios_internos':pd.DataFrame(vocab_i),
            'vocabularios_externos':pd.DataFrame(vocab_o),'detalles':detalles,'candidatos':candidatos}


def guardar(df,r):
    resumenes,clases = {},[]
    y=df.grado.to_numpy()
    for condicion in ['Original','Solo_titulos','Agresivo','Seleccion_anidada']:
        pred=r['predicciones']['prediccion_'+condicion].to_numpy()
        folds=r['folds'][r['folds'].condicion==condicion]
        reporte=classification_report(y,pred,labels=CLASES,output_dict=True,zero_division=0)
        resumenes[condicion]={'f1_macro_media_outer':float(folds.F1_macro.mean()),
                             'f1_macro_std_outer':float(folds.F1_macro.std(ddof=1)),
                             'metricas_oof':metricas(y,pred),'reporte_oof':reporte,
                             'matriz_confusion':confusion_matrix(y,pred,labels=CLASES).tolist()}
        clases.extend({'condicion':condicion,'grado':c,**reporte[c]} for c in CLASES)
    resumen={'protocolo':'seleccion_modelo_titulos_anidada_grupos_v1',
             'fecha_utc':datetime.now(timezone.utc).isoformat(),'sha256_script':sha256_archivo(__file__),
             'corpus':{'seleccion':CORPUS_SELECCIONADO,'archivo':RUTA_ENTRADA,
                       'sha256':df.attrs['sha256_entrada'],'total':len(df),
                       'grupos':int(r['predicciones'].grupo_cv.nunique()),'filas_descartadas':0,
                       'distribucion':{str(k):int(v) for k,v in df.grado.value_counts().items()}},
             'candidatos':r['candidatos'],'condicion_seleccion_predefinida':'Solo_titulos',
             'regla_seleccion':'F1 interno medio mayor, desviación menor, nombre ascendente',
             'parametros':{'tfidf':TFIDF_CONFIG,'stopwords':STOPWORDS_ES,'seed':SEED,
                           'seed_modelos_y_smote':'42+fold_externo','smote_k':'min(2,n_min_train-1)',
                           'max_iter_logistica':3000,'max_iter_svc':10000,'dual_svc':'auto',
                           'patrones_titulo':PATRONES_TITULO,'patron_agresivo':PATRON_TOKEN.pattern},
             'folds_externos':len(r['detalles']),'particiones':r['detalles'],
             'modelos_seleccionados':r['folds'].query("condicion == 'Seleccion_anidada'").modelo.value_counts().to_dict(),
             'resultados':resumenes,'orden_clases':CLASES,
             'versiones':{p:version(p) for p in ['numpy','pandas','scikit-learn','imbalanced-learn']},
             'limitaciones':[
                 'La predicción externa evalúa un procedimiento de selección, no un único clasificador final.',
                 'El nivel Solo_titulos está fijado antes de evaluar; no se escoge según los resultados externos.',
                 'Comparar niveles es descriptivo y no demuestra causalmente fuga ni superioridad estadística.',
                 'El enmascaramiento agresivo elimina también contenido disciplinar válido.',
                 'Los grupos se definen con el texto original; no es prueba en universidades desconocidas.',
                 'Las métricas ponderan perfiles. La desviación entre folds no es intervalo de confianza.',
                 'Cambiar candidatos tras observar resultados requiere evaluación futura independiente.',
                 'No exporta un modelo para despliegue ni actualiza automáticamente el reporte 07.']}
    if sha256_archivo(RUTA_ENTRADA)!=df.attrs['sha256_entrada']:
        raise ValueError('El corpus cambió durante la ejecución; repite el análisis.')
    contenido=json.dumps(resumen,ensure_ascii=False,indent=2,allow_nan=False)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    nombres={'folds':'comparacion_condiciones_outer.csv','rankings':'ranking_modelos_inner.csv',
             'scores':'puntuaciones_modelos_inner.csv','predicciones':'predicciones_oof_modelo_seleccionado.csv',
             'textos':'auditoria_enmascaramiento.csv','particiones':'auditoria_particiones.csv',
             'vocabularios_internos':'vocabularios_internos.csv','vocabularios_externos':'vocabularios_externos.csv'}
    for clave,nombre in nombres.items():
        r[clave].to_csv(OUT_DIR/nombre,index=False,encoding='utf-8-sig')
    r['folds'].query("condicion == 'Seleccion_anidada'").to_csv(
        OUT_DIR/'validacion_modelo_seleccionado_outer.csv',index=False,encoding='utf-8-sig')
    r['folds'].query("condicion != 'Seleccion_anidada'").to_csv(
        OUT_DIR/'comparacion_niveles_enmascaramiento.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(clases).to_csv(OUT_DIR/'metricas_por_clase.csv',index=False,encoding='utf-8-sig')
    matriz=np.asarray(resumenes['Seleccion_anidada']['matriz_confusion'])
    pd.DataFrame(matriz,index=CLASES,columns=CLASES).rename_axis('grado_real').to_csv(
        OUT_DIR/'matriz_confusion_modelo_seleccionado.csv',encoding='utf-8-sig')
    (OUT_DIR/'resumen_seleccion_modelo_robusta.json').write_text(contenido,encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,5))
    for c in resumenes:
        f=r['folds'][r['folds'].condicion==c]
        ax.plot(f.fold_externo,f.F1_macro,marker='o',label=c)
    ax.set(xlabel='Fold externo',ylabel='F1 macro',ylim=(0,1.03),title='Sensibilidad y selección anidada de modelo')
    ax.set_xticks(sorted(r['folds'].fold_externo.unique()))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR/'comparacion_modelos_robustos.png',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay(matriz,display_labels=CLASES).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d')
    ax.set(title='Selección anidada — predicciones externas',xlabel='Predicción',ylabel='Grado real')
    fig.tight_layout()
    fig.savefig(OUT_DIR/'matriz_confusion_modelo_seleccionado.png',dpi=160)
    plt.close(fig)
    return resumen


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus',choices=['actual','v2'])
    args=parser.parse_args(argv)
    configurar_corpus(args.corpus or solicitar_corpus())
    print(f'\nAUDITORÍA 03 — SELECCIÓN ANIDADA\nEntrada: {RUTA_ENTRADA}\nSalida: {OUT_DIR}')
    df=cargar_corpus()
    resumen=guardar(df,ejecutar(df))
    for condicion,r in resumen['resultados'].items():
        print(f"{condicion}: F1 medio externo={r['f1_macro_media_outer']:.4f}")
    print(f'Resultados: {OUT_DIR}')
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except ImportError as exc:
        print(f'Dependencia ausente o incompatible: {exc}',file=sys.stderr)
        sys.exit(1)
    except (ValueError,FileNotFoundError,ConvergenceWarning) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(1)
