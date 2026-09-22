# -*- coding: utf-8 -*-
"""Consolida las salidas actuales de los pasos 01–06 sin volver a entrenar.

Desde src/Fase3_Analisis:
    python 07_generar_reporte.py --corpus actual
Sin --corpus muestra el menú actual/V2. Requiere pandas y scikit-learn.

Lee exclusivamente las rutas del corpus elegido. Exige los seis resúmenes y
los CSV predictivos; verifica SHA-256, número de perfiles y distribución de
clases. Recalcula métricas de predicciones OOF y contrasta folds/resúmenes.
No carga resultados históricos como respaldo ni rellena métricas ausentes.
Si algo falta o pertenece a otra entrada, informa qué paso repetir y se detiene
antes de escribir el reporte. No ejecuta scraping, entrenamiento ni GWO.

Salidas en src/data/resultados_cientificos:
    resultados_finales_{actual|v2}.json
    reporte_final_{actual|v2}.md
    resumen_metricas_{actual|v2}.csv

El JSON conserva los seis resúmenes íntegros y la procedencia de los archivos.
El Markdown explica los resultados sin confundir F1 con precisión, similitud
con equivalencia semántica, ni ajuste exploratorio con evaluación predictiva.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SRC_ROOT = Path(__file__).resolve().parent.parent
CLASES = ['Civil', 'Ejecución', 'Informática']
PASOS = {
    '01': '01_proyeccion_pca_lda.py',
    '02': '02_homogeneidad_significancia.py',
    '03': '03_diferenciacion_lexica.py',
    '04': '04_seleccion_caracteristicas_gwo.py',
    '05': '05_validacion_gwo.py',
    '06': '06_modelo_final_robusto.py',
}


def sha256_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def solicitar_corpus():
    print('\nSelecciona el corpus:\n1. Actual — salida corregida del encoding\n2. V2 — corpus histórico')
    opciones = {'1': 'actual', 'actual': 'actual', '2': 'v2', 'v2': 'v2'}
    while True:
        try:
            respuesta = input('Opción [1/2]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit('Selección cancelada. Usa --corpus actual o --corpus v2.') from None
        if respuesta in opciones:
            return opciones[respuesta]
        print('Opción no válida. Escribe 1 o 2.')


def definir_rutas(corpus):
    base = Path(SRC_ROOT) / 'data' / 'resultados_cientificos'
    entrada = Path(SRC_ROOT) / 'data' / 'processed' / (
        'perfiles_egreso_etiquetado_actual_corregido.csv' if corpus == 'actual'
        else 'perfiles_egreso_etiquetado_v2.csv')
    gwo = base / 'gwo' / corpus
    anidada = gwo / 'validacion_anidada'
    robusto = base / 'modelo_final_robusto' / corpus
    resumenes = {
        '01': base / 'visualizaciones_exploratorias' / f'resumen_pca_lda_{corpus}.json',
        '02': base / 'homogeneidad_semantica' / f'resumen_homogeneidad_{corpus}.json',
        '03': base / 'diferenciacion_lexica' / f'resumen_diferenciacion_lexica_{corpus}.json',
        '04': gwo / 'resumen_gwo.json',
        '05': anidada / 'resumen_validacion_anidada.json',
        '06': robusto / 'resumen_modelo_final_robusto.json',
    }
    tablas = {
        '04_resultados': gwo / 'gwo_resultados.csv',
        '04_features': gwo / 'gwo_features_seleccionadas.csv',
        '05_folds': anidada / 'folds.csv',
        '05_predicciones': anidada / 'predicciones.csv',
        '06_folds': robusto / 'metricas_folds_modelo_final.csv',
        '06_predicciones': robusto / 'predicciones_oof_modelo_final.csv',
    }
    return entrada, base, resumenes, tablas


def problema(paso, mensaje, corpus):
    raise ValueError(f'Paso {paso}: {mensaje}\nRepite: python .\\{PASOS[paso]} --corpus {corpus}')


def cerca(a, b, contexto, paso, corpus, tolerancia=1e-6):
    try:
        valido = math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tolerancia
    except (TypeError, ValueError):
        valido = False
    if not valido:
        problema(paso, f'{contexto} no coincide: {a!r} frente a {b!r}.', corpus)


def exigir_columnas(df, columnas, paso, corpus):
    faltan = set(columnas) - set(df.columns)
    if faltan:
        problema(paso, f'faltan columnas {sorted(faltan)}.', corpus)


def validar_procedencia(resumenes, corpus, huella, df):
    distribucion = {str(k): int(v) for k,v in df.grado.value_counts().items()}
    for paso, r in resumenes.items():
        try:
            if paso in {'01', '02', '03', '04'}:
                c = r['corpus']
                seleccion, digest, n, reparto = c['seleccion'], c['sha256'], c['total'], c['distribucion']
            elif paso == '05':
                seleccion, digest, n, reparto = r['corpus'], r['sha256_entrada'], r['n_perfiles'], r['distribucion_grados']
            else:
                c = r['corpus']
                seleccion, digest, n, reparto = c['seleccion'], r['sha256_entrada'], c['total_perfiles'], c['distribucion']
        except (KeyError, TypeError):
            problema(paso, 'resumen antiguo o incompatible: falta la procedencia del corpus.', corpus)
        if seleccion != corpus or digest != huella or n != len(df) or reparto != distribucion:
            problema(paso, 'el resumen corresponde a otro corpus o a una versión anterior del CSV.', corpus)


def validar_predicciones(pred, df, columna_fold, columnas_pred, paso, corpus):
    exigir_columnas(pred, ['fila_datos', 'grupo_cv', 'grado', columna_fold, *columnas_pred], paso, corpus)
    if len(pred) != len(df) or sorted(pred.fila_datos.tolist()) != list(range(1, len(df)+1)):
        problema(paso, 'debe existir una predicción por fila, sin duplicados ni omisiones.', corpus)
    pred = pred.sort_values('fila_datos').reset_index(drop=True)
    if pred.grado.tolist() != df.grado.tolist():
        problema(paso, 'las etiquetas de las predicciones no coinciden con las del corpus.', corpus)
    if pred.grupo_cv.isna().any() or pred.grupo_cv.astype(str).str.strip().eq('').any():
        problema(paso, 'faltan identificadores de grupo.', corpus)
    if pred.groupby('grupo_cv')[columna_fold].nunique().gt(1).any():
        problema(paso, 'un grupo aparece en más de una partición de prueba.', corpus)
    if pred.groupby('grupo_cv').grado.nunique().gt(1).any():
        problema(paso, 'hay grupos con grados contradictorios.', corpus)
    ids = pred[columna_fold]
    if ids.isna().any() or set(ids) != set(range(1, ids.nunique()+1)) or ids.nunique() < 2:
        problema(paso, 'identificadores de fold inválidos.', corpus)
    for _, bloque in pred.groupby(columna_fold):
        if set(bloque.grado) != set(CLASES):
            problema(paso, 'una partición de prueba no contiene las tres clases.', corpus)
    for columna in columnas_pred:
        if not set(pred[columna]).issubset(CLASES):
            problema(paso, f'predicciones inválidas en {columna}.', corpus)
    return pred


def verificar_metricas(pred, folds, columna_pred, columna_fold, columna_f1, columna_acc,
                       columna_n_test, columna_n_train, paso, corpus):
    exigir_columnas(folds, [columna_fold, columna_f1, columna_acc, columna_n_test, columna_n_train], paso, corpus)
    if folds[columna_fold].duplicated().any() or set(folds[columna_fold]) != set(pred[columna_fold]):
        problema(paso, 'las métricas y las predicciones tienen folds distintos.', corpus)
    valores = []
    for _, fila in folds.sort_values(columna_fold).iterrows():
        bloque = pred[pred[columna_fold] == fila[columna_fold]]
        f1 = f1_score(bloque.grado, bloque[columna_pred], labels=CLASES, average='macro', zero_division=0)
        acc = accuracy_score(bloque.grado, bloque[columna_pred])
        cerca(fila[columna_f1], f1, 'F1 por fold', paso, corpus)
        cerca(fila[columna_acc], acc, 'Accuracy por fold', paso, corpus)
        cerca(fila[columna_n_test], len(bloque), 'N_test', paso, corpus, 0)
        cerca(fila[columna_n_train], len(pred)-len(bloque), 'N_train', paso, corpus, 0)
        valores.append(float(f1))
    return {
        'n_folds': len(valores), 'f1_macro_media_folds': float(pd.Series(valores).mean()),
        'f1_macro_std_folds': float(pd.Series(valores).std(ddof=1)),
        'f1_macro_oof': float(f1_score(pred.grado, pred[columna_pred], labels=CLASES, average='macro', zero_division=0)),
        'accuracy_oof': float(accuracy_score(pred.grado, pred[columna_pred])),
        'f1_por_fold': valores,
    }


def consolidar(corpus):
    entrada, base, rutas, rutas_tablas = definir_rutas(corpus)
    requeridos = {'corpus': entrada, **rutas, **rutas_tablas}
    faltantes = [(k, p) for k,p in requeridos.items() if not p.is_file()]
    if faltantes:
        lineas = ['Faltan archivos para consolidar el corpus seleccionado:']
        lineas.extend(f'  {k}: {p}' for k,p in faltantes)
        pasos = sorted({k[:2] for k,_ in faltantes if k != 'corpus'})
        lineas.extend(f'Repite: python .\\{PASOS[p]} --corpus {corpus}' for p in pasos)
        if any(k == 'corpus' for k,_ in faltantes):
            lineas.append('Completa primero la fase 2 para el corpus seleccionado.')
        raise FileNotFoundError('\n'.join(lineas))
    huellas = {k: sha256_archivo(p) for k,p in requeridos.items()}
    df = pd.read_csv(entrada, sep=None, engine='python', encoding='utf-8-sig', keep_default_na=False)
    if not {'grado', 'perfil_egreso'}.issubset(df.columns) or df.empty:
        raise ValueError('El corpus debe contener grado y perfil_egreso y al menos una fila.')
    if set(df.grado) != set(CLASES) or df.perfil_egreso.astype(str).str.strip().eq('').any():
        raise ValueError('El corpus contiene grados inválidos o perfiles vacíos; revisa fase 2.')
    for columna in ['estado_registro', 'estado_etiquetado']:
        if columna in df and df[columna].astype(str).str.upper().str.strip().isin(['REVISAR','ERROR']).any():
            raise ValueError(f'El corpus contiene filas pendientes en {columna}; revisa fase 2.')
    def invalido(valor):
        raise ValueError(f'Número JSON no finito: {valor}')
    resumenes = {paso: json.loads(p.read_text(encoding='utf-8-sig'), parse_constant=invalido)
                 for paso,p in rutas.items()}
    validar_procedencia(resumenes, corpus, huellas['corpus'], df)
    tablas = {k: pd.read_csv(p, encoding='utf-8-sig') for k,p in rutas_tablas.items()}
    # Las revisiones viejas de 05/06 no pueden pasar como validación actual.
    if resumenes['05'].get('protocolo') != 'validacion_anidada_gwo_por_grupos_v1':
        problema('05', 'protocolo de validación anidada no reconocido.', corpus)
    if resumenes['06'].get('protocolo') != 'complementnb_smote_titulos_enmascarados_grupos_v1':
        problema('06', 'protocolo del modelo enmascarado no reconocido.', corpus)
    p5 = validar_predicciones(tablas['05_predicciones'], df, 'fold_prueba_externa',
                             ['prediccion_TFIDF_completo', 'prediccion_GWO'], '05', corpus)
    p6 = validar_predicciones(tablas['06_predicciones'], df, 'Fold', ['grado_predicho'], '06', corpus)
    if (p5.grupo_cv.tolist() != p6.grupo_cv.tolist()
            or p5.fold_prueba_externa.tolist() != p6.Fold.tolist()):
        raise ValueError('Los pasos 05 y 06 usan grupos o folds distintos. Repite ambos con las versiones actualizadas.')
    cerca(resumenes['05']['n_grupos'], p5.grupo_cv.nunique(), 'Número de grupos', '05', corpus, 0)
    cerca(resumenes['06']['corpus']['total_grupos'], p6.grupo_cv.nunique(), 'Número de grupos', '06', corpus, 0)
    exigir_columnas(tablas['05_folds'], ['modelo'], '05', corpus)
    if set(tablas['05_folds'].modelo) != {'TFIDF_completo', 'GWO'}:
        problema('05', 'deben estar los dos modelos: TFIDF_completo y GWO.', corpus)
    metricas = []
    for nombre in ['TFIDF_completo', 'GWO']:
        folds = tablas['05_folds'][tablas['05_folds'].modelo == nombre]
        pred = p5.rename(columns={'fold_prueba_externa': 'fold_externo'})
        met = verificar_metricas(pred, folds, 'prediccion_'+nombre, 'fold_externo', 'F1_macro',
                                 'accuracy', 'n_test', 'n_train', '05', corpus)
        r = resumenes['05']['modelos'][nombre]
        cerca(met['n_folds'], resumenes['05']['parametros']['folds_externos_efectivos'], 'Folds efectivos', '05', corpus, 0)
        for campo, esperado in [('f1_macro_media_folds', r['media_folds']['F1_macro']),
                                ('f1_macro_std_folds', r['desviacion_folds_ddof1']['F1_macro']),
                                ('f1_macro_oof', r['metricas_predicciones_externas_conjuntas']['F1_macro']),
                                ('accuracy_oof', r['metricas_predicciones_externas_conjuntas']['accuracy'])]:
            cerca(met[campo], esperado, campo, '05', corpus)
        metricas.append({'paso': '05', 'modelo': nombre, 'texto': 'original', **met})
    met = verificar_metricas(p6, tablas['06_folds'], 'grado_predicho', 'Fold', 'F1_macro',
                             'Accuracy', 'N_test', 'N_train', '06', corpus)
    cerca(met['n_folds'], resumenes['06']['validacion']['n_splits'], 'Folds efectivos', '06', corpus, 0)
    for campo in ['f1_macro_media_folds','f1_macro_std_folds','f1_macro_oof','accuracy_oof']:
        cerca(met[campo], resumenes['06']['resultados_principales'][campo], campo, '06', corpus)
    metricas.append({'paso': '06', 'modelo': 'ComplementNB_SMOTE', 'texto': 'titulos_enmascarados', **met})
    # GWO exploratorio queda separado de las métricas de prueba externa.
    r4 = resumenes['04']
    exigir_columnas(tablas['04_resultados'], ['Modelo','n_features','F1_media','F1_std','n_folds','alcance'], '04', corpus)
    exigir_columnas(tablas['04_features'], ['feature'], '04', corpus)
    if len(tablas['04_resultados']) != 2 or set(tablas['04_resultados'].alcance) != {'exploratorio'}:
        problema('04', 'se esperan dos resultados exploratorios.', corpus)
    n_sel = r4['seleccion']['features_seleccionadas']
    n_orig = r4['seleccion']['features_entrada']
    if not 0 < n_sel < n_orig:
        problema('04', 'recuento de características inválido.', corpus)
    cerca(n_sel, tablas['04_features'].feature.nunique(), 'Features seleccionadas únicas', '04', corpus, 0)
    cerca(n_sel, len(tablas['04_features']), 'Filas de features', '04', corpus, 0)
    exploratorios = []
    for nombre, prefijo, n in [('Baseline','f1_baseline_comparacion',n_orig), ('GWO','f1_gwo_comparacion',n_sel)]:
        filas = tablas['04_resultados'][tablas['04_resultados'].Modelo.str.startswith(nombre)]
        if len(filas) != 1:
            problema('04', f'falta el resultado único de {nombre}.', corpus)
        valores = pd.Series(r4['cv'][prefijo], dtype=float)
        if len(valores)<2 or not valores.between(0,1).all():
            problema('04', 'F1 exploratorios inválidos.', corpus)
        fila = filas.iloc[0]
        for columna, esperado in [('n_features',n), ('n_folds',len(valores)), ('F1_media',valores.mean()), ('F1_std',valores.std(ddof=1))]:
            cerca(fila[columna], esperado, columna, '04', corpus)
        exploratorios.append({'modelo': nombre, 'n_features': int(n),
                              'f1_media': float(valores.mean()), 'f1_std': float(valores.std(ddof=1))})
    test = resumenes['02']['test_permutacion']
    p = test['p_valor']
    cerca(p, (test['permutaciones_extremas']+1)/(test['n_permutaciones']+1), 'p-valor corregido', '02', corpus, 1e-8)
    if not 0 <= p <= 1 or bool(test['significativo']) != (p < test['alpha']):
        problema('02', 'p-valor o decisión de significancia inconsistente.', corpus)
    media_base, media_gwo, media_rob = [m['f1_macro_media_folds'] for m in metricas]
    reporte = {
        'metadata': {'version': '6.0', 'fecha_generacion_utc': datetime.now(timezone.utc).isoformat(),
                     'estado': 'resultados_consolidados_verificados', 'sha256_script': sha256_archivo(__file__)},
        'corpus': {'seleccion': corpus, 'archivo': str(entrada.resolve()), 'sha256': huellas['corpus'],
                   'total_perfiles': len(df), 'grupos_validacion': int(p5.grupo_cv.nunique()),
                   'distribucion': {str(k):int(v) for k,v in df.grado.value_counts().items()}},
        'analisis_semantico': {'pca_lda': resumenes['01'], 'homogeneidad': resumenes['02'], 'diferenciacion_lexica': resumenes['03']},
        'gwo_exploratorio': {'resumen': r4, 'metricas': exploratorios},
        'gwo_validacion_anidada': resumenes['05'],
        'resultado_robusto_final': resumenes['06'],
        'metricas_evaluacion': metricas,
        'comparaciones_descriptivas': {
            'delta_f1_medio_gwo_menos_baseline': media_gwo-media_base,
            'delta_f1_medio_enmascarado_menos_baseline': media_rob-media_base,
            'nota': 'Mismos perfiles, grupos y folds. Diferencias descriptivas; no se realizó un test de superioridad.'},
        'advertencias_metodologicas': [
            'PCA y LDA son visualizaciones del corpus completo; LDA utiliza etiquetas y no demuestra rendimiento predictivo.',
            'La similitud TF-IDF mide coincidencia léxica ponderada; no demuestra equivalencia de competencias.',
            'El p-valor del paso 02 corresponde a unidades agrupadas, distintas de los pares descriptivos de perfiles.',
            'La ausencia de significancia no demuestra igualdad entre grados.',
            'Los términos distintivos describen este corpus; no son competencias exclusivas ni pruebas de significancia.',
            'Paso 04 es exploratorio: selección e IDF usan información global. Paso 05 evalúa selección dentro de train externo.',
            'Paso 06 usa ComplementNB + SMOTE con títulos enmascarados y sin GWO; no es GWO con enmascaramiento.',
            'F1 macro no equivale a precisión ni accuracy. Media de folds y F1 conjunto son agregaciones diferentes.',
            'La desviación de los folds no es un intervalo de confianza. No se demuestra superioridad con una diferencia de medias.',
            'Agrupar perfiles vinculados reduce dependencia conocida, pero no constituye prueba en universidades desconocidas.',
            'Enmascarar títulos no elimina todas las pistas institucionales o de grado.',
            'Las métricas ponderan perfiles, no grupos. La clase minoritaria limita la validación.',
            'Cambiar el protocolo después de observar resultados requiere una evaluación futura independiente.'],
        'fuentes': {k: {'archivo': str(p.resolve()), 'sha256': huellas[k]} for k,p in requeridos.items()},
    }
    # No mezclar archivos que cambian mientras se prepara el reporte.
    for k,p in requeridos.items():
        if sha256_archivo(p) != huellas[k]:
            raise ValueError(f'El archivo {p} cambió durante la lectura. Repite el reporte.')
    return reporte, base


def celda(valor):
    return str(valor).replace('|', '\\|').replace('\n', ' ')


def renderizar_markdown(r):
    corpus = r['corpus']
    analisis = r['analisis_semantico']
    test = analisis['homogeneidad']['test_permutacion']
    met = r['metricas_evaluacion']
    filas = [f"# Reporte de resultados — corpus {corpus['seleccion']}", '',
             f"Generado: {r['metadata']['fecha_generacion_utc']}", '',
             f"**{corpus['total_perfiles']} perfiles; {corpus['grupos_validacion']} grupos de validación.**", '',
             '| Grado | Perfiles |', '|---|---:|']
    filas.extend(f'| {c} | {corpus["distribucion"][c]} |' for c in CLASES)
    filas.extend(['', '## Evaluación predictiva', '',
                  'F1 macro entre 0 y 1. OOF reúne una predicción de prueba por perfil. '
                  'La desviación entre folds no es un intervalo de confianza.', '',
                  '| Paso y modelo | Folds | F1 medio | Desv. folds | F1 OOF | Accuracy OOF |',
                  '|---|---:|---:|---:|---:|---:|'])
    nombres = ['05 — TF-IDF completo + SMOTE + ComplementNB', '05 — GWO + SMOTE + ComplementNB',
               '06 — Títulos enmascarados + SMOTE + ComplementNB']
    for nombre, m in zip(nombres, met):
        filas.append(f"| {nombre} | {m['n_folds']} | {m['f1_macro_media_folds']:.4f} | "
                     f"{m['f1_macro_std_folds']:.4f} | {m['f1_macro_oof']:.4f} | {m['accuracy_oof']:.4f} |")
    dif = r['comparaciones_descriptivas']
    parametros_gwo = r['gwo_validacion_anidada']['parametros']
    filas.extend(['', f"GWO menos baseline: **{dif['delta_f1_medio_gwo_menos_baseline']:+.4f}** en F1 medio. "
                  f"Enmascarado menos baseline: **{dif['delta_f1_medio_enmascarado_menos_baseline']:+.4f}**. "
                  'Son diferencias descriptivas; no prueban superioridad estadística.', '',
                  'El paso 05 repite la selección de características dentro de cada entrenamiento externo. '
                  'El paso 06 evalúa un modelo fijo con títulos enmascarados, sin selección GWO.', '',
                  f"Búsqueda GWO del paso 05: {parametros_gwo['epochs']} iteraciones y "
                  f"{parametros_gwo['poblacion']} lobos por entrenamiento externo.", '',
                  '## Selección GWO exploratoria (paso 04)', '',
                  '| Modelo | Características | F1 medio exploratorio |', '|---|---:|---:|'])
    for m in r['gwo_exploratorio']['metricas']:
        filas.append(f"| {m['modelo']} | {m['n_features']} | {m['f1_media']:.4f} |")
    exploracion = r['gwo_exploratorio']['resumen']['optimizador']
    filas.extend(['', f"Búsqueda exploratoria: {exploracion['epochs']} iteraciones y "
                  f"{exploracion['poblacion']} lobos.", '',
                  'Estas cifras usan un vocabulario global y una selección previa a la comparación; '
                  'no se interpretan como rendimiento de prueba independiente.', '', '## Estructura del corpus', '',
                  f"PCA, varianza explicada en dos componentes: "
                  f"{analisis['pca_lda']['pca']['varianza_explicada_total_2d']:.2%}. "
                  'LDA utiliza las etiquetas y su gráfico es exploratorio.', '',
                  f"Test de permutación: p = **{test['p_valor']:.8f}**, "
                  f"{test['n_permutaciones']} permutaciones; "
                  f"{test['agrupacion']['n_unidades']} unidades del test. "
                  + ('Alcanza' if test['significativo'] else 'No alcanza')
                  + f" el umbral declarado α = {test['alpha']}. "
                  'El resultado describe asociación bajo este protocolo; no demuestra igualdad o equivalencia de competencias.', '',
                  '### Similitud coseno entre centroides', '', '| Grados | Similitud |', '|---|---:|'])
    matriz = analisis['homogeneidad']['similitud_centroides']
    for i,c in enumerate(CLASES):
        for otro in CLASES[i+1:]:
            filas.append(f'| {c}–{otro} | {matriz[c][otro]:.4f} |')
    filas.extend(['', '### Vocabulario distintivo', '',
                  'Hasta cinco términos por grado, ordenados por la razón de prevalencia del paso 03. '
                  'No implican competencias exclusivas.', ''])
    for c in CLASES:
        terminos = analisis['diferenciacion_lexica']['top_terminos_por_grado'].get(c, [])
        lista = ', '.join(celda(t['termino']) for t in terminos[:5]) or 'sin términos que cumplan el criterio'
        filas.append(f'- **{c}:** {lista}.')
    mascaras = r['resultado_robusto_final']['control_fuga_lexica']
    filas.extend(['', '## Enmascaramiento', '',
                  f"Se eliminaron {mascaras['denominaciones_enmascaradas']} menciones de títulos en "
                  f"{mascaras['documentos_con_titulos_enmascarados']} perfiles. "
                  'La auditoría del paso 06 conserva los textos originales y transformados.', '',
                  '## Alcance y límites', ''])
    filas.extend('- '+a for a in r['advertencias_metodologicas'])
    filas.extend(['', '## Procedencia', '', f"SHA-256 del corpus: `{corpus['sha256']}`.", '',
                  'Se verificaron el corpus de los seis resúmenes, la cobertura de predicciones y '
                  'la concordancia de métricas con sus CSV. El JSON conserva los resúmenes completos y '
                  'las rutas y huellas de todos los archivos leídos. Esta comprobación no sustituye '
                  'la revisión de la extracción y las etiquetas del corpus.', ''])
    return '\n'.join(filas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--corpus', choices=['actual', 'v2'])
    args = parser.parse_args(argv)
    corpus = args.corpus or solicitar_corpus()
    print(f'\nFASE 3 — REPORTE CONSOLIDADO ({corpus})')
    reporte, base = consolidar(corpus)
    # Preparar todas las representaciones antes de escribir cualquiera.
    contenido_json = json.dumps(reporte, ensure_ascii=False, indent=2, allow_nan=False)
    contenido_md = renderizar_markdown(reporte)
    tabla = pd.DataFrame([{k:v for k,v in m.items() if k != 'f1_por_fold'}
                          for m in reporte['metricas_evaluacion']])
    base.mkdir(parents=True, exist_ok=True)
    salidas = [base/f'resultados_finales_{corpus}.json', base/f'reporte_final_{corpus}.md',
               base/f'resumen_metricas_{corpus}.csv']
    salidas[0].write_text(contenido_json, encoding='utf-8')
    salidas[1].write_text(contenido_md, encoding='utf-8')
    tabla.to_csv(salidas[2], index=False, encoding='utf-8-sig')
    print(f"Verificados: {reporte['corpus']['total_perfiles']} perfiles; "
          f"{reporte['corpus']['grupos_validacion']} grupos; pasos 01–06.")
    print(tabla[['paso','modelo','f1_macro_media_folds','f1_macro_oof']].to_string(index=False))
    for salida in salidas:
        print(f'Generado: {salida}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        print(f'ERROR: no se generó el reporte. {exc}\n'
              'Revisa que los pasos 01–06 estén actualizados y ejecutados para el mismo corpus.', file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f'Dependencia ausente o incompatible: {exc}', file=sys.stderr)
        sys.exit(1)
