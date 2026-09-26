# -*- coding: utf-8 -*-

from __future__ import annotations
import re
import shutil
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
import unicodedata
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


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
    global RUTA_TERMINOS, RUTA_AUDITORIA, RUTA_RESUMEN, RUTA_GRAFICO
    if corpus not in {"actual", "v2"}:
        raise ValueError("Corpus no válido: usa actual o v2.")
    CORPUS_SELECCIONADO = corpus
    nombre = ("perfiles_egreso_etiquetado_actual_corregido.csv"
              if corpus == "actual" else "perfiles_egreso_etiquetado_v2.csv")
    RUTA_ENTRADA = os.path.join(SRC_ROOT, "data", "processed", nombre)
    RESULTADOS_DIR = os.path.join(SRC_ROOT, "data", "resultados_cientificos", "diferenciacion_lexica")
    def ruta(nombre, extension):
        return os.path.join(RESULTADOS_DIR, f"{nombre}_{corpus}.{extension}")
    RUTA_TERMINOS = ruta("terminos_distintivos", "csv")
    RUTA_AUDITORIA = ruta("auditoria_terminos_excluidos", "csv")
    RUTA_RESUMEN = ruta("resumen_diferenciacion_lexica", "json")
    RUTA_GRAFICO = ruta("terminos_distintivos", "png")


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

TOP_N = 8
CLASES = ["Civil", "Ejecución", "Informática"]
VERSION_ANALISIS = "expresiones_contexto_auditado_v3"

# Suavizado para evitar divisiones por cero.
ALPHA = 0.5

# Un término debe aparecer en al menos dos perfiles
# de la clase para entrar en el ranking.
MIN_DOCUMENTOS_CLASE = 2

VECTORIZADOR_CONFIG = {
    "max_features": None,
    "ngram_range": (1, 4),
    "min_df": 2,
    "max_df": 1.0,
    "lowercase": True,
    "strip_accents": "unicode",
    "binary": True,
}


# =============================================================================
# 3. STOPWORDS PARA INTERPRETACIÓN
# =============================================================================
#
# Se utilizan SOLO en este análisis descriptivo.
#
# NO modifican:
# - el corpus de entrada;
# - GWO;
# - ComplementNB;
# - las validaciones.
# =============================================================================

STOPWORDS_ES = {
    "a", "al", "algo", "algunas", "algunos",
    "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde",
    "donde", "dos", "durante", "e", "el",
    "ella", "ellas", "ellos", "en", "entre",
    "era", "es", "esa", "esas", "ese", "eso",
    "esos", "esta", "estas", "este", "esto",
    "estos", "fue", "ha", "hacia", "hasta",
    "hay", "la", "las", "le", "les", "lo",
    "los", "mas", "me", "mi", "mis", "muy",
    "no", "nos", "o", "para", "pero", "por",
    "porque", "que", "se", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "tambien",
    "tanto", "te", "tiene", "todo", "todos",
    "tras", "tu", "un", "una", "uno", "unos",
    "y", "ya",    "catolica",
    "ucv",
    "cristiana",
    "cristianas",
    "cristiano",
    "cristianos",
    "catolicas",
    "catolico" ,
    "catolicos", 
    "distintas",
    "distintos",
    "distinto",
    "distinta",
    "alto",
    "altos",
    "alta",
    "altas",
    "sello",
    "sellos",
    "diversa",
    "diversas",
    "diverso",
    "diversos",
    "diversidad",
    "diversidades",
    "buenas",
    "buena",
    "bueno",
    "buenos",
    "destacar",
    "destacarse",
    "destacando",
    "destacado",
    "destacan",
    "partir",
    "acuerdo",
    "acuerdos",
    "generar",
    "generando",
    "generado",
    "generan",
    "persona",
    "componentes",
    "permitan",
    "critica",
    "ambientes",
    "laboral",
    "persona",
    "estudios",
    "junto",
    "podra",
    "contribuyendo", 
    "ano",
    "anos",
    "año",
    "años",
    "concepcion",
    "santiago", "todas", "todos", "toda", "todo", "universidad", "universidades", "instituto", "institutos",
    "magister", "magisters", "valparaiso", "chile", "chiles", "informatico", "informaticos", "informaticas", 
    "cualquiera", "cualesquiera", "cualquiera", "cualesquiera", "cualquier", "cualesquier", "cualquiera",
    "humana", "humanas", "humano", "humanos", "ingenieria", "ingenierias", "ingeniero", "ingenieros", "ingeniera", "ingenieras",
}

# Estas palabras permiten descartar bigramas poco informativos.
# No se borran del texto antes de formar los pares.
STOPWORDS_FUNCIONALES = set("""
a al algo algunas algunos ante antes como con contra
cual cuando de del desde donde durante e el ella ellas ellos
en entre era es esa esas ese eso esos esta estas este esto
estos fue ha hacia hasta hay la las le les lo los
mas me mi mis muy no nos o para pero por porque que
se ser si sin sobre son su sus tambien tanto te tiene
todo todos toda todas tras tu un una uno unos unas y ya
cualquier cualquiera cualesquiera
""".split())

STOPWORDS_ES.update({
    "todas",
    "cualquier",
})


# =============================================================================
# 4. TÉRMINOS EXCLUIDOS DE LA INTERPRETACIÓN
# =============================================================================

TOKENS_GRADO = {
    "civil",
    "ejecucion",
    "informatica",
    "informatico",
    "informaticos",
    "informaticas",
    "ingenieria",
    "ingeniero",
    "ingenieros",
    "ingeniera",
    "ingenieras",
}

# Exclusión descriptiva de términos aislados
# que no identifican una competencia o ámbito específico.
TERMINOS_GENERICOS_INTERPRETACION = {
    "desenvolverse",
    "continuo",
    "eficiente",
    "promueve", "desempena", "demuestra", "solida", "solido",
    "superior", "capaz", "capaces", "titulado", "titulados",
    "egresado", "egresados", "posee", "sera", "seran",
}


TOKENS_INSTITUCIONALES = {
    "universidad",
    "instituto",
    "magister",
    "valparaiso",
    "chile"

}

TOKENS_INSTITUCIONALES.update({
    "cristiana", "cristianas", "cristiano", "cristianos",
    "catolica", "catolicas", "catolico", "catolicos",
    "ucv", "concepcion", "santiago", "universidades", "institutos",
})


# =============================================================================
# 5. UTILIDADES
# =============================================================================

def normalizar_texto(
    texto: str,
) -> str:

    texto = str(texto).lower()

    texto = unicodedata.normalize(
        "NFD",
        texto,
    )

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return texto


def clasificar_exclusion(
    termino: str,
) -> str | None:

    normalizado = normalizar_texto(
        termino
    )

    if normalizado in TERMINOS_GENERICOS_INTERPRETACION:
        return "termino_generico_aislado"

    tokens = set(
        normalizado.split()
    )

    if tokens.intersection(
        TOKENS_GRADO
    ):
        return "denominacion_grado"

    if tokens.intersection(
        TOKENS_INSTITUCIONALES
    ):
        return "contexto_institucional_geografico"

    return None


# =============================================================================
# 6. CARGA DEL CORPUS
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
# 7. MATRIZ DE PRESENCIA DOCUMENTAL
# =============================================================================

# Conectores permitidos DENTRO de las expresiones, nunca en sus extremos.
# No se inventan uniones: todas las secuencias proceden del texto original.
# Se rechazan coordinaciones con y/e para evitar fragmentos de dos ideas.
CONECTORES_INTERNOS = set('de del la las el los en con para'.split())
EXCLUIR_UNIGRAMAS = {normalizar_texto(p) for p in STOPWORDS_ES}
FUNCIONALES = {normalizar_texto(p) for p in STOPWORDS_FUNCIONALES}
COLUMNAS = ['grado', 'termino', 'log2_ratio_prevalencia', 'documentos_grado',
            'total_documentos_grado', 'documentos_resto', 'total_documentos_resto',
            'prevalencia_grado', 'prevalencia_resto', 'instituciones_grado',
            'grupos_apoyo_grado']


def secuencias_originales(texto):
    """Devuelve clave normalizada, expresión literal y segmento de contexto."""
    # Números, guiones, puntuación y saltos son límites. No se saltan palabras.
    for segmento in re.split(r'[^\w\s]+|[\r\n]+|\d+|_', str(texto)):
        palabras = re.findall(r'[^\W\d_]+', segmento, flags=re.UNICODE)
        for inicio in range(len(palabras)):
            for largo in range(1, VECTORIZADOR_CONFIG['ngram_range'][1] + 1):
                trozo = palabras[inicio:inicio + largo]
                if len(trozo) != largo:
                    break
                literal = ' '.join(trozo)
                yield normalizar_texto(literal), literal, segmento.strip()


# Reglas exploratorias revisadas con los CSV de evidencias. Iguales para las clases.
# El ranking visual prioriza expresiones; los unigramas válidos siguen en el CSV completo.
MIN_PALABRAS_GRAFICO = 2
FUNCIONALES.update('dicho dicha dichos dichas mismo misma mismos mismas segun cuyo cuya cuyos cuyas'.split())
SIGLAS_CORTAS = {'ti', 'ia', 'sql', 'api'}
TOKENS_IDENTIDAD = {'identitario', 'identitaria', 'identitarios', 'identitarias',
                   'institucion', 'institucional', 'institucionales', 'carrera', 'carreras'}
EXTREMOS_VAGOS = {'ambito', 'ambitos', 'conocimiento', 'conocimientos',
                 'diverso', 'diversos', 'diversa', 'diversas', 'acorde',
                 'actua', 'actuan', 'dar', 'adaptandose',
                 'alto', 'alta', 'altos', 'altas', 'toma'}
MARCADORES_MENU = ('historia institucional', 'conoce nuestros campus',
                  'agrupaciones estudiantiles', 'programa proenta',
                  'acreditacion institucional', 'solicita informacion',
                  'politica de privacidad', 'todos los derechos reservados')
ALIAS_INSTITUCIONES = {
    'universidad de la santisima concepcion': 'universidad catolica de la santisima concepcion',
    'universidad catolica de la santisima concepcion': 'universidad catolica de la santisima concepcion',
}


def motivo_contexto(contexto):
    texto = normalizar_texto(contexto)
    # Excluye todo el segmento de sello explícito, evitando que sobrevivan
    # subfrases institucionales sin el token que provocó el filtro.
    if 'sello identitario' in texto or 'vision cristiana' in texto:
        return 'segmento_de_identidad_institucional'
    # Dos señales evitan descartar por una sola mención académica legítima.
    return ('segmento_con_indicios_de_menu' if
            sum(m in texto for m in MARCADORES_MENU) >= 2 else None)


def institucion_canonica(fila):
    nombre = ' '.join(normalizar_texto(str(fila.get('universidad', ''))).split())
    host = (urlparse(str(fila.get('url', ''))).hostname or '').lower()
    # Alias comprobados en las fuentes aportadas. No agrupa otros dominios automáticamente.
    if host == 'ucsc.cl' or host.endswith('.ucsc.cl'):
        return 'universidad catolica de la santisima concepcion'
    return ALIAS_INSTITUCIONES.get(nombre, nombre)


def motivo_candidato(termino):
    palabras = termino.split()
    if set(palabras) & TOKENS_IDENTIDAD:
        return 'identidad_institucional_declarada'
    if any(len(p) < 3 and p not in FUNCIONALES and p not in SIGLAS_CORTAS for p in palabras):
        return 'fragmento_corto_no_reconocido'
    if len(palabras) > 1 and (palabras[0] in EXTREMOS_VAGOS or palabras[-1] in EXTREMOS_VAGOS):
        return 'extremo_vago_declarado'
    if len(palabras) == 1:
        if len(termino) < 2 or termino in EXCLUIR_UNIGRAMAS:
            return 'stopword_o_unigrama_corto'
    else:
        if palabras[0] in FUNCIONALES or palabras[-1] in FUNCIONALES:
            return 'extremo_funcional'
        if any(p in FUNCIONALES and p not in CONECTORES_INTERNOS
               for p in palabras[1:-1]):
            return 'expresion_funcional'
        if sum(p not in FUNCIONALES for p in palabras) < 2:
            return 'menos_de_dos_palabras_con_contenido'
        if (palabras[0] in TERMINOS_GENERICOS_INTERPRETACION or
                palabras[-1] in TERMINOS_GENERICOS_INTERPRETACION):
            return 'extremo_generico_declarado'
    return clasificar_exclusion(termino)


def construir_matriz(df):
    """Presencia binaria; conserva todos los candidatos con min_df=2."""
    documentos = []
    originales = {}
    exclusiones = Counter()
    for texto in df['perfil_egreso'].astype(str):
        aceptados = set()
        descartados = set()
        for termino, literal, contexto in secuencias_originales(texto):
            motivo = motivo_contexto(contexto) or motivo_candidato(termino)
            if motivo:
                descartados.add((termino, motivo))
            else:
                aceptados.add(termino)
                originales.setdefault(termino, Counter())[literal.lower()] += 1
        documentos.append(sorted(aceptados))
        exclusiones.update(descartados)
    # El analizador recibe candidatos ya extraídos; no vuelve a tokenizarlos.
    vectorizador = CountVectorizer(analyzer=lambda x: x, lowercase=False,
                                  token_pattern=None, binary=True,
                                  min_df=VECTORIZADOR_CONFIG['min_df'],
                                  max_df=VECTORIZADOR_CONFIG['max_df'],
                                  max_features=VECTORIZADOR_CONFIG['max_features'])
    try:
        X = vectorizador.fit_transform(documentos)
    except ValueError as error:
        raise ValueError('No hay vocabulario con el soporte requerido. '
                         'Revisa los textos y los filtros; no se generaron términos artificiales.') from error
    vectorizador.etiquetas_originales_ = {
        t: sorted(c, key=lambda v: (-c[v], v))[0] for t, c in originales.items()}
    vectorizador.exclusiones_candidatos_ = exclusiones
    return vectorizador, X, vectorizador.get_feature_names_out()


def calcular_puntuacion(docs_clase, docs_resto, n_clase, n_resto):
    if ALPHA <= 0 or min(n_clase, n_resto) <= 0:
        raise ValueError('El suavizado y los tamaños de clase deben ser positivos.')
    return np.log2(((docs_clase + ALPHA) / (n_clase + 2 * ALPHA)) /
                   ((docs_resto + ALPHA) / (n_resto + 2 * ALPHA)))


def grupos_descriptivos(df):
    """Une grupos declarados y textos idénticos normalizados; no es CV."""
    padre = list(range(len(df)))
    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    vistos = {}
    for i, (_, fila) in enumerate(df.iterrows()):
        texto = ' '.join(normalizar_texto(fila['perfil_egreso']).split())
        claves = [('texto', texto)]
        grupo = str(fila.get('grupo_perfil', '')).strip()
        if grupo:
            claves.append(('declarado', grupo))
        for clave in claves:
            if clave in vistos:
                padre[raiz(i)] = raiz(vistos[clave])
            else:
                vistos[clave] = i
    return [raiz(i) for i in range(len(df))]


def contiene_secuencia(larga, corta):
    a, b = larga.split(), corta.split()
    return len(a) > len(b) and any(a[i:i + len(b)] == b
                                    for i in range(len(a) - len(b) + 1))


def reducir_redundancia(candidatos, soportes):
    """Prefiere la expresión larga solo cuando aparece en los mismos perfiles."""
    # Igual soporte GLOBAL implica igual puntuación; no fusiona significados.
    orden = sorted(candidatos, key=lambda r: (-len(r['termino'].split()), r['termino']))
    retenidos, excluidos, por_soporte = [], [], {}
    for fila in orden:
        firma = soportes[fila['termino']]
        representante = next((r['termino'] for r in por_soporte.get(firma, [])
                              if contiene_secuencia(r['termino'], fila['termino'])), None)
        if representante:
            excluidos.append({**fila, 'motivo_exclusion': 'subexpresion_mismo_soporte',
                               'representada_por': representante})
        else:
            retenidos.append(fila)
            por_soporte.setdefault(firma, []).append(fila)
    return retenidos, excluidos


def generar_rankings(df, X, vocabulario):
    etiquetas = df['grado'].astype(str).to_numpy()
    grupos = grupos_descriptivos(df)
    instituciones = [institucion_canonica(fila) for _, fila in df.iterrows()]
    csc = X.tocsc()
    soportes = {str(t): frozenset(map(int, csc.indices[csc.indptr[j]:csc.indptr[j+1]]))
                for j, t in enumerate(vocabulario)}
    resultados, excluidos, completos = [], [], []
    for clase in CLASES:
        mascara = etiquetas == clase
        n = int(mascara.sum())
        conteos = np.asarray(X[mascara].sum(axis=0)).ravel()
        resto = np.asarray(X[~mascara].sum(axis=0)).ravel()
        scores = calcular_puntuacion(conteos, resto, n, len(df) - n)
        candidatos = []
        for j, termino in enumerate(vocabulario):
            termino = str(termino)
            indices = [i for i in soportes[termino] if mascara[i]]
            fila = dict(zip(COLUMNAS, [clase, termino, float(scores[j]), int(conteos[j]),
                n, int(resto[j]), len(df)-n, float(conteos[j]/n),
                float(resto[j]/(len(df)-n)),
                len({instituciones[i] for i in indices if instituciones[i]}),
                len({grupos[i] for i in indices})]))
            completos.append(fila)
            motivo = ('unigrama_conservado_en_csv_completo' if len(termino.split()) < MIN_PALABRAS_GRAFICO
                      else 'soporte_insuficiente_en_clase' if conteos[j] < MIN_DOCUMENTOS_CLASE
                      else 'prevalencia_suavizada_no_superior_al_resto' if scores[j] <= 0
                      else None)
            if motivo:
                excluidos.append({**fila, 'motivo_exclusion': motivo, 'representada_por': ''})
            else:
                candidatos.append(fila)
        candidatos, redundantes = reducir_redundancia(candidatos, soportes)
        excluidos.extend(redundantes)
        candidatos.sort(key=lambda r: (-r['log2_ratio_prevalencia'],
                                        -r['documentos_grado'], r['termino']))
        resultados.extend({**r, 'ranking': i} for i, r in enumerate(candidatos[:TOP_N], 1))
    return (pd.DataFrame(resultados, columns=COLUMNAS + ['ranking']),
            pd.DataFrame(excluidos, columns=COLUMNAS + ['motivo_exclusion', 'representada_por']),
            pd.DataFrame(completos, columns=COLUMNAS), soportes)


def seleccionar_compartidos(completos, soportes):
    """Al menos dos perfiles en CADA clase; orden por cobertura mínima."""
    candidatos = []
    for termino, filas in completos.groupby('termino', sort=True):
        if len(termino.split()) >= MIN_PALABRAS_GRAFICO and len(filas) == len(CLASES) and (filas['documentos_grado'] >= MIN_DOCUMENTOS_CLASE).all():
            candidatos.append({'termino': termino,
                'cobertura_minima': float(filas['prevalencia_grado'].min()),
                'cobertura_media_clases': float(filas['prevalencia_grado'].mean())})
    candidatos, _ = reducir_redundancia(candidatos, soportes)
    candidatos.sort(key=lambda r: (-r['cobertura_minima'],
                                    -r['cobertura_media_clases'], r['termino']))
    orden = {r['termino']: i for i, r in enumerate(candidatos[:TOP_N], 1)}
    seleccion = completos[completos.termino.isin(orden)].copy()
    seleccion['ranking'] = seleccion.termino.map(orden)
    return seleccion.sort_values(['ranking', 'grado'])


def ruta_extra(nombre, extension):
    return os.path.join(RESULTADOS_DIR, f'{nombre}_{CORPUS_SELECCIONADO}.{extension}')


def respaldar_salidas():
    previos = [p for p in Path(RESULTADOS_DIR).glob(f'*_{CORPUS_SELECCIONADO}.*')
               if p.is_file() and p.suffix in {'.csv', '.json', '.png'}]
    if not previos:
        return None
    carpeta = Path(RESULTADOS_DIR) / 'respaldos' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    carpeta.mkdir(parents=True)
    for p in previos:
        shutil.copy2(p, carpeta / p.name)
    return str(carpeta)


def generar_graficos(resultados, compartidos, etiquetas, df):
    colores = {'Civil': '#2573A6', 'Ejecución': '#C26727', 'Informática': '#26806A'}
    maximo = max(1.0, float(resultados.log2_ratio_prevalencia.max())) if not resultados.empty else 1.0
    def dibujar(ax, clase):
        datos = resultados[resultados.grado == clase].sort_values('ranking')
        ax.set_title(f'{clase} · {int((df.grado == clase).sum())} perfiles',
                     loc='left', fontsize=15, fontweight='bold', pad=14)
        if datos.empty:
            ax.text(.5, .5, 'Sin términos que cumplan los criterios',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_yticks([])
        else:
            y = np.arange(len(datos))
            ax.barh(y, datos.log2_ratio_prevalencia, color=colores[clase], height=.64)
            ax.set_yticks(y, [etiquetas.get(t, t) for t in datos.termino], fontsize=11)
            ax.invert_yaxis()
            for i, (_, fila) in enumerate(datos.iterrows()):
                nota = (f"{int(fila.documentos_grado)}/{int(fila.total_documentos_grado)}"
                        f" · resto {int(fila.documentos_resto)}/{int(fila.total_documentos_resto)}"
                        f"\n{int(fila.instituciones_grado)} inst. en la clase")
                ax.text(fila.log2_ratio_prevalencia + maximo * .02, i, nota, va='center', fontsize=9)
        ax.set_xlim(0, maximo * 1.55)
        ax.set_xlabel('Log₂ de la razón de prevalencias suavizadas · clase / resto', fontsize=10)
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), layout='constrained')
    for ax, clase in zip(axes, CLASES):
        dibujar(ax, clase)
    fig.suptitle('Expresiones distintivas por grado\n'
                 'Presencia en perfiles; mayor valor = mayor presencia relativa', fontsize=17)
    fig.supxlabel('Selección descriptiva · no implica competencias exclusivas ni significancia estadística', fontsize=10)
    fig.savefig(RUTA_GRAFICO, dpi=220)
    plt.close(fig)
    for clase in CLASES:
        fig, ax = plt.subplots(figsize=(13, 6.3), layout='constrained')
        dibujar(ax, clase)
        fig.suptitle('Expresiones distintivas · análisis descriptivo', fontsize=16)
        # Las barras ponderan perfiles; el CSV informa los grupos de apoyo.
        fig.supxlabel('Conteos por perfil; consultar grupos e instituciones de apoyo en el CSV. '
                       'No son competencias exclusivas.', fontsize=9)
        fig.savefig(ruta_extra('terminos_distintivos_' + normalizar_texto(clase), 'png'), dpi=220)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6.5), layout='constrained')
    terminos = list(compartidos.sort_values('ranking').termino.drop_duplicates())
    if terminos:
        valores, anotaciones = [], []
        for t in terminos:
            filas = compartidos[compartidos.termino == t].set_index('grado')
            valores.append([100 * float(filas.loc[c, 'prevalencia_grado']) for c in CLASES])
            anotaciones.append([f"{100*filas.loc[c,'prevalencia_grado']:.0f}%\n"
                                f"({int(filas.loc[c,'documentos_grado'])}/{int(filas.loc[c,'total_documentos_grado'])})"
                                for c in CLASES])
        im = ax.imshow(valores, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(3), CLASES)
        ax.set_yticks(range(len(terminos)), [etiquetas.get(t, t) for t in terminos])
        for i in range(len(terminos)):
            for j in range(3):
                ax.text(j, i, anotaciones[i][j], ha='center', va='center', fontsize=10,
                        color='white' if valores[i][j] >= 60 else '#182B40')
        fig.colorbar(im, ax=ax, label='Porcentaje de perfiles que contienen la expresión')
    else:
        ax.text(.5, .5, 'Sin expresiones presentes en al menos dos perfiles de cada clase',
                ha='center', va='center', transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
    ax.set_title('Vocabulario compartido entre los tres grados', fontsize=16, pad=18)
    fig.supxlabel('Orden por la menor cobertura entre clases. Coincidencia textual no implica equivalencia de competencias.', fontsize=9)
    fig.savefig(ruta_extra('terminos_compartidos', 'png'), dpi=220)
    plt.close(fig)


def exportar_evidencias(df, seleccionados, soportes):
    evidencias = []
    for termino in sorted(seleccionados):
        for i in sorted(soportes[termino]):
            fila = df.iloc[i]
            ejemplo = next((ctx for t, literal, ctx in secuencias_originales(fila.perfil_egreso)
                            if t == termino and not motivo_contexto(ctx)), '')
            evidencias.append({'termino': termino, 'fila_csv': i + 2,
                'grado': fila.grado, 'universidad': fila.get('universidad', ''),
                'carrera': fila.get('carrera', ''), 'url': fila.get('url', ''),
                'grupo_perfil': fila.get('grupo_perfil', ''), 'fragmento_original': ejemplo})
    pd.DataFrame(evidencias, columns=['termino', 'fila_csv', 'grado', 'universidad',
        'carrera', 'url', 'grupo_perfil', 'fragmento_original']).to_csv(
        ruta_extra('evidencias_terminos', 'csv'), index=False, encoding='utf-8-sig')


def guardar_resumen(df, vectorizador, resultados, excluidos, compartidos, respaldo):
    distribucion = {str(k): int(v) for k, v in df.grado.value_counts().items()}
    grupos = df.get('grupo_perfil', pd.Series(dtype=str)).astype(str).str.strip()
    conteos = grupos[grupos.ne('')].value_counts()
    resumen = {
        'fecha_ejecucion_utc': datetime.now(timezone.utc).isoformat(),
        'version_analisis': VERSION_ANALISIS,
        'sha256_script': sha256_archivo(__file__),
        'corpus': {'seleccion': CORPUS_SELECCIONADO, 'archivo_entrada': RUTA_ENTRADA,
                   'sha256': df.attrs['sha256_entrada'], 'total': len(df),
                   'distribucion': distribucion, 'filas_descartadas': 0},
        'metodo': {'top_maximo_por_grado': TOP_N,
            'criterio_puntuacion': 'expresiones de 2 a 4 palabras; log2_ratio_prevalencia > 0; al menos 2 perfiles de la clase',
            'formula': 'log2(((docs_grado+alpha)/(n_grado+2*alpha))/((docs_resto+alpha)/(n_resto+2*alpha)))',
            'prevalencias_csv': 'proporciones sin suavizar; la puntuación aplica alpha',
            'desempate': 'puntuación descendente, documentos del grado descendente, término alfabético',
            'unidad_analisis': 'presencia binaria por perfil; se conservan todas las filas',
            'vectorizador': 'CountVectorizer(binary=True) con analizador de candidatos contiguos',
            'max_features': None, 'features_generadas': len(vectorizador.get_feature_names_out()),
            'ngram_range': [1, 4], 'min_df': 2, 'max_df': 1.0,
            'min_documentos_clase': MIN_DOCUMENTOS_CLASE, 'suavizado_alpha': ALPHA,
            'redundancia': 'se omite una subexpresión solo si tiene idéntico conjunto de perfiles de aparición que una expresión más larga elegible',
            'compartidos': 'al menos 2 perfiles de cada clase; orden por cobertura mínima descendente, media entre clases descendente y término',
            'grupos_apoyo': 'unión de grupo_perfil declarado y textos idénticos normalizados; conteo descriptivo, no grupos CV',
            'instituciones_apoyo': 'nombres normalizados y alias UCSC comprobados en nombres/URL; otros alias pueden persistir'},
        'filtrado_interpretativo': {
            'min_palabras_grafico': MIN_PALABRAS_GRAFICO,
            'unigramas': 'conservados en prevalencias_todos_terminos, fuera de las figuras',
            'tokens_identidad': sorted(TOKENS_IDENTIDAD),
            'extremos_vagos': sorted(EXTREMOS_VAGOS),
            'marcadores_menu': list(MARCADORES_MENU),
            'umbral_marcadores_menu': 2,
            'marcadores_segmento_identidad': ['sello identitario', 'vision cristiana'],
            'alias_instituciones': ALIAS_INSTITUCIONES,
            'tokens_grado': sorted(TOKENS_GRADO),
            'tokens_institucionales': sorted(TOKENS_INSTITUCIONALES),
            'lista_stopwords': sorted(EXCLUIR_UNIGRAMAS),
            'stopwords_funcionales': sorted(FUNCIONALES),
            'conectores_internos': sorted(CONECTORES_INTERNOS),
            'terminos_genericos_declarados': sorted(TERMINOS_GENERICOS_INTERPRETACION),
            'orden': 'extraer secuencias originales, filtrar candidatos, min_df, puntuar, reducir redundancia y seleccionar',
            'afecta_corpus_original': False, 'afecta_gwo': False,
            'registros_excluidos': len(excluidos),
            'exclusiones_por_motivo': {str(k): int(v) for k,v in excluidos.motivo_exclusion.value_counts().items()},
            'auditoria_candidatos': ruta_extra('auditoria_candidatos_filtrados', 'csv')},
        'top_terminos_por_grado': {c: resultados[resultados.grado == c].to_dict('records') for c in CLASES},
        'terminos_compartidos': compartidos.to_dict('records'),
        'grupos_perfil_repetidos': {str(k): int(v) for k,v in conteos.items() if v > 1},
        'respaldo_salidas_anteriores': respaldo,
        'interpretacion': {'uso': 'exploración léxica complementaria revisada tras observar resultados',
                          'advertencia': 'Las expresiones son secuencias textuales, no competencias reconocidas automáticamente.'},
        'limitaciones': [
            f"La clase Ejecución contiene {distribucion.get('Ejecución',0)} perfiles.",
            'Las listas y reglas interpretativas se revisaron tras observar resultados: análisis exploratorio, no confirmatorio.',
            'No modifica el CSV de entrada ni los experimentos predictivos; no exige repetir GWO.',
            'Mantiene palabras funcionales internas permitidas y no salta palabras ni cruza puntuación.',
            'No usa un analizador gramatical ni lematiza: puede conservar frases incompletas y variantes morfológicas.',
            'La relevancia académica requiere revisar los fragmentos y fuentes del CSV de evidencias.',
            'Las figuras se restringen a expresiones de 2 a 4 palabras; no representan todas las palabras distintivas.',
            'Se filtran segmentos con dos o más indicios de menú; no sustituye corregir la extracción original.',
            'Los alias UCSC se unifican solo para contar instituciones; no se eliminan perfiles ni se cambian grupos CV.',
            'El soporte por perfiles puede reflejar textos vinculados o una sola institución; revisar columnas de apoyo.',
            'El filtrado institucional usa listas declaradas y no garantiza retirar todos los nombres propios.',
            'Un término distintivo no es exclusivo de una clase. No hay prueba de significancia por término.',
            'El gráfico compartido mide presencia textual y no equivalencia de competencias.',
            'El reporte 07 puede leer el top distintivo; la figura compartida es un resultado complementario.'
        ]}
    with open(RUTA_RESUMEN, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, allow_nan=False)
    return resumen


def main():
    parser = argparse.ArgumentParser(description='Expresiones distintivas y compartidas: análisis descriptivo.')
    parser.add_argument('--corpus', choices=['actual', 'v2'], help='Si se omite, muestra menú.')
    args = parser.parse_args()
    configurar_corpus(args.corpus or solicitar_corpus())
    df = cargar_corpus()
    print(f'CORPUS {CORPUS_SELECCIONADO.upper()}: {len(df)} perfiles\nEntrada: {RUTA_ENTRADA}')
    print('SHA-256:', df.attrs['sha256_entrada'])
    vectorizador, X, vocabulario = construir_matriz(df)
    resultados, excluidos, completos, soportes = generar_rankings(df, X, vocabulario)
    compartidos = seleccionar_compartidos(completos, soportes)
    for tabla in [resultados, excluidos, completos, compartidos]:
        tabla['expresion_original'] = tabla.termino.map(vectorizador.etiquetas_originales_)
    if sha256_archivo(RUTA_ENTRADA) != df.attrs['sha256_entrada']:
        raise ValueError('El CSV cambió durante el análisis. Repite la ejecución.')
    os.makedirs(RESULTADOS_DIR, exist_ok=True)
    respaldo = respaldar_salidas()
    for tabla, ruta in [(resultados, RUTA_TERMINOS), (excluidos, RUTA_AUDITORIA),
                        (completos, ruta_extra('prevalencias_todos_terminos', 'csv')),
                        (compartidos, ruta_extra('terminos_compartidos', 'csv'))]:
        tabla.to_csv(ruta, index=False, encoding='utf-8-sig')
    pd.DataFrame([{'termino': t, 'motivo_exclusion': m, 'documentos': n}
                  for (t,m), n in sorted(vectorizador.exclusiones_candidatos_.items())],
                 columns=['termino','motivo_exclusion','documentos']).to_csv(
        ruta_extra('auditoria_candidatos_filtrados', 'csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame([{'fila_csv': i + 2, 'universidad_original': fila.get('universidad', ''),
                   'universidad_canonica': institucion_canonica(fila), 'url': fila.get('url', '')}
                  for i, (_, fila) in enumerate(df.iterrows())]).to_csv(
        ruta_extra('auditoria_instituciones', 'csv'), index=False, encoding='utf-8-sig')
    exportar_evidencias(df, set(resultados.termino) | set(compartidos.termino), soportes)
    generar_graficos(resultados, compartidos, vectorizador.etiquetas_originales_, df)
    guardar_resumen(df, vectorizador, resultados, excluidos, compartidos, respaldo)
    for clase in CLASES:
        print(f'\n{clase}:')
        datos = resultados[resultados.grado == clase]
        if datos.empty:
            print('Sin expresiones que cumplan los criterios; no se completa artificialmente el top.')
        for _, r in datos.iterrows():
            print(f"  {r.expresion_original}: log2={r.log2_ratio_prevalencia:.3f}; "
                  f"{r.documentos_grado}/{r.total_documentos_grado} perfiles; "
                  f"resto {r.documentos_resto}/{r.total_documentos_resto}; "
                  f"{r.instituciones_grado} instituciones; {r.grupos_apoyo_grado} grupos de apoyo")
    print('\nResultados:', RESULTADOS_DIR)
    if respaldo:
        print('Salidas anteriores respaldadas:', respaldo)
    print('Se generaron el gráfico conjunto, tres gráficos por clase, la vista compartida y sus CSV.')
    print('Análisis descriptivo; revisar evidencias antes de interpretar expresiones como competencias.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f'ERROR: {error}') from None
