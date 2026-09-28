# -*- coding: utf-8 -*-
"""
EJECUTOR DE DEMOSTRACIÓN PARA LA DEFENSA
========================================

Objetivo
--------
Mostrar el pipeline completo del Proyecto de Título de forma ordenada,
entendible y compatible con una demostración de aproximadamente 5 minutos.

Flujo por defecto
-----------------
FASE 1
  - Ejecuta la recolección y etiquetado en modo DEMO (3 fuentes).

FASE 2
  - Audita el corpus actual.
  - Revisa/corrige encoding de forma controlada.

FASE 3
  01. PCA / LDA
  02. Homogeneidad y prueba de permutación
  03. Diferenciación léxica
  04. GWO exploratorio
  05. Validación anidada: por defecto NO se recalcula porque es el paso lento.
      Se leen y verifican sus resultados oficiales existentes.
  06. Modelo robusto con títulos enmascarados
  07. Reporte consolidado

Uso recomendado para la defensa
--------------------------------
    python .\ejecutar_demo_defensa.py

Opciones útiles
---------------
    --detalle completo
        Muestra toda la salida de los scripts hijos.

    --ejecutar-paso-05
        Recalcula la validación anidada completa. NO se recomienda durante
        la defensa porque vuelve a ejecutar GWO en cada fold externo.

    --rapido
        También reutiliza el resultado oficial del paso 04 GWO, útil como
        plan de emergencia si el tiempo disponible es muy corto.

    --limite-demo N
        Cambia la cantidad de fuentes de la Fase 1 demo. Por defecto: 3.

Notas
-----
- Este archivo no hace git add, commit ni push.
- La Fase 1 demo demuestra funcionamiento con pocas fuentes.
- Desde Fase 2 se trabaja con el corpus actual completo ya recolectado.
- El paso 05 solo se presenta como resultado reutilizado si su SHA-256
  coincide con el corpus actual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

FASE2 = ROOT / "src" / "Fase2_Procesamiento"
FASE3 = ROOT / "src" / "Fase3_Analisis"

CORPUS_ACTUAL = (
    ROOT
    / "src"
    / "data"
    / "processed"
    / "perfiles_egreso_etiquetado_actual_corregido.csv"
)

RESUMEN_GWO = (
    ROOT
    / "src"
    / "data"
    / "resultados_cientificos"
    / "gwo"
    / "actual"
    / "resumen_gwo.json"
)

RESUMEN_VALIDACION = (
    ROOT
    / "src"
    / "data"
    / "resultados_cientificos"
    / "gwo"
    / "actual"
    / "validacion_anidada"
    / "resumen_validacion_anidada.json"
)


# =============================================================================
# COLORES
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
AZUL = "\033[94m"
CIAN = "\033[96m"
VERDE = "\033[92m"
AMARILLO = "\033[93m"
ROJO = "\033[91m"
GRIS = "\033[90m"

USAR_COLOR = True


def color(texto: str, codigo: str) -> str:
    if not USAR_COLOR:
        return texto
    return codigo + texto + RESET


def configurar_terminal(sin_color: bool) -> None:
    global USAR_COLOR
    USAR_COLOR = not sin_color

    if os.name == "nt":
        # Activa secuencias ANSI en terminales modernas de Windows/VS Code.
        try:
            os.system("")
        except Exception:
            pass


# =============================================================================
# PRESENTACIÓN EN CONSOLA
# =============================================================================

def linea(caracter: str = "═", ancho: int = 78) -> str:
    return caracter * ancho


def portada() -> None:
    print()
    print(color(linea(), AZUL))
    print(color("  DEMOSTRACIÓN DE CÓDIGO — PROYECTO DE TÍTULO", BOLD + CIAN))
    print(color("  Descubrimiento de patrones en perfiles de egreso de Informática", BOLD))
    print(color(linea(), AZUL))
    print()
    print("  Flujo:  FASE 1  →  FASE 2  →  FASE 3  →  REPORTE FINAL")
    print("  Modo :  defensa de 5 minutos")
    print("  Python:  " + sys.executable)
    print()
    print(color(
        "  IMPORTANTE: el paso 05 se reutiliza por defecto porque recalcular "
        "la validación anidada completa demora demasiado para la defensa.",
        AMARILLO,
    ))
    print()


def encabezado(fase: str, titulo: str) -> None:
    print()
    print(color(linea("─"), AZUL))
    print(color("  " + fase + " — " + titulo, BOLD + CIAN))
    print(color(linea("─"), AZUL))


def ficha(que_hace: str, herramientas: str, observar: str) -> None:
    print(color("  QUÉ HACE      : ", BOLD) + que_hace)
    print(color("  HERRAMIENTAS  : ", BOLD) + herramientas)
    print(color("  QUÉ OBSERVAR  : ", BOLD) + observar)
    print()


def mensaje(texto: str, tipo: str = "info") -> None:
    colores = {
        "ok": VERDE,
        "aviso": AMARILLO,
        "error": ROJO,
        "info": CIAN,
    }
    simbolos = {
        "ok": "✓",
        "aviso": "!",
        "error": "✗",
        "info": "→",
    }
    print(
        "  "
        + color(simbolos.get(tipo, "→"), colores.get(tipo, CIAN))
        + " "
        + texto
    )


def ruta_visible(ruta: Path) -> str:
    try:
        relativa = ruta.relative_to(ROOT)
    except ValueError:
        return str(ruta)
    return str(relativa).replace("/", "\\")


def comando_visible(script: Path, argumentos: list[str]) -> str:
    resto = " ".join(argumentos)
    comando = "python .\\" + ruta_visible(script)
    if resto:
        comando += " " + resto
    return comando


# =============================================================================
# EJECUCIÓN CONTROLADA
# =============================================================================

def ejecutar_python(
    script: Path,
    argumentos: list[str],
    filtros: tuple[str, ...],
    detalle: str,
) -> float:
    if not script.is_file():
        raise FileNotFoundError("No se encontró el script: " + str(script))

    comando = [
        sys.executable,
        "-u",
        str(script),
        *argumentos,
    ]

    print(color("  COMANDO       : ", BOLD) + comando_visible(script, argumentos))
    print(color("  ESTADO        : ", BOLD) + "ejecutando...")
    print()

    entorno = os.environ.copy()
    entorno["PYTHONUTF8"] = "1"
    entorno["PYTHONIOENCODING"] = "utf-8"
    entorno["PYTHONUNBUFFERED"] = "1"

    inicio = time.perf_counter()
    ultimas_lineas: list[str] = []
    impresas = 0

    proceso = subprocess.Popen(
        comando,
        cwd=str(ROOT),
        env=entorno,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proceso.stdout is not None

    filtros_normalizados = tuple(x.casefold() for x in filtros)

    for linea_salida in proceso.stdout:
        texto = linea_salida.rstrip()
        if texto:
            ultimas_lineas.append(texto)
            ultimas_lineas = ultimas_lineas[-18:]

        mostrar = detalle == "completo"

        if not mostrar and texto:
            texto_normalizado = texto.casefold()
            mostrar = any(
                patron in texto_normalizado
                for patron in filtros_normalizados
            )

        if mostrar and texto:
            print(color("  │ ", GRIS) + texto)
            impresas += 1

    codigo = proceso.wait()
    duracion = time.perf_counter() - inicio

    if codigo != 0:
        print()
        mensaje(
            "El paso terminó con código " + str(codigo) + ".",
            "error",
        )
        if detalle != "completo" and ultimas_lineas:
            print(color("  Últimas líneas del error:", ROJO))
            for texto in ultimas_lineas:
                print("    " + texto)
        raise RuntimeError(
            "Falló: " + comando_visible(script, argumentos)
        )

    if impresas == 0:
        mensaje("Ejecución completada sin incidencias.", "ok")

    print()
    mensaje(
        "Paso completado en {:.1f} s.".format(duracion),
        "ok",
    )

    return duracion


# =============================================================================
# TRAZABILIDAD
# =============================================================================

def sha256_archivo(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def cargar_json(ruta: Path) -> dict:
    if not ruta.is_file():
        raise FileNotFoundError("No se encontró: " + str(ruta))
    return json.loads(ruta.read_text(encoding="utf-8-sig"))


def verificar_sha_resumen(
    resumen: dict,
    sha_guardado: str,
    descripcion: str,
) -> None:
    if not CORPUS_ACTUAL.is_file():
        raise FileNotFoundError(
            "No existe el corpus actual corregido: " + str(CORPUS_ACTUAL)
        )

    sha_actual = sha256_archivo(CORPUS_ACTUAL)

    if sha_guardado.lower() != sha_actual.lower():
        raise ValueError(
            descripcion
            + " corresponde a otro corpus. "
            + "No se mostrará como resultado oficial."
        )


# =============================================================================
# RESULTADOS REUTILIZADOS
# =============================================================================

def mostrar_gwo_existente() -> None:
    encabezado(
        "FASE 3 / PASO 04",
        "Selección de características con GWO — resultado reutilizado",
    )
    ficha(
        "Busca un subconjunto de las características TF-IDF.",
        "Grey Wolf Optimizer, 30 lobos y 100 iteraciones.",
        "Cuántas variables conserva y si mantiene el rendimiento exploratorio.",
    )

    resumen = cargar_json(RESUMEN_GWO)
    corpus = resumen["corpus"]

    verificar_sha_resumen(
        resumen,
        corpus["sha256"],
        "El resumen GWO",
    )

    seleccion = resumen["seleccion"]
    baseline = resumen["cv"]["f1_baseline_comparacion"]
    gwo = resumen["cv"]["f1_gwo_comparacion"]

    media_baseline = sum(baseline) / len(baseline)
    media_gwo = sum(gwo) / len(gwo)

    mensaje(
        "Modo rápido: no se recalcula GWO durante la demostración.",
        "aviso",
    )
    mensaje(
        "{} → {} características ({:.1f}% menos).".format(
            seleccion["features_entrada"],
            seleccion["features_seleccionadas"],
            seleccion["reduccion_porcentaje"],
        ),
        "ok",
    )
    mensaje(
        "F1 exploratorio: baseline {:.4f} | GWO {:.4f}.".format(
            media_baseline,
            media_gwo,
        ),
        "ok",
    )


def mostrar_validacion_existente() -> None:
    encabezado(
        "FASE 3 / PASO 05",
        "Validación anidada — resultado oficial reutilizado",
    )
    ficha(
        "Evalúa GWO sin permitir que la selección observe el conjunto de prueba externo.",
        "4 folds externos, búsqueda interna, TF-IDF, SMOTE y ComplementNB.",
        "Comparación robusta entre TF-IDF completo y GWO.",
    )

    resumen = cargar_json(RESUMEN_VALIDACION)

    verificar_sha_resumen(
        resumen,
        resumen["sha256_entrada"],
        "La validación anidada",
    )

    if resumen.get("protocolo") != "validacion_anidada_gwo_por_grupos_v1":
        raise ValueError(
            "El paso 05 existente usa un protocolo no reconocido."
        )

    completo = resumen["modelos"]["TFIDF_completo"]
    gwo = resumen["modelos"]["GWO"]

    f1_completo = completo[
        "metricas_predicciones_externas_conjuntas"
    ]["F1_macro"]

    f1_gwo = gwo[
        "metricas_predicciones_externas_conjuntas"
    ]["F1_macro"]

    mensaje(
        "No se recalcula en vivo: GWO se repite dentro de cada fold y excede el tiempo de defensa.",
        "aviso",
    )
    mensaje(
        "TF-IDF completo — F1 OOF: {:.4f}.".format(f1_completo),
        "ok",
    )
    mensaje(
        "GWO — F1 OOF: {:.4f}.".format(f1_gwo),
        "ok",
    )
    mensaje(
        "La mejora es agregada y descriptiva; no fue uniforme en todos los folds.",
        "info",
    )


# =============================================================================
# FLUJO DE LA DEMOSTRACIÓN
# =============================================================================

def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecutor ordenado de la demostración de código para la defensa."
        )
    )

    parser.add_argument(
        "--detalle",
        choices=["resumido", "completo"],
        default="resumido",
        help=(
            "resumido muestra solo líneas relevantes; completo muestra toda "
            "la salida de cada script."
        ),
    )

    parser.add_argument(
        "--ejecutar-paso-05",
        action="store_true",
        help=(
            "Recalcula la validación anidada completa. No recomendado para "
            "una defensa de 5 minutos."
        ),
    )

    parser.add_argument(
        "--rapido",
        action="store_true",
        help=(
            "También reutiliza el resultado oficial del paso 04 GWO para "
            "ahorrar aproximadamente un minuto."
        ),
    )

    parser.add_argument(
        "--limite-demo",
        type=int,
        default=3,
        help="Fuentes a procesar en Fase 1 demo. Por defecto: 3.",
    )

    parser.add_argument(
        "--sin-color",
        action="store_true",
        help="Desactiva colores ANSI.",
    )

    return parser


def main() -> int:
    args = construir_parser().parse_args()

    if args.limite_demo < 1:
        raise ValueError("--limite-demo debe ser al menos 1.")

    configurar_terminal(args.sin_color)
    portada()

    inicio_total = time.perf_counter()

    # -------------------------------------------------------------------------
    # FASE 1
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 1",
        "Recolección y etiquetado",
    )
    ficha(
        "Obtiene perfiles de egreso desde fuentes oficiales y los etiqueta.",
        "Requests, Selenium o PDF según la fuente; reglas de validación y etiquetado.",
        "Fuentes procesadas, método usado, estado OK y categoría asignada.",
    )

    ejecutar_python(
        ROOT / "ejecutar_fase1.py",
        [
            "--modo",
            "demo",
            "--limite",
            str(args.limite_demo),
        ],
        (
            "[1/",
            "[2/",
            "[3/",
            "extracción exitosa",
            "fuentes procesadas",
            "ok                 :",
            "revisar",
            "error",
            "registros aceptados",
            "registros excluidos",
            "civil       :",
            "informática :",
            "ejecución   :",
            "fase 1 completada correctamente",
        ),
        args.detalle,
    )

    print()
    mensaje(
        "La Fase 1 anterior es una DEMO de extracción. "
        "A partir de Fase 2 usamos el corpus actual completo de 63 perfiles.",
        "aviso",
    )

    # -------------------------------------------------------------------------
    # FASE 2 — AUDITORÍA
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 2 / PASO 01",
        "Preparación y auditoría del corpus",
    )
    ficha(
        "Comprueba que el corpus esté estructuralmente correcto antes del análisis.",
        "pandas, reglas de auditoría y SHA-256.",
        "63 registros, clases válidas, sin vacíos ni duplicados exactos.",
    )

    ejecutar_python(
        FASE2 / "01_preparacion_corpus.py",
        ["--corpus", "actual"],
        (
            "registros encontrados",
            "registros esperados",
            "civil       :",
            "informática :",
            "ejecución   :",
            "perfiles vacíos",
            "grados fuera",
            "problemas de encoding",
            "duplicados exactos",
            "corpus de entrada intacto",
            "fase 2 completada correctamente",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # FASE 2 — ENCODING
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 2 / PASO 02",
        "Corrección controlada de encoding",
    )
    ficha(
        "Busca caracteres dañados y corrige solo los casos detectados.",
        "Reglas de mojibake, validación de estructura y SHA-256.",
        "Cantidad de registros corregidos y corpus candidato final.",
    )

    ejecutar_python(
        FASE2 / "02_corregir_encoding_v3.py",
        ["--corpus", "actual"],
        (
            "registros totales",
            "registros corregidos",
            "mojibake restante",
            "civil       :",
            "informática :",
            "ejecución   :",
            "estado corpus de entrada",
            "corrección generada correctamente",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # FASE 3 — 01
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 3 / PASO 01",
        "Representación y visualización PCA / LDA",
    )
    ficha(
        "Convierte los perfiles en números y genera dos visualizaciones del corpus.",
        "TF-IDF, PCA y LDA.",
        "400 características y generación correcta de las proyecciones.",
    )

    ejecutar_python(
        FASE3 / "01_proyeccion_pca_lda.py",
        ["--corpus", "actual"],
        (
            "corpus: 63",
            "tf-idf generado",
            "grupos compartidos",
            "proyección lda es supervisada",
            "proyección pca/lda generada correctamente",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # FASE 3 — 02
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 3 / PASO 02",
        "Homogeneidad y significancia",
    )
    ficha(
        "Compara cuánto se parecen los perfiles dentro y entre categorías.",
        "Similitud coseno, centroides y 5.000 permutaciones.",
        "62 unidades del test y finalización de la prueba estadística.",
    )

    ejecutar_python(
        FASE3 / "02_homogeneidad_significancia.py",
        ["--corpus", "actual"],
        (
            "corpus: 63",
            "unidades del test",
            "ejecutando test de permutación",
            "permutaciones : 5000",
            "significativo :",
            "análisis de homogeneidad completado correctamente",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # FASE 3 — 03
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 3 / PASO 03",
        "Diferenciación léxica",
    )
    ficha(
        "Busca expresiones relativamente más presentes en cada categoría.",
        "Prevalencia documental, expresiones de 2 a 4 palabras y auditoría de soporte.",
        "Ejemplos de expresiones para Civil, Informática y Ejecución.",
    )

    ejecutar_python(
        FASE3 / "03_diferenciacion_lexica.py",
        ["--corpus", "actual"],
        (
            "corpus actual: 63",
            "civil:",
            "soluciones computacionales",
            "ejecución:",
            "automatizando procesos",
            "informática:",
            "buenas prácticas",
            "se generaron el gráfico conjunto",
            "análisis descriptivo",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # FASE 3 — 04
    # -------------------------------------------------------------------------
    if args.rapido:
        mostrar_gwo_existente()
    else:
        encabezado(
            "FASE 3 / PASO 04",
            "Selección de características con GWO",
        )
        ficha(
            "Busca si podemos trabajar con menos variables sin perder capacidad de diferenciación.",
            "Grey Wolf Optimizer, 30 lobos, 100 iteraciones y TF-IDF.",
            "Reducción de 400 a 318 características y comparación exploratoria.",
        )

        ejecutar_python(
            FASE3 / "04_seleccion_caracteristicas_gwo.py",
            ["--corpus", "actual"],
            (
                "gwo exploratorio",
                "corpus: 63",
                "grupos: 62",
                "baseline (400 características)",
                "iniciando gwo",
                "gwo finalizado",
                "baseline (400 feat)",
                "gwo (318 feat)",
                "resultados:",
            ),
            args.detalle,
        )

    # -------------------------------------------------------------------------
    # FASE 3 — 05
    # -------------------------------------------------------------------------
    if args.ejecutar_paso_05:
        encabezado(
            "FASE 3 / PASO 05",
            "Validación anidada GWO",
        )
        ficha(
            "Evalúa la selección de variables sin permitir que GWO observe la prueba externa.",
            "4 folds externos, búsqueda interna, SMOTE y ComplementNB.",
            "F1 OOF de TF-IDF completo y GWO.",
        )

        ejecutar_python(
            FASE3 / "05_validacion_gwo.py",
            ["--corpus", "actual"],
            (
                "fase 3 — validación anidada gwo",
                "perfiles: 63",
                "fold externo",
                "f1 externo",
                "tfidf_completo:",
                "gwo: f1 macro",
                "resultados guardados",
                "no existe una única máscara",
            ),
            args.detalle,
        )
    else:
        mostrar_validacion_existente()

    # -------------------------------------------------------------------------
    # FASE 3 — 06
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 3 / PASO 06",
        "Modelo robusto con títulos enmascarados",
    )
    ficha(
        "Oculta los nombres explícitos de las carreras y vuelve a evaluar.",
        "TF-IDF, SMOTE solo en entrenamiento, ComplementNB y 4 folds.",
        "76 menciones ocultadas y F1 OOF del modelo enmascarado.",
    )

    ejecutar_python(
        FASE3 / "06_modelo_final_robusto.py",
        ["--corpus", "actual"],
        (
            "fase 3 — modelo robusto",
            "perfiles: 63",
            "títulos enmascarados",
            "fold 1/",
            "fold 2/",
            "fold 3/",
            "fold 4/",
            "f1 macro medio por fold",
            "f1 macro conjunto",
            "resultados guardados",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # FASE 3 — 07
    # -------------------------------------------------------------------------
    encabezado(
        "FASE 3 / PASO 07",
        "Verificación y reporte consolidado",
    )
    ficha(
        "Verifica que las salidas anteriores pertenezcan al mismo corpus y consolida las métricas.",
        "SHA-256, JSON, CSV, métricas OOF y reporte Markdown.",
        "TF-IDF, GWO y modelo enmascarado reunidos en un único reporte final.",
    )

    ejecutar_python(
        FASE3 / "07_generar_reporte.py",
        ["--corpus", "actual"],
        (
            "fase 3 — reporte consolidado",
            "verificados: 63 perfiles",
            "paso             modelo",
            "tfidf_completo",
            "gwo",
            "complementnb_smote",
            "generado:",
        ),
        args.detalle,
    )

    # -------------------------------------------------------------------------
    # CIERRE
    # -------------------------------------------------------------------------
    duracion_total = time.perf_counter() - inicio_total

    print()
    print(color(linea(), VERDE))
    print(color("  DEMOSTRACIÓN COMPLETADA", BOLD + VERDE))
    print(color(linea(), VERDE))
    print()

    mensaje(
        "Tiempo total del ejecutor: {:.1f} s ({:.2f} min).".format(
            duracion_total,
            duracion_total / 60.0,
        ),
        "ok",
    )

    print()
    print(color("  DESCUBRIMIENTO PRINCIPAL", BOLD + CIAN))
    print(
        "  No son perfiles completamente iguales ni completamente distintos:"
    )
    print(
        "  comparten una base común, pero conservan patrones textuales"
    )
    print(
        "  asociados al tipo de grado."
    )
    print()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        mensaje("Demostración interrumpida por el usuario.", "aviso")
        raise SystemExit(130)
    except Exception as error:
        print()
        print(color(linea(), ROJO))
        print(color("  ERROR DURANTE LA DEMOSTRACIÓN", BOLD + ROJO))
        print(color(linea(), ROJO))
        print()
        mensaje(str(error), "error")
        raise SystemExit(1)
