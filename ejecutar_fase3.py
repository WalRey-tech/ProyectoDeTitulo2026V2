# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# =============================================================================
# 1. RUTAS GENERALES
# =============================================================================

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "src"

FASE3 = SRC / "Fase3_Analisis"

DATASET = (
    SRC
    / "data"
    / "processed"
    / "perfiles_egreso_etiquetado_v2.csv"
)

RESULTADOS = (
    SRC
    / "data"
    / "resultados_cientificos"
)

GWO_RESULTADOS = (
    RESULTADOS
    / "gwo"
)

GWO_FEATURES = (
    GWO_RESULTADOS
    / "gwo_features_seleccionadas.csv"
)


# =============================================================================
# 2. SCRIPTS OFICIALES DE FASE 3
# =============================================================================

SCRIPT_01 = (
    FASE3
    / "01_proyeccion_pca_lda.py"
)

SCRIPT_02 = (
    FASE3
    / "02_homogeneidad_significancia.py"
)

SCRIPT_03 = (
    FASE3
    / "03_diferenciacion_lexica.py"
)

SCRIPT_04 = (
    FASE3
    / "04_seleccion_caracteristicas_gwo.py"
)

SCRIPT_05 = (
    FASE3
    / "05_validacion_gwo.py"
)

SCRIPT_06 = (
    FASE3
    / "06_modelo_final_robusto.py"
)

SCRIPT_07 = (
    FASE3
    / "07_generar_reporte.py"
)


# =============================================================================
# 3. UTILIDADES
# =============================================================================

def verificar_archivo(
    ruta: Path,
    descripcion: str,
) -> None:

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {descripcion}:\n"
            f"{ruta}"
        )


def ejecutar(
    script: Path,
    numero: int,
    descripcion: str,
) -> None:

    verificar_archivo(
        script,
        f"el script {numero:02d}",
    )

    print()
    print("=" * 78)

    print(
        f"PASO {numero}/7 — {descripcion}"
    )

    print("=" * 78)

    print(
        f"Script: {script.name}"
    )

    print("-" * 78)

    subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=str(ROOT),
        check=True,
    )

    print()

    print(
        f"✓ Paso {numero} completado correctamente."
    )


# =============================================================================
# 4. VALIDACIÓN DE ESTRUCTURA
# =============================================================================

def verificar_proyecto() -> None:

    verificar_archivo(
        DATASET,
        "el corpus científico V2",
    )

    scripts = [
        SCRIPT_01,
        SCRIPT_02,
        SCRIPT_03,
        SCRIPT_04,
        SCRIPT_05,
        SCRIPT_06,
        SCRIPT_07,
    ]

    for script in scripts:

        verificar_archivo(
            script,
            script.name,
        )


# =============================================================================
# 5. MODO COMPLETO
# =============================================================================
#
# Ejecuta nuevamente GWO.
#
# Uso:
#
# python ejecutar_fase3.py --modo completo
#
# =============================================================================

def ejecutar_modo_completo() -> None:

    ejecutar(
        SCRIPT_01,
        1,
        "Visualización exploratoria PCA / LDA",
    )

    ejecutar(
        SCRIPT_02,
        2,
        "Homogeneidad y significancia semántica",
    )

    ejecutar(
        SCRIPT_03,
        3,
        "Diferenciación léxica interpretable",
    )

    ejecutar(
        SCRIPT_04,
        4,
        "Selección de características mediante GWO",
    )

    ejecutar(
        SCRIPT_05,
        5,
        "Validación de características seleccionadas por GWO",
    )

    ejecutar(
        SCRIPT_06,
        6,
        "Modelo robusto final con control de fuga léxica",
    )

    ejecutar(
        SCRIPT_07,
        7,
        "Generación del reporte científico consolidado",
    )


# =============================================================================
# 6. MODO VALIDAR
# =============================================================================
#
# No vuelve a ejecutar las 100 épocas de GWO.
# Reutiliza la selección oficial ya generada.
#
# Uso:
#
# python ejecutar_fase3.py
#
# o:
#
# python ejecutar_fase3.py --modo validar
#
# =============================================================================

def ejecutar_modo_validar() -> None:

    verificar_archivo(
        GWO_FEATURES,
        "las características GWO existentes",
    )

    ejecutar(
        SCRIPT_01,
        1,
        "Visualización exploratoria PCA / LDA",
    )

    ejecutar(
        SCRIPT_02,
        2,
        "Homogeneidad y significancia semántica",
    )

    ejecutar(
        SCRIPT_03,
        3,
        "Diferenciación léxica interpretable",
    )

    print()
    print("=" * 78)

    print(
        "PASO 4/7 — Selección de características mediante GWO"
    )

    print("=" * 78)

    print(
        "Modo validar: se reutiliza la selección GWO existente."
    )

    print(
        f"Archivo: {GWO_FEATURES}"
    )

    print(
        "✓ Paso 4 reutilizado correctamente."
    )

    ejecutar(
        SCRIPT_05,
        5,
        "Validación de características seleccionadas por GWO",
    )

    ejecutar(
        SCRIPT_06,
        6,
        "Modelo robusto final con control de fuga léxica",
    )

    ejecutar(
        SCRIPT_07,
        7,
        "Generación del reporte científico consolidado",
    )


# =============================================================================
# 7. FLUJO PRINCIPAL
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Ejecutor oficial de la Fase 3 "
            "del Proyecto de Título."
        )
    )

    parser.add_argument(
        "--modo",
        choices=[
            "completo",
            "validar",
        ],
        default="validar",
        help=(
            "'completo' ejecuta nuevamente GWO; "
            "'validar' reutiliza la selección GWO "
            "existente. Por defecto: validar."
        ),
    )

    args = parser.parse_args()

    verificar_proyecto()

    print()
    print("=" * 78)

    print(
        "FASE 3 — ANÁLISIS CIENTÍFICO"
    )

    print("=" * 78)

    print(
        f"Proyecto   : {ROOT}"
    )

    print(
        f"Corpus     : {DATASET}"
    )

    print(
        f"Resultados : {RESULTADOS}"
    )

    print(
        f"Modo       : {args.modo}"
    )

    print()

    print(
        "Pipeline oficial:"
    )

    print(
        "  01. PCA / LDA"
    )

    print(
        "  02. Homogeneidad semántica"
    )

    print(
        "  03. Diferenciación léxica"
    )

    print(
        "  04. Selección GWO"
    )

    print(
        "  05. Validación GWO"
    )

    print(
        "  06. Modelo robusto final"
    )

    print(
        "  07. Reporte consolidado"
    )

    if args.modo == "completo":

        print()
        print(
            "Se ejecutará nuevamente "
            "la optimización GWO."
        )

        ejecutar_modo_completo()

    else:

        print()
        print(
            "Se reutilizará la selección "
            "GWO existente."
        )

        ejecutar_modo_validar()

    print()
    print("=" * 78)

    print(
        "FASE 3 COMPLETADA CORRECTAMENTE"
    )

    print("=" * 78)

    print()

    print(
        "Resultados disponibles en:"
    )

    print(
        RESULTADOS
    )

    return 0


# =============================================================================
# 8. CONTROL DE ERRORES
# =============================================================================

if __name__ == "__main__":

    try:

        raise SystemExit(
            main()
        )

    except subprocess.CalledProcessError as error:

        print()
        print("=" * 78)

        print(
            "ERROR DURANTE LA EJECUCIÓN DE FASE 3"
        )

        print("=" * 78)

        print(
            "Un script terminó con "
            f"código {error.returncode}."
        )

        raise SystemExit(
            error.returncode
        )

    except Exception as error:

        print()
        print("=" * 78)

        print(
            "ERROR DE CONFIGURACIÓN DE FASE 3"
        )

        print("=" * 78)

        print(
            error
        )

        raise SystemExit(
            1
        )