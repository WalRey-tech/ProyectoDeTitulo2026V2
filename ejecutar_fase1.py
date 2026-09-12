# -*- coding: utf-8 -*-

"""
ORQUESTADOR OFICIAL — FASE 1
============================

Coordina:

    1. Recolección web controlada
    2. Etiquetado y auditoría

MODOS
-----

DEMO
    Diseñado para demostración durante la defensa.

    Por defecto procesa 3 fuentes:

        python ejecutar_fase1.py --modo demo

COMPLETO
    Procesa todas las fuentes configuradas:

        python ejecutar_fase1.py --modo completo


IMPORTANTE
----------

El corpus científico:

    src/data/processed/perfiles_egreso_etiquetado_v2.csv

NO es generado ni modificado por este pipeline.

El script calcula su SHA-256 antes y después de la ejecución
para comprobarlo de manera reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys

from collections import Counter
from pathlib import Path


# =============================================================================
# 1. RUTAS
# =============================================================================

ROOT = Path(
    __file__
).resolve().parent


FASE1_DIR = (
    ROOT
    / "src"
    / "Fase1_Recoleccion"
)


MAIN_SCRAPING = (
    FASE1_DIR
    / "main.py"
)


ETIQUETADOR = (
    FASE1_DIR
    / "etiquetador.py"
)


DATA_DIR = (
    ROOT
    / "src"
    / "data"
)


RAW_DIR = (
    DATA_DIR
    / "raw"
)


PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)


CORPUS_V2 = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)


# =============================================================================
# 2. CONFIGURACIÓN
# =============================================================================

LIMITE_DEMO_POR_DEFECTO = 3


# =============================================================================
# 3. HASH SHA-256
# =============================================================================

def calcular_sha256(
    ruta: Path,
) -> str | None:
    """
    Calcula SHA-256 de un archivo.

    Se utiliza para comprobar que el corpus V2
    permanece exactamente igual antes y después.
    """

    if not ruta.exists():

        return None


    sha = hashlib.sha256()


    with open(
        ruta,
        "rb",
    ) as archivo:

        while True:

            bloque = archivo.read(
                1024 * 1024
            )


            if not bloque:

                break


            sha.update(
                bloque
            )


    return sha.hexdigest().upper()


# =============================================================================
# 4. EJECUCIÓN DE SUBPROCESOS
# =============================================================================

def ejecutar_script(
    titulo: str,
    comando: list[str],
):
    """
    Ejecuta un paso utilizando el mismo Python
    del entorno virtual activo.

    La salida del programa se muestra directamente
    en la consola.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        titulo
    )

    print(
        "=" * 78
    )


    print(
        "Comando:"
    )

    print(
        " ".join(
            comando
        )
    )

    print(
        "-" * 78
    )


    entorno = os.environ.copy()

    # Facilita impresión correcta de caracteres UTF-8.
    entorno[
        "PYTHONUTF8"
    ] = "1"


    resultado = subprocess.run(

        comando,

        cwd=ROOT,

        env=entorno,

        check=False,

    )


    if resultado.returncode != 0:

        raise RuntimeError(
            f"El paso terminó con código "
            f"{resultado.returncode}"
        )


# =============================================================================
# 5. LECTURA SEGURA DE CSV
# =============================================================================

def leer_csv(
    ruta: Path,
) -> list[dict]:
    """
    Lee uno de los CSV generados por el pipeline.

    Los archivos utilizan:

        sep=";"
        encoding="utf-8-sig"
    """

    if not ruta.exists():

        return []


    with open(
        ruta,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as archivo:

        lector = csv.DictReader(
            archivo,
            delimiter=";",
        )


        return list(
            lector
        )


# =============================================================================
# 6. RUTAS SEGÚN MODO
# =============================================================================

def obtener_archivos_modo(
    modo: str,
):
    """
    Devuelve las salidas esperadas según el modo.
    """

    if modo == "demo":

        raw = (
            RAW_DIR
            / "perfiles_egreso_raw_demo.csv"
        )

        etiquetado = (
            PROCESSED_DIR
            / "perfiles_egreso_etiquetado_demo.csv"
        )

        auditoria = (
            PROCESSED_DIR
            / "auditoria_etiquetado_demo.csv"
        )


    else:

        raw = (
            RAW_DIR
            / "perfiles_egreso_raw_actual.csv"
        )

        etiquetado = (
            PROCESSED_DIR
            / "perfiles_egreso_etiquetado_actual.csv"
        )

        auditoria = (
            PROCESSED_DIR
            / "auditoria_etiquetado_actual.csv"
        )


    return (
        raw,
        etiquetado,
        auditoria,
    )


# =============================================================================
# 7. RESUMEN DE LA RECOLECCIÓN
# =============================================================================

def resumir_raw(
    ruta: Path,
) -> dict:
    """
    Resume resultados de la etapa de scraping.
    """

    filas = leer_csv(
        ruta
    )


    estados = Counter(

        str(
            fila.get(
                "estado_registro",
                "",
            )
        ).strip()
        or "SIN_ESTADO"

        for fila in filas

    )


    metodos = Counter(

        str(
            fila.get(
                "metodo_usado",
                "",
            )
        ).strip()
        or "sin_metodo"

        for fila in filas

    )


    estrategias = Counter(

        str(
            fila.get(
                "estrategia_extraccion",
                "",
            )
        ).strip()
        or "sin_resultado"

        for fila in filas

    )


    return {

        "total":
            len(
                filas
            ),

        "estados":
            estados,

        "metodos":
            metodos,

        "estrategias":
            estrategias,
    }


# =============================================================================
# 8. RESUMEN DEL ETIQUETADO
# =============================================================================

def resumir_etiquetado(
    ruta_etiquetado: Path,
    ruta_auditoria: Path,
) -> dict:
    """
    Resume el dataset etiquetado y su auditoría.
    """

    aceptados = leer_csv(
        ruta_etiquetado
    )


    excluidos = leer_csv(
        ruta_auditoria
    )


    grados = Counter(

        str(
            fila.get(
                "grado",
                "",
            )
        ).strip()
        or "SIN_GRADO"

        for fila in aceptados

    )


    motivos_exclusion = Counter(

        str(
            fila.get(
                "motivo_exclusion_etiquetado",
                "",
            )
        ).strip()
        or "sin_motivo"

        for fila in excluidos

    )


    return {

        "aceptados":
            len(
                aceptados
            ),

        "excluidos":
            len(
                excluidos
            ),

        "grados":
            grados,

        "motivos_exclusion":
            motivos_exclusion,
    }


# =============================================================================
# 9. COMPROBACIÓN DE ARCHIVOS
# =============================================================================

def comprobar_archivo(
    ruta: Path,
    descripcion: str,
):
    """
    Verifica que un paso realmente haya generado
    el archivo esperado.
    """

    if not ruta.exists():

        raise FileNotFoundError(
            f"No se generó {descripcion}:\n"
            f"{ruta}"
        )


# =============================================================================
# 10. ARGUMENTOS
# =============================================================================

def construir_parser():
    """
    Interfaz de línea de comandos.
    """

    parser = argparse.ArgumentParser(

        description=(
            "Orquestador de la Fase 1: "
            "scraping + etiquetado."
        )

    )


    parser.add_argument(

        "--modo",

        choices=[
            "demo",
            "completo",
        ],

        default="demo",

        help=(
            "demo procesa pocas fuentes; "
            "completo procesa todo config.py."
        ),

    )


    parser.add_argument(

        "--limite",

        type=int,

        default=None,

        help=(
            "Cantidad máxima de fuentes. "
            "En demo se utilizan 3 por defecto."
        ),

    )


    return parser


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    """
    Ejecuta la Fase 1 completa.
    """

    parser = construir_parser()

    args = parser.parse_args()


    # =========================================================================
    # VALIDACIONES PREVIAS
    # =========================================================================

    if not MAIN_SCRAPING.exists():

        parser.error(
            f"No existe:\n{MAIN_SCRAPING}"
        )


    if not ETIQUETADOR.exists():

        parser.error(
            f"No existe:\n{ETIQUETADOR}"
        )


    if (
        args.limite is not None
        and args.limite <= 0
    ):

        parser.error(
            "--limite debe ser mayor que 0"
        )


    limite = args.limite


    if (
        args.modo == "demo"
        and limite is None
    ):

        limite = (
            LIMITE_DEMO_POR_DEFECTO
        )


    (
        ruta_raw,
        ruta_etiquetada,
        ruta_auditoria,
    ) = obtener_archivos_modo(
        args.modo
    )


    # =========================================================================
    # HASH V2 ANTES
    # =========================================================================

    sha_v2_antes = calcular_sha256(
        CORPUS_V2
    )


    # =========================================================================
    # ENCABEZADO
    # =========================================================================

    print(
        "=" * 78
    )

    print(
        "FASE 1 — PIPELINE DE RECOLECCIÓN Y ETIQUETADO"
    )

    print(
        "=" * 78
    )


    print(
        f"Proyecto            : {ROOT}"
    )

    print(
        f"Modo                : {args.modo}"
    )


    if limite is not None:

        print(
            f"Fuentes solicitadas : {limite}"
        )

    else:

        print(
            "Fuentes solicitadas : todas"
        )


    print(
        f"Python              : {sys.executable}"
    )


    print(
        "\nPipeline:"
    )

    print(
        "  1. Scraping seguro y auditable"
    )

    print(
        "  2. Etiquetado y auditoría"
    )


    print(
        "\nCorpus científico protegido:"
    )

    print(
        f"  {CORPUS_V2}"
    )


    if sha_v2_antes:

        print(
            f"  SHA-256 antes: {sha_v2_antes}"
        )

    else:

        print(
            "  ADVERTENCIA: corpus V2 no encontrado."
        )


    # =========================================================================
    # PASO 1 — SCRAPING
    # =========================================================================

    comando_scraping = [

        sys.executable,

        str(
            MAIN_SCRAPING
        ),

        "--modo",

        args.modo,

    ]


    if limite is not None:

        comando_scraping.extend(
            [
                "--limite",
                str(
                    limite
                ),
            ]
        )


    try:

        ejecutar_script(

            "PASO 1/2 — RECOLECCIÓN WEB CONTROLADA",

            comando_scraping,

        )

    except Exception as error:

        print(
            "\n"
            + "=" * 78
        )

        print(
            "ERROR EN FASE 1"
        )

        print(
            "=" * 78
        )

        print(
            f"Falló el scraping: {error}"
        )

        sys.exit(
            1
        )


    comprobar_archivo(

        ruta_raw,

        "el archivo de recolección",

    )


    # =========================================================================
    # PASO 2 — ETIQUETADO
    # =========================================================================

    comando_etiquetador = [

        sys.executable,

        str(
            ETIQUETADOR
        ),

        "--modo",

        args.modo,

    ]


    try:

        ejecutar_script(

            "PASO 2/2 — ETIQUETADO Y AUDITORÍA",

            comando_etiquetador,

        )

    except Exception as error:

        print(
            "\n"
            + "=" * 78
        )

        print(
            "ERROR EN FASE 1"
        )

        print(
            "=" * 78
        )

        print(
            f"Falló el etiquetado: {error}"
        )

        sys.exit(
            1
        )


    comprobar_archivo(

        ruta_etiquetada,

        "el dataset etiquetado",

    )


    comprobar_archivo(

        ruta_auditoria,

        "la auditoría de etiquetado",

    )


    # =========================================================================
    # RESÚMENES
    # =========================================================================

    resumen_raw = resumir_raw(
        ruta_raw
    )


    resumen_etiquetado = (
        resumir_etiquetado(
            ruta_etiquetada,
            ruta_auditoria,
        )
    )


    # =========================================================================
    # HASH V2 DESPUÉS
    # =========================================================================

    sha_v2_despues = calcular_sha256(
        CORPUS_V2
    )


    v2_intacto = (

        sha_v2_antes is not None

        and

        sha_v2_despues is not None

        and

        sha_v2_antes
        == sha_v2_despues

    )


    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESUMEN DE FASE 1"
    )

    print(
        "=" * 78
    )


    print(
        "\nRecolección:"
    )

    print(
        f"  Fuentes procesadas : "
        f"{resumen_raw['total']}"
    )


    print(
        f"  OK                 : "
        f"{resumen_raw['estados'].get('OK', 0)}"
    )

    print(
        f"  REVISAR            : "
        f"{resumen_raw['estados'].get('REVISAR', 0)}"
    )

    print(
        f"  ERROR              : "
        f"{resumen_raw['estados'].get('ERROR', 0)}"
    )


    print(
        "\nEtiquetado:"
    )

    print(
        f"  Aceptados : "
        f"{resumen_etiquetado['aceptados']}"
    )

    print(
        f"  Excluidos : "
        f"{resumen_etiquetado['excluidos']}"
    )


    print(
        "\nDistribución del conjunto aceptado:"
    )


    for grado in [
        "Civil",
        "Informática",
        "Ejecución",
    ]:

        print(
            f"  {grado:<12}: "
            f"{resumen_etiquetado['grados'].get(grado, 0)}"
        )


    # =========================================================================
    # VERIFICACIÓN DEL CORPUS V2
    # =========================================================================

    print(
        "\n"
        + "-" * 78
    )

    print(
        "VERIFICACIÓN DEL CORPUS CIENTÍFICO V2"
    )

    print(
        "-" * 78
    )


    if sha_v2_antes is None:

        print(
            "  No fue posible verificar el V2 "
            "porque el archivo no existía antes de la ejecución."
        )

    else:

        print(
            f"  SHA-256 antes   : {sha_v2_antes}"
        )

        print(
            f"  SHA-256 después : {sha_v2_despues}"
        )


        if v2_intacto:

            print(
                "  Estado           : ✓ V2 INTACTO"
            )

        else:

            print(
                "  Estado           : ✗ EL V2 CAMBIÓ"
            )


    # =========================================================================
    # ARCHIVOS
    # =========================================================================

    print(
        "\nArchivos de esta ejecución:"
    )

    print(
        f"  Raw       : {ruta_raw}"
    )

    print(
        f"  Etiquetado: {ruta_etiquetada}"
    )

    print(
        f"  Auditoría : {ruta_auditoria}"
    )


    print(
        "\nInterpretación:"
    )

    print(
        "  - La ejecución en vivo representa el estado actual de Internet."
    )

    print(
        "  - Los registros dudosos o fallidos no se incorporan silenciosamente."
    )

    print(
        "  - El corpus V2 permanece separado para garantizar reproducibilidad."
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "FASE 1 COMPLETADA CORRECTAMENTE"
    )

    print(
        "=" * 78
    )


# =============================================================================
# 12. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()