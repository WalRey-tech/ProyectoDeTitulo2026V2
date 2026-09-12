# -*- coding: utf-8 -*-

"""
FASE 1 — RECOLECCIÓN CONTROLADA DE PERFILES DE EGRESO
======================================================

Este script coordina el scraping de las fuentes declaradas en config.py.

MODOS DE EJECUCIÓN
------------------

1. DEMO

   Diseñado para la defensa o pruebas rápidas.

   Por defecto procesa solamente 5 fuentes y genera:

       src/data/raw/perfiles_egreso_raw_demo.csv
       src/data/raw/resumen_scraping_demo.json

2. COMPLETO

   Procesa todas las fuentes configuradas y genera:

       src/data/raw/perfiles_egreso_raw_actual.csv
       src/data/raw/resumen_scraping_actual.json


IMPORTANTE
----------

Este script NO modifica:

    src/data/processed/perfiles_egreso_etiquetado_v2.csv

Ese archivo corresponde al corpus científico congelado utilizado
para reproducir los resultados oficiales de la investigación.
"""

from __future__ import annotations

import argparse
import csv
import json

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

import pandas as pd


# =============================================================================
# 1. IMPORTS LOCALES
# =============================================================================

# Permite ejecutar:
#
#   python main.py
#
# desde Fase1_Recoleccion,
# y también ejecutar:
#
#   python .\src\Fase1_Recoleccion\main.py
#
# desde la raíz del proyecto.

try:
    from .config import SITES
    from .scraper import scrapear_sitio

except ImportError:
    from config import SITES
    from scraper import scrapear_sitio


# ftfy es opcional.
try:
    from ftfy import fix_text

except ImportError:
    fix_text = None


# =============================================================================
# 2. RUTAS DEL PROYECTO
# =============================================================================

DIRECTORIO_ACTUAL = Path(
    __file__
).resolve().parent


SRC_ROOT = (
    DIRECTORIO_ACTUAL
    .parent
)


DATA_DIR = (
    SRC_ROOT
    / "data"
)


RAW_DIR = (
    DATA_DIR
    / "raw"
)


# Corpus científico oficial.
#
# Se declara únicamente para dejar explícito que
# Fase 1 NO debe escribir sobre él.

CORPUS_CIENTIFICO_V2 = (
    DATA_DIR
    / "processed"
    / "perfiles_egreso_etiquetado_v2.csv"
)


# =============================================================================
# 3. PARÁMETROS
# =============================================================================

LONGITUD_MINIMA_REFERENCIA = 150

LIMITE_DEMO_POR_DEFECTO = 5


# =============================================================================
# 4. CORRECCIÓN DE CODIFICACIÓN
# =============================================================================

def corregir_mojibake(texto):
    """
    Corrige errores habituales de codificación.

    Ejemplos:

        InformÃ¡tica -> Informática
        diseÃ±ar     -> diseñar
        tecnolÃ³gica -> tecnológica

    No modifica valores no textuales.
    """

    if not isinstance(
        texto,
        str,
    ):
        return texto


    texto = texto.strip()


    # -------------------------------------------------------------------------
    # Método preferido: ftfy
    # -------------------------------------------------------------------------

    if fix_text is not None:

        try:
            return fix_text(
                texto
            )

        except Exception:
            pass


    # -------------------------------------------------------------------------
    # Fallback simple
    # -------------------------------------------------------------------------

    try:

        marcas = [
            "Ã",
            "Â",
            "â",
            "�",
        ]

        if any(
            marca in texto
            for marca in marcas
        ):

            return (
                texto
                .encode(
                    "latin1"
                )
                .decode(
                    "utf-8"
                )
            )

    except Exception:
        pass


    return texto


# =============================================================================
# 5. LIMPIEZA DEL DATAFRAME
# =============================================================================

def limpiar_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza el DataFrame obtenido desde el scraper.

    No cambia el significado del contenido.
    """

    df = df.copy()

    df = df.fillna(
        ""
    )


    # -------------------------------------------------------------------------
    # Corrección de encoding en columnas textuales
    # -------------------------------------------------------------------------

    for columna in df.select_dtypes(
        include=[
            "object",
        ]
    ).columns:

        df[columna] = (
            df[columna]
            .apply(
                corregir_mojibake
            )
        )


    # -------------------------------------------------------------------------
    # Garantizar columna perfil_egreso
    # -------------------------------------------------------------------------

    if "perfil_egreso" not in df.columns:

        df[
            "perfil_egreso"
        ] = ""


    # -------------------------------------------------------------------------
    # Normalizar espacios
    # -------------------------------------------------------------------------

    df[
        "perfil_egreso"
    ] = (

        df[
            "perfil_egreso"
        ]
        .astype(
            str
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()

    )


    # -------------------------------------------------------------------------
    # Calcular longitud después de limpiar
    # -------------------------------------------------------------------------

    df[
        "largo_perfil"
    ] = (

        df[
            "perfil_egreso"
        ]
        .str.len()

    )


    return df


# =============================================================================
# 6. NORMALIZACIÓN DE BOOLEANOS
# =============================================================================

def convertir_a_bool(
    valor,
) -> bool:
    """
    Convierte distintas representaciones de verdadero/falso.
    """

    if isinstance(
        valor,
        bool,
    ):
        return valor


    texto = str(
        valor
    ).strip().lower()


    return texto in {
        "true",
        "1",
        "si",
        "sí",
        "yes",
        "verdadero",
    }


# =============================================================================
# 7. ESTADO FINAL DE CADA REGISTRO
# =============================================================================

def determinar_estado_registro(
    fila,
) -> str:
    """
    Clasifica cada extracción en:

        OK
        REVISAR
        ERROR

    ERROR
        No se pudo obtener un perfil o ocurrió un error
        de adquisición/extracción.

    REVISAR
        Existe contenido, pero requiere validación manual.

    OK
        Existe contenido suficiente y no posee alertas
        metodológicas relevantes.
    """

    perfil = str(
        fila.get(
            "perfil_egreso",
            "",
        )
    ).strip()


    error = str(
        fila.get(
            "error",
            "",
        )
    ).strip()


    requiere_revision = convertir_a_bool(
        fila.get(
            "requiere_revision",
            False,
        )
    )


    largo = len(
        perfil
    )


    # -------------------------------------------------------------------------
    # ERROR
    # -------------------------------------------------------------------------

    if error or not perfil:

        return "ERROR"


    # -------------------------------------------------------------------------
    # REVISAR
    # -------------------------------------------------------------------------

    if (
        requiere_revision
        or largo < LONGITUD_MINIMA_REFERENCIA
    ):

        return "REVISAR"


    # -------------------------------------------------------------------------
    # OK
    # -------------------------------------------------------------------------

    return "OK"


# =============================================================================
# 8. ORDEN DE COLUMNAS
# =============================================================================

def ordenar_columnas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ordena las columnas manteniendo toda la trazabilidad
    generada por scraper.py.
    """

    columnas_preferidas = [

        "indice_fuente",

        "fecha_extraccion_utc",

        "estado_registro",

        "universidad",

        "tipo_institucion",

        "carrera",

        "tipo_carrera",

        "url_original",

        "url",

        "url_final",

        "selector",

        "metodo_usado",

        "estrategia_extraccion",

        "termino_detectado",

        "etiqueta_titulo",

        "robots_estado",

        "content_type",

        "perfil_egreso",

        "largo_perfil",

        "requiere_revision",

        "motivo_revision",

        "advertencias",

        "error",
    ]


    columnas_presentes = [

        columna
        for columna
        in columnas_preferidas

        if columna
        in df.columns

    ]


    # -------------------------------------------------------------------------
    # Si en el futuro scraper.py agrega columnas nuevas,
    # no las eliminamos.
    # -------------------------------------------------------------------------

    columnas_extra = [

        columna
        for columna
        in df.columns

        if columna
        not in columnas_presentes

    ]


    return df[
        columnas_presentes
        + columnas_extra
    ]


# =============================================================================
# 9. VALIDACIÓN GENERAL
# =============================================================================

def validar_dataframe(
    df: pd.DataFrame,
):
    """
    Muestra un resumen general de la recolección.
    """

    print(
        "\n"
        + "=" * 72
    )

    print(
        "VALIDACIÓN GENERAL DE LA RECOLECCIÓN"
    )

    print(
        "=" * 72
    )


    total = len(
        df
    )


    estados = (

        df[
            "estado_registro"
        ]
        .value_counts()

    )


    cantidad_ok = int(
        estados.get(
            "OK",
            0,
        )
    )


    cantidad_revision = int(
        estados.get(
            "REVISAR",
            0,
        )
    )


    cantidad_error = int(
        estados.get(
            "ERROR",
            0,
        )
    )


    print(
        f"Fuentes procesadas : {total}"
    )

    print(
        f"OK                 : {cantidad_ok}"
    )

    print(
        f"REVISAR            : {cantidad_revision}"
    )

    print(
        f"ERROR              : {cantidad_error}"
    )


    # -------------------------------------------------------------------------
    # Métodos de adquisición
    # -------------------------------------------------------------------------

    if "metodo_usado" in df.columns:

        print(
            "\nMétodos de adquisición:"
        )

        print(
            df[
                "metodo_usado"
            ]
            .replace(
                "",
                "sin_metodo",
            )
            .value_counts()
            .to_string()
        )


    # -------------------------------------------------------------------------
    # Estrategias de extracción
    # -------------------------------------------------------------------------

    if (
        "estrategia_extraccion"
        in df.columns
    ):

        print(
            "\nEstrategias de extracción:"
        )

        print(
            df[
                "estrategia_extraccion"
            ]
            .replace(
                "",
                "sin_resultado",
            )
            .value_counts()
            .to_string()
        )


    # -------------------------------------------------------------------------
    # Estado robots.txt
    # -------------------------------------------------------------------------

    if "robots_estado" in df.columns:

        print(
            "\nEstado robots.txt:"
        )

        print(
            df[
                "robots_estado"
            ]
            .replace(
                "",
                "sin_estado",
            )
            .value_counts()
            .to_string()
        )


    print(
        "=" * 72
    )


# =============================================================================
# 10. VALIDACIÓN DE CALIDAD
# =============================================================================

def validar_calidad_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Ejecuta controles de calidad.

    IMPORTANTE:

    Las anomalías se reportan.
    No se eliminan registros automáticamente.
    """

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CONTROL DE CALIDAD DEL DATASET"
    )

    print(
        "=" * 72
    )


    resumen = {}


    # =========================================================================
    # 1. DUPLICADOS UNIVERSIDAD + CARRERA
    # =========================================================================

    duplicados_carrera = pd.DataFrame()


    if {
        "universidad",
        "carrera",
    }.issubset(
        df.columns
    ):

        duplicados_carrera = (

            df[
                df.duplicated(
                    subset=[
                        "universidad",
                        "carrera",
                    ],
                    keep=False,
                )
            ]

            .sort_values(
                by=[
                    "universidad",
                    "carrera",
                ]
            )

        )


    resumen[
        "duplicados_universidad_carrera"
    ] = int(
        len(
            duplicados_carrera
        )
    )


    print(
        "Duplicados universidad + carrera: "
        f"{len(duplicados_carrera)}"
    )


    if not duplicados_carrera.empty:

        columnas = [

            columna
            for columna
            in [
                "universidad",
                "carrera",
                "url",
            ]

            if columna
            in duplicados_carrera.columns

        ]


        print(

            duplicados_carrera[
                columnas
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )

        )


    # =========================================================================
    # 2. URLS DUPLICADAS
    # =========================================================================

    duplicados_url = pd.DataFrame()


    if "url" in df.columns:

        duplicados_url = (

            df[
                df.duplicated(
                    subset=[
                        "url",
                    ],
                    keep=False,
                )
            ]

            .sort_values(
                by=[
                    "url",
                ]
            )

        )


    resumen[
        "duplicados_url"
    ] = int(
        len(
            duplicados_url
        )
    )


    print(
        "\nDuplicados por URL: "
        f"{len(duplicados_url)}"
    )


    # =========================================================================
    # 3. PERFILES EXACTAMENTE REPETIDOS
    # =========================================================================

    duplicados_texto = pd.DataFrame()


    if "perfil_egreso" in df.columns:

        no_vacios = (

            df[
                df[
                    "perfil_egreso"
                ]
                .str.strip()
                .ne(
                    ""
                )
            ]

        )


        duplicados_texto = (

            no_vacios[
                no_vacios.duplicated(
                    subset=[
                        "perfil_egreso",
                    ],
                    keep=False,
                )
            ]

        )


    resumen[
        "perfiles_texto_duplicado"
    ] = int(
        len(
            duplicados_texto
        )
    )


    print(
        "Perfiles con texto exactamente repetido: "
        f"{len(duplicados_texto)}"
    )


    # =========================================================================
    # 4. POSIBLES PROBLEMAS DE CODIFICACIÓN
    # =========================================================================

    patron_encoding = (
        r"Ã|Â|â|�"
    )


    problemas_encoding = pd.DataFrame()


    if not df.empty:

        columnas_texto = (

            df.select_dtypes(
                include=[
                    "object",
                ]
            ).columns

        )


        if len(
            columnas_texto
        ) > 0:

            mascara_encoding = (

                df[
                    columnas_texto
                ]

                .astype(
                    str
                )

                .apply(
                    lambda columna:
                    columna.str.contains(
                        patron_encoding,
                        regex=True,
                        na=False,
                    )
                )

                .any(
                    axis=1
                )

            )


            problemas_encoding = (

                df[
                    mascara_encoding
                ]

            )


    resumen[
        "problemas_encoding"
    ] = int(
        len(
            problemas_encoding
        )
    )


    print(
        "Filas con posibles problemas de encoding: "
        f"{len(problemas_encoding)}"
    )


    # =========================================================================
    # 5. PERFILES CORTOS
    # =========================================================================

    perfiles_cortos = (

        df[
            (
                df[
                    "largo_perfil"
                ] > 0
            )
            &
            (
                df[
                    "largo_perfil"
                ]
                < LONGITUD_MINIMA_REFERENCIA
            )
        ]

    )


    resumen[
        "perfiles_cortos"
    ] = int(
        len(
            perfiles_cortos
        )
    )


    print(
        "Perfiles cortos "
        f"(< {LONGITUD_MINIMA_REFERENCIA} caracteres): "
        f"{len(perfiles_cortos)}"
    )


    # =========================================================================
    # 6. PERFILES VACÍOS
    # =========================================================================

    perfiles_vacios = (

        df[
            df[
                "perfil_egreso"
            ]
            .str.strip()
            .eq(
                ""
            )
        ]

    )


    resumen[
        "perfiles_vacios"
    ] = int(
        len(
            perfiles_vacios
        )
    )


    print(
        "Perfiles vacíos: "
        f"{len(perfiles_vacios)}"
    )


    # =========================================================================
    # 7. REGISTROS QUE REQUIEREN REVISIÓN
    # =========================================================================

    requiere_revision = (

        df[
            df[
                "estado_registro"
            ]
            .eq(
                "REVISAR"
            )
        ]

    )


    resumen[
        "requieren_revision"
    ] = int(
        len(
            requiere_revision
        )
    )


    print(
        "Registros que requieren revisión: "
        f"{len(requiere_revision)}"
    )


    if not requiere_revision.empty:

        columnas = [

            columna
            for columna
            in [
                "universidad",
                "carrera",
                "metodo_usado",
                "estrategia_extraccion",
                "termino_detectado",
                "largo_perfil",
                "motivo_revision",
            ]

            if columna
            in requiere_revision.columns

        ]


        print(
            "\nRegistros que requieren revisión:"
        )


        print(

            requiere_revision[
                columnas
            ]
            .to_string(
                index=False
            )

        )


    # =========================================================================
    # 8. REGISTROS CON ERROR
    # =========================================================================

    registros_error = (

        df[
            df[
                "estado_registro"
            ]
            .eq(
                "ERROR"
            )
        ]

    )


    resumen[
        "registros_error"
    ] = int(
        len(
            registros_error
        )
    )


    print(
        "Registros con ERROR: "
        f"{len(registros_error)}"
    )


    if not registros_error.empty:

        columnas = [

            columna
            for columna
            in [
                "universidad",
                "carrera",
                "robots_estado",
                "motivo_revision",
                "error",
            ]

            if columna
            in registros_error.columns

        ]


        print(
            "\nRegistros con error:"
        )


        print(

            registros_error[
                columnas
            ]
            .to_string(
                index=False
            )

        )


    print(
        "=" * 72
    )


    return resumen


# =============================================================================
# 11. RUTAS DE SALIDA
# =============================================================================

def obtener_rutas_salida(
    modo: str,
):
    """
    Define archivos diferentes para demo
    y para recolección completa.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    if modo == "demo":

        csv_path = (
            RAW_DIR
            / "perfiles_egreso_raw_demo.csv"
        )

        resumen_path = (
            RAW_DIR
            / "resumen_scraping_demo.json"
        )

    else:

        csv_path = (
            RAW_DIR
            / "perfiles_egreso_raw_actual.csv"
        )

        resumen_path = (
            RAW_DIR
            / "resumen_scraping_actual.json"
        )


    return (
        csv_path,
        resumen_path,
    )


# =============================================================================
# 12. GUARDAR CSV
# =============================================================================

def guardar_csv(
    df: pd.DataFrame,
    ruta: Path,
):
    """
    Guarda el resultado crudo y auditable.
    """

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    df.to_csv(

        ruta,

        index=False,

        sep=";",

        encoding="utf-8-sig",

        quoting=csv.QUOTE_ALL,

        lineterminator="\n",

    )


# =============================================================================
# 13. GUARDAR RESUMEN JSON
# =============================================================================

def guardar_resumen(
    df: pd.DataFrame,
    resumen_calidad: dict,
    ruta: Path,
    modo: str,
    total_configurado: int,
):
    """
    Genera un resumen JSON para auditoría.
    """

    conteo_estados = {

        str(clave):
            int(valor)

        for clave, valor
        in (

            df[
                "estado_registro"
            ]
            .value_counts()
            .items()

        )
    }


    metodos = {}


    if "metodo_usado" in df.columns:

        metodos = {

            str(clave):
                int(valor)

            for clave, valor
            in (

                df[
                    "metodo_usado"
                ]
                .replace(
                    "",
                    "sin_metodo",
                )
                .value_counts()
                .items()

            )
        }


    estrategias = {}


    if (
        "estrategia_extraccion"
        in df.columns
    ):

        estrategias = {

            str(clave):
                int(valor)

            for clave, valor
            in (

                df[
                    "estrategia_extraccion"
                ]
                .replace(
                    "",
                    "sin_resultado",
                )
                .value_counts()
                .items()

            )
        }


    robots = {}


    if "robots_estado" in df.columns:

        robots = {

            str(clave):
                int(valor)

            for clave, valor
            in (

                df[
                    "robots_estado"
                ]
                .replace(
                    "",
                    "sin_estado",
                )
                .value_counts()
                .items()

            )
        }


    resumen = {

        "fase":
            "Fase 1 - Recolección",

        "modo":
            modo,

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fuentes_configuradas":
            int(
                total_configurado
            ),

        "fuentes_procesadas":
            int(
                len(
                    df
                )
            ),

        "estados":
            conteo_estados,

        "metodos_adquisicion":
            metodos,

        "estrategias_extraccion":
            estrategias,

        "robots_txt":
            robots,

        "control_calidad":
            resumen_calidad,

        "corpus_cientifico_v2_modificado":
            False,

        "corpus_cientifico_v2":
            str(
                CORPUS_CIENTIFICO_V2
            ),
    }


    with open(
        ruta,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            resumen,
            archivo,
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# 14. SELECCIÓN DE FUENTES
# =============================================================================

def seleccionar_fuentes(
    modo: str,
    limite: int | None,
):
    """
    Determina cuántas fuentes serán procesadas.

    DEMO:
        5 fuentes por defecto.

    COMPLETO:
        todas las fuentes por defecto.
    """

    fuentes = list(
        SITES
    )


    if limite is not None:

        if limite <= 0:

            raise ValueError(
                "--limite debe ser mayor que 0"
            )


        return fuentes[
            :limite
        ]


    if modo == "demo":

        return fuentes[
            :LIMITE_DEMO_POR_DEFECTO
        ]


    return fuentes


# =============================================================================
# 15. ARGUMENTOS
# =============================================================================

def construir_parser():
    """
    Define los argumentos disponibles.
    """

    parser = argparse.ArgumentParser(

        description=(
            "Fase 1 - Recolección controlada "
            "de perfiles de egreso."
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
            "completo procesa todas."
        ),

    )


    parser.add_argument(

        "--limite",

        type=int,

        default=None,

        help=(
            "Cantidad máxima de fuentes a procesar. "
            "En demo se usan 5 por defecto."
        ),

    )


    return parser


# =============================================================================
# 16. FLUJO PRINCIPAL
# =============================================================================

def main():
    """
    Orquesta la Fase 1.
    """

    parser = construir_parser()

    args = parser.parse_args()


    try:

        fuentes = seleccionar_fuentes(
            args.modo,
            args.limite,
        )

    except ValueError as error:

        parser.error(
            str(
                error
            )
        )


    (
        ruta_csv,
        ruta_resumen,
    ) = obtener_rutas_salida(
        args.modo
    )


    # =========================================================================
    # ENCABEZADO
    # =========================================================================

    print(
        "=" * 78
    )

    print(
        "FASE 1 — RECOLECCIÓN CONTROLADA"
    )

    print(
        "=" * 78
    )


    print(
        f"Modo                    : {args.modo}"
    )

    print(
        f"Fuentes configuradas    : {len(SITES)}"
    )

    print(
        f"Fuentes a procesar      : {len(fuentes)}"
    )

    print(
        f"Salida CSV              : {ruta_csv}"
    )


    print(
        "\nSEGURIDAD / REPRODUCIBILIDAD:"
    )

    print(
        "  - Solo se procesan URLs previamente configuradas."
    )

    print(
        "  - TLS/SSL permanece habilitado."
    )

    print(
        "  - Se aplican timeout, límites y controles de destino."
    )

    print(
        "  - Los errores no se corrigen ocultándolos."
    )

    print(
        "  - Esta ejecución NO modifica el corpus científico V2."
    )

    print(
        f"  - Corpus V2 protegido: {CORPUS_CIENTIFICO_V2}"
    )


    resultados = []


    # =========================================================================
    # SCRAPING
    # =========================================================================

    for indice, site in enumerate(
        fuentes,
        start=1,
    ):

        universidad = site.get(
            "universidad",
            "",
        )

        carrera = site.get(
            "carrera",
            "",
        )


        print(
            "\n"
            + "-" * 78
        )


        print(
            f"[{indice}/{len(fuentes)}] "
            f"{universidad} — {carrera}"
        )


        print(
            "-" * 78
        )


        try:

            data = scrapear_sitio(
                site
            )

        except Exception as error:

            # -----------------------------------------------------------------
            # Última barrera defensiva.
            #
            # Un problema inesperado en una universidad
            # no debe destruir toda la ejecución.
            # -----------------------------------------------------------------

            data = {

                "universidad":
                    universidad,

                "tipo_institucion":
                    site.get(
                        "tipo_institucion",
                        "",
                    ),

                "carrera":
                    carrera,

                "tipo_carrera":
                    site.get(
                        "tipo_carrera",
                        "",
                    ),

                "url":
                    site.get(
                        "url",
                        "",
                    ),

                "url_original":
                    site.get(
                        "url",
                        "",
                    ),

                "url_final":
                    "",

                "selector":
                    site.get(
                        "selector",
                        "",
                    ),

                "metodo_usado":
                    "",

                "estrategia_extraccion":
                    "",

                "termino_detectado":
                    "",

                "etiqueta_titulo":
                    "",

                "perfil_egreso":
                    "",

                "largo_perfil":
                    0,

                "requiere_revision":
                    True,

                "motivo_revision":
                    "error_no_controlado",

                "robots_estado":
                    "",

                "content_type":
                    "",

                "advertencias":
                    "",

                "error":
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
            }


            print(
                "   ERROR NO CONTROLADO: "
                f"{error}"
            )


        # ---------------------------------------------------------------------
        # Metadatos de trazabilidad
        # ---------------------------------------------------------------------

        data[
            "indice_fuente"
        ] = indice


        data[
            "fecha_extraccion_utc"
        ] = datetime.now(
            timezone.utc
        ).isoformat()


        resultados.append(
            data
        )


    # =========================================================================
    # DATAFRAME
    # =========================================================================

    df = pd.DataFrame(
        resultados
    )


    df = limpiar_dataframe(
        df
    )


    df[
        "estado_registro"
    ] = df.apply(
        determinar_estado_registro,
        axis=1,
    )


    df = ordenar_columnas(
        df
    )


    # =========================================================================
    # VALIDACIONES
    # =========================================================================

    validar_dataframe(
        df
    )


    resumen_calidad = validar_calidad_dataset(
        df
    )


    # =========================================================================
    # GUARDADO
    # =========================================================================

    guardar_csv(
        df,
        ruta_csv,
    )


    guardar_resumen(

        df=df,

        resumen_calidad=resumen_calidad,

        ruta=ruta_resumen,

        modo=args.modo,

        total_configurado=len(
            SITES
        ),
    )


    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RECOLECCIÓN FINALIZADA"
    )

    print(
        "=" * 78
    )


    print(
        "\nArchivos generados:"
    )

    print(
        f"  CSV     : {ruta_csv}"
    )

    print(
        f"  Resumen : {ruta_resumen}"
    )


    print(
        "\nIMPORTANTE:"
    )

    print(
        "  Estos archivos corresponden a una "
        "recolección actual de Internet."
    )

    print(
        "  NO sustituyen automáticamente el corpus "
        "científico V2."
    )

    print(
        "  Los registros REVISAR/ERROR deben ser "
        "auditados antes de cualquier uso experimental."
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "FASE 1 — RECOLECCIÓN COMPLETADA"
    )

    print(
        "=" * 78
    )


# =============================================================================
# 17. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()