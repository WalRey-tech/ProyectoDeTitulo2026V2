# -*- coding: utf-8 -*-

"""
FASE 1 — ETIQUETADO CONTROLADO DEL CORPUS RECOLECTADO
======================================================

Este script transforma la salida del scraping en un conjunto etiquetado
según el tipo de grado académico utilizado en la investigación.

REGLA DE ETIQUETADO
-------------------

La etiqueta se construye exclusivamente desde el nombre de la carrera:

1. Si el nombre contiene "civil":
       -> Civil

2. Si el nombre contiene "ejecución":
       -> Ejecución

3. En los demás casos:
       -> Informática

IMPORTANTE
----------

El tipo de institución NO determina automáticamente el grado.

Por ejemplo:

    Instituto Profesional + Ingeniería en Informática
    -> Informática

No se transforma automáticamente en "Ejecución".

Esto mantiene la definición de clases utilizada en el corpus científico.

MODOS
-----

DEMO:

    entrada:
        src/data/raw/perfiles_egreso_raw_demo.csv

    salida:
        src/data/processed/perfiles_egreso_etiquetado_demo.csv

COMPLETO:

    entrada:
        src/data/raw/perfiles_egreso_raw_actual.csv

    salida:
        src/data/processed/perfiles_egreso_etiquetado_actual.csv


MUY IMPORTANTE
--------------

Este script NO modifica:

    perfiles_egreso_etiquetado_v2.csv

El corpus V2 permanece congelado para reproducir los resultados
oficiales de la tesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

import pandas as pd


# =============================================================================
# 1. RUTAS
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


PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)


CORPUS_CIENTIFICO_V2 = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)


# =============================================================================
# 2. NORMALIZACIÓN
# =============================================================================

def normalizar_texto(
    texto,
) -> str:
    """
    Normaliza texto solamente para reglas internas.

    Convierte:

        Ejecución
        EJECUCION
        ejecución

    a una forma comparable.

    El texto original NO se modifica.
    """

    texto = str(
        texto or ""
    ).strip().lower()


    texto = unicodedata.normalize(
        "NFKD",
        texto,
    )


    texto = "".join(
        caracter
        for caracter in texto
        if not unicodedata.combining(
            caracter
        )
    )


    return texto


# =============================================================================
# 3. CLASIFICACIÓN DE GRADO
# =============================================================================

def clasificar_grado(
    nombre_carrera,
) -> str:
    """
    Clasifica una carrera en las tres categorías utilizadas
    en la investigación.

    Prioridad:

        Civil
        Ejecución
        Informática

    La clasificación depende del nombre de la carrera,
    no del tipo de institución.
    """

    carrera = normalizar_texto(
        nombre_carrera
    )


    # -------------------------------------------------------------------------
    # CIVIL
    # -------------------------------------------------------------------------

    if "civil" in carrera:

        return "Civil"


    # -------------------------------------------------------------------------
    # EJECUCIÓN
    # -------------------------------------------------------------------------

    if "ejecu" in carrera:

        return "Ejecución"


    # -------------------------------------------------------------------------
    # INFORMÁTICA
    # -------------------------------------------------------------------------

    return "Informática"


# =============================================================================
# 4. DETECCIÓN DE PROGRAMAS FUERA DEL ALCANCE
# =============================================================================

def detectar_programa_fuera_alcance(
    nombre_carrera,
) -> tuple[bool, str]:
    """
    Detecta programas técnicos o de analista que no corresponden
    al universo de carreras profesionales estudiadas.

    IMPORTANTE:

    No excluye:

        Ingeniería de Ejecución para Técnicos de Nivel Superior

    porque sigue siendo una carrera de ingeniería.
    """

    carrera = normalizar_texto(
        nombre_carrera
    )


    # -------------------------------------------------------------------------
    # TÉCNICO como nombre principal de programa
    # -------------------------------------------------------------------------

    if carrera.startswith(
        "tecnico "
    ):

        return (
            True,
            "programa_tecnico",
        )


    # -------------------------------------------------------------------------
    # ANALISTA
    # -------------------------------------------------------------------------

    if "analista" in carrera:

        return (
            True,
            "programa_analista",
        )


    return (
        False,
        "",
    )


# =============================================================================
# 5. ESTADO DEL SCRAPING
# =============================================================================

def obtener_estado_registro(
    fila,
) -> str:
    """
    Obtiene el estado generado por main.py.

    Si se procesa un archivo antiguo sin columna estado_registro,
    se infiere un estado mínimo desde perfil_egreso.
    """

    estado = str(
        fila.get(
            "estado_registro",
            "",
        )
    ).strip().upper()


    if estado in {
        "OK",
        "REVISAR",
        "ERROR",
    }:

        return estado


    perfil = str(
        fila.get(
            "perfil_egreso",
            "",
        )
    ).strip()


    if perfil:

        return "OK"


    return "ERROR"


# =============================================================================
# 6. AUDITORÍA DE REGISTROS
# =============================================================================

def preparar_dataset(
    df: pd.DataFrame,
):
    """
    Separa los registros aceptados de los excluidos.

    Solo los registros OK se etiquetan automáticamente.

    REVISAR y ERROR permanecen en una auditoría separada.
    """

    df = df.copy()


    # -------------------------------------------------------------------------
    # Validar columnas esenciales
    # -------------------------------------------------------------------------

    columnas_requeridas = {
        "carrera",
        "perfil_egreso",
    }


    faltantes = (
        columnas_requeridas
        - set(
            df.columns
        )
    )


    if faltantes:

        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(
                sorted(
                    faltantes
                )
            )
        )


    # -------------------------------------------------------------------------
    # Estado efectivo
    # -------------------------------------------------------------------------

    df[
        "estado_etiquetado"
    ] = df.apply(
        obtener_estado_registro,
        axis=1,
    )


    # -------------------------------------------------------------------------
    # Detectar programas fuera de alcance
    # -------------------------------------------------------------------------

    detecciones = (

        df[
            "carrera"
        ]
        .apply(
            detectar_programa_fuera_alcance
        )

    )


    df[
        "fuera_alcance"
    ] = [

        resultado[
            0
        ]

        for resultado
        in detecciones

    ]


    df[
        "motivo_fuera_alcance"
    ] = [

        resultado[
            1
        ]

        for resultado
        in detecciones

    ]


    # -------------------------------------------------------------------------
    # Motivo de exclusión
    # -------------------------------------------------------------------------

    def motivo_exclusion(
        fila,
    ) -> str:

        if fila[
            "estado_etiquetado"
        ] == "ERROR":

            return (
                "error_en_recoleccion"
            )


        if fila[
            "estado_etiquetado"
        ] == "REVISAR":

            return (
                "requiere_revision_manual"
            )


        if bool(
            fila[
                "fuera_alcance"
            ]
        ):

            return str(
                fila[
                    "motivo_fuera_alcance"
                ]
            )


        perfil = str(
            fila.get(
                "perfil_egreso",
                "",
            )
        ).strip()


        if not perfil:

            return (
                "perfil_vacio"
            )


        return ""


    df[
        "motivo_exclusion_etiquetado"
    ] = df.apply(
        motivo_exclusion,
        axis=1,
    )


    # -------------------------------------------------------------------------
    # Aceptados
    # -------------------------------------------------------------------------

    aceptados = (

        df[
            df[
                "motivo_exclusion_etiquetado"
            ]
            .eq(
                ""
            )
        ]
        .copy()

    )


    # -------------------------------------------------------------------------
    # Excluidos
    # -------------------------------------------------------------------------

    excluidos = (

        df[
            df[
                "motivo_exclusion_etiquetado"
            ]
            .ne(
                ""
            )
        ]
        .copy()

    )


    return (
        aceptados,
        excluidos,
    )


# =============================================================================
# 7. APLICAR ETIQUETAS
# =============================================================================

def etiquetar_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega la columna grado.
    """

    df = df.copy()


    df[
        "grado"
    ] = (

        df[
            "carrera"
        ]
        .apply(
            clasificar_grado
        )

    )


    # -------------------------------------------------------------------------
    # Colocar grado inmediatamente después de carrera
    # -------------------------------------------------------------------------

    columnas = list(
        df.columns
    )


    columnas.remove(
        "grado"
    )


    posicion = (
        columnas.index(
            "carrera"
        )
        + 1
    )


    columnas.insert(
        posicion,
        "grado",
    )


    return df[
        columnas
    ]


# =============================================================================
# 8. RUTAS SEGÚN MODO
# =============================================================================

def obtener_rutas(
    modo: str,
):
    """
    Define archivos de entrada y salida.

    Ninguna ruta corresponde al corpus V2.
    """

    if modo == "demo":

        entrada = (
            RAW_DIR
            / "perfiles_egreso_raw_demo.csv"
        )

        salida = (
            PROCESSED_DIR
            / "perfiles_egreso_etiquetado_demo.csv"
        )

        auditoria = (
            PROCESSED_DIR
            / "auditoria_etiquetado_demo.csv"
        )

        resumen = (
            PROCESSED_DIR
            / "resumen_etiquetado_demo.json"
        )


    else:

        entrada = (
            RAW_DIR
            / "perfiles_egreso_raw_actual.csv"
        )

        salida = (
            PROCESSED_DIR
            / "perfiles_egreso_etiquetado_actual.csv"
        )

        auditoria = (
            PROCESSED_DIR
            / "auditoria_etiquetado_actual.csv"
        )

        resumen = (
            PROCESSED_DIR
            / "resumen_etiquetado_actual.json"
        )


    return (
        entrada,
        salida,
        auditoria,
        resumen,
    )


# =============================================================================
# 9. GUARDADO
# =============================================================================

def guardar_csv(
    df: pd.DataFrame,
    ruta: Path,
):
    """
    Guarda CSV con formato consistente.
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
# 10. RESUMEN JSON
# =============================================================================

def guardar_resumen(
    aceptados: pd.DataFrame,
    excluidos: pd.DataFrame,
    ruta: Path,
    modo: str,
    entrada: Path,
):
    """
    Guarda una síntesis auditable del etiquetado.
    """

    distribucion = {}


    if (
        not aceptados.empty
        and "grado"
        in aceptados.columns
    ):

        distribucion = {

            str(clase):
                int(cantidad)

            for clase, cantidad
            in (

                aceptados[
                    "grado"
                ]
                .value_counts()
                .items()

            )
        }


    motivos_exclusion = {}


    if not excluidos.empty:

        motivos_exclusion = {

            str(motivo):
                int(cantidad)

            for motivo, cantidad
            in (

                excluidos[
                    "motivo_exclusion_etiquetado"
                ]
                .value_counts()
                .items()

            )
        }


    resumen = {

        "fase":
            "Fase 1 - Etiquetado",

        "modo":
            modo,

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "archivo_entrada":
            str(
                entrada
            ),

        "registros_aceptados":
            int(
                len(
                    aceptados
                )
            ),

        "registros_excluidos":
            int(
                len(
                    excluidos
                )
            ),

        "distribucion_grado":
            distribucion,

        "motivos_exclusion":
            motivos_exclusion,

        "regla_etiquetado": {

            "civil":
                "nombre de carrera contiene 'civil'",

            "ejecucion":
                "nombre de carrera contiene 'ejecu'",

            "informatica":
                "resto de carreras aceptadas",
        },

        "tipo_institucion_define_grado":
            False,

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
# 11. ARGUMENTOS
# =============================================================================

def construir_parser():
    """
    Construye la interfaz de línea de comandos.
    """

    parser = argparse.ArgumentParser(

        description=(
            "Etiquetado controlado de perfiles "
            "de egreso recolectados."
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
            "demo usa raw_demo; "
            "completo usa raw_actual."
        ),

    )


    return parser


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    """
    Ejecuta el etiquetado.
    """

    parser = construir_parser()

    args = parser.parse_args()


    (
        ruta_entrada,
        ruta_salida,
        ruta_auditoria,
        ruta_resumen,
    ) = obtener_rutas(
        args.modo
    )


    print(
        "=" * 78
    )

    print(
        "FASE 1 — ETIQUETADO CONTROLADO"
    )

    print(
        "=" * 78
    )


    print(
        f"Modo                  : {args.modo}"
    )

    print(
        f"Entrada               : {ruta_entrada}"
    )

    print(
        f"Salida etiquetada     : {ruta_salida}"
    )

    print(
        f"Auditoría exclusiones : {ruta_auditoria}"
    )


    print(
        "\nREGLAS:"
    )

    print(
        "  Civil       -> carrera contiene 'civil'"
    )

    print(
        "  Ejecución   -> carrera contiene 'ejecu'"
    )

    print(
        "  Informática -> resto de carreras aceptadas"
    )

    print(
        "  El tipo de institución NO define automáticamente el grado."
    )


    print(
        "\nCONTROL:"
    )

    print(
        "  Solo registros OK se etiquetan automáticamente."
    )

    print(
        "  REVISAR y ERROR quedan en auditoría."
    )

    print(
        "  Programas técnicos/analistas quedan fuera del conjunto etiquetado."
    )

    print(
        "  El corpus científico V2 NO se modifica."
    )


    # =========================================================================
    # ENTRADA
    # =========================================================================

    if not ruta_entrada.exists():

        print(
            "\nERROR:"
        )

        print(
            f"No existe el archivo de entrada:\n{ruta_entrada}"
        )

        print(
            "\nEjecuta primero el scraping correspondiente."
        )

        return


    df = pd.read_csv(
        ruta_entrada,
        sep=";",
        encoding="utf-8-sig",
    )


    print(
        f"\nRegistros recibidos: {len(df)}"
    )


    # =========================================================================
    # FILTRO / AUDITORÍA
    # =========================================================================

    try:

        (
            aceptados,
            excluidos,
        ) = preparar_dataset(
            df
        )

    except ValueError as error:

        print(
            f"\nERROR DE ESTRUCTURA: {error}"
        )

        return


    # =========================================================================
    # ETIQUETADO
    # =========================================================================

    aceptados = etiquetar_dataset(
        aceptados
    )


    # =========================================================================
    # RESULTADOS
    # =========================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULTADO DEL ETIQUETADO"
    )

    print(
        "=" * 78
    )


    print(
        f"Registros aceptados : {len(aceptados)}"
    )

    print(
        f"Registros excluidos : {len(excluidos)}"
    )


    if not aceptados.empty:

        print(
            "\nDistribución por grado:"
        )

        print(
            aceptados[
                "grado"
            ]
            .value_counts()
            .to_string()
        )


        if (
            "tipo_institucion"
            in aceptados.columns
        ):

            print(
                "\nDistribución grado × institución:"
            )

            print(
                pd.crosstab(
                    aceptados[
                        "grado"
                    ],
                    aceptados[
                        "tipo_institucion"
                    ],
                )
            )


    if not excluidos.empty:

        print(
            "\nMotivos de exclusión:"
        )

        print(
            excluidos[
                "motivo_exclusion_etiquetado"
            ]
            .value_counts()
            .to_string()
        )


        columnas = [

            columna
            for columna
            in [
                "universidad",
                "carrera",
                "estado_etiquetado",
                "motivo_exclusion_etiquetado",
            ]

            if columna
            in excluidos.columns

        ]


        print(
            "\nRegistros excluidos:"
        )

        print(
            excluidos[
                columnas
            ]
            .to_string(
                index=False
            )
        )


    # =========================================================================
    # GUARDAR
    # =========================================================================

    guardar_csv(
        aceptados,
        ruta_salida,
    )


    guardar_csv(
        excluidos,
        ruta_auditoria,
    )


    guardar_resumen(
        aceptados=aceptados,
        excluidos=excluidos,
        ruta=ruta_resumen,
        modo=args.modo,
        entrada=ruta_entrada,
    )


    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Dataset etiquetado : {ruta_salida}"
    )

    print(
        f"  Auditoría          : {ruta_auditoria}"
    )

    print(
        f"  Resumen            : {ruta_resumen}"
    )


    print(
        "\nIMPORTANTE:"
    )

    print(
        "  Esta salida corresponde a datos obtenidos "
        "desde Internet actualmente."
    )

    print(
        "  No sustituye el corpus experimental V2."
    )

    print(
        "  La construcción de la etiqueta usa el nombre "
        "de la carrera, no el texto del perfil."
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "ETIQUETADO COMPLETADO"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()