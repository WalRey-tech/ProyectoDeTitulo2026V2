# -*- coding: utf-8 -*-

"""
FASE 2 — PREPARACIÓN Y AUDITORÍA DEL CORPUS CIENTÍFICO V2
==========================================================

OBJETIVO
--------

Esta etapa verifica la integridad estructural y textual del corpus
científico congelado:

    src/data/processed/perfiles_egreso_etiquetado_v2.csv

La Fase 2:

    - verifica cantidad de registros;
    - verifica distribución por grado;
    - detecta perfiles vacíos;
    - detecta posibles problemas de codificación;
    - detecta duplicados;
    - genera una copia normalizada para auditoría;
    - calcula SHA-256 antes y después;
    - NO modifica el corpus V2.

IMPORTANTE
----------

La copia generada por esta fase NO reemplaza la entrada de Fase 3.

La Fase 3 oficial continúa utilizando directamente:

    perfiles_egreso_etiquetado_v2.csv

Esta etapa tampoco ejecuta:

    - TF-IDF;
    - SMOTE;
    - clasificación;
    - GWO;
    - PCA/LDA;
    - homogeneidad;
    - test de permutación.

Esas responsabilidades pertenecen a Fase 3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata

from collections import Counter
from datetime import datetime, timezone
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


PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)


RUTA_CORPUS_V2 = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)


RUTA_COPIA_AUDITORIA = (
    PROCESSED_DIR
    / "perfiles_egreso_preparado_v2_auditoria.csv"
)


RUTA_RESUMEN = (
    PROCESSED_DIR
    / "resumen_preparacion_corpus_v2.json"
)


# =============================================================================
# 2. ESTRUCTURA OFICIAL ESPERADA
# =============================================================================

TOTAL_ESPERADO = 61


DISTRIBUCION_ESPERADA = {
    "Civil": 31,
    "Informática": 25,
    "Ejecución": 5,
}


COLUMNAS_REQUERIDAS = {
    "grado",
    "perfil_egreso",
}


CLASES_VALIDAS = set(
    DISTRIBUCION_ESPERADA.keys()
)


# =============================================================================
# 3. UTILIDADES
# =============================================================================

def texto_seguro(
    valor,
) -> str:
    """
    Convierte un valor en texto evitando transformar NaN en 'nan'.
    """

    if pd.isna(
        valor
    ):
        return ""

    return str(
        valor
    )


def calcular_sha256_archivo(
    ruta: Path,
) -> str | None:
    """
    Calcula SHA-256 de un archivo completo.
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


def calcular_sha256_texto(
    texto: str,
) -> str:
    """
    Calcula SHA-256 de un texto individual.

    Sirve para detectar perfiles idénticos sin modificar su contenido.
    """

    return hashlib.sha256(
        texto.encode(
            "utf-8"
        )
    ).hexdigest().upper()


# =============================================================================
# 4. NORMALIZACIÓN TEXTUAL NO DESTRUCTIVA
# =============================================================================

def normalizar_perfil(
    texto,
) -> str:
    """
    Genera una versión normalizada únicamente para auditoría.

    Operaciones:

        - Unicode NFC;
        - elimina caracteres zero-width;
        - convierte NBSP en espacio normal;
        - colapsa espacios repetidos;
        - elimina espacios iniciales/finales.

    NO:

        - transforma a minúsculas;
        - elimina stopwords;
        - lematiza;
        - elimina palabras;
        - cambia el corpus V2.
    """

    texto = texto_seguro(
        texto
    )


    texto = unicodedata.normalize(
        "NFC",
        texto,
    )


    texto = texto.replace(
        "\u200b",
        "",
    )


    texto = texto.replace(
        "\ufeff",
        "",
    )


    texto = texto.replace(
        "\xa0",
        " ",
    )


    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )


    return texto.strip()


# =============================================================================
# 5. CLAVES NORMALIZADAS PARA AUDITORÍA DE DUPLICADOS
# =============================================================================

def normalizar_clave(
    texto,
) -> str:
    """
    Normalización utilizada únicamente para comparar identificadores.

    Por ejemplo:

        Universidad de Chile
        universidad de chile

    son tratados como equivalentes para detectar duplicados.
    """

    texto = texto_seguro(
        texto
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


    texto = re.sub(
        r"\s+",
        " ",
        texto,
    )


    return texto.strip()


# =============================================================================
# 6. DETECCIÓN DE POSIBLE MOJIBAKE
# =============================================================================

def tiene_posible_mojibake(
    texto,
) -> bool:
    """
    Detecta marcas frecuentes de problemas de codificación.
    """

    texto = texto_seguro(
        texto
    )


    marcas = (
        "Ã",
        "Â",
        "â€",
        "â€™",
        "â€œ",
        "â€",
        "�",
    )


    return any(
        marca in texto
        for marca in marcas
    )


# =============================================================================
# 7. CARGA DEL CORPUS
# =============================================================================

def cargar_corpus() -> pd.DataFrame:
    """
    Carga el corpus científico V2.

    Se utiliza autodetección de separador para conservar compatibilidad
    con versiones CSV separadas por coma o punto y coma.
    """

    if not RUTA_CORPUS_V2.exists():

        raise FileNotFoundError(
            "No se encontró el corpus científico V2:\n"
            f"{RUTA_CORPUS_V2}"
        )


    df = pd.read_csv(
        RUTA_CORPUS_V2,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


    faltantes = (
        COLUMNAS_REQUERIDAS
        - set(
            df.columns
        )
    )


    if faltantes:

        raise ValueError(
            "El corpus V2 no contiene las columnas obligatorias: "
            + ", ".join(
                sorted(
                    faltantes
                )
            )
        )


    return df


# =============================================================================
# 8. AUDITORÍA DEL CORPUS
# =============================================================================

def auditar_corpus(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Genera una copia auditable del corpus.

    Ninguna fila es eliminada.
    """

    salida = df.copy()


    # =========================================================================
    # TEXTO ORIGINAL Y NORMALIZADO
    # =========================================================================

    salida[
        "perfil_normalizado_auditoria"
    ] = (

        salida[
            "perfil_egreso"
        ]
        .apply(
            normalizar_perfil
        )

    )


    salida[
        "largo_perfil_original"
    ] = (

        salida[
            "perfil_egreso"
        ]
        .apply(
            lambda valor:
                len(
                    texto_seguro(
                        valor
                    )
                )
        )

    )


    salida[
        "largo_perfil_normalizado"
    ] = (

        salida[
            "perfil_normalizado_auditoria"
        ]
        .str.len()

    )


    salida[
        "normalizacion_modifico_texto"
    ] = (

        salida.apply(
            lambda fila:
                texto_seguro(
                    fila[
                        "perfil_egreso"
                    ]
                )
                != fila[
                    "perfil_normalizado_auditoria"
                ],
            axis=1,
        )

    )


    salida[
        "sha256_perfil_normalizado"
    ] = (

        salida[
            "perfil_normalizado_auditoria"
        ]
        .apply(
            calcular_sha256_texto
        )

    )


    # =========================================================================
    # PERFIL VACÍO
    # =========================================================================

    salida[
        "perfil_vacio"
    ] = (

        salida[
            "perfil_normalizado_auditoria"
        ]
        .eq(
            ""
        )

    )


    # =========================================================================
    # CLASE FUERA DEL CATÁLOGO
    # =========================================================================

    salida[
        "grado_normalizado_auditoria"
    ] = (

        salida[
            "grado"
        ]
        .apply(
            lambda valor:
                texto_seguro(
                    valor
                ).strip()
        )

    )


    salida[
        "grado_fuera_catalogo"
    ] = (

        ~salida[
            "grado_normalizado_auditoria"
        ]
        .isin(
            CLASES_VALIDAS
        )

    )


    # =========================================================================
    # POSIBLE PROBLEMA DE CODIFICACIÓN
    # =========================================================================

    columnas_textuales_revisables = [

        columna
        for columna
        in [
            "universidad",
            "carrera",
            "grado",
            "perfil_egreso",
        ]

        if columna
        in salida.columns

    ]


    def revisar_encoding_fila(
        fila,
    ) -> bool:

        return any(

            tiene_posible_mojibake(
                fila[
                    columna
                ]
            )

            for columna
            in columnas_textuales_revisables

        )


    salida[
        "posible_problema_encoding"
    ] = salida.apply(
        revisar_encoding_fila,
        axis=1,
    )


    # =========================================================================
    # DUPLICADO DE PERFIL
    # =========================================================================

    perfil_no_vacio = (

        salida[
            "perfil_normalizado_auditoria"
        ]
        .ne(
            ""
        )

    )


    salida[
        "perfil_duplicado_exacto"
    ] = False


    salida.loc[
        perfil_no_vacio,
        "perfil_duplicado_exacto",
    ] = (

        salida.loc[
            perfil_no_vacio,
            "perfil_normalizado_auditoria",
        ]
        .duplicated(
            keep=False
        )

    )


    # =========================================================================
    # DUPLICADO UNIVERSIDAD + CARRERA
    # =========================================================================

    salida[
        "universidad_carrera_duplicada"
    ] = False


    if {
        "universidad",
        "carrera",
    }.issubset(
        salida.columns
    ):

        clave_universidad = (

            salida[
                "universidad"
            ]
            .apply(
                normalizar_clave
            )

        )


        clave_carrera = (

            salida[
                "carrera"
            ]
            .apply(
                normalizar_clave
            )

        )


        claves = pd.DataFrame(
            {
                "_universidad":
                    clave_universidad,

                "_carrera":
                    clave_carrera,
            }
        )


        claves_validas = (

            clave_universidad.ne(
                ""
            )
            &
            clave_carrera.ne(
                ""
            )

        )


        salida.loc[
            claves_validas,
            "universidad_carrera_duplicada",
        ] = (

            claves.loc[
                claves_validas
            ]
            .duplicated(
                subset=[
                    "_universidad",
                    "_carrera",
                ],
                keep=False,
            )
            .values

        )


    # =========================================================================
    # URL DUPLICADA
    # =========================================================================

    salida[
        "url_duplicada"
    ] = False


    if "url" in salida.columns:

        urls = (

            salida[
                "url"
            ]
            .apply(
                lambda valor:
                    texto_seguro(
                        valor
                    ).strip()
            )

        )


        urls_validas = urls.ne(
            ""
        )


        salida.loc[
            urls_validas,
            "url_duplicada",
        ] = (

            urls.loc[
                urls_validas
            ]
            .duplicated(
                keep=False
            )

        )


    # =========================================================================
    # OBSERVACIONES POR FILA
    # =========================================================================

    def construir_observaciones(
        fila,
    ) -> str:

        observaciones = []


        if bool(
            fila[
                "perfil_vacio"
            ]
        ):

            observaciones.append(
                "perfil_vacio"
            )


        if bool(
            fila[
                "grado_fuera_catalogo"
            ]
        ):

            observaciones.append(
                "grado_fuera_catalogo"
            )


        if bool(
            fila[
                "posible_problema_encoding"
            ]
        ):

            observaciones.append(
                "posible_problema_encoding"
            )


        if bool(
            fila[
                "perfil_duplicado_exacto"
            ]
        ):

            observaciones.append(
                "perfil_duplicado_exacto"
            )


        if bool(
            fila[
                "universidad_carrera_duplicada"
            ]
        ):

            observaciones.append(
                "universidad_carrera_duplicada"
            )


        if bool(
            fila[
                "url_duplicada"
            ]
        ):

            observaciones.append(
                "url_duplicada"
            )


        return " | ".join(
            observaciones
        )


    salida[
        "observaciones_fase2"
    ] = salida.apply(
        construir_observaciones,
        axis=1,
    )


    salida[
        "estado_fase2"
    ] = (

        salida[
            "observaciones_fase2"
        ]
        .apply(
            lambda texto:
                "OK"
                if not texto
                else "REVISAR"
        )

    )


    # =========================================================================
    # DISTRIBUCIÓN
    # =========================================================================

    distribucion = {

        str(
            grado
        ):
            int(
                cantidad
            )

        for grado, cantidad
        in (

            salida[
                "grado_normalizado_auditoria"
            ]
            .value_counts()
            .to_dict()
            .items()

        )

    }


    # =========================================================================
    # MÉTRICAS DE LONGITUD
    # =========================================================================

    largos = (

        salida[
            "largo_perfil_normalizado"
        ]

    )


    estadisticas_longitud = {

        "minimo":
            int(
                largos.min()
            )
            if len(
                largos
            )
            else 0,

        "mediana":
            float(
                largos.median()
            )
            if len(
                largos
            )
            else 0,

        "promedio":
            round(
                float(
                    largos.mean()
                ),
                2,
            )
            if len(
                largos
            )
            else 0,

        "maximo":
            int(
                largos.max()
            )
            if len(
                largos
            )
            else 0,
    }


    # =========================================================================
    # VALIDACIÓN DE LA ESTRUCTURA OFICIAL
    # =========================================================================

    total_correcto = (

        len(
            salida
        )
        == TOTAL_ESPERADO

    )


    distribucion_correcta = (

        distribucion
        == DISTRIBUCION_ESPERADA

    )


    perfiles_vacios = int(

        salida[
            "perfil_vacio"
        ]
        .sum()

    )


    estructura_oficial_ok = (

        total_correcto
        and distribucion_correcta
        and perfiles_vacios == 0

    )


    resumen = {

        "total_registros":
            int(
                len(
                    salida
                )
            ),

        "total_esperado":
            TOTAL_ESPERADO,

        "total_correcto":
            total_correcto,

        "distribucion_observada":
            distribucion,

        "distribucion_esperada":
            DISTRIBUCION_ESPERADA,

        "distribucion_correcta":
            distribucion_correcta,

        "perfiles_vacios":
            perfiles_vacios,

        "grados_fuera_catalogo":
            int(
                salida[
                    "grado_fuera_catalogo"
                ].sum()
            ),

        "posibles_problemas_encoding":
            int(
                salida[
                    "posible_problema_encoding"
                ].sum()
            ),

        "perfiles_duplicados_exactos":
            int(
                salida[
                    "perfil_duplicado_exacto"
                ].sum()
            ),

        "duplicados_universidad_carrera":
            int(
                salida[
                    "universidad_carrera_duplicada"
                ].sum()
            ),

        "urls_duplicadas":
            int(
                salida[
                    "url_duplicada"
                ].sum()
            ),

        "normalizaciones_que_modificaron_representacion":
            int(
                salida[
                    "normalizacion_modifico_texto"
                ].sum()
            ),

        "estado_fase2":
            {

                str(
                    estado
                ):
                    int(
                        cantidad
                    )

                for estado, cantidad
                in (

                    salida[
                        "estado_fase2"
                    ]
                    .value_counts()
                    .to_dict()
                    .items()

                )
            },

        "estadisticas_longitud":
            estadisticas_longitud,

        "estructura_oficial_ok":
            estructura_oficial_ok,
    }


    return (
        salida,
        resumen,
    )


# =============================================================================
# 9. GUARDADO
# =============================================================================

def guardar_csv_auditoria(
    df: pd.DataFrame,
):
    """
    Guarda una copia derivada.

    Nunca escribe sobre el corpus V2 original.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    df.to_csv(

        RUTA_COPIA_AUDITORIA,

        index=False,

        sep=";",

        encoding="utf-8-sig",

        quoting=csv.QUOTE_ALL,

        lineterminator="\n",

    )


def guardar_resumen(
    resumen_auditoria: dict,
    sha_antes: str | None,
    sha_despues: str | None,
):
    """
    Guarda el resumen de auditoría en JSON.
    """

    resumen = {

        "fase":
            "Fase 2 - Preparación y auditoría textual",

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "corpus_cientifico":
            str(
                RUTA_CORPUS_V2
            ),

        "salida_auditoria":
            str(
                RUTA_COPIA_AUDITORIA
            ),

        "sha256_v2_antes":
            sha_antes,

        "sha256_v2_despues":
            sha_despues,

        "v2_intacto":
            (
                sha_antes is not None
                and sha_antes == sha_despues
            ),

        "auditoria":
            resumen_auditoria,

        "rol_fase2":
            (
                "Preparación y auditoría no destructiva "
                "del corpus científico V2."
            ),

        "entrada_oficial_fase3":
            "perfiles_egreso_etiquetado_v2.csv",

        "salida_fase2_usada_como_entrada_fase3":
            False,

        "operaciones_no_realizadas_en_fase2": [
            "TF-IDF",
            "SMOTE",
            "clasificación supervisada",
            "GWO",
            "PCA",
            "LDA",
            "test de homogeneidad",
            "test de permutación",
        ],
    }


    with open(
        RUTA_RESUMEN,
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
# 10. IMPRESIÓN DE RESUMEN
# =============================================================================

def imprimir_resumen(
    resumen: dict,
):
    """
    Muestra un resumen legible para consola y defensa.
    """

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULTADO DE LA AUDITORÍA"
    )

    print(
        "=" * 78
    )


    print(
        f"Registros encontrados : "
        f"{resumen['total_registros']}"
    )

    print(
        f"Registros esperados   : "
        f"{resumen['total_esperado']}"
    )


    print(
        "\nDistribución observada:"
    )


    distribucion = resumen[
        "distribucion_observada"
    ]


    for grado in [
        "Civil",
        "Informática",
        "Ejecución",
    ]:

        print(
            f"  {grado:<12}: "
            f"{distribucion.get(grado, 0)}"
        )


    print(
        "\nControles textuales:"
    )

    print(
        f"  Perfiles vacíos                    : "
        f"{resumen['perfiles_vacios']}"
    )

    print(
        f"  Grados fuera del catálogo          : "
        f"{resumen['grados_fuera_catalogo']}"
    )

    print(
        f"  Posibles problemas de encoding     : "
        f"{resumen['posibles_problemas_encoding']}"
    )

    print(
        f"  Perfiles duplicados exactos        : "
        f"{resumen['perfiles_duplicados_exactos']}"
    )

    print(
        f"  Universidad + carrera duplicadas   : "
        f"{resumen['duplicados_universidad_carrera']}"
    )

    print(
        f"  URLs duplicadas                    : "
        f"{resumen['urls_duplicadas']}"
    )

    print(
        f"  Textos normalizados formalmente    : "
        f"{resumen['normalizaciones_que_modificaron_representacion']}"
    )


    print(
        "\nLongitud de perfiles:"
    )


    estadisticas = resumen[
        "estadisticas_longitud"
    ]


    print(
        f"  Mínimo   : {estadisticas['minimo']}"
    )

    print(
        f"  Mediana  : {estadisticas['mediana']}"
    )

    print(
        f"  Promedio : {estadisticas['promedio']}"
    )

    print(
        f"  Máximo   : {estadisticas['maximo']}"
    )


    print(
        "\nEstado por registro:"
    )


    for estado, cantidad in resumen[
        "estado_fase2"
    ].items():

        print(
            f"  {estado:<8}: {cantidad}"
        )


    print(
        "\nValidación estructural:"
    )


    if resumen[
        "estructura_oficial_ok"
    ]:

        print(
            "  ✓ CORPUS V2 COHERENTE CON LA ESTRUCTURA OFICIAL"
        )

    else:

        print(
            "  ✗ EL CORPUS V2 NO COINCIDE CON LA ESTRUCTURA ESPERADA"
        )


# =============================================================================
# 11. ARGUMENTOS
# =============================================================================

def construir_parser():
    """
    Interfaz de línea de comandos.
    """

    parser = argparse.ArgumentParser(

        description=(
            "Fase 2 - Preparación y auditoría "
            "no destructiva del corpus V2."
        )

    )


    parser.add_argument(

        "--permitir-estructura-distinta",

        action="store_true",

        help=(
            "No termina con error si el corpus no contiene "
            "exactamente 61 registros con distribución 31/25/5. "
            "Útil solamente para auditorías exploratorias."
        ),

    )


    return parser


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    """
    Ejecuta Fase 2.
    """

    parser = construir_parser()

    args = parser.parse_args()


    print(
        "=" * 78
    )

    print(
        "FASE 2 — PREPARACIÓN Y AUDITORÍA DEL CORPUS V2"
    )

    print(
        "=" * 78
    )


    print(
        f"Entrada oficial : {RUTA_CORPUS_V2}"
    )

    print(
        f"Salida auditoría: {RUTA_COPIA_AUDITORIA}"
    )

    print(
        f"Resumen JSON    : {RUTA_RESUMEN}"
    )


    print(
        "\nALCANCE:"
    )

    print(
        "  - Verificación estructural y textual."
    )

    print(
        "  - Normalización sobre una copia para auditoría."
    )

    print(
        "  - El corpus V2 original no se modifica."
    )

    print(
        "  - No se realizan experimentos de aprendizaje automático."
    )

    print(
        "  - Fase 3 continúa utilizando directamente el V2 original."
    )


    # =========================================================================
    # SHA ANTES
    # =========================================================================

    sha_antes = calcular_sha256_archivo(
        RUTA_CORPUS_V2
    )


    if sha_antes is None:

        print(
            "\nERROR:"
        )

        print(
            "No se encontró el corpus V2."
        )

        raise SystemExit(
            1
        )


    print(
        f"\nSHA-256 V2 antes : {sha_antes}"
    )


    # =========================================================================
    # CARGA Y AUDITORÍA
    # =========================================================================

    try:

        df = cargar_corpus()


        (
            df_auditoria,
            resumen,
        ) = auditar_corpus(
            df
        )


    except Exception as error:

        print(
            "\nERROR CRÍTICO:"
        )

        print(
            str(
                error
            )
        )

        raise SystemExit(
            1
        )


    # =========================================================================
    # GUARDAR COPIA DE AUDITORÍA
    # =========================================================================

    guardar_csv_auditoria(
        df_auditoria
    )


    # =========================================================================
    # SHA DESPUÉS
    # =========================================================================

    sha_despues = calcular_sha256_archivo(
        RUTA_CORPUS_V2
    )


    # =========================================================================
    # GUARDAR RESUMEN
    # =========================================================================

    guardar_resumen(
        resumen_auditoria=resumen,
        sha_antes=sha_antes,
        sha_despues=sha_despues,
    )


    # =========================================================================
    # RESULTADOS
    # =========================================================================

    imprimir_resumen(
        resumen
    )


    print(
        "\n"
        + "-" * 78
    )

    print(
        "VERIFICACIÓN DE INTEGRIDAD DEL V2"
    )

    print(
        "-" * 78
    )


    print(
        f"SHA-256 antes   : {sha_antes}"
    )

    print(
        f"SHA-256 después : {sha_despues}"
    )


    if (
        sha_antes
        == sha_despues
    ):

        print(
            "Estado           : ✓ V2 INTACTO"
        )

    else:

        print(
            "Estado           : ✗ EL CORPUS V2 CAMBIÓ"
        )

        raise SystemExit(
            1
        )


    # =========================================================================
    # VALIDACIÓN ESTRICTA
    # =========================================================================

    if (
        not resumen[
            "estructura_oficial_ok"
        ]
        and not args.permitir_estructura_distinta
    ):

        print(
            "\nERROR:"
        )

        print(
            "El corpus no coincide con la estructura científica "
            "oficial esperada."
        )

        print(
            "Revise los resultados antes de continuar."
        )

        raise SystemExit(
            2
        )


    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\nArchivos generados:"
    )

    print(
        f"  Auditoría CSV : {RUTA_COPIA_AUDITORIA}"
    )

    print(
        f"  Resumen JSON  : {RUTA_RESUMEN}"
    )


    print(
        "\nIMPORTANTE:"
    )

    print(
        "  El archivo preparado es un artefacto de auditoría."
    )

    print(
        "  NO reemplaza al corpus científico V2."
    )

    print(
        "  Fase 3 continúa leyendo directamente "
        "perfiles_egreso_etiquetado_v2.csv."
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "FASE 2 COMPLETADA CORRECTAMENTE"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()