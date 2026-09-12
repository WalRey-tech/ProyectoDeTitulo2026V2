# -*- coding: utf-8 -*-

"""
FASE 2 — CORRECCIÓN CONTROLADA DE ENCODING
==========================================

Genera un corpus candidato V3 a partir del corpus científico V2.

OBJETIVOS
---------

- Detectar y corregir mojibake conocido en perfil_egreso.
- NO modificar el corpus V2.
- Mantener exactamente los mismos 61 registros.
- Mantener exactamente la misma distribución de clases.
- Registrar cuáles filas fueron modificadas.
- Verificar que no queden marcas conocidas de mojibake.

IMPORTANTE
----------

El V3 generado es inicialmente un CANDIDATO.

No reemplaza automáticamente al corpus científico V2.

Antes de adoptar V3 como corpus oficial se debe volver a ejecutar
Fase 3 y comparar los resultados.
"""

from __future__ import annotations

import csv
import hashlib
import json

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


PROCESSED_DIR = (
    SRC_ROOT
    / "data"
    / "processed"
)


RUTA_V2 = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)


RUTA_V3 = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v3.csv"
)


RUTA_AUDITORIA = (
    PROCESSED_DIR
    / "auditoria_correccion_encoding_v3.csv"
)


RUTA_RESUMEN = (
    PROCESSED_DIR
    / "resumen_correccion_encoding_v3.json"
)


# =============================================================================
# 2. ESTRUCTURA CIENTÍFICA ESPERADA
# =============================================================================

TOTAL_ESPERADO = 61


DISTRIBUCION_ESPERADA = {
    "Civil": 31,
    "Informática": 25,
    "Ejecución": 5,
}


# =============================================================================
# 3. UTILIDADES
# =============================================================================

def texto_seguro(
    valor,
) -> str:
    """
    Convierte un valor a texto sin transformar NaN en la cadena 'nan'.
    """

    if pd.isna(
        valor
    ):
        return ""

    return str(
        valor
    )


def sha256_archivo(
    ruta: Path,
) -> str:
    """
    Calcula SHA-256 de un archivo.
    """

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
# 4. DETECCIÓN DE MOJIBAKE
# =============================================================================

def tiene_mojibake(
    texto,
) -> bool:
    """
    Detecta marcas frecuentes de UTF-8 interpretado incorrectamente.

    Se utilizan códigos Unicode explícitos para evitar depender
    de la codificación de PowerShell o del editor.
    """

    texto = texto_seguro(
        texto
    )


    marcas = (

        "\u00c3",          # Ã

        "\u00c2",          # Â

        "\u00e2\u20ac",    # â€

        "\ufffd",          # carácter de reemplazo Unicode

    )


    return any(
        marca in texto
        for marca in marcas
    )


# =============================================================================
# 5. CORRECCIÓN CONTROLADA
# =============================================================================

def corregir_mojibake(
    texto,
) -> str:
    """
    Corrige únicamente secuencias conocidas de mojibake.

    La transformación es explícita y auditable.

    No se realiza una recodificación genérica sobre todo el corpus,
    porque eso podría alterar texto que ya está correctamente escrito.
    """

    resultado = texto_seguro(
        texto
    )


    # =========================================================================
    # ORDEN IMPORTANTE
    #
    # Primero se reemplazan secuencias completas.
    # El carácter residual Â se elimina al final.
    # =========================================================================

    reemplazos = {

        # ---------------------------------------------------------------------
        # Vocales minúsculas
        # ---------------------------------------------------------------------

        "\u00c3\u00a1": "\u00e1",  # á
        "\u00c3\u00a9": "\u00e9",  # é
        "\u00c3\u00ad": "\u00ed",  # í
        "\u00c3\u00b3": "\u00f3",  # ó
        "\u00c3\u00ba": "\u00fa",  # ú


        # ---------------------------------------------------------------------
        # Vocales mayúsculas
        #
        # Algunas secuencias contienen caracteres de control U+008x/U+009x,
        # por eso se expresan mediante escapes Unicode.
        # ---------------------------------------------------------------------

        "\u00c3\u0081": "\u00c1",  # Á
        "\u00c3\u0089": "\u00c9",  # É
        "\u00c3\u008d": "\u00cd",  # Í
        "\u00c3\u0093": "\u00d3",  # Ó
        "\u00c3\u009a": "\u00da",  # Ú


        # ---------------------------------------------------------------------
        # Ñ / ñ
        # ---------------------------------------------------------------------

        "\u00c3\u00b1": "\u00f1",  # ñ
        "\u00c3\u0091": "\u00d1",  # Ñ


        # ---------------------------------------------------------------------
        # Ü / ü
        # ---------------------------------------------------------------------

        "\u00c3\u00bc": "\u00fc",  # ü
        "\u00c3\u009c": "\u00dc",  # Ü


        # ---------------------------------------------------------------------
        # Signos frecuentes
        # ---------------------------------------------------------------------

        "\u00c2\u00bf": "\u00bf",  # ¿
        "\u00c2\u00a1": "\u00a1",  # ¡


        # ---------------------------------------------------------------------
        # Espacios mal recodificados
        # ---------------------------------------------------------------------

        "\u00c2\u00a0": " ",
        "\u00c2 ": " ",

    }


    for origen, destino in reemplazos.items():

        resultado = resultado.replace(
            origen,
            destino,
        )


    # =========================================================================
    # Â residual
    # =========================================================================

    resultado = resultado.replace(
        "\u00c2",
        "",
    )


    return resultado


# =============================================================================
# 6. CARGA DEL CORPUS
# =============================================================================

def cargar_v2() -> pd.DataFrame:
    """
    Carga el corpus V2.
    """

    if not RUTA_V2.exists():

        raise FileNotFoundError(
            "No existe el corpus científico V2:\n"
            f"{RUTA_V2}"
        )


    df = pd.read_csv(
        RUTA_V2,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


    columnas_requeridas = {
        "grado",
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


    return df


# =============================================================================
# 7. VALIDACIÓN DE ESTRUCTURA
# =============================================================================

def obtener_distribucion(
    df: pd.DataFrame,
) -> dict:
    """
    Obtiene distribución de clases.
    """

    return {

        str(
            grado
        ):
            int(
                cantidad
            )

        for grado, cantidad
        in (

            df[
                "grado"
            ]
            .value_counts()
            .to_dict()
            .items()

        )

    }


def validar_estructura_v2(
    df: pd.DataFrame,
):
    """
    Comprueba que el V2 sea el corpus esperado.
    """

    if len(
        df
    ) != TOTAL_ESPERADO:

        raise ValueError(
            "El corpus V2 no contiene los "
            f"{TOTAL_ESPERADO} registros esperados. "
            f"Encontrados: {len(df)}"
        )


    distribucion = obtener_distribucion(
        df
    )


    if distribucion != DISTRIBUCION_ESPERADA:

        raise ValueError(
            "La distribución de clases del V2 no coincide "
            "con la distribución científica esperada.\n"
            f"Observada: {distribucion}\n"
            f"Esperada : {DISTRIBUCION_ESPERADA}"
        )


# =============================================================================
# 8. GENERACIÓN DE V3
# =============================================================================

def generar_v3(
    df_v2: pd.DataFrame,
):
    """
    Genera V3 corrigiendo únicamente perfiles con mojibake detectable.
    """

    df_v3 = df_v2.copy()


    auditoria = []


    for indice, fila in df_v2.iterrows():

        original = texto_seguro(
            fila[
                "perfil_egreso"
            ]
        )


        # ---------------------------------------------------------------------
        # Si no hay marcas sospechosas, el texto queda exactamente igual.
        # ---------------------------------------------------------------------

        if not tiene_mojibake(
            original
        ):

            continue


        corregido = corregir_mojibake(
            original
        )


        # ---------------------------------------------------------------------
        # Si la función no produjo cambios, tampoco alteramos el registro.
        # ---------------------------------------------------------------------

        if corregido == original:

            continue


        df_v3.at[
            indice,
            "perfil_egreso",
        ] = corregido


        auditoria.append(
            {
                "indice":
                    int(
                        indice
                    ),

                "universidad":
                    fila.get(
                        "universidad",
                        "",
                    ),

                "carrera":
                    fila.get(
                        "carrera",
                        "",
                    ),

                "grado":
                    fila.get(
                        "grado",
                        "",
                    ),

                "largo_original":
                    int(
                        len(
                            original
                        )
                    ),

                "largo_corregido":
                    int(
                        len(
                            corregido
                        )
                    ),

                "mojibake_antes":
                    bool(
                        tiene_mojibake(
                            original
                        )
                    ),

                "mojibake_despues":
                    bool(
                        tiene_mojibake(
                            corregido
                        )
                    ),

                "texto_modificado":
                    True,
            }
        )


    return (
        df_v3,
        pd.DataFrame(
            auditoria
        ),
    )


# =============================================================================
# 9. VERIFICAR QUE SOLO CAMBIÓ PERFIL_EGRESO
# =============================================================================

def verificar_columnas_no_modificadas(
    df_v2: pd.DataFrame,
    df_v3: pd.DataFrame,
):
    """
    Verifica que ninguna columna distinta de perfil_egreso haya cambiado.
    """

    columnas_comparar = [

        columna
        for columna
        in df_v2.columns

        if columna != "perfil_egreso"

    ]


    for columna in columnas_comparar:

        serie_v2 = (

            df_v2[
                columna
            ]
            .fillna(
                ""
            )
            .astype(
                str
            )

        )


        serie_v3 = (

            df_v3[
                columna
            ]
            .fillna(
                ""
            )
            .astype(
                str
            )

        )


        if not serie_v2.equals(
            serie_v3
        ):

            raise RuntimeError(
                "ERROR: se modificó una columna que debía permanecer "
                f"intacta: {columna}"
            )


# =============================================================================
# 10. GUARDADO
# =============================================================================

def guardar_v3(
    df_v3: pd.DataFrame,
):
    """
    Guarda el corpus candidato V3.
    """

    df_v3.to_csv(

        RUTA_V3,

        index=False,

        sep=";",

        encoding="utf-8-sig",

        quoting=csv.QUOTE_ALL,

        lineterminator="\n",

    )


def guardar_auditoria(
    df_auditoria: pd.DataFrame,
):
    """
    Guarda las filas modificadas.
    """

    # Si por alguna razón no hubiera cambios,
    # mantenemos columnas conocidas.

    if df_auditoria.empty:

        df_auditoria = pd.DataFrame(
            columns=[
                "indice",
                "universidad",
                "carrera",
                "grado",
                "largo_original",
                "largo_corregido",
                "mojibake_antes",
                "mojibake_despues",
                "texto_modificado",
            ]
        )


    df_auditoria.to_csv(

        RUTA_AUDITORIA,

        index=False,

        sep=";",

        encoding="utf-8-sig",

        quoting=csv.QUOTE_ALL,

        lineterminator="\n",

    )


# =============================================================================
# 11. RESUMEN JSON
# =============================================================================

def guardar_resumen(
    *,
    hash_v2: str,
    hash_v2_despues: str,
    hash_v3: str,
    df_v3: pd.DataFrame,
    registros_corregidos: int,
    mojibake_restante: int,
):
    """
    Guarda resumen reproducible de la transformación.
    """

    resumen = {

        "fase":
            "Fase 2 - Corrección controlada de encoding",

        "fecha_generacion_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "entrada_v2":
            str(
                RUTA_V2
            ),

        "salida_v3":
            str(
                RUTA_V3
            ),

        "auditoria":
            str(
                RUTA_AUDITORIA
            ),

        "registros_total":
            int(
                len(
                    df_v3
                )
            ),

        "registros_corregidos":
            int(
                registros_corregidos
            ),

        "mojibake_restante":
            int(
                mojibake_restante
            ),

        "distribucion_grado":
            obtener_distribucion(
                df_v3
            ),

        "sha256_v2_antes":
            hash_v2,

        "sha256_v2_despues":
            hash_v2_despues,

        "sha256_v3":
            hash_v3,

        "v2_intacto":
            (
                hash_v2
                == hash_v2_despues
            ),

        "v3_es_candidato":
            True,

        "v3_reemplaza_v2_automaticamente":
            False,

        "nota":
            (
                "V3 corrige únicamente problemas conocidos de encoding. "
                "Debe validarse nuevamente con Fase 3 antes de ser adoptado "
                "como corpus científico oficial."
            ),
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
# 12. MAIN
# =============================================================================

def main():
    """
    Ejecuta la generación controlada del candidato V3.
    """

    print(
        "=" * 78
    )

    print(
        "FASE 2 — CORRECCIÓN CONTROLADA DE ENCODING"
    )

    print(
        "=" * 78
    )


    print(
        f"Entrada V2 : {RUTA_V2}"
    )

    print(
        f"Salida V3  : {RUTA_V3}"
    )


    # =========================================================================
    # HASH V2 ANTES
    # =========================================================================

    if not RUTA_V2.exists():

        print(
            "\nERROR:"
        )

        print(
            "No existe el corpus V2."
        )

        raise SystemExit(
            1
        )


    hash_v2_antes = sha256_archivo(
        RUTA_V2
    )


    print(
        f"\nSHA V2     : {hash_v2_antes}"
    )


    # =========================================================================
    # CARGA Y VALIDACIÓN
    # =========================================================================

    try:

        df_v2 = cargar_v2()


        validar_estructura_v2(
            df_v2
        )


        (
            df_v3,
            df_auditoria,
        ) = generar_v3(
            df_v2
        )


        verificar_columnas_no_modificadas(
            df_v2,
            df_v3,
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
    # VALIDACIONES DEL CANDIDATO
    # =========================================================================

    if len(
        df_v3
    ) != len(
        df_v2
    ):

        raise RuntimeError(
            "ERROR: cambió el número de registros."
        )


    distribucion_v2 = obtener_distribucion(
        df_v2
    )


    distribucion_v3 = obtener_distribucion(
        df_v3
    )


    if distribucion_v2 != distribucion_v3:

        raise RuntimeError(
            "ERROR: cambió la distribución de clases."
        )


    if distribucion_v3 != DISTRIBUCION_ESPERADA:

        raise RuntimeError(
            "ERROR: la distribución de V3 no coincide "
            "con 31/25/5."
        )


    # =========================================================================
    # MOJIBAKE RESTANTE
    # =========================================================================

    mojibake_restante = int(

        df_v3[
            "perfil_egreso"
        ]
        .apply(
            tiene_mojibake
        )
        .sum()

    )


    # =========================================================================
    # GUARDADO
    # =========================================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    guardar_v3(
        df_v3
    )


    guardar_auditoria(
        df_auditoria
    )


    # =========================================================================
    # HASHES FINALES
    # =========================================================================

    hash_v2_despues = sha256_archivo(
        RUTA_V2
    )


    hash_v3 = sha256_archivo(
        RUTA_V3
    )


    # =========================================================================
    # VERIFICACIÓN V2
    # =========================================================================

    if hash_v2_antes != hash_v2_despues:

        raise RuntimeError(
            "ERROR: el corpus V2 fue modificado."
        )


    # =========================================================================
    # RESUMEN JSON
    # =========================================================================

    guardar_resumen(

        hash_v2=hash_v2_antes,

        hash_v2_despues=hash_v2_despues,

        hash_v3=hash_v3,

        df_v3=df_v3,

        registros_corregidos=len(
            df_auditoria
        ),

        mojibake_restante=mojibake_restante,

    )


    # =========================================================================
    # CONSOLA
    # =========================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULTADO"
    )

    print(
        "=" * 78
    )


    print(
        f"Registros totales     : {len(df_v3)}"
    )

    print(
        f"Registros corregidos  : {len(df_auditoria)}"
    )

    print(
        f"Mojibake restante     : {mojibake_restante}"
    )


    print(
        "\nDistribución:"
    )


    for grado in [
        "Civil",
        "Informática",
        "Ejecución",
    ]:

        print(
            f"  {grado:<12}: "
            f"{distribucion_v3.get(grado, 0)}"
        )


    print(
        f"\nSHA V2 después : {hash_v2_despues}"
    )

    print(
        f"SHA V3         : {hash_v3}"
    )


    print(
        "\nEstado V2      : ✓ INTACTO"
    )


    # =========================================================================
    # VALIDACIÓN FINAL DEL ENCODING
    # =========================================================================

    if mojibake_restante != 0:

        print(
            "\n"
            + "=" * 78
        )

        print(
            "ATENCIÓN"
        )

        print(
            "=" * 78
        )

        print(
            "Todavía existen perfiles con marcas conocidas de mojibake."
        )

        print(
            "V3 NO debe considerarse corregido todavía."
        )

        raise SystemExit(
            2
        )


    print(
        "\nArchivos generados:"
    )

    print(
        f"  V3 candidato : {RUTA_V3}"
    )

    print(
        f"  Auditoría    : {RUTA_AUDITORIA}"
    )

    print(
        f"  Resumen      : {RUTA_RESUMEN}"
    )


    print(
        "\nIMPORTANTE:"
    )

    print(
        "  V3 es todavía un corpus candidato."
    )

    print(
        "  V2 permanece intacto."
    )

    print(
        "  Antes de adoptar V3 debemos volver a ejecutar Fase 3 "
        "y comparar los resultados."
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "CORRECCIÓN V3 GENERADA CORRECTAMENTE"
    )

    print(
        "=" * 78
    )


# =============================================================================
# 13. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()