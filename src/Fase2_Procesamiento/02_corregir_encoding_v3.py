# -*- coding: utf-8 -*-

"""
FASE 2 — CORRECCIÓN CONTROLADA DE ENCODING

Sin argumentos muestra un menú; también admite --corpus actual o --corpus v2.
Actual lee perfiles_egreso_etiquetado_actual.csv (referencia: 63, 32/26/5)
y genera perfiles_egreso_etiquetado_actual_corregido.csv.
V2 lee perfiles_egreso_etiquetado_v2.csv (referencia: 61, 31/25/5)
y conserva la salida histórica perfiles_egreso_etiquetado_v3.csv.
Corrige únicamente mojibake conocido en perfil_egreso y registra los cambios.
Mantiene las filas, las clases y el archivo de entrada; verifica su SHA-256.
Ambos modos leen el etiquetado original: la copia preparada es solo auditoría.
La salida es un candidato y no cambia las rutas de Fase 3 automáticamente.

"""

from __future__ import annotations

import argparse
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


RUTA_ENTRADA = (
    PROCESSED_DIR
    / "perfiles_egreso_etiquetado_v2.csv"
)


RUTA_SALIDA = (
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

CORPUS_SELECCIONADO = "v2"


def configurar_corpus(corpus: str):
    """Selecciona rutas y expectativas; nunca escribe sobre la entrada."""
    global CORPUS_SELECCIONADO, RUTA_ENTRADA, RUTA_SALIDA, RUTA_AUDITORIA
    global RUTA_RESUMEN, TOTAL_ESPERADO, DISTRIBUCION_ESPERADA
    if corpus not in {"actual", "v2"}:
        raise ValueError("Corpus no válido.")
    CORPUS_SELECCIONADO = corpus
    sufijo_salida = "actual_corregido" if corpus == "actual" else "v3"
    RUTA_ENTRADA = PROCESSED_DIR / f"perfiles_egreso_etiquetado_{corpus}.csv"
    RUTA_SALIDA = PROCESSED_DIR / f"perfiles_egreso_etiquetado_{sufijo_salida}.csv"
    RUTA_AUDITORIA = PROCESSED_DIR / f"auditoria_correccion_encoding_{sufijo_salida}.csv"
    RUTA_RESUMEN = PROCESSED_DIR / f"resumen_correccion_encoding_{sufijo_salida}.json"
    TOTAL_ESPERADO = 61
    DISTRIBUCION_ESPERADA = {"Civil": 31, "Informática": 25, "Ejecución": 5}
    if corpus == "actual":
        # El corpus vivo depende de los aceptados, no del número de fuentes.
        ruta_resumen = PROCESSED_DIR / "resumen_etiquetado_actual.json"
        if not ruta_resumen.exists():
            raise FileNotFoundError(
                "Falta resumen_etiquetado_actual.json. Ejecuta primero "
                "etiquetador.py --modo completo con el raw actualizado."
            )
        resumen = json.loads(ruta_resumen.read_text(encoding="utf-8-sig"))
        if resumen.get("modo") != "completo":
            raise ValueError("El resumen debe corresponder al etiquetado completo.")
        TOTAL_ESPERADO = resumen["registros_aceptados"]
        DISTRIBUCION_ESPERADA = resumen["distribucion_grado"]
        if (type(TOTAL_ESPERADO) is not int or TOTAL_ESPERADO <= 0
                or not isinstance(DISTRIBUCION_ESPERADA, dict)
                or not set(DISTRIBUCION_ESPERADA).issubset({"Civil", "Informática", "Ejecución"})
                or any(type(v) is not int or v < 0 for v in DISTRIBUCION_ESPERADA.values())
                or sum(DISTRIBUCION_ESPERADA.values()) != TOTAL_ESPERADO):
            raise ValueError("El resumen de etiquetado contiene recuentos inválidos.")


def solicitar_corpus() -> str:
    """Elige explícitamente la entrada de esta ejecución."""
    print("\nCORPUS PARA FASE 2")
    print("1. Actual — recuentos del resumen de etiquetado completo")
    print("2. V2 — corpus anterior (61 esperados)")
    while True:
        try:
            opcion = input("Selecciona 1 o 2: ").strip().lower()
        except EOFError:
            raise SystemExit("Indica --corpus actual o --corpus v2.") from None
        except KeyboardInterrupt:
            raise SystemExit("\nOperación cancelada.") from None
        if opcion in {"1", "actual"}:
            return "actual"
        if opcion in {"2", "v2"}:
            return "v2"
        print("Opción inválida. Ingresa 1 o 2.")


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

def cargar_corpus() -> pd.DataFrame:
    """
    Carga el corpus de entrada.
    """

    if not RUTA_ENTRADA.exists():

        raise FileNotFoundError(
            "No existe el corpus de entrada:\n"
            f"{RUTA_ENTRADA}"
        )


    df = pd.read_csv(
        RUTA_ENTRADA,
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


def validar_estructura(
    df: pd.DataFrame,
):
    """
    Comprueba que la entrada corresponda al corpus elegido.
    """

    if len(
        df
    ) != TOTAL_ESPERADO:

        raise ValueError(
            "El corpus de entrada no contiene los "
            f"{TOTAL_ESPERADO} registros esperados. "
            f"Encontrados: {len(df)}"
        )


    distribucion = obtener_distribucion(
        df
    )


    if distribucion != DISTRIBUCION_ESPERADA:

        raise ValueError(
            "La distribución de clases del corpus de entrada no coincide "
            "con la distribución científica esperada.\n"
            f"Observada: {distribucion}\n"
            f"Esperada : {DISTRIBUCION_ESPERADA}"
        )


# =============================================================================
# 8. GENERACIÓN DE candidato corregido
# =============================================================================

def generar_candidato(
    df_entrada: pd.DataFrame,
):
    """
    Genera candidato corregido corrigiendo únicamente perfiles con mojibake detectable.
    """

    df_salida = df_entrada.copy()


    auditoria = []


    for indice, fila in df_entrada.iterrows():

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


        df_salida.at[
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
        df_salida,
        pd.DataFrame(
            auditoria
        ),
    )


# =============================================================================
# 9. VERIFICAR QUE SOLO CAMBIÓ PERFIL_EGRESO
# =============================================================================

def verificar_columnas_no_modificadas(
    df_entrada: pd.DataFrame,
    df_salida: pd.DataFrame,
):
    """
    Verifica que ninguna columna distinta de perfil_egreso haya cambiado.
    """

    columnas_comparar = [

        columna
        for columna
        in df_entrada.columns

        if columna != "perfil_egreso"

    ]


    for columna in columnas_comparar:

        serie_v2 = (

            df_entrada[
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

            df_salida[
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

def guardar_candidato(
    df_salida: pd.DataFrame,
):
    """
    Guarda el corpus candidato corregido.
    """

    df_salida.to_csv(

        RUTA_SALIDA,

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
    hash_entrada: str,
    hash_entrada_despues: str,
    hash_salida: str,
    df_salida: pd.DataFrame,
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

        "corpus_seleccionado": CORPUS_SELECCIONADO,
        "entrada":
            str(
                RUTA_ENTRADA
            ),

        "salida":
            str(
                RUTA_SALIDA
            ),

        "auditoria":
            str(
                RUTA_AUDITORIA
            ),

        "registros_total":
            int(
                len(
                    df_salida
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
                df_salida
            ),

        "sha256_entrada_antes":
            hash_entrada,

        "sha256_entrada_despues":
            hash_entrada_despues,

        "sha256_salida":
            hash_salida,

        "entrada_intacta":
            (
                hash_entrada
                == hash_entrada_despues
            ),

        "salida_es_candidato":
            True,

        "reemplaza_entrada_automaticamente":
            False,

        "nota":
            (
                "candidato corregido corrige únicamente problemas conocidos de encoding. "
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
    Ejecuta la generación controlada del candidato corregido.
    """
    parser = argparse.ArgumentParser(description="Corrección controlada de encoding.")
    parser.add_argument("--corpus", choices=["actual", "v2"], default=None,
                        help="Dataset a corregir; si se omite, muestra un menú.")
    args = parser.parse_args()
    configurar_corpus(args.corpus or solicitar_corpus())

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
        f"Entrada elegida : {RUTA_ENTRADA}"
    )

    print(
        f"Salida corregida: {RUTA_SALIDA}"
    )


    # =========================================================================
    # HASH corpus de entrada ANTES
    # =========================================================================

    if not RUTA_ENTRADA.exists():

        print(
            "\nERROR:"
        )

        print(
            "No existe el corpus de entrada."
        )

        raise SystemExit(
            1
        )


    hash_entrada_antes = sha256_archivo(
        RUTA_ENTRADA
    )


    print(
        f"\nSHA corpus de entrada     : {hash_entrada_antes}"
    )


    # =========================================================================
    # CARGA Y VALIDACIÓN
    # =========================================================================

    try:

        df_entrada = cargar_corpus()


        validar_estructura(
            df_entrada
        )


        (
            df_salida,
            df_auditoria,
        ) = generar_candidato(
            df_entrada
        )


        verificar_columnas_no_modificadas(
            df_entrada,
            df_salida,
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
        df_salida
    ) != len(
        df_entrada
    ):

        raise RuntimeError(
            "ERROR: cambió el número de registros."
        )


    distribucion_entrada = obtener_distribucion(
        df_entrada
    )


    distribucion_salida = obtener_distribucion(
        df_salida
    )


    if distribucion_entrada != distribucion_salida:

        raise RuntimeError(
            "ERROR: cambió la distribución de clases."
        )


    if distribucion_salida != DISTRIBUCION_ESPERADA:

        raise RuntimeError(
            "ERROR: la distribución de candidato corregido no coincide "
            f"con la esperada: {DISTRIBUCION_ESPERADA}."
        )


    # =========================================================================
    # MOJIBAKE RESTANTE
    # =========================================================================

    mojibake_restante = int(

        df_salida[
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


    guardar_candidato(
        df_salida
    )


    guardar_auditoria(
        df_auditoria
    )


    # =========================================================================
    # HASHES FINALES
    # =========================================================================

    hash_entrada_despues = sha256_archivo(
        RUTA_ENTRADA
    )


    hash_salida = sha256_archivo(
        RUTA_SALIDA
    )


    # =========================================================================
    # VERIFICACIÓN corpus de entrada
    # =========================================================================

    if hash_entrada_antes != hash_entrada_despues:

        raise RuntimeError(
            "ERROR: el corpus de entrada fue modificado."
        )


    # =========================================================================
    # RESUMEN JSON
    # =========================================================================

    guardar_resumen(

        hash_entrada=hash_entrada_antes,

        hash_entrada_despues=hash_entrada_despues,

        hash_salida=hash_salida,

        df_salida=df_salida,

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
        f"Registros totales     : {len(df_salida)}"
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
            f"{distribucion_salida.get(grado, 0)}"
        )


    print(
        f"\nSHA corpus de entrada después : {hash_entrada_despues}"
    )

    print(
        f"SHA candidato corregido         : {hash_salida}"
    )


    print(
        "\nEstado corpus de entrada      : ✓ INTACTO"
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
            "candidato corregido NO debe considerarse corregido todavía."
        )

        raise SystemExit(
            2
        )


    print(
        "\nArchivos generados:"
    )

    print(
        f"  candidato corregido candidato : {RUTA_SALIDA}"
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
        "  candidato corregido es todavía un corpus candidato."
    )

    print(
        "  corpus de entrada permanece intacto."
    )

    print(
        "  Antes de adoptar candidato corregido debemos volver a ejecutar Fase 3 "
        "y comparar los resultados."
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "CORRECCIÓN GENERADA CORRECTAMENTE"
    )

    print(
        "=" * 78
    )


# =============================================================================
# 13. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()