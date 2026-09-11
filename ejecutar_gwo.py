from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================

ROOT = Path(__file__).resolve().parent

SRC = ROOT / "src"

GWO_DIR = (
    SRC
    / "Fase3_Analisis"
    / "GWO"
)

DATASET = (
    SRC
    / "data"
    / "processed"
    / "perfiles_egreso_etiquetado_v2.csv"
)

RESULTADOS_DIR = (
    SRC
    / "data"
    / "resultados_cientificos"
    / "gwo"
)

FEATURES = (
    RESULTADOS_DIR
    / "gwo_features_seleccionadas.csv"
)

SCRIPT_SELECCION = (
    GWO_DIR
    / "07_seleccion_caracteristicas_gwo.py"
)

SCRIPT_VALIDACION = (
    GWO_DIR
    / "08_validacion_gwo.py"
)


# =============================================================================
# FUNCIONES
# =============================================================================

def verificar_archivo(ruta: Path, descripcion: str) -> None:
    """Comprueba que un archivo requerido exista."""

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {descripcion}:\n{ruta}"
        )


def ejecutar(ruta: Path) -> None:
    """Ejecuta un script usando el mismo Python del entorno activo."""

    verificar_archivo(
        ruta,
        f"el script {ruta.name}",
    )

    print("\n" + "=" * 72)
    print(f"Ejecutando: {ruta.name}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, str(ruta)],
        cwd=ROOT,
        check=True,
    )


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta la metodología GWO integrada al proyecto de título."
        )
    )

    parser.add_argument(
        "--modo",
        choices=("validar", "completo"),
        default="validar",
        help=(
            "validar: utiliza las características GWO existentes; "
            "completo: vuelve a ejecutar la selección GWO y luego valida."
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print("MÓDULO GWO — PROYECTO DE TÍTULO")
    print("=" * 72)

    verificar_archivo(
        DATASET,
        "el corpus V2",
    )

    RESULTADOS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Corpus     : {DATASET}")
    print(f"Resultados : {RESULTADOS_DIR}")
    print(f"Modo       : {args.modo}")

    if args.modo == "completo":

        print(
            "\nModo COMPLETO:"
            "\n1. Selección de características mediante GWO."
            "\n2. Validación 5-fold y 10-fold."
        )

        ejecutar(
            SCRIPT_SELECCION
        )

    else:

        verificar_archivo(
            FEATURES,
            "el archivo de características GWO",
        )

        print(
            "\nModo VALIDAR:"
            "\nSe utilizarán las características GWO ya seleccionadas."
        )

    ejecutar(
        SCRIPT_VALIDACION
    )

    print("\n" + "=" * 72)
    print("PROCESO GWO TERMINADO CORRECTAMENTE")
    print("=" * 72)

    print(
        "\nRevise los resultados en:"
        f"\n{RESULTADOS_DIR}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )