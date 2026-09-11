# Módulo GWO — Grupo 8

Este paquete permite reproducir la selección de características mediante **Grey Wolf Optimizer (GWO)** y comprobar la estabilidad del resultado con validación cruzada.

## Qué contiene

```text
GWO_Grupo_8_Estudiantes/
├── README.md
├── requirements.txt
├── ejecutar_gwo.py
└── src/
    ├── gwo_feature_selection.py
    ├── cv10_gwo.py
    └── data/
        ├── processed/
        │   └── perfiles_egreso_etiquetado_v2.csv
        └── resultados_cientificos/
            ├── resultados de referencia
            └── salidas generadas por los scripts
```

El corpus incluido contiene **61 perfiles de egreso**: 31 de Ingeniería Civil Informática, 25 de Ingeniería en Informática y 5 de Ingeniería de Ejecución en Informática.

## Instalación en Windows

Abrir PowerShell dentro de la carpeta descomprimida y ejecutar:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell impide activar el entorno, se puede ejecutar directamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso recomendado para comenzar

Primero comprobar los resultados existentes, sin repetir la búsqueda GWO:

```powershell
python ejecutar_gwo.py --modo validar
```

Este modo usa las **249 características ya seleccionadas** y ejecuta la comparación con 5 y 10 pliegues. Genera:

- `cv5_gwo_resultados.csv`
- `cv10_gwo_resultados.csv`
- `cv10_gwo_resultados.png`
- `cv10_gwo_f1_clase.png`

## Ejecutar la optimización completa

```powershell
python ejecutar_gwo.py --modo completo
```

Este modo:

1. construye 400 características TF–IDF;
2. ejecuta GWO para seleccionar un subconjunto;
3. guarda las características seleccionadas;
4. evalúa el subconjunto con 5 pliegues;
5. ejecuta la comprobación adicional con 10 pliegues.

La optimización completa usa 100 épocas y una población de 30 lobos, por lo que puede tardar bastante según el computador.

## Interpretación correcta

- El resultado de **5 pliegues** es exploratorio. En la ejecución de referencia, GWO seleccionó 249 de 400 características y obtuvo un macro-F1 de 0,841.
- La comprobación de **10 pliegues** mostró un macro-F1 aproximado de 0,732 y un F1 de 0,400 para la clase minoritaria.
- La caída de rendimiento muestra sensibilidad al particionado. Por ello, el resultado de 0,841 no debe presentarse como una estimación definitiva de generalización.
- La selección GWO se realizó sobre el corpus disponible antes de la comprobación final. Para una evaluación estrictamente insesgada, GWO debe ejecutarse dentro de cada pliegue externo mediante validación anidada.

## Semillas y reproducibilidad

Los scripts usan `random_state = 42`. Aun así, diferencias de versiones de Python, `scikit-learn`, `imbalanced-learn` o `mealpy` pueden producir pequeñas variaciones.

## Archivos que no deben editarse al comenzar

- `src/data/processed/perfiles_egreso_etiquetado_v2.csv`
- `src/data/resultados_cientificos/gwo_features_seleccionadas.csv`

Se recomienda ejecutar primero `--modo validar`, revisar las salidas y luego experimentar con parámetros en una copia del paquete.

