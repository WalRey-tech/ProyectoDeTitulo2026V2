# Reporte de resultados — corpus actual

Generado: 2026-09-23T01:24:08.035957+00:00

**63 perfiles; 62 grupos de validación.**

| Grado | Perfiles |
|---|---:|
| Civil | 32 |
| Ejecución | 5 |
| Informática | 26 |

## Evaluación predictiva

F1 macro entre 0 y 1. OOF reúne una predicción de prueba por perfil. La desviación entre folds no es un intervalo de confianza.

| Paso y modelo | Folds | F1 medio | Desv. folds | F1 OOF | Accuracy OOF |
|---|---:|---:|---:|---:|---:|
| 05 — TF-IDF completo + SMOTE + ComplementNB | 4 | 0.5605 | 0.2515 | 0.5815 | 0.6667 |
| 05 — GWO + SMOTE + ComplementNB | 4 | 0.7221 | 0.1345 | 0.7111 | 0.7302 |
| 06 — Títulos enmascarados + SMOTE + ComplementNB | 4 | 0.4552 | 0.1247 | 0.4981 | 0.6190 |

GWO menos baseline: **+0.1616** en F1 medio. Enmascarado menos baseline: **-0.1053**. Son diferencias descriptivas; no prueban superioridad estadística.

El paso 05 repite la selección de características dentro de cada entrenamiento externo. El paso 06 evalúa un modelo fijo con títulos enmascarados, sin selección GWO.

Búsqueda GWO del paso 05: 100 iteraciones y 30 lobos por entrenamiento externo.

## Selección GWO exploratoria (paso 04)

| Modelo | Características | F1 medio exploratorio |
|---|---:|---:|
| Baseline | 400 | 0.6358 |
| GWO | 279 | 0.8360 |

Búsqueda exploratoria: 100 iteraciones y 30 lobos.

Estas cifras usan un vocabulario global y una selección previa a la comparación; no se interpretan como rendimiento de prueba independiente.

## Estructura del corpus

PCA, varianza explicada en dos componentes: 9.71%. LDA utiliza las etiquetas y su gráfico es exploratorio.

Test de permutación: p = **0.00599880**, 5000 permutaciones; 62 unidades del test. Alcanza el umbral declarado α = 0.05. El resultado describe asociación bajo este protocolo; no demuestra igualdad o equivalencia de competencias.

### Similitud coseno entre centroides

| Grados | Similitud |
|---|---:|
| Civil–Ejecución | 0.6480 |
| Civil–Informática | 0.8350 |
| Ejecución–Informática | 0.6387 |

### Vocabulario distintivo

Hasta cinco términos por grado, ordenados por la razón de prevalencia del paso 03. No implican competencias exclusivas.

- **Civil:** estudios, destacan, fundamentos, basicas ciencias, disciplina.
- **Ejecución:** cristiana, desenvolverse, respeto, concepcion, lenguajes programacion.
- **Informática:** laboral, buenas, buenas practicas, componentes, contribuyendo.

## Enmascaramiento

Se eliminaron 68 menciones de títulos en 40 perfiles. La auditoría del paso 06 conserva los textos originales y transformados.

## Alcance y límites

- PCA y LDA son visualizaciones del corpus completo; LDA utiliza etiquetas y no demuestra rendimiento predictivo.
- La similitud TF-IDF mide coincidencia léxica ponderada; no demuestra equivalencia de competencias.
- El p-valor del paso 02 corresponde a unidades agrupadas, distintas de los pares descriptivos de perfiles.
- La ausencia de significancia no demuestra igualdad entre grados.
- Los términos distintivos describen este corpus; no son competencias exclusivas ni pruebas de significancia.
- Paso 04 es exploratorio: selección e IDF usan información global. Paso 05 evalúa selección dentro de train externo.
- Paso 06 usa ComplementNB + SMOTE con títulos enmascarados y sin GWO; no es GWO con enmascaramiento.
- F1 macro no equivale a precisión ni accuracy. Media de folds y F1 conjunto son agregaciones diferentes.
- La desviación de los folds no es un intervalo de confianza. No se demuestra superioridad con una diferencia de medias.
- Agrupar perfiles vinculados reduce dependencia conocida, pero no constituye prueba en universidades desconocidas.
- Enmascarar títulos no elimina todas las pistas institucionales o de grado.
- Las métricas ponderan perfiles, no grupos. La clase minoritaria limita la validación.
- Cambiar el protocolo después de observar resultados requiere una evaluación futura independiente.

## Procedencia

SHA-256 del corpus: `dc822ace9d53064eab3bc5279f82bc43a608d034750601af10321036918af59b`.

Se verificaron el corpus de los seis resúmenes, la cobertura de predicciones y la concordancia de métricas con sus CSV. El JSON conserva los resúmenes completos y las rutas y huellas de todos los archivos leídos. Esta comprobación no sustituye la revisión de la extracción y las etiquetas del corpus.
