# -*- coding: utf-8 -*-

"""
EXTRACCIÓN CONTROLADA DE PERFILES DE EGRESO
============================================

Este módulo recibe HTML ya descargado y busca el contenido asociado
al perfil de egreso.

La extracción se realiza en tres niveles:

1. selector_manual
   Se utiliza el selector CSS definido explícitamente en config.py.

2. contexto_prioritario
   Se buscan encabezados claramente asociados a un perfil de egreso.

3. contexto_secundario
   Se buscan secciones relacionadas, como competencias o descripción
   de la carrera.

IMPORTANTE:
El contenido secundario NO se considera automáticamente equivalente
a un perfil de egreso. Se marca como "requiere_revision=True" para
mantener trazabilidad metodológica.

Este módulo NO realiza conexiones de red ni ejecuta JavaScript.
Solamente procesa HTML como datos.
"""

from __future__ import annotations

import unicodedata

from bs4 import BeautifulSoup
from bs4.element import Tag

from utils import limpiar_texto


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

# ---------------------------------------------------------------------------
# Expresiones que identifican directamente un perfil de egreso.
# ---------------------------------------------------------------------------

PALABRAS_CLAVE_PRIORITARIAS = [
    "perfil de egreso",
    "perfil del egresado",
    "perfil de egresado",
    "perfil de la egresada",
    "perfil del titulado",
    "perfil profesional",
    "competencias de egreso",
]


# ---------------------------------------------------------------------------
# Expresiones relacionadas, pero que NO necesariamente representan
# un perfil de egreso completo.
#
# Por eso los resultados encontrados mediante estas expresiones quedan
# marcados para revisión.
# ---------------------------------------------------------------------------

PALABRAS_CLAVE_SECUNDARIAS = [
    "competencias",
    "el titulado",
    "la titulada",
    "qué aprenderás",
    "que aprenderas",
    "sobre la carrera",
    "descripción de la carrera",
    "descripcion de la carrera",
    "desempeño profesional",
    "desempeno profesional",
    "campo ocupacional",
    "campo laboral",
]


# ---------------------------------------------------------------------------
# Etiquetas que normalmente representan encabezados reales.
# ---------------------------------------------------------------------------

ETIQUETAS_TITULO = [
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "strong",
    "b",
]


# ---------------------------------------------------------------------------
# <span> se conserva únicamente como fallback.
#
# No se mezcla con los encabezados principales porque las páginas modernas
# utilizan <span> para menús, botones y otros componentes que pueden provocar
# falsos positivos.
# ---------------------------------------------------------------------------

ETIQUETAS_TITULO_FALLBACK = [
    "span",
]


# ---------------------------------------------------------------------------
# Bloques que pueden contener texto académico útil.
# ---------------------------------------------------------------------------

ETIQUETAS_CONTENIDO = [
    "p",
    "ul",
    "ol",
    "div",
]


# ---------------------------------------------------------------------------
# Parámetros conservadores.
# ---------------------------------------------------------------------------

LONGITUD_REFERENCIA_PERFIL = 150

MAX_BLOQUES_HERMANOS = 12

MAX_ELEMENTOS_FALLBACK = 30

MAX_LONGITUD_TITULO = 250


# =============================================================================
# 2. NORMALIZACIÓN
# =============================================================================

def normalizar_para_busqueda(
    texto: str,
) -> str:
    """
    Normaliza texto únicamente para comparaciones internas.

    Ejemplo:
        "Perfil de Egreso"
        "PERFIL DE EGRESO"
        "perfil de egreso"

    terminan representados de forma equivalente.

    La normalización NO modifica el texto final guardado.
    """

    if not texto:
        return ""

    texto = limpiar_texto(
        str(texto)
    ).lower()

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
# 3. UTILIDADES DE TEXTO
# =============================================================================

def extraer_texto_nodo(
    nodo,
) -> str:
    """
    Obtiene únicamente el texto visible de un nodo HTML.

    No ejecuta código HTML o JavaScript.
    """

    if not nodo:
        return ""

    texto = nodo.get_text(
        " ",
        strip=True,
    )

    return limpiar_texto(
        texto
    )


def unir_fragmentos(
    fragmentos: list[str],
) -> str:
    """
    Une fragmentos evitando duplicados exactos.

    Mantiene el orden original de aparición.
    """

    resultado = []

    vistos = set()

    for fragmento in fragmentos:

        fragmento = limpiar_texto(
            fragmento
        )

        if not fragmento:
            continue

        clave = fragmento.casefold()

        if clave in vistos:
            continue

        vistos.add(
            clave
        )

        resultado.append(
            fragmento
        )

    return limpiar_texto(
        " ".join(
            resultado
        )
    )


# =============================================================================
# 4. FILTRO DE ZONAS NO DESEADAS
# =============================================================================

def esta_en_zona_no_contenido(
    nodo,
) -> bool:
    """
    Evita interpretar elementos de navegación como perfil de egreso.

    Ejemplos:
        nav
        footer
        header
        aside
    """

    if not nodo:
        return False

    for padre in nodo.parents:

        nombre = getattr(
            padre,
            "name",
            None,
        )

        if nombre in {
            "nav",
            "footer",
            "header",
            "aside",
        }:

            return True

    return False


# =============================================================================
# 5. EXTRACCIÓN CONTROLADA DE BLOQUES
# =============================================================================

def extraer_bloque_controlado(
    nodo,
) -> str:
    """
    Extrae texto desde un bloque HTML sin asumir automáticamente
    que todo <div> corresponde al perfil.

    Para <div>, se prefieren párrafos y elementos de lista internos.
    """

    if not nodo:
        return ""

    if nodo.name in {
        "p",
        "ul",
        "ol",
    }:

        return extraer_texto_nodo(
            nodo
        )


    if nodo.name == "div":

        fragmentos = []

        elementos = nodo.find_all(
            [
                "p",
                "li",
            ],
            recursive=True,
        )

        for elemento in elementos:

            texto = extraer_texto_nodo(
                elemento
            )

            if texto:
                fragmentos.append(
                    texto
                )


        texto_interno = unir_fragmentos(
            fragmentos
        )


        if texto_interno:
            return texto_interno


        # Algunos sitios escriben directamente dentro de un div.
        # En ese caso se permite usar su texto, pero únicamente si
        # tiene un tamaño razonable.
        texto_div = extraer_texto_nodo(
            nodo
        )

        if (
            texto_div
            and len(
                texto_div
            ) <= 5000
        ):

            return texto_div


    return ""


# =============================================================================
# 6. BÚSQUEDA DE CONTENIDO CERCANO
# =============================================================================

def buscar_contenido_cercano(
    titulo,
) -> str:
    """
    Busca contenido asociado inmediatamente a un encabezado.

    Estrategia A:
        Examina hermanos posteriores.

    Estrategia B:
        Si la página posee una estructura anidada, busca elementos
        siguientes dentro del mismo contenedor principal.

    La búsqueda está limitada para evitar capturar grandes zonas
    no relacionadas de la página.
    """

    if not titulo:
        return ""


    # =========================================================================
    # A. HERMANOS POSTERIORES
    # =========================================================================

    fragmentos = []

    hermano = titulo.find_next_sibling()

    bloques_revisados = 0


    while (
        hermano
        and bloques_revisados
        < MAX_BLOQUES_HERMANOS
    ):

        if not isinstance(
            hermano,
            Tag,
        ):

            hermano = (
                hermano.find_next_sibling()
            )

            continue


        bloques_revisados += 1


        # Un nuevo encabezado principal normalmente representa
        # el término de la sección actual.
        if hermano.name in {
            "h1",
            "h2",
            "h3",
        }:

            break


        if (
            hermano.name
            in ETIQUETAS_CONTENIDO
            and not esta_en_zona_no_contenido(
                hermano
            )
        ):

            texto = (
                extraer_bloque_controlado(
                    hermano
                )
            )

            if texto:

                fragmentos.append(
                    texto
                )


        hermano = (
            hermano.find_next_sibling()
        )


    texto_hermanos = unir_fragmentos(
        fragmentos
    )


    if texto_hermanos:

        return texto_hermanos


    # =========================================================================
    # B. BÚSQUEDA CONTROLADA DENTRO DEL MISMO CONTENEDOR
    # =========================================================================

    contenedor = titulo.find_parent(
        [
            "section",
            "article",
            "main",
            "div",
        ]
    )


    elementos_revisados = 0

    fragmentos = []


    for candidato in titulo.find_all_next(
        [
            "h1",
            "h2",
            "h3",
            "p",
            "ul",
            "ol",
            "div",
        ],
        limit=MAX_ELEMENTOS_FALLBACK,
    ):

        if candidato is titulo:
            continue


        # Si se identificó un contenedor lógico,
        # no permitimos que la búsqueda salga de él.
        if (
            contenedor is not None
            and contenedor
            not in candidato.parents
        ):

            break


        elementos_revisados += 1


        if candidato.name in {
            "h1",
            "h2",
            "h3",
        }:

            break


        if esta_en_zona_no_contenido(
            candidato
        ):

            continue


        texto = extraer_bloque_controlado(
            candidato
        )


        if texto:

            fragmentos.append(
                texto
            )


        if (
            elementos_revisados
            >= MAX_ELEMENTOS_FALLBACK
        ):

            break


    return unir_fragmentos(
        fragmentos
    )


# =============================================================================
# 7. DETECCIÓN POR CONTEXTO
# =============================================================================

def encontrar_palabra_clave(
    texto_titulo: str,
    palabras_clave: list[str],
) -> str:
    """
    Devuelve la palabra clave que produjo la coincidencia.
    """

    texto_normalizado = (
        normalizar_para_busqueda(
            texto_titulo
        )
    )


    if not texto_normalizado:

        return ""


    for palabra in palabras_clave:

        palabra_normalizada = (
            normalizar_para_busqueda(
                palabra
            )
        )

        if (
            palabra_normalizada
            in texto_normalizado
        ):

            return palabra


    return ""


def buscar_por_contexto(
    soup,
    palabras_clave: list[str],
) -> dict:
    """
    Busca encabezados asociados al perfil.

    Devuelve tanto el texto como la palabra que originó
    la extracción, permitiendo auditar posteriormente
    cómo se obtuvo cada registro.
    """

    # =========================================================================
    # A. ENCABEZADOS REALES
    # =========================================================================

    titulos = soup.find_all(
        ETIQUETAS_TITULO
    )


    for titulo in titulos:

        if esta_en_zona_no_contenido(
            titulo
        ):

            continue


        texto_titulo = titulo.get_text(
            " ",
            strip=True,
        )


        if (
            not texto_titulo
            or len(
                texto_titulo
            ) > MAX_LONGITUD_TITULO
        ):

            continue


        palabra_detectada = (
            encontrar_palabra_clave(
                texto_titulo,
                palabras_clave,
            )
        )


        if not palabra_detectada:

            continue


        texto_extraido = (
            buscar_contenido_cercano(
                titulo
            )
        )


        if texto_extraido:

            return {
                "texto":
                    texto_extraido,

                "termino_detectado":
                    palabra_detectada,

                "etiqueta_titulo":
                    titulo.name,
            }


    # =========================================================================
    # B. FALLBACK EN <span>
    # =========================================================================

    spans = soup.find_all(
        ETIQUETAS_TITULO_FALLBACK
    )


    for titulo in spans:

        if esta_en_zona_no_contenido(
            titulo
        ):

            continue


        texto_titulo = titulo.get_text(
            " ",
            strip=True,
        )


        if (
            not texto_titulo
            or len(
                texto_titulo
            ) > MAX_LONGITUD_TITULO
        ):

            continue


        palabra_detectada = (
            encontrar_palabra_clave(
                texto_titulo,
                palabras_clave,
            )
        )


        if not palabra_detectada:

            continue


        texto_extraido = (
            buscar_contenido_cercano(
                titulo
            )
        )


        if texto_extraido:

            return {
                "texto":
                    texto_extraido,

                "termino_detectado":
                    palabra_detectada,

                "etiqueta_titulo":
                    titulo.name,
            }


    return {
        "texto":
            "",

        "termino_detectado":
            "",

        "etiqueta_titulo":
            "",
    }


# =============================================================================
# 8. CONSTRUCCIÓN DEL RESULTADO
# =============================================================================

def construir_resultado(
    texto: str = "",
    estrategia: str = "sin_resultado",
    termino_detectado: str = "",
    selector_usado: str = "",
    etiqueta_titulo: str = "",
    requiere_revision: bool = False,
    motivo_revision: str = "",
    error_extraccion: str = "",
) -> dict:
    """
    Construye una salida auditable y uniforme.
    """

    texto = limpiar_texto(
        texto
    )


    longitud = len(
        texto
    )


    # Un texto demasiado corto no necesariamente es incorrecto,
    # pero debe ser revisado antes de considerarlo científicamente válido.
    if (
        texto
        and longitud
        < LONGITUD_REFERENCIA_PERFIL
    ):

        requiere_revision = True

        if not motivo_revision:

            motivo_revision = (
                "texto_menor_a_150_caracteres"
            )


    return {

        "texto":
            texto,

        "estrategia":
            estrategia,

        "termino_detectado":
            termino_detectado,

        "selector_usado":
            selector_usado,

        "etiqueta_titulo":
            etiqueta_titulo,

        "longitud_texto":
            longitud,

        "requiere_revision":
            bool(
                requiere_revision
            ),

        "motivo_revision":
            motivo_revision,

        "error_extraccion":
            error_extraccion,
    }


# =============================================================================
# 9. EXTRACCIÓN PRINCIPAL CON METADATOS
# =============================================================================

def extraer_con_metadatos(
    html: str,
    selector: str = "",
) -> dict:
    """
    Extrae contenido manteniendo trazabilidad.

    Orden:

        1. selector manual
        2. contexto prioritario
        3. contexto secundario
        4. sin resultado

    La estrategia secundaria siempre queda marcada para revisión.
    """

    if not html:

        return construir_resultado(
            error_extraccion=(
                "html_vacio"
            )
        )


    try:

        soup = BeautifulSoup(
            html,
            "lxml",
        )


        # =====================================================================
        # Eliminar elementos que nunca son relevantes para el texto académico.
        # =====================================================================

        for etiqueta in soup.find_all(
            [
                "script",
                "style",
                "noscript",
                "template",
                "svg",
            ]
        ):

            etiqueta.decompose()


        selector_limpio = (
            str(
                selector or ""
            ).strip()
        )


        error_selector = ""


        # =====================================================================
        # PASO 1 — SELECTOR MANUAL
        # =====================================================================

        if selector_limpio not in {
            "",
            "-",
        }:

            try:

                nodos = soup.select(
                    selector_limpio
                )

            except Exception as error:

                nodos = []

                error_selector = (
                    f"selector_css_invalido: "
                    f"{error}"
                )


            if nodos:

                fragmentos = []

                for nodo in nodos:

                    texto_nodo = (
                        extraer_texto_nodo(
                            nodo
                        )
                    )

                    if texto_nodo:

                        fragmentos.append(
                            texto_nodo
                        )


                texto = unir_fragmentos(
                    fragmentos
                )


                if texto:

                    return construir_resultado(
                        texto=texto,
                        estrategia=(
                            "selector_manual"
                        ),
                        selector_usado=(
                            selector_limpio
                        ),
                        requiere_revision=False,
                    )


        # =====================================================================
        # PASO 2 — CONTEXTO PRIORITARIO
        # =====================================================================

        prioritario = buscar_por_contexto(
            soup,
            PALABRAS_CLAVE_PRIORITARIAS,
        )


        if prioritario[
            "texto"
        ]:

            return construir_resultado(
                texto=prioritario[
                    "texto"
                ],
                estrategia=(
                    "contexto_prioritario"
                ),
                termino_detectado=(
                    prioritario[
                        "termino_detectado"
                    ]
                ),
                selector_usado=(
                    selector_limpio
                ),
                etiqueta_titulo=(
                    prioritario[
                        "etiqueta_titulo"
                    ]
                ),
                requiere_revision=False,
                error_extraccion=(
                    error_selector
                ),
            )


        # =====================================================================
        # PASO 3 — CONTEXTO SECUNDARIO
        # =====================================================================

        secundario = buscar_por_contexto(
            soup,
            PALABRAS_CLAVE_SECUNDARIAS,
        )


        if secundario[
            "texto"
        ]:

            return construir_resultado(
                texto=secundario[
                    "texto"
                ],
                estrategia=(
                    "contexto_secundario"
                ),
                termino_detectado=(
                    secundario[
                        "termino_detectado"
                    ]
                ),
                selector_usado=(
                    selector_limpio
                ),
                etiqueta_titulo=(
                    secundario[
                        "etiqueta_titulo"
                    ]
                ),
                requiere_revision=True,
                motivo_revision=(
                    "contenido_relacionado_pero_no_"
                    "necesariamente_perfil_de_egreso"
                ),
                error_extraccion=(
                    error_selector
                ),
            )


        # =====================================================================
        # SIN RESULTADO
        # =====================================================================

        return construir_resultado(
            estrategia=(
                "sin_resultado"
            ),
            selector_usado=(
                selector_limpio
            ),
            requiere_revision=True,
            motivo_revision=(
                "no_se_encontro_contenido"
            ),
            error_extraccion=(
                error_selector
            ),
        )


    except Exception as error:

        return construir_resultado(
            estrategia=(
                "error"
            ),
            selector_usado=(
                str(
                    selector or ""
                ).strip()
            ),
            requiere_revision=True,
            motivo_revision=(
                "error_durante_extraccion"
            ),
            error_extraccion=(
                str(
                    error
                )
            ),
        )


# =============================================================================
# 10. COMPATIBILIDAD CON EL SCRAPER ACTUAL
# =============================================================================

def extraer_por_css(
    html: str,
    selector: str = "",
) -> str:
    """
    Wrapper de compatibilidad.

    El scraper antiguo espera que extraer_por_css()
    devuelva solamente un string.

    Por ahora mantenemos ese comportamiento para no romper Fase 1.

    La nueva versión de scraper.py utilizará directamente
    extraer_con_metadatos() y almacenará toda la trazabilidad.
    """

    resultado = extraer_con_metadatos(
        html,
        selector,
    )

    return resultado[
        "texto"
    ]