from bs4 import BeautifulSoup
from utils import limpiar_texto


PALABRAS_CLAVE_PRIORITARIAS = [
    "perfil de egreso",
    "perfil del egresado",
    "perfil profesional",
]

PALABRAS_CLAVE_SECUNDARIAS = [
    "competencias",
    "el titulado",
    "descripción de la carrera",
]

ETIQUETAS_TITULO = ["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b"]
ETIQUETAS_CONTENIDO = ["p", "ul", "ol", "li", "div"]


def extraer_texto_nodo(nodo) -> str:
    if not nodo:
        return ""

    texto = nodo.get_text(" ", strip=True)
    return limpiar_texto(texto)


def buscar_contenido_cercano(titulo) -> str:
    """
    Busca contenido cercano después de un título.
    Sirve para estructuras tipo:
    <h2>Perfil de egreso</h2>
    <div class="list-content"><p>Texto...</p></div>
    """

    contenido = []

    # 1. Buscar hermanos posteriores
    hermano = titulo.find_next_sibling()

    while hermano:
        if hermano.name in ["h1", "h2", "h3"]:
            break

        if hermano.name in ETIQUETAS_CONTENIDO:
            texto = extraer_texto_nodo(hermano)
            if texto:
                contenido.append(texto)

        hermano = hermano.find_next_sibling()

    if contenido:
        return " ".join(contenido)

    # 2. Si no hay hermanos útiles, buscar el siguiente bloque de contenido
    siguiente = titulo.find_next(["p", "ul", "ol", "div"])

    if siguiente:
        return extraer_texto_nodo(siguiente)

    return ""


def buscar_por_contexto(soup, palabras_clave) -> str:
    """
    Busca títulos que contengan palabras clave y extrae el contenido asociado.
    """

    titulos = soup.find_all(ETIQUETAS_TITULO)

    for titulo in titulos:
        texto_titulo = titulo.get_text(" ", strip=True).lower()

        if any(palabra in texto_titulo for palabra in palabras_clave):
            texto_extraido = buscar_contenido_cercano(titulo)

            if texto_extraido:
                return texto_extraido

    return ""


def extraer_por_css(html: str, selector: str = "") -> str:
    """
    Extracción híbrida:
    1. Si hay selector manual, intenta extraer con CSS.
    2. Si falla, busca por contexto usando palabras clave.
    3. Prioriza Perfil de egreso antes que competencias/campo laboral.
    """

    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")

        # 1. Selector manual
        if selector and str(selector).strip() not in ["", "-"]:
            nodos = soup.select(selector)

            if nodos:
                texto = " ".join(
                    extraer_texto_nodo(nodo)
                    for nodo in nodos
                    if extraer_texto_nodo(nodo)
                )

                if texto:
                    return limpiar_texto(texto)

        # 2. Contexto prioritario
        texto_contexto = buscar_por_contexto(soup, PALABRAS_CLAVE_PRIORITARIAS)

        if texto_contexto:
            return texto_contexto

        # 3. Contexto secundario
        texto_secundario = buscar_por_contexto(soup, PALABRAS_CLAVE_SECUNDARIAS)

        if texto_secundario:
            return texto_secundario

    except Exception as e:
        print(f"   [Error en extracción: {e}]")

    return ""