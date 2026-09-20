from bs4 import BeautifulSoup
from utils import limpiar_texto

# =============================================================================
# DICCIONARIOS DE BÚSQUEDA (El "Cerebro" del Scraper)
# =============================================================================

# Estas palabras son el Santo Grial. Si el scraper ve un título con esto, 
# asume inmediatamente que el texto que sigue es el perfil real.
PALABRAS_CLAVE_PRIORITARIAS = [
    "perfil de egreso",
    "perfil del egresado",
    "perfil profesional",
    "qué aprenderás",        
    "sobre la carrera"       
]

# Si el scraper no encuentra las prioritarias, intentará raspar estas secciones 
# para no volver con las manos vacías (aportan mucho valor semántico al modelo).
PALABRAS_CLAVE_SECUNDARIAS = [
    "competencias",
    "el titulado",
    "descripción de la carrera",
    "campo ocupacional",     
    "campo laboral",
    "desempeño profesional"
]

# Dónde buscará el scraper los títulos (agregamos 'span' porque algunos 
# sitios web modernos no usan h2 o h3, solo texto con estilos).
ETIQUETAS_TITULO = ["h1", "h2", "h3", "h4", "h5", "h6", "strong", "b", "span"]

# Dónde asumirá el scraper que está el texto largo a extraer.
ETIQUETAS_CONTENIDO = ["p", "ul", "ol", "li", "div"]


# =============================================================================
# FUNCIONES DE EXTRACCIÓN Y LIMPIEZA
# =============================================================================

def extraer_texto_nodo(nodo) -> str:
    """
    Toma un bloque de HTML (nodo) y le arranca todo el código, 
    dejando solo el texto humano limpio y sin espacios extra.
    """
    if not nodo:
        return ""

    texto = nodo.get_text(" ", strip=True)
    return limpiar_texto(texto)


def buscar_contenido_cercano(titulo) -> str:
    """
    El núcleo de la inteligencia del scraper. Cuando encuentra un título 
    (ej: <h2>Perfil de egreso</h2>), esta función baja línea por línea 
    capturando todos los párrafos (<p>) o listas (<ul>) hasta que choca 
    con el siguiente título importante. Así evitamos traer basura de la web.
    """
    contenido = []

    # 1. Buscar los elementos que están justo debajo del título (hermanos)
    hermano = titulo.find_next_sibling()

    while hermano:
        # Si choca con otro título grande, se detiene para no mezclar temas
        if hermano.name in ["h1", "h2", "h3"]:
            break

        # Si es un párrafo o lista, extrae el texto y lo guarda en la bolsa
        if hermano.name in ETIQUETAS_CONTENIDO:
            texto = extraer_texto_nodo(hermano)
            if texto:
                contenido.append(texto)

        # Avanza al siguiente bloque
        hermano = hermano.find_next_sibling()

    # Si logró juntar párrafos de esta manera, los une y los entrega
    if contenido:
        return " ".join(contenido)

    # 2. Plan B: Si la página está mal programada y no hay "hermanos", 
    # busca a la fuerza el siguiente párrafo o bloque que exista en todo el documento.
    siguiente = titulo.find_next(["p", "ul", "ol", "div"])

    if siguiente:
        return extraer_texto_nodo(siguiente)

    return ""


def buscar_por_contexto(soup, palabras_clave) -> str:
    """
    Escanea toda la página buscando si algún título hace match con nuestras 
    palabras clave. Si encuentra uno, manda a llamar a 'buscar_contenido_cercano'.
    """
    # Recopila todos los títulos de la página
    titulos = soup.find_all(ETIQUETAS_TITULO)

    for titulo in titulos:
        # Pasa el título a minúsculas para compararlo sin problemas
        texto_titulo = titulo.get_text(" ", strip=True).lower()

        # Revisa si alguna de nuestras palabras clave está dentro de ese título
        if any(palabra in texto_titulo for palabra in palabras_clave):
            # ¡Bingo! Encontró el título. Ahora extrae el texto de abajo.
            texto_extraido = buscar_contenido_cercano(titulo)

            if texto_extraido:
                return texto_extraido

    return ""


def extraer_por_css(html: str, selector: str = "") -> str:
    """
    El director de orquesta. Decide qué método usar para extraer la data.
    Es una extracción "híbrida" en 3 pasos para maximizar la efectividad.
    """
    if not html:
        return ""

    try:
        # Convierte el HTML crudo en un objeto navegable
        soup = BeautifulSoup(html, "lxml")

        # PASO 1: Selector Manual (Si ustedes pusieron algo en config.py)
        # Verifica que el selector no esté vacío ni sea un guion
        if selector and str(selector).strip() not in ["", "-"]:
            nodos = soup.select(selector)

            if nodos:
                # Junta todo el texto encontrado por el selector CSS
                texto = " ".join(
                    extraer_texto_nodo(nodo)
                    for nodo in nodos
                    if extraer_texto_nodo(nodo)
                )

                if texto:
                    return limpiar_texto(texto)

        # PASO 2: Inteligencia Automática (Contexto Prioritario)
        # Si no hay selector (como en casi todo el config.py actual), busca Perfiles de Egreso.
        texto_contexto = buscar_por_contexto(soup, PALABRAS_CLAVE_PRIORITARIAS)

        if texto_contexto:
            return texto_contexto

        # PASO 3: Inteligencia Secundaria (Contexto Secundario)
        # Si la universidad no puso un "Perfil de Egreso", busca "Competencias" o "Campo Laboral".
        texto_secundario = buscar_por_contexto(soup, PALABRAS_CLAVE_SECUNDARIAS)

        if texto_secundario:
            return texto_secundario

    except Exception as e:
        # Si algo explota (la página es muy rara), imprime el error para que sepamos qué pasó
        print(f"   [Error en extracción: {e}]")

    # Si los 3 pasos fallan, devuelve vacío para que lo sepamos y lo peguemos a mano
    return ""