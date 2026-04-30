from bs4 import BeautifulSoup
from utils import limpiar_texto

# El diccionario de palabras que el "sabueso" buscará en los títulos de las webs
PALABRAS_CLAVE = [
    "perfil de egreso", 
    "perfil del egresado", 
    "perfil profesional", 
    "competencias", 
    "el titulado", 
    "campo ocupacional",
    "campo laboral",
    "descripción de la carrera"
]

def buscar_por_palabras_clave(soup):
    """
    Busca automáticamente el texto rastreando títulos que contengan palabras clave
    y extrae los párrafos que le siguen.
    """
    # 1. Buscar en todas las etiquetas que suelen ser títulos o subtítulos
    etiquetas_titulos = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b'])
    
    for etiqueta in etiquetas_titulos:
        texto_etiqueta = etiqueta.get_text().lower()
        
        # Si el título tiene alguna de nuestras palabras clave...
        if any(palabra in texto_etiqueta for palabra in PALABRAS_CLAVE):
            contenido = []
            # Buscar los elementos hermanos que vienen justo debajo del título
            hermano = etiqueta.find_next_sibling()
            
            while hermano:
                # Si topamos con otro título principal, dejamos de extraer
                if hermano.name in ['h1', 'h2', 'h3']:
                    break
                
                # Si es un párrafo o una lista de competencias, lo guardamos
                if hermano.name in ['p', 'ul', 'li', 'div']:
                    texto_limpio = limpiar_texto(hermano.get_text(separator=" "))
                    if texto_limpio:
                        contenido.append(texto_limpio)
                        
                hermano = hermano.find_next_sibling()
            
            # Si logró recolectar texto debajo del título, lo une y lo devuelve
            if contenido:
                return " ".join(contenido)
                
    return ""

def extraer_por_css(html: str, selector: str) -> str:
    """
    Función híbrida: Intenta usar el buscador automático primero. 
    Si falla y hay un selector manual, usa el selector.
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "lxml")
        
        # 1. BÚSQUEDA AUTOMÁTICA INTELIGENTE
        # Se activa si dejas el selector vacío ("") o le pones un guión ("-") en config.py
        if not selector or selector.strip() in ["", "-"]:
            texto_automatico = buscar_por_palabras_clave(soup)
            if texto_automatico:
                return texto_automatico
        
        # 2. PLAN B: BÚSQUEDA MANUAL (Solo si pusiste algo en config.py y el automático falló)
        if selector and selector.strip() not in ["", "-"]:
            nodo = soup.select_one(selector)
            if nodo:
                return limpiar_texto(nodo.get_text(separator=" "))
                
    except Exception as e:
        print(f"   [Error en extracción: {e}]")
        
    return ""