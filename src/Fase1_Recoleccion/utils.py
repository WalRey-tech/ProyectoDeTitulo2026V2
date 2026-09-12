import re

def limpiar_texto(texto: str) -> str:
    """
    Limpia el texto extraído del HTML eliminando espacios redundantes,
    saltos de línea y caracteres invisibles que ensucian el análisis NLP.
    """
    # Prevención de errores si el texto llega vacío o como un dato nulo
    if not texto:
        return ""
        
    # 1. Eliminar caracteres invisibles clásicos del web scraping
    # \u200b es un "Zero-width space" (espacio de ancho cero) muy común en webs
    # \xa0 es un "Non-breaking space" (espacio duro de HTML: &nbsp;)
    texto = texto.replace('\u200b', '').replace('\xa0', ' ')
    
    # 2. Aplastar múltiples espacios, saltos de línea y tabs en un solo espacio
    texto = re.sub(r"\s+", " ", texto)
    
    # 3. Quitar espacios accidentales al inicio y al final
    return texto.strip()