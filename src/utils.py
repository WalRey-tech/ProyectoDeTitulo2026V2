import re

def limpiar_texto(texto: str) -> str:
    return re.sub(r"\s+", " ", texto).strip()