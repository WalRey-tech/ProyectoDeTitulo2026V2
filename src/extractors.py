from bs4 import BeautifulSoup
from utils import limpiar_texto

def extraer_por_css(html, selector):
    soup = BeautifulSoup(html, "lxml")
    nodo = soup.select_one(selector)
    if nodo:
        return limpiar_texto(nodo.get_text())
    return ""