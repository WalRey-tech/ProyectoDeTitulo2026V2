import requests
import time
import random
# Silencia las advertencias de seguridad al entrar a Ues con certificados vencidos
import urllib3 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from extractors import extraer_por_css

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0" # Agregué Firefox para mayor camuflaje
]

def scrapear_sitio(site):
    headers = {
        "User-Agent": random.choice(USER_AGENTS)
    }

    # Pausa para no colapsar los servidores
    time.sleep(random.uniform(2, 5)) 

    # Se agregó timeout=15 por si el servidor de un CFT está muy lento
    response = requests.get(site["url"], headers=headers, timeout=15, verify=False)
    
    # Esto obliga al bot a lanzar un error si la U borró la página (Error 404) 
    # o si nos bloquearon (Error 403), en lugar de tragar en silencio.
    response.raise_for_status()

    html = response.text

    # Extraer la información
    perfil = extraer_por_css(html, site.get("selector", ""))

    if not perfil:
        print(f"   ⚠️ ALERTA: La página cargó, pero no se extrajo texto. Revisa el selector '{site.get('selector')}' para {site.get('universidad')}.")

    return {
        "universidad": site.get("universidad", ""),
        "tipo_institucion": site.get("tipo_institucion", ""),
        "carrera": site.get("carrera", ""),
        "tipo_carrera": site.get("tipo_carrera", ""),
        "url": site.get("url", ""),
        "perfil": perfil,
        "selector": site.get("selector", "")
    }