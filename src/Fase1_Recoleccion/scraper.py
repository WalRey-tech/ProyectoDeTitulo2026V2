import requests
import time
import random
import urllib3

# Desactiva las advertencias de seguridad en la consola. 
# Útil porque muchas páginas de universidades tienen certificados SSL vencidos.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Importamos la función inteligente que armamos en extractors.py
from extractors import extraer_por_css

# Lista de "disfraces" para que las páginas web crean que somos un usuario real 
# navegando desde distintos navegadores, y no nos bloqueen por ser un bot.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
]

def obtener_headers():
    """Elige un User-Agent al azar para cada petición."""
    return {
        "User-Agent": random.choice(USER_AGENTS)
    }

def extraer_con_requests(site):
    """
    Método 1: Requests. Es súper rápido y ligero, pero no puede leer 
    páginas que requieran cargar JavaScript (como las de React o Angular).
    """
    headers = obtener_headers()

    # Pausa aleatoria entre 2 y 5 segundos para no saturar los servidores de las Ues.
    time.sleep(random.uniform(2, 5))

    verificar_ssl = site.get("verificar_ssl", True)

    # Hace la petición a la página web
    response = requests.get(
        site["url"],
        headers=headers,
        timeout=30,  # Si la página tarda más de 30 seg, aborta para no quedarse pegado
        verify=verificar_ssl
    )

    # Si hay un error 404 o 500, lanza una alerta
    response.raise_for_status()

    # Extrae el HTML y se lo pasa a nuestro "cerebro" (extractors.py)
    html = response.text
    perfil = extraer_por_css(html, site.get("selector", ""))

    return perfil


def extraer_con_selenium(site):
    """
    Método 2: Selenium. Abre un navegador Chrome invisible. 
    Es más lento, pero es infalible para páginas modernas que usan mucho JavaScript.
    """
    options = Options()
    options.add_argument("--headless=new")  # Navegador invisible (no abre ventana)
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    # Si la universidad tiene el SSL vencido, le decimos a Selenium que lo ignore
    verificar_ssl = site.get("verificar_ssl", True)
    if not verificar_ssl:
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-insecure-localhost")

    driver = webdriver.Chrome(options=options)
    
    # IMPORTANTE: Límite de 30 segundos para que no se quede cargando infinito
    driver.set_page_load_timeout(30)

    try:
        driver.get(site["url"])

        # Espera 3 segundos para asegurar que el JavaScript de la página termine de renderizar
        time.sleep(3)

        # Saca el HTML final generado y lo manda a nuestro extractor
        html = driver.page_source
        perfil = extraer_por_css(html, site.get("selector", ""))

        return perfil

    finally:
        # Siempre cierra el navegador, incluso si hay un error, para no gastar toda la RAM
        driver.quit()


def scrapear_sitio(site):
    """
    El controlador principal. Toma la decisión de qué método usar según el config.py.
    Tiene un plan de respaldo: si un método falla, intenta con el otro.
    """
    perfil = ""
    metodo_usado = ""

    try:
        tipo_extraccion = site.get("tipo_extraccion", "css")

        # RUTA A: Intenta primero con Selenium
        if tipo_extraccion == "selenium":
            perfil = extraer_con_selenium(site)
            metodo_usado = "selenium"

            # Plan de Respaldo: Si Selenium no trajo nada, intenta con Requests
            if not perfil:
                print(
                    f"Selenium no extrajo texto para {site.get('universidad')}. "
                    f"Probando requests..."
                )
                perfil = extraer_con_requests(site)
                metodo_usado = "requests"

        # RUTA B: Intenta primero con Requests (Ruta por defecto)
        else:
            perfil = extraer_con_requests(site)
            metodo_usado = "requests"

            # Plan de Respaldo: Si Requests no trajo nada, intenta con Selenium
            if not perfil:
                print(
                    f"Requests no extrajo texto para {site.get('universidad')}. "
                    f"Probando Selenium..."
                )
                perfil = extraer_con_selenium(site)
                metodo_usado = "selenium"

        # Si agotó ambas opciones y sigue vacío, lanza una alerta visual en la consola
        if not perfil:
            print(
                f"ALERTA: No se extrajo texto. "
                f"Revisa el selector o la página para {site.get('universidad')}."
            )

        # Retorna el diccionario final que se convertirá en una fila del CSV.
        # CAMBIO CRÍTICO: Se cambió 'perfil' a 'perfil_egreso' para compatibilidad con la Fase 2.
        return {
            "universidad": site.get("universidad", ""),
            "tipo_institucion": site.get("tipo_institucion", ""),
            "carrera": site.get("carrera", ""),
            "tipo_carrera": site.get("tipo_carrera", ""),
            "url": site.get("url", ""),
            "perfil_egreso": perfil,  
            "selector": site.get("selector", ""),
            "metodo_usado": metodo_usado,
            "error": ""
        }

    except Exception as error:
        # Si algo se rompe completamente, guarda el error en el CSV en lugar de botar el programa
        return {
            "universidad": site.get("universidad", ""),
            "tipo_institucion": site.get("tipo_institucion", ""),
            "carrera": site.get("carrera", ""),
            "tipo_carrera": site.get("tipo_carrera", ""),
            "url": site.get("url", ""),
            "perfil_egreso": "",
            "selector": site.get("selector", ""),
            "metodo_usado": metodo_usado,
            "error": str(error)
        }