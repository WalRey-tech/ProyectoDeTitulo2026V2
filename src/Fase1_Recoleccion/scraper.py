import requests
import time
import random
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from extractors import extraer_por_css

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
]


def obtener_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS)
    }


def extraer_con_requests(site):
    headers = obtener_headers()

    time.sleep(random.uniform(2, 5))

    verificar_ssl = site.get("verificar_ssl", True)

    response = requests.get(
        site["url"],
        headers=headers,
        timeout=10,
        verify=verificar_ssl
    )

    response.raise_for_status()

    html = response.text
    perfil = extraer_por_css(html, site.get("selector", ""))

    return perfil


def extraer_con_selenium(site):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(site["url"])

        time.sleep(3)

        html = driver.page_source
        perfil = extraer_por_css(html, site.get("selector", ""))

        return perfil

    finally:
        driver.quit()


def scrapear_sitio(site):
    perfil = ""
    metodo_usado = ""

    try:
        tipo_extraccion = site.get("tipo_extraccion", "css")

        if tipo_extraccion == "selenium":
            perfil = extraer_con_selenium(site)
            metodo_usado = "selenium"

            if not perfil:
                print(
                    f"⚠️ Selenium no extrajo texto para {site.get('universidad')}. "
                    f"Probando requests..."
                )
                perfil = extraer_con_requests(site)
                metodo_usado = "requests"

        else:
            perfil = extraer_con_requests(site)
            metodo_usado = "requests"

            if not perfil:
                print(
                    f"⚠️ Requests no extrajo texto para {site.get('universidad')}. "
                    f"Probando Selenium..."
                )
                perfil = extraer_con_selenium(site)
                metodo_usado = "selenium"

        if not perfil:
            print(
                f"⚠️ ALERTA: No se extrajo texto. "
                f"Revisa el selector '{site.get('selector')}' para {site.get('universidad')}."
            )

        return {
            "universidad": site.get("universidad", ""),
            "tipo_institucion": site.get("tipo_institucion", ""),
            "carrera": site.get("carrera", ""),
            "tipo_carrera": site.get("tipo_carrera", ""),
            "url": site.get("url", ""),
            "perfil": perfil,
            "selector": site.get("selector", ""),
            "metodo_usado": metodo_usado,
            "error": ""
        }

    except Exception as error:
        return {
            "universidad": site.get("universidad", ""),
            "tipo_institucion": site.get("tipo_institucion", ""),
            "carrera": site.get("carrera", ""),
            "tipo_carrera": site.get("tipo_carrera", ""),
            "url": site.get("url", ""),
            "perfil": "",
            "selector": site.get("selector", ""),
            "metodo_usado": metodo_usado,
            "error": str(error)
        }