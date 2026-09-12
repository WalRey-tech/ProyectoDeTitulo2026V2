import random
import time

import requests
import urllib3

<<<<<<< HEAD
# Desactiva las advertencias de seguridad en la consola. 
# Útil porque muchas páginas de universidades tienen certificados SSL vencidos.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Importamos la función inteligente que armamos en extractors.py
=======
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

>>>>>>> 88833ed (Actualiza scraper y configuración de sitios)
from extractors import extraer_por_css
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# Lista de User-Agent para reducir bloqueos simples por parte de los sitios.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
]


def obtener_headers():
    """Elige un User-Agent al azar para cada petición."""
    return {"User-Agent": random.choice(USER_AGENTS)}


def extraer_con_requests(site):
    """
    Método 1: Requests.
    Solo se usa con selectores CSS, porque extraer_por_css() trabaja con CSS.
    """
    tipo_selector = site.get("tipo_selector", "css").lower()

    if tipo_selector != "css":
        raise ValueError(
            f"Requests solo admite selectores CSS en este scraper. "
            f"Se recibió tipo_selector='{tipo_selector}'."
        )

    headers = obtener_headers()
    time.sleep(random.uniform(2, 5))

    verificar_ssl = site.get("verificar_ssl", True)

    response = requests.get(
        site["url"],
        headers=headers,
        timeout=30,
        verify=verificar_ssl,
    )
    response.raise_for_status()

    html = response.text
    return extraer_por_css(html, site.get("selector", ""))


def extraer_con_selenium(site):
    """
    Método 2: Selenium.

    Admite:
    - tipo_selector='css'            -> un elemento mediante CSS
    - tipo_selector='xpath'          -> un elemento mediante XPath
    - tipo_selector='xpath_multiple' -> varios elementos mediante XPath
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    verificar_ssl = site.get("verificar_ssl", True)
    if not verificar_ssl:
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-insecure-localhost")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)

    try:
        driver.get(site["url"])

        # Pequeña espera inicial para páginas que terminan de renderizar con JavaScript.
        time.sleep(3)

        selector = site.get("selector", "").strip()
        tipo_selector = site.get("tipo_selector", "css").lower().strip()

        if not selector:
            raise ValueError("El sitio no tiene definido un selector.")

        print(f"   Tipo selector: {tipo_selector}")
        print(f"   Selector: {selector}")

        try:
            if tipo_selector == "xpath":
                elemento = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                return elemento.text.strip()

            if tipo_selector == "xpath_multiple":
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )

                elementos = driver.find_elements(By.XPATH, selector)
                textos = [
                    elemento.text.strip()
                    for elemento in elementos
                    if elemento.text.strip()
                ]
                return "\n\n".join(textos)

            if tipo_selector == "css":
                elemento = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                return elemento.text.strip()

            raise ValueError(
                f"tipo_selector no válido: '{tipo_selector}'. "
                "Usa 'css', 'xpath' o 'xpath_multiple'."
            )

        except TimeoutException as error:
            raise TimeoutException(
                f"No se encontró el elemento en 15 segundos. "
                f"tipo_selector='{tipo_selector}', selector='{selector}', "
                f"url_final='{driver.current_url}'."
            ) from error

    finally:
        driver.quit()


def scrapear_sitio(site):
    """
    Extrae un perfil de egreso usando Requests y/o Selenium.

    Reglas:
    - XPath y XPath múltiple se procesan únicamente con Selenium.
    - CSS puede usar Requests y Selenium como respaldo.
    """
    perfil = ""
    metodo_usado = ""
    errores = []

    tipo_extraccion = site.get("tipo_extraccion", "requests").lower().strip()
    tipo_selector = site.get("tipo_selector", "css").lower().strip()

    if tipo_selector in {"xpath", "xpath_multiple"}:
        orden_metodos = ["selenium"]
    elif tipo_extraccion == "selenium":
        orden_metodos = ["selenium", "requests"]
    else:
        orden_metodos = ["requests", "selenium"]

    for metodo in orden_metodos:
        try:
            print(f"   Intentando extracción mediante {metodo}...")

            if metodo == "requests":
                perfil = extraer_con_requests(site)
            else:
                perfil = extraer_con_selenium(site)

            metodo_usado = metodo

            if perfil and perfil.strip():
                print(f"   Extracción exitosa mediante {metodo}.")
                break

            errores.append(f"{metodo}: no se encontró texto")
            print(f"   {metodo} no encontró contenido. Probando respaldo...")

        except Exception as error:
            errores.append(f"{metodo}: {error}")
            print(f"   Falló {metodo}: {error}")

            if metodo != orden_metodos[-1]:
                print("   Probando método de respaldo...")

    if not perfil:
        print(
            f"ALERTA: No se extrajo texto para "
            f"{site.get('universidad')} - {site.get('carrera')}"
        )

    return {
        "universidad": site.get("universidad", ""),
        "tipo_institucion": site.get("tipo_institucion", ""),
        "carrera": site.get("carrera", ""),
        "tipo_carrera": site.get("tipo_carrera", ""),
        "url": site.get("url", ""),
        "perfil_egreso": perfil,
        "selector": site.get("selector", ""),
        "tipo_selector": tipo_selector,
        "metodo_usado": metodo_usado,
        "error": "" if perfil else " | ".join(errores),
    }
