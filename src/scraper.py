import requests
from extractors import extraer_por_css

def scrapear_sitio(site):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
    }

    response = requests.get(site["url"], headers=headers)
    html = response.text

    perfil = extraer_por_css(html, site["selector"])

    return {
        "universidad": site["universidad"],
        "tipo_institucion": site.get("tipo_institucion", ":"),
        "carrera": site["carrera"],
        "tipo_carrera": site.get("tipo_carrera", ":"),
        "url": site["url"],
        "perfil": perfil[:500],
        "selector": site["selector"]

    }