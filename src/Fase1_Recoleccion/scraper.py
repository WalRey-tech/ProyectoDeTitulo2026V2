# -*- coding: utf-8 -*-

"""
SCRAPER CONTROLADO Y AUDITABLE — FASE 1
=======================================

Principios:

- Solo se permiten URLs HTTPS públicas.
- La validación TLS/SSL permanece siempre activa.
- No se intenta evadir autenticación, CAPTCHA ni bloqueos HTTP.
- Se consulta robots.txt y se respetan prohibiciones explícitas.
- Se utilizan pausas y timeouts.
- Se limita el tamaño máximo de las respuestas.
- Se controla el tipo de contenido recibido.
- Requests se utiliza como método liviano.
- Selenium se utiliza para páginas que necesitan JavaScript.
- Se registra trazabilidad de adquisición y extracción.

El scraping en vivo NO modifica el corpus científico V2.
"""

from __future__ import annotations

import ipaddress
import socket
import time
import urllib.robotparser

from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)

import requests

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# Permite ejecutar:
#
#   python main.py
#
# desde Fase1_Recoleccion,
# y también importar el módulo desde la raíz del proyecto.
try:
    from .extractors import extraer_con_metadatos
except ImportError:
    from extractors import extraer_con_metadatos


# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================

BOT_NAME = "ProyectoTituloPerfilesEgreso"

USER_AGENT = (
    "ProyectoTituloPerfilesEgreso/1.0 "
    "(investigacion-academica; acceso-a-paginas-publicas)"
)

TIMEOUT_CONEXION = 10
TIMEOUT_LECTURA = 30
TIMEOUT_SELENIUM = 30

PAUSA_ENTRE_SOLICITUDES = 2.0
ESPERA_RENDER_SELENIUM = 3.0

MAX_REDIRECCIONES = 5

# Evita descargar accidentalmente respuestas excesivamente grandes.
MAX_HTML_BYTES = 10 * 1024 * 1024       # 10 MB
MAX_ROBOTS_BYTES = 512 * 1024           # 512 KB

CONTENT_TYPES_HTML = {
    "text/html",
    "application/xhtml+xml",
}


# Parámetros de publicidad/seguimiento que pueden eliminarse
# sin cambiar la página académica solicitada.
PARAMETROS_TRACKING_EXACTOS = {
    "gclid",
    "dclid",
    "fbclid",
    "gbraid",
    "wbraid",
    "gad_source",
    "gad_campaignid",
    "mc_cid",
    "mc_eid",
}

PREFIJOS_TRACKING = (
    "utm_",
    "hsa_",
)


# Evita descargar robots.txt varias veces para el mismo dominio
# durante una misma ejecución.
ROBOTS_CACHE = {}


# =============================================================================
# 2. ERRORES CONTROLADOS
# =============================================================================

class ErrorScraping(Exception):
    """
    Error general controlado del scraper.
    """


class ErrorSeguridadScraping(ErrorScraping):
    """
    La solicitud fue rechazada por una regla de seguridad.
    """


class ErrorNoRecuperable(ErrorScraping):
    """
    Error frente al cual no corresponde intentar otro método
    como una forma de evasión.
    """


# =============================================================================
# 3. LIMPIEZA DE URL
# =============================================================================

def limpiar_parametros_tracking(url: str) -> str:
    """
    Elimina únicamente parámetros conocidos de seguimiento.

    Ejemplo:

        ?utm_source=google&gclid=...

    se elimina.

    Pero parámetros funcionales como:

        ?page_id=35

    se conservan.
    """

    partes = urlparse(
        str(url or "").strip()
    )

    parametros_limpios = []

    for clave, valor in parse_qsl(
        partes.query,
        keep_blank_values=True,
    ):

        clave_lower = clave.lower()

        if clave_lower in PARAMETROS_TRACKING_EXACTOS:
            continue

        if any(
            clave_lower.startswith(prefijo)
            for prefijo in PREFIJOS_TRACKING
        ):
            continue

        parametros_limpios.append(
            (clave, valor)
        )

    nueva_query = urlencode(
        parametros_limpios,
        doseq=True,
    )

    return urlunparse(
        (
            partes.scheme,
            partes.netloc,
            partes.path,
            partes.params,
            nueva_query,
            partes.fragment,
        )
    )


# =============================================================================
# 4. VALIDACIÓN DE DESTINO
# =============================================================================

def _ip_es_publica(ip_texto: str) -> bool:
    """
    Determina si una dirección corresponde a una IP pública/global.
    """

    try:
        ip = ipaddress.ip_address(
            ip_texto
        )

        return ip.is_global

    except ValueError:
        return False


def validar_url_publica(url: str) -> str:
    """
    Control preventivo del destino.

    Requisitos:

    - HTTPS obligatorio.
    - Sin usuario/contraseña incrustados.
    - Hostname válido.
    - Sin localhost.
    - Sin redes privadas/locales/reservadas.
    - Puerto HTTPS estándar.

    Esto reduce el riesgo de que el scraper termine visitando
    recursos internos o locales.
    """

    url = str(
        url or ""
    ).strip()

    if not url:
        raise ErrorSeguridadScraping(
            "URL vacía"
        )

    partes = urlparse(
        url
    )

    # -------------------------------------------------------------------------
    # HTTPS obligatorio
    # -------------------------------------------------------------------------

    if partes.scheme.lower() != "https":
        raise ErrorSeguridadScraping(
            "Solo se permiten URLs HTTPS"
        )

    # -------------------------------------------------------------------------
    # No admitimos credenciales dentro de una URL
    # -------------------------------------------------------------------------

    if partes.username or partes.password:
        raise ErrorSeguridadScraping(
            "No se permiten credenciales dentro de la URL"
        )

    hostname = partes.hostname

    if not hostname:
        raise ErrorSeguridadScraping(
            "La URL no contiene un hostname válido"
        )

    hostname = (
        hostname
        .rstrip(".")
        .lower()
    )

    # -------------------------------------------------------------------------
    # Bloqueo de hosts locales
    # -------------------------------------------------------------------------

    if (
        hostname == "localhost"
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
    ):
        raise ErrorSeguridadScraping(
            "No se permiten hosts locales"
        )

    # -------------------------------------------------------------------------
    # Puerto
    # -------------------------------------------------------------------------

    try:
        puerto = partes.port

    except ValueError as error:

        raise ErrorSeguridadScraping(
            f"Puerto inválido: {error}"
        ) from error


    if puerto not in (
        None,
        443,
    ):
        raise ErrorSeguridadScraping(
            f"Puerto HTTPS no permitido: {puerto}"
        )

    # -------------------------------------------------------------------------
    # Si el hostname es directamente una dirección IP
    # -------------------------------------------------------------------------

    try:

        ip_literal = ipaddress.ip_address(
            hostname
        )

        if not ip_literal.is_global:
            raise ErrorSeguridadScraping(
                "No se permiten direcciones IP "
                "privadas, locales o reservadas"
            )

        return url

    except ValueError:
        pass

    # -------------------------------------------------------------------------
    # Comprobación DNS preventiva
    # -------------------------------------------------------------------------

    try:

        resultados = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM,
        )

    except socket.gaierror as error:

        raise ErrorSeguridadScraping(
            "No fue posible resolver el dominio: "
            f"{error}"
        ) from error


    ips = {
        resultado[4][0]
        for resultado in resultados
        if resultado[4]
    }


    if not ips:

        raise ErrorSeguridadScraping(
            "El dominio no resolvió a ninguna IP"
        )


    ips_no_publicas = [
        ip
        for ip in ips
        if not _ip_es_publica(ip)
    ]


    if ips_no_publicas:

        raise ErrorSeguridadScraping(
            "El dominio resolvió a una IP no pública: "
            + ", ".join(
                sorted(
                    ips_no_publicas
                )
            )
        )


    return url


# =============================================================================
# 5. CABECERAS Y PAUSA
# =============================================================================

def obtener_headers() -> dict:
    """
    Identificación estable y transparente.

    No se rotan navegadores ficticios para aparentar usuarios diferentes.
    """

    return {

        "User-Agent":
            USER_AGENT,

        "Accept":
            "text/html,"
            "application/xhtml+xml;q=0.9,"
            "*/*;q=0.1",

        "Accept-Language":
            "es-CL,es;q=0.9",

        "Connection":
            "close",
    }


def pausa_respetuosa():
    """
    Reduce la frecuencia de solicitudes al servidor.
    """

    time.sleep(
        PAUSA_ENTRE_SOLICITUDES
    )


# =============================================================================
# 6. REQUESTS CON REDIRECCIONES CONTROLADAS
# =============================================================================

def _obtener_respuesta_con_redirecciones(
    session: requests.Session,
    url: str,
):
    """
    Las redirecciones no se siguen automáticamente.

    Cada nuevo destino se valida ANTES de visitarlo.
    """

    url_actual = validar_url_publica(
        url
    )

    for _ in range(
        MAX_REDIRECCIONES + 1
    ):

        response = session.get(
            url_actual,

            headers=obtener_headers(),

            timeout=(
                TIMEOUT_CONEXION,
                TIMEOUT_LECTURA,
            ),

            # SIEMPRE activo.
            verify=True,

            # Las procesamos nosotros.
            allow_redirects=False,

            stream=True,
        )


        if 300 <= response.status_code < 400:

            location = response.headers.get(
                "Location"
            )

            response.close()


            if not location:

                raise ErrorNoRecuperable(
                    "Redirección HTTP sin cabecera Location"
                )


            siguiente_url = urljoin(
                url_actual,
                location,
            )

            url_actual = validar_url_publica(
                siguiente_url
            )

            continue


        return (
            response,
            url_actual,
        )


    raise ErrorNoRecuperable(
        "Se superó el máximo de "
        f"{MAX_REDIRECCIONES} redirecciones"
    )


# =============================================================================
# 7. LÍMITE DE TAMAÑO
# =============================================================================

def _leer_respuesta_limitada(
    response: requests.Response,
    max_bytes: int,
) -> bytes:
    """
    Descarga la respuesta por bloques.

    Si supera el límite permitido, se interrumpe.
    """

    content_length = response.headers.get(
        "Content-Length"
    )


    if content_length:

        try:

            declarado = int(
                content_length
            )

            if declarado > max_bytes:

                raise ErrorNoRecuperable(
                    "Respuesta demasiado grande: "
                    f"{declarado} bytes"
                )

        except ValueError:
            # Si la cabecera es incorrecta,
            # igualmente controlamos el tamaño real.
            pass


    bloques = []

    total = 0


    for bloque in response.iter_content(
        chunk_size=64 * 1024,
    ):

        if not bloque:
            continue


        total += len(
            bloque
        )


        if total > max_bytes:

            raise ErrorNoRecuperable(
                "Respuesta supera el límite de "
                f"{max_bytes} bytes"
            )


        bloques.append(
            bloque
        )


    return b"".join(
        bloques
    )


def _decodificar_respuesta(
    response: requests.Response,
    contenido: bytes,
) -> str:
    """
    Convierte la respuesta HTTP en texto.
    """

    encoding = (
        response.encoding
        or "utf-8"
    )


    try:

        return contenido.decode(
            encoding,
            errors="replace",
        )

    except LookupError:

        return contenido.decode(
            "utf-8",
            errors="replace",
        )


# =============================================================================
# 8. ROBOTS.TXT
# =============================================================================

def verificar_robots(
    url: str,
    session: requests.Session,
):
    """
    Consulta robots.txt.

    Si existe una prohibición explícita, no se realiza scraping.

    Si robots.txt no existe o no puede comprobarse, la situación
    queda registrada para auditoría, pero no se interpreta
    automáticamente como una prohibición.
    """

    partes = urlparse(
        url
    )

    origen = (
        f"{partes.scheme}://"
        f"{partes.netloc}"
    )


    cache = ROBOTS_CACHE.get(
        origen
    )


    if cache is None:

        robots_url = urljoin(
            origen,
            "/robots.txt",
        )


        try:

            response, robots_final = (
                _obtener_respuesta_con_redirecciones(
                    session,
                    robots_url,
                )
            )


            with response:

                status = response.status_code


                if status in {
                    404,
                    410,
                }:

                    cache = {

                        "estado":
                            "no_publicado",

                        "parser":
                            None,

                        "detalle":
                            "robots.txt no publicado "
                            f"(HTTP {status})",
                    }


                elif status == 200:

                    contenido = (
                        _leer_respuesta_limitada(
                            response,
                            MAX_ROBOTS_BYTES,
                        )
                    )

                    texto = (
                        _decodificar_respuesta(
                            response,
                            contenido,
                        )
                    )


                    parser = (
                        urllib.robotparser
                        .RobotFileParser()
                    )

                    parser.set_url(
                        robots_final
                    )

                    parser.parse(
                        texto.splitlines()
                    )


                    cache = {

                        "estado":
                            "publicado",

                        "parser":
                            parser,

                        "detalle":
                            "robots.txt consultado: "
                            f"{robots_final}",
                    }


                else:

                    cache = {

                        "estado":
                            "no_verificable",

                        "parser":
                            None,

                        "detalle":
                            "robots.txt respondió "
                            f"HTTP {status}",
                    }


        except (
            requests.RequestException,
            ErrorScraping,
        ) as error:

            cache = {

                "estado":
                    "no_verificable",

                "parser":
                    None,

                "detalle":
                    "No fue posible verificar "
                    f"robots.txt: {error}",
            }


        ROBOTS_CACHE[
            origen
        ] = cache


    parser = cache[
        "parser"
    ]


    if parser is not None:

        permitido = parser.can_fetch(
            BOT_NAME,
            url,
        )


        if permitido:

            return (
                True,
                "permitido",
                cache["detalle"],
            )


        return (
            False,
            "denegado",
            "robots.txt no permite acceder a "
            f"{url} con {BOT_NAME}",
        )


    return (
        True,
        cache["estado"],
        cache["detalle"],
    )


# =============================================================================
# 9. EXTRACCIÓN CON REQUESTS
# =============================================================================

def extraer_con_requests(
    site: dict,
    url: str,
    session: requests.Session,
):
    """
    Descarga HTML mediante Requests.

    TLS permanece obligatorio.
    """

    pausa_respetuosa()


    response, url_final = (
        _obtener_respuesta_con_redirecciones(
            session,
            url,
        )
    )


    with response:

        status = response.status_code


        # Estos estados no deben intentar "evadirse"
        # usando otro cliente.
        if status in {
            401,
            403,
            404,
            410,
            429,
        }:

            raise ErrorNoRecuperable(
                f"HTTP {status}: "
                "acceso no disponible"
            )


        if status >= 400:

            raise ErrorNoRecuperable(
                f"HTTP {status}"
            )


        content_type = (
            response
            .headers
            .get(
                "Content-Type",
                "",
            )
            .split(";")[0]
            .strip()
            .lower()
        )


        # Si el servidor declara explícitamente un tipo no HTML,
        # no lo procesamos como página web.
        if (
            content_type
            and content_type
            not in CONTENT_TYPES_HTML
        ):

            raise ErrorNoRecuperable(
                "Content-Type no HTML: "
                f"{content_type}"
            )


        contenido = (
            _leer_respuesta_limitada(
                response,
                MAX_HTML_BYTES,
            )
        )


        html = (
            _decodificar_respuesta(
                response,
                contenido,
            )
        )


    resultado = extraer_con_metadatos(
        html,
        site.get(
            "selector",
            "",
        ),
    )


    return (
        resultado,
        url_final,
        content_type,
    )


# =============================================================================
# 10. SELENIUM
# =============================================================================

def _crear_driver():
    """
    Crea Chrome headless manteniendo las protecciones normales.

    Intencionalmente NO se utiliza:

        --ignore-certificate-errors
        --no-sandbox
    """

    options = Options()


    options.add_argument(
        "--headless=new"
    )

    options.add_argument(
        "--disable-gpu"
    )

    options.add_argument(
        "--window-size=1920,1080"
    )

    options.add_argument(
        "--disable-extensions"
    )

    options.add_argument(
        "--disable-notifications"
    )

    options.add_argument(
        "--disable-sync"
    )

    options.add_argument(
        "--no-first-run"
    )

    options.add_argument(
        f"--user-agent={USER_AGENT}"
    )


    # No necesitamos descargar archivos ni mostrar notificaciones.
    options.add_experimental_option(

        "prefs",

        {
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_setting_values.popups": 2,
            "profile.default_content_setting_values.automatic_downloads": 2,
            "safebrowsing.enabled": True,
        },
    )


    driver = webdriver.Chrome(
        options=options
    )


    driver.set_page_load_timeout(
        TIMEOUT_SELENIUM
    )


    return driver


def _detectar_pagina_error_certificado(
    html: str,
) -> bool:
    """
    Detecta indicadores habituales de la página de error TLS de Chrome.
    """

    texto = str(
        html or ""
    ).lower()


    indicadores = [

        "net::err_cert_",

        "your connection is not private",

        "privacy error",

        "tu conexión no es privada",

        "error de privacidad",
    ]


    return any(
        indicador in texto
        for indicador in indicadores
    )


def extraer_con_selenium(
    site: dict,
    url: str,
):
    """
    Renderiza páginas que requieren JavaScript.

    La URL se valida antes de abrirla y la URL final
    vuelve a comprobarse después de cualquier redirección.
    """

    validar_url_publica(
        url
    )


    pausa_respetuosa()


    driver = _crear_driver()


    try:

        driver.get(
            url
        )


        time.sleep(
            ESPERA_RENDER_SELENIUM
        )


        url_final = (
            driver.current_url
        )


        validar_url_publica(
            url_final
        )


        html = (
            driver.page_source
            or ""
        )


        if _detectar_pagina_error_certificado(
            html
        ):

            raise ErrorSeguridadScraping(
                "Chrome detectó un error "
                "de certificado TLS"
            )


        tamano_html = len(
            html.encode(
                "utf-8",
                errors="ignore",
            )
        )


        if tamano_html > MAX_HTML_BYTES:

            raise ErrorNoRecuperable(
                "HTML renderizado supera "
                f"{MAX_HTML_BYTES} bytes"
            )


        resultado = (
            extraer_con_metadatos(
                html,
                site.get(
                    "selector",
                    "",
                ),
            )
        )


        return (
            resultado,
            url_final,
            "text/html",
        )


    finally:

        # Siempre cerramos Chrome,
        # incluso si ocurre una excepción.
        driver.quit()


# =============================================================================
# 11. ESTRUCTURA DE RESULTADO
# =============================================================================

def _resultado_vacio(
    site: dict,
    url_original: str,
    url_solicitada: str,
) -> dict:
    """
    Estructura uniforme para todos los sitios.
    """

    return {

        "universidad":
            site.get(
                "universidad",
                "",
            ),

        "tipo_institucion":
            site.get(
                "tipo_institucion",
                "",
            ),

        "carrera":
            site.get(
                "carrera",
                "",
            ),

        "tipo_carrera":
            site.get(
                "tipo_carrera",
                "",
            ),

        # Compatibilidad con el main.py actual.
        "url":
            url_solicitada,

        "url_original":
            url_original,

        "url_final":
            "",

        "selector":
            site.get(
                "selector",
                "",
            ),

        "metodo_usado":
            "",

        "estrategia_extraccion":
            "",

        "termino_detectado":
            "",

        "etiqueta_titulo":
            "",

        "perfil_egreso":
            "",

        "largo_perfil":
            0,

        "requiere_revision":
            True,

        "motivo_revision":
            "",

        "robots_estado":
            "",

        "content_type":
            "",

        "advertencias":
            "",

        "error":
            "",
    }


# =============================================================================
# 12. ORQUESTACIÓN DE UN SITIO
# =============================================================================

def scrapear_sitio(
    site: dict,
) -> dict:
    """
    Ejecuta la adquisición completa de un sitio.

    Requests y Selenium son métodos alternativos de adquisición.

    Selenium puede actuar como respaldo cuando una página necesita
    JavaScript, pero NO se utiliza para evadir:

    - robots.txt;
    - certificados inválidos;
    - HTTP 401;
    - HTTP 403;
    - HTTP 429;
    - controles de URL.
    """

    url_original = str(
        site.get(
            "url",
            "",
        )
    ).strip()


    url_solicitada = (
        limpiar_parametros_tracking(
            url_original
        )
    )


    salida = _resultado_vacio(
        site,
        url_original,
        url_solicitada,
    )


    advertencias = []

    errores_intentos = []


    # -------------------------------------------------------------------------
    # Compatibilidad con config.py antiguo
    # -------------------------------------------------------------------------
    #
    # Algunas fuentes antiguas contienen verificar_ssl=False.
    #
    # La opción se registra para auditoría, pero ya NO desactiva TLS.
    # -------------------------------------------------------------------------

    if site.get(
        "verificar_ssl"
    ) is False:

        advertencias.append(
            "config_verificar_ssl_false_"
            "ignorado_tls_obligatorio"
        )


    # -------------------------------------------------------------------------
    # Validar URL
    # -------------------------------------------------------------------------

    try:

        validar_url_publica(
            url_solicitada
        )

    except ErrorSeguridadScraping as error:

        salida[
            "motivo_revision"
        ] = (
            "url_bloqueada_por_seguridad"
        )

        salida[
            "advertencias"
        ] = " | ".join(
            advertencias
        )

        salida[
            "error"
        ] = (
            f"seguridad_url: {error}"
        )

        return salida


    # -------------------------------------------------------------------------
    # Sesión Requests
    # -------------------------------------------------------------------------

    with requests.Session() as session:

        # ---------------------------------------------------------------------
        # robots.txt
        # ---------------------------------------------------------------------

        (
            permitido,
            robots_estado,
            robots_detalle,
        ) = verificar_robots(
            url_solicitada,
            session,
        )


        salida[
            "robots_estado"
        ] = robots_estado


        if robots_estado in {
            "no_publicado",
            "no_verificable",
        }:

            advertencias.append(
                robots_detalle
            )


        if not permitido:

            salida[
                "motivo_revision"
            ] = (
                "robots_txt_no_permite_acceso"
            )

            salida[
                "advertencias"
            ] = " | ".join(
                advertencias
            )

            salida[
                "error"
            ] = (
                robots_detalle
            )

            return salida


        # ---------------------------------------------------------------------
        # Orden de adquisición
        # ---------------------------------------------------------------------

        tipo_extraccion = str(
            site.get(
                "tipo_extraccion",
                "css",
            )
        ).strip().lower()


        if tipo_extraccion == "selenium":

            orden_metodos = [
                "selenium",
                "requests",
            ]

        else:

            orden_metodos = [
                "requests",
                "selenium",
            ]


        # ---------------------------------------------------------------------
        # Intentos
        # ---------------------------------------------------------------------

        for metodo in orden_metodos:

            try:

                print(
                    "   Intentando extracción "
                    f"mediante {metodo}..."
                )


                if metodo == "requests":

                    (
                        resultado,
                        url_final,
                        content_type,
                    ) = extraer_con_requests(
                        site,
                        url_solicitada,
                        session,
                    )

                else:

                    (
                        resultado,
                        url_final,
                        content_type,
                    ) = extraer_con_selenium(
                        site,
                        url_solicitada,
                    )


                texto = (
                    resultado
                    .get(
                        "texto",
                        "",
                    )
                    .strip()
                )


                # -----------------------------------------------------------------
                # ÉXITO
                # -----------------------------------------------------------------

                if texto:

                    salida[
                        "url_final"
                    ] = url_final

                    salida[
                        "metodo_usado"
                    ] = metodo

                    salida[
                        "estrategia_extraccion"
                    ] = resultado.get(
                        "estrategia",
                        "",
                    )

                    salida[
                        "termino_detectado"
                    ] = resultado.get(
                        "termino_detectado",
                        "",
                    )

                    salida[
                        "etiqueta_titulo"
                    ] = resultado.get(
                        "etiqueta_titulo",
                        "",
                    )

                    salida[
                        "perfil_egreso"
                    ] = texto

                    salida[
                        "largo_perfil"
                    ] = resultado.get(
                        "longitud_texto",
                        len(
                            texto
                        ),
                    )

                    salida[
                        "requiere_revision"
                    ] = bool(
                        resultado.get(
                            "requiere_revision",
                            False,
                        )
                    )

                    salida[
                        "motivo_revision"
                    ] = resultado.get(
                        "motivo_revision",
                        "",
                    )

                    salida[
                        "content_type"
                    ] = content_type


                    error_extraccion = (
                        resultado.get(
                            "error_extraccion",
                            "",
                        )
                    )


                    if error_extraccion:

                        advertencias.append(
                            error_extraccion
                        )


                    # Si Requests falló y Selenium funcionó,
                    # conservamos esa información como auditoría.
                    advertencias.extend(
                        errores_intentos
                    )


                    salida[
                        "advertencias"
                    ] = " | ".join(
                        advertencias
                    )

                    salida[
                        "error"
                    ] = ""


                    estado = (
                        "REVISAR"
                        if salida[
                            "requiere_revision"
                        ]
                        else "OK"
                    )


                    print(
                        "   Extracción exitosa "
                        f"mediante {metodo} "
                        f"[{estado}] "
                        f"("
                        f"{salida['estrategia_extraccion']}"
                        f")."
                    )


                    return salida


                # -----------------------------------------------------------------
                # NO ENCONTRÓ TEXTO
                # -----------------------------------------------------------------

                detalle = resultado.get(
                    "error_extraccion",
                    "",
                )


                mensaje = (
                    f"{metodo}: "
                    "no se encontró contenido utilizable"
                )


                if detalle:

                    mensaje += (
                        f" ({detalle})"
                    )


                errores_intentos.append(
                    mensaje
                )


                print(
                    f"   {metodo} no encontró "
                    "contenido. Probando respaldo..."
                )


            # ---------------------------------------------------------------------
            # ERRORES QUE NO DEBEN EVADIRSE
            # ---------------------------------------------------------------------

            except (
                ErrorSeguridadScraping,
                ErrorNoRecuperable,
                requests.exceptions.SSLError,
            ) as error:

                salida[
                    "motivo_revision"
                ] = (
                    "error_no_recuperable"
                )


                salida[
                    "advertencias"
                ] = " | ".join(
                    advertencias
                    + errores_intentos
                )


                salida[
                    "error"
                ] = (
                    f"{metodo}: {error}"
                )


                print(
                    "   Fallo no recuperable "
                    f"en {metodo}: {error}"
                )


                return salida


            # ---------------------------------------------------------------------
            # ERROR DE RED
            # ---------------------------------------------------------------------

            except requests.RequestException as error:

                errores_intentos.append(
                    f"{metodo}: "
                    f"error de red: {error}"
                )


                print(
                    f"   Falló {metodo}: "
                    f"{error}"
                )


                print(
                    "   Probando método "
                    "de respaldo..."
                )


            # ---------------------------------------------------------------------
            # ERROR GENERAL
            # ---------------------------------------------------------------------

            except Exception as error:

                errores_intentos.append(
                    f"{metodo}: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )


                print(
                    f"   Falló {metodo}: "
                    f"{error}"
                )


                print(
                    "   Probando método "
                    "de respaldo..."
                )


    # =========================================================================
    # SIN RESULTADO
    # =========================================================================

    salida[
        "motivo_revision"
    ] = (
        "sin_perfil_extraido"
    )


    salida[
        "advertencias"
    ] = " | ".join(
        advertencias
    )


    salida[
        "error"
    ] = " | ".join(
        errores_intentos
    )


    print(
        "   ALERTA: No se extrajo texto para "
        f"{site.get('universidad', '')} - "
        f"{site.get('carrera', '')}"
    )


    return salida