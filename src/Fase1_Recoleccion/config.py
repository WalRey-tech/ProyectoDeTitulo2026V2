SITES = [
    {
        "universidad": "Universidad de Santiago de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería de Ejecución en Computación e Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.usach.cl/carreras/ingenieria-de-ejecucion-en-computacion-e-informatica",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""  
    },
    {
        "universidad": "Universidad de Santiago de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.usach.cl/carreras/ingenieria-civil-en-informatica-plan-comun",
        "tipo_extraccion": "selenium",
        "tipo_selector": "xpath",
        "selector": "//div[contains(@class, 'field--name-body')]/h2[contains(., 'Plan Común')]/following-sibling::p[1]"
    },
    {
        "universidad": "Universidad de Valparaíso",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería en Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uv.cl/carreras/ingenieria-en-informatica",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#tm-main .uk-text-justify"
    },
    {
        "universidad": "Universidad de Valparaíso",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uv.cl/carreras/ingenieria-civil-informatica",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".uk-panel p"
    },
    {
        "universidad": "Universidad de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Computación",
        "tipo_carrera": "Profesional",
        "url": "https://admisionuchile.cl/career/ingenieria-civil-en-computacion/",
        "tipo_extraccion": "requests",
        "tipo_selector": "css",
        "selector": ".elementor-element-329fe8e2",
        "selector_estricto": True,
        "selector_coincidencias": 1,
        "longitud_minima": 150,
        "marcadores_requeridos": [
            "conciben, diseñan",
            "Software",
            "arquitectura de hardware"
        ],
        "marcadores_prohibidos": [
            "Arancel",
            "Matrícula"
        ]
    },
    {
        "universidad": "Universidad de Atacama",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Computación e Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uda.cl/index.php/ingenieria-civil-en-computacion-e-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".et_pb_toggle_content p"
    },
    {
        "universidad": "Universidad de la Serena",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Computación e Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.userena.cl/carreras/ingenieria-civil-en-computacion-e-informatica",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#section-id-1724720371470"
    },
    {
        "universidad": "Universidad de Talca",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Computación",
        "tipo_carrera": "Profesional",
        "url": "https://admision.utalca.cl/carreras/ingenieria-civil-en-computacion/",
        # Bloque del perfil completo, verificado en la página de admisión.
        "tipo_extraccion": "requests",
        "tipo_selector": "css",
        "selector": ".elementor-element-9289d86",
        "selector_estricto": True,
        "longitud_minima": 150
    },
    {
        "universidad": "Universidad de Talca",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería en Informática Empresarial",
        "tipo_carrera": "Profesional",
        "url": "https://admision.utalca.cl/carreras/ingenieria-en-informatica-empresarial/",
        "tipo_extraccion": "requests",
        "tipo_selector": "css",
        "selector": ".elementor-element-9289d86",
        "selector_estricto": True,
        "selector_coincidencias": 1,
        "longitud_minima": 150,
        "marcadores_requeridos": [
            "Informática Empresarial",
            "gestión empresarial",
            "toma de decisiones"
        ],
        "marcadores_prohibidos": [
            "Arancel",
            "Matrícula"
        ]
    },
    {
        "universidad": "Universidad del Biobío",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil en Informática",
        "tipo_carrera": "Profesional",
        "url": "https://ubiobio.cl/admision/Todas_las_Carreras/17/Ingenieria_Civil_en_Informatica_Concepcion/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "",
        "verificar_ssl": False
    },
    {
        "universidad": "Universidad del Biobío",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria de Ejecución en Computación e Informática",
        "tipo_carrera": "Profesional",
        "url": "https://ubiobio.cl/admision/Ciencias_Empresariales/21/Ingenieria_de_Ejecucion_en_Computacion_e_Informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "",
        "verificar_ssl": False
    },
    {
        "universidad": "Universidad de la Frontera",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ufro.cl/ingenieria-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".elementor-element-0713830"
    },
    {
        "universidad": "Universidad de la Frontera",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ufro.cl/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".elementor-widget-container p"
    },
    {
        "universidad": "Universidad de los Lagos",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ulagos.cl/Carreras/ingenieria-civil-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".row.perfil"
    },
    {
        "universidad": "Universidad de Magallanes",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.umag.cl/?page_id=35",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#nav-perfil p",
        "verificar_ssl": False
    },
    {
        "universidad": "Universidad de Magallanes",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.umag.cl/?page_id=27",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".elementor-element-516655f",
        "verificar_ssl": False
    },
    {
        "universidad": "Universidad Tecnologica Metropolitana",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://conocetucarrera.utem.cl/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "-"
    },
    {
        "universidad": "Universidad Tecnologica Metropolitana",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Computacion mencion Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://conocetucarrera.utem.cl/ingenieria-civil-en-computacion-mencion-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "-"
    },
    {
        "universidad": "Universidad de Ohiggins",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Computacion",
        "tipo_carrera": "Profesional",
        "url": "https://www.uoh.cl/escuela-de-ingenieria/carreras/ingenieria-civil-en-computacion/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad de Bernardo Ohiggins",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.ubo.cl/facultades/facultad-de-ingenieria-ciencia-y-tecnologia/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".tab-content p"
    },
    {
        "universidad": "Universidad de Aysen",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://uaysen.cl/departamentos/ciencias-naturales-y-tecnologia/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Pontificia Universidad Catolica de Valparaiso",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.pucv.cl/pucv/pregrado/ingenieria-en-informatica",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#menuPreguntas"
    },
    {
        "universidad": "Universidad de Concepcion",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.udec.cl/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".fusion-builder-column-0"
    },
    {
        "universidad": "Universidad Tecnica Federico Santa Maria",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://usm.cl/admision/carreras/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "elementor-widget-container p"
    },
    {
        "universidad": "Universidad Tecnica Federico Santa Maria",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://usm.cl/admision/carreras/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".elementor-element-6c130331"
    },
    {
        "universidad": "Universidad Catolica del Norte",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ucn.cl/carreras/tecnologia-computacion/ingenieria-en-computacion-e-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad Catolica del Norte",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ucn.cl/carreras/tecnologia-computacion/ingenieria-civil-en-computacion-e-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad Católica de la Santísima Concepción",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://ingenieria.ucsc.cl/carreras/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad Católica de la Santísima Concepción",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria de Ejecucion en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://it.ucsc.cl/carreras/ingenieria-de-ejecucion-en-informatica/",
        "tipo_extraccion": "requests",
        "tipo_selector": "css",
        "selector": ".graduate-profile > p",
        "selector_texto_contiene": "El/la Ingeniero/a de Ejecución en Informática egresado/a",
        "selector_estricto": True,
        "selector_coincidencias": 1,
        "longitud_minima": 150,
        "marcadores_requeridos": [
            "desarrollo de software",
            "bases de datos"
        ],
        "marcadores_prohibidos": [
            "Técnico(a) Universitario(a)",
            "El/la Técnico(a)",
            "Perfil de ingreso"
        ],
        "modalidad": "Presencial; diurna; 8 semestres",
        "grupo_perfil": "ucsc_ejecucion_informatica",
        "id_programa": "ucsc_ejecucion_presencial"
    },
    {
        "universidad": "Universidad Catolica de Temuco",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uct.cl/ing-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#elementor-tab-content-1201"
    },
    {
        "universidad": "Universidad de las Américas",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://ingenieria.udla.cl/carreras/ingenieria-en-informatica/",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "#perfil-egreso",
        "requiere_revision": True,
        "motivo_revision": "El HTML mezcla denominaciones y contenido de periodismo; el PDF oficial enlazado devuelve 404. Pendiente obtener documento vigente.",
        "marcadores_prohibidos": [
            "periodísticos"
        ],
        "longitud_minima": 150
    },
    {
        "universidad": "Universidad Andres Bello",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.unab.cl/carreras/ingenieria-en-computacion-e-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#acordeon-3-2"
    },
    {
        "universidad": "Universidad Autonoma de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uautonoma.cl/carreras/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad Autonoma de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uautonoma.cl/carreras/ingenieria-en-informatica-online-regular/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#tab4"
    },
    {
        "universidad": "Universidad Central de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ucentral.cl/carrera/ingenieria-civil-en-computacion-e-informatica/",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "div[role='region']"
    },
    {
        "universidad": "Universidad Central de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica y Sistemas Inteligentes",
        "tipo_carrera": "Profesional",
        "url": "https://advance.ucentral.cl/programas/ingenieria-en-informatica-y-sistemas-inteligentes/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#e-n-accordion-item-1911"
    },
    {
        "universidad": "Universidad Mayor",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://advance.umayor.cl/educacion-online/ingenieria-informatica-plan-regular?utm_source=google&utm_medium=cpa&utm_campaign=SNB_AD_00807156_ON_20267&utm_content=RSA1_ex&utm_term=&gad_source=1&gad_campaignid=23076000607&gbraid=0AAAAADtiOMI1iMfo5ZY2WadRR71Q1zVIJ&gclid=CjwKCAjwhqfPBhBWEiwAZo196q-QOm6zr2L98M0Gp4yAELJa9zXl4ci5MQzLJJrfvV5vChRPKrEUeBoCY6wQAvD_BwE",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".box-acordeones > div:nth-child(1) p",
        "verificar_ssl": False
    },
    {
        "universidad": "Universidad Mayor",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.umayor.cl/um/carreras/ingenieria-civil-en-computacion-e-informatica-santiago",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".panel p"
    },
    {
        "universidad": "Universidad Diego Portales",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica y Telecomunicaciones",
        "tipo_carrera": "Profesional",
        "url": "https://admision.udp.cl/carrera/ingenieria-civil-en-informatica-y-telecomunicaciones/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad Diego Portales",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Informatica y Gestion",
        "tipo_carrera": "Profesional",
        "url": "https://admision.udp.cl/carrera/ingenieria-en-informatica-y-gestion/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "-"  # 
    },
    {
        "universidad": "Universidad Adolfo Ibanez",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.uai.cl/admision/carreras/ingenieria-civil-informatica",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "-"  # Brayan
    },
    {
        "universidad": "Universidad del Desarrollo",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica e Innovacion Tecnologica",
        "tipo_carrera": "Profesional",
        "url": "https://ingenieria.udd.cl/carrera/ingenieria-civil-en-informatica-e-innovacion-tecnologica/perfil-del-alumno/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".container.text"
    },
    {
        "universidad": "Universidad del Desarrollo",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica e Inteligencia Artificial",
        "tipo_carrera": "Profesional",
        "url": "https://ingenieria.udd.cl/carrera/ingenieria-civil-en-informatica-e-inteligencia-artificial/malla/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".text-row__content-inner p"
    },
    {
        "universidad": "Universidad Finis Terrae",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica y Telecomunicaciones",
        "tipo_carrera": "Profesional",
        "url": "https://admision.finis.cl/carrera/ingenieria-civil-informatica-y-telecomunicaciones/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ".fusion-text.fusion-text-4",
        "verificar_ssl": False
    },
    {
        "universidad": "Universidad Alberto Hurtado",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uahurtado.cl/carreras/ingenieria-civil-plan-comun-informatica/",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "#descripcion_carrera"
    },
    {
        "universidad": "Universidad de los Andes",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Ciencias de la Computacion",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uandes.cl/carreras/area-ingenieria-y-administracion/ingenieria-civil-en-ciencias-de-la-computacion",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#ContentplaceholderMain_C087_Col01 p"
    },
    {
        "universidad": "Universidad Gabriela Mistral",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ugm.cl/carreras/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "-" 
    },
    {
        "universidad": "INACAP",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://portal.inacap.cl/carreras/area-informatica-ciberseguridad-y-telecomunicaciones/ingenieria-en-informatica",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "#perfil-egreso"
    },
    {
        "universidad": "Duoc UC",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.duoc.cl/carreras/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "#ficha-tecnica"
    },
    {
        "universidad": "AIEP",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.aiep.cl/admision/carrera/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Santo Tomas",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.tupuedes.cl/carreras/instituto-profesional/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "IACC",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.iacc.cl/carrera/ingenieria-en-informatica/?origen=google_ads_carreras&utm_campaign=39844489-_Search_CARRERA_ING_INFORMATICA&utm_source=ppc&utm_medium=google&utm_term=cpl&utm_content=ingenieria-en-informatica&utm_term=ingenier%C3%ADa%20en%20sistemas%20universidades&utm_campaign=IACC_Search_Carrera_Ingenieria_Inform%C3%A1tica&utm_source=adwords&utm_medium=ppc&hsa_acc=3643935547&hsa_cam=21457092417&hsa_grp=167767224914&hsa_ad=705590150535&hsa_src=g&hsa_tgt=kwd-589776510461&hsa_kw=ingenier%C3%ADa%20en%20sistemas%20universidades&hsa_mt=b&hsa_net=adwords&hsa_ver=3&gad_source=1&gad_campaignid=21457092417&gbraid=0AAAAAqi2tcnc4mlF6prfKC1hCFytIsM7a&gclid=CjwKCAjw46HPBhAMEiwASZpLRHfuoZDNIblNry7ovo2v8t2FkJLsAKlMxZYtNTqT4XptYu05OZwVSxoCjDsQAvD_BwE",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Iplacex",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.iplacex.cl/carreras/carreras-100-online/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "IP Los Leones",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica y Ciberseguridad",
        "tipo_carrera": "Profesional",
        "url": "https://ipleones.cl/carreras/ingenieria-en-informatica-y-ciberseguridad/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": ""
    },
    {
        "universidad": "Universidad Catolica del Maule",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Ejecucion en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        # PDF oficial enlazado como "Perfil de Egreso" en la página de admisión.
        "url": "https://files.griddo.ucm.cl/ingenieria-de-ejecucion-en-computacion-e-informatica.pdf",
        "url_pagina_origen": "https://www.ucm.cl/prenovato/ingenieria-ejecucion-en-computacion-e-informatica/",
        "tipo_extraccion": "pdf",
        "tipo_selector": "pdf",
        "selector": "",
        # Página 1: perfil. Página 2: competencias. Página 3: malla (excluida).
        "pdf_paginas": [1, 2],
        "pdf_marcadores": ["Perfil de egreso", "Competencias Profesionales", "Competencias Genéricas"],
        "longitud_minima": 150
    },

    {
        "universidad": "Universidad Católica de la Santísima Concepción",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria de Ejecucion en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://advance.ucsc.cl/carreras/ingenieria-de-ejecucion-en-informatica/",
        "tipo_extraccion": "requests",
        "tipo_selector": "css",
        "selector": ".graduate-profile > p",
        "selector_texto_contiene": "El/la Ingeniero/a de Ejecución en Informática egresado/a",
        "selector_estricto": True,
        "selector_coincidencias": 1,
        "longitud_minima": 150,
        "marcadores_requeridos": [
            "desarrollo de software",
            "bases de datos"
        ],
        "marcadores_prohibidos": [
            "Técnico(a) Universitario(a)",
            "El/la Técnico(a)",
            "Perfil de ingreso"
        ],
        "modalidad": "Online; continuidad de estudios; 8 trimestres",
        "grupo_perfil": "ucsc_ejecucion_informatica",
        "id_programa": "ucsc_ejecucion_online"
    },
    {
        "universidad": "IP Chile",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.ipchile.cl/carreras/ingenieria-en-informatica/",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "#accordionInfoCarrera" 
    },
    {
        "universidad": "Universidad Austral de Chile",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.uach.cl/admision/valdivia/ingenieria-civil-en-informatica",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "" 
    },
    {
        "universidad": "Instituto profesional Virginio Gomez",
        "tipo_institucion": "Instituto Profesional",
        "carrera": "Ingenieria en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://admision.virginiogomez.cl/carreras/ingenieria-en-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "" 
    },
    {
        "universidad": "Universidad de Playa Ancha",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria Civil en Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://www.upla.cl/admision/carreras-profesionales/facultad-de-ingenieria/ingenieria-civil-informatica/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "" 
    },

     {
        "universidad": "Universidad de Los Lagos",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería en Informática para Técnicos de Nivel Superior",
        "tipo_carrera": "Profesional",
        "url": "https://admision.ulagos.cl/sin-licenciatura/ingenieria-en-informatica-para-tecnicos-de-nivel-superior/",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "h2, p" 
    },

    {
        "universidad": "Universidad Gabriela Mistral",
        "tipo_institucion": "Universidad",
        "carrera": "Ingenieria en Computacion e Informatica",
        "tipo_carrera": "Profesional",
        "url": "https://advance.ugm.cl/carrera/ingenieria-en-computacion/",
        "tipo_extraccion": "css",
        "tipo_selector": "css",
        "selector": "" 
    },
    {
        "universidad": "Universidad San Sebastian",
        "tipo_institucion": "Universidad",
        "carrera": "Ingeniería Civil Informática",
        "tipo_carrera": "Profesional",
        "url": "https://admision.uss.cl/carreras/ingenieria-civil-informatica",
        "tipo_extraccion": "selenium",
        "tipo_selector": "css",
        "selector": "#carrera-perfil-egreso" 
    },
]