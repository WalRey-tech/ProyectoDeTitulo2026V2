"""
05_VISUALIZACION_INTERACTIVA.PY
-----------------------------------------------------------------------
Fase: 3. Análisis
Propósito: 
    Renderizar el espacio semántico (PCA) en un entorno web interactivo,
    aplicando estándares de visualización analítica de datos (cuadrículas, 
    ejes cartesianos y distribuciones marginales).
    
    Este dashboard HTML sirve como prueba de concepto (PoC) visual antes 
    de llevar los datos a la aplicación web final en Next.js.
"""

# ==========================================
# IMPORTACIÓN DE LIBRERÍAS
# ==========================================
import os
import pandas as pd
import plotly.express as px

# ==========================================
# CONFIGURACIÓN DE RUTAS INTELIGENTES
# ==========================================
# Ubicamos la carpeta actual (Fase3_Analisis)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas apuntando al Data Lake central
RUTA_ENTRADA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv"))
RUTA_HTML = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "mapa_interactivo.html"))

if __name__ == "__main__":
    print("🚀 Iniciando motor de visualización analítica avanzada...")

    # ==========================================
    # 1. CARGA Y PREPARACIÓN DE DATOS
    # ==========================================
    print(f"📂 Cargando datos analíticos desde: {RUTA_ENTRADA}")
    if not os.path.exists(RUTA_ENTRADA):
        print("❌ Error: Faltan los datos de clustering. Ejecuta '01_clustering.py' primero.")
        exit()

    df = pd.read_csv(RUTA_ENTRADA, sep=';', encoding='utf-8')

    # Bautizamos los clusters según los hallazgos previos
    nombres_oficiales = {
        0: "Ingeniería de Software y Desarrollo",
        1: "Ciencia de Datos y Analytics",
        2: "Arquitectura y Gestión de Sistemas",
        3: "Ciberseguridad y Control de Riesgos"
    }

    df['Perfil_Formativo'] = df['Cluster'].map(nombres_oficiales)
    df['Etiqueta_Interactiva'] = "<b>" + df['universidad'] + "</b><br>" + df['carrera']

    # ==========================================
    # 2. RENDERIZADO DEL MAPA ANALÍTICO
    # ==========================================
    print("📈 Generando plano cartesiano y distribuciones marginales...")

    # Generamos la figura interactiva con Plotly Express
    fig = px.scatter(
        df,
        x='Coordenada_X',            
        y='Coordenada_Y',            
        color='Perfil_Formativo',    
        hover_name='Etiqueta_Interactiva', 
        hover_data={                 
            'tipo_institucion': True,
            'Coordenada_X': ':.2f',  # Muestra el número matemático redondeado a 2 decimales
            'Coordenada_Y': ':.2f',  
            'Perfil_Formativo': False 
        },
        marginal_x="box", # Diagrama de caja estadístico en el techo
        marginal_y="box", # Diagrama de caja estadístico a la derecha
        title="<b>Análisis Dimensional de Perfiles de Egreso Informáticos en Chile (2026)</b><br><sup>Agrupamiento K-Means sobre Embeddings NLP comprimidos mediante PCA</sup>",
        color_discrete_sequence=px.colors.qualitative.Prism, # Paleta técnica intensa
    )

    # ==========================================
    # 3. REFINAMIENTO ESTÉTICO (ESTILO INGENIERÍA)
    # ==========================================
    # Ajustamos los puntos
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey'), opacity=0.85),
        selector=dict(mode='markers')
    )

    # Configuramos la cuadrícula (Grid) y los ejes para un look matemático/científico
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
        plot_bgcolor="#f0f3f5", # Fondo gris técnico muy claro
        paper_bgcolor="white",  # Bordes blancos
        
        # Eje Horizontal (X)
        xaxis=dict(
            title="<b>Componente Principal 1 (Varianza Semántica Dominante)</b>",
            showgrid=True,
            gridcolor='white',
            gridwidth=1.5,
            zeroline=True,
            zerolinecolor='#2c3e50', # Línea central oscura
            zerolinewidth=2,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=True # Cierra la caja del gráfico por arriba
        ),
        
        # Eje Vertical (Y)
        yaxis=dict(
            title="<b>Componente Principal 2 (Varianza Semántica Secundaria)</b>",
            showgrid=True,
            gridcolor='white',
            gridwidth=1.5,
            zeroline=True,
            zerolinecolor='#2c3e50', # Línea central oscura
            zerolinewidth=2,
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=True # Cierra la caja del gráfico por la derecha
        ),
        
        legend_title_text='<b>Clasificación del Modelo (Clusters)</b>',
        hoverlabel=dict(bgcolor="white", font_size=13)
    )

    # ==========================================
    # 4. EXPORTACIÓN A HTML
    # ==========================================
    os.makedirs(os.path.dirname(RUTA_HTML), exist_ok=True)
    fig.write_html(RUTA_HTML)
    
    print("\n" + "="*60)
    print("✅ RENDERIZADO COMPLETADO")
    print(f"🌐 Abre este archivo en tu navegador web para explorar el mapa:")
    print(f"--> {RUTA_HTML}")
    print("="*60)