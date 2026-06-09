"""
05_VISUALIZACION_INTERACTIVA.PY
-----------------------------------------------------------------------
Fase: 3. Análisis

Propósito:
    Generar un dashboard HTML interactivo para visualizar los resultados
    del clustering semántico de perfiles de egreso.
"""

import os
import pandas as pd
import plotly.express as px
import plotly.io as pio


# ==========================================
# RUTAS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_ENTRADA = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_con_clusters.csv")
)

RUTA_HTML = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "processed", "dashboard_clusters.html")
)


# ==========================================
# CONFIGURACIÓN DE CLUSTERS
# ==========================================
NOMBRES_CLUSTERS = {
    0: "Gestión Tecnológica y Sistemas Organizacionales",
    1: "Ciencia de Datos y Desarrollo de Software",
    2: "Ingeniería de Software y Computación Aplicada",
    3: "Ciberseguridad y Gestión de Riesgos"
}

COLORES_CLUSTERS = {
    0: "#8B5CF6",
    1: "#3B82F6",
    2: "#14B8A6",
    3: "#F97316"
}

COLORES_INSTITUCION = {
    "Universidad": "#6366F1",
    "Instituto Profesional": "#FF6B4A",
    "CFT": "#20C997"
}

PALABRAS_CLAVE = {
    0: [
        "información", "organización", "proyecto", "sistema",
        "social", "solución", "tecnología", "tecnológico"
    ],
    1: [
        "ciencia", "dato", "desarrollo", "proyecto",
        "software", "solución", "tecnología", "tecnológico"
    ],
    2: [
        "calidad", "computación", "dato", "información",
        "proyecto", "sistema", "software", "solución"
    ],
    3: [
        "amenaza", "ciberseguridad", "información", "organización",
        "proyecto", "riesgo", "seguridad", "sistema"
    ]
}


# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def generar_tags(palabras, color):
    tags = ""
    for palabra in palabras:
        tags += (
            f'<span class="tag" '
            f'style="background:{color}22; border-color:{color}; color:{color};">'
            f'{palabra}</span>'
        )
    return tags


def generar_ejemplos(df_cluster):
    ejemplos = df_cluster[["universidad", "carrera"]].head(5).values.tolist()
    html = ""
    for universidad, carrera in ejemplos:
        html += f"<li>{universidad} — {carrera}</li>"
    return html


def generar_leyenda_html(labels, values, color_map):
    total = sum(values)
    bloques = ""

    for label, value in zip(labels, values):
        color = color_map.get(label, "#94A3B8")
        porcentaje = (value / total * 100) if total > 0 else 0

        bloques += f"""
        <div class="legend-item">
            <span class="legend-dot" style="background:{color};"></span>
            <span class="legend-text">{label} ({value})</span>
            <span class="legend-value">{porcentaje:.1f}%</span>
        </div>
        """

    return f'<div class="custom-legend">{bloques}</div>'


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("🚀 Generando dashboard visual de clusters...")

    if not os.path.exists(RUTA_ENTRADA):
        print("❌ Error: No existe perfiles_con_clusters.csv.")
        print("➡️ Ejecuta primero: python .\\01_clustering.py")
        exit()

    df = pd.read_csv(
        RUTA_ENTRADA,
        sep=";",
        encoding="utf-8-sig"
    )

    df.columns = df.columns.str.strip()

    columnas_requeridas = [
        "universidad",
        "carrera",
        "tipo_institucion",
        "Cluster",
        "Coordenada_X",
        "Coordenada_Y"
    ]

    for columna in columnas_requeridas:
        if columna not in df.columns:
            print(f"❌ Error: Falta la columna requerida: {columna}")
            print("Columnas disponibles:")
            print(df.columns.tolist())
            exit()

    df["Perfil_Formativo"] = df["Cluster"].map(NOMBRES_CLUSTERS)
    df["Cluster_Texto"] = "Cluster " + df["Cluster"].astype(str)

    total_programas = len(df)
    total_tipos = df["tipo_institucion"].nunique()
    total_perfiles = df["Cluster"].nunique()

    # ==========================================
    # CONFIGURACIÓN GENERAL DE PLOTLY
    # ==========================================
    CONFIG_PLOTLY = {
        "displayModeBar": False,
        "responsive": True
    }

    # ==========================================
    # DATOS PARA DONA 1
    # ==========================================
    orden_instituciones = ["Universidad", "Instituto Profesional", "CFT"]
    tipo_counts = df["tipo_institucion"].value_counts().reindex(orden_instituciones, fill_value=0).reset_index()
    tipo_counts.columns = ["tipo_institucion", "cantidad"]

    fig_tipo = px.pie(
        tipo_counts,
        names="tipo_institucion",
        values="cantidad",
        hole=0.58,
        color="tipo_institucion",
        color_discrete_map=COLORES_INSTITUCION
    )

    fig_tipo.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=13),
        margin=dict(l=0, r=0, t=0, b=0),
        height=280,
        showlegend=False
    )

    fig_tipo.update_traces(
        textposition="inside",
        textinfo="percent",
        marker=dict(line=dict(color="#0F172A", width=2))
    )

    leyenda_tipo = generar_leyenda_html(
        tipo_counts["tipo_institucion"].tolist(),
        tipo_counts["cantidad"].tolist(),
        COLORES_INSTITUCION
    )

    # ==========================================
    # DATOS PARA DONA 2
    # ==========================================
    orden_perfiles = [
        "Gestión Tecnológica y Sistemas Organizacionales",
        "Ciencia de Datos y Desarrollo de Software",
        "Ingeniería de Software y Computación Aplicada",
        "Ciberseguridad y Gestión de Riesgos"
    ]

    perfil_counts = (
        df["Perfil_Formativo"]
        .value_counts()
        .reindex(orden_perfiles, fill_value=0)
        .reset_index()
    )
    perfil_counts.columns = ["Perfil_Formativo", "cantidad"]

    mapa_colores_perfiles = {
        nombre: COLORES_CLUSTERS[cluster]
        for cluster, nombre in NOMBRES_CLUSTERS.items()
    }

    fig_perfil = px.pie(
        perfil_counts,
        names="Perfil_Formativo",
        values="cantidad",
        hole=0.58,
        color="Perfil_Formativo",
        color_discrete_map=mapa_colores_perfiles
    )

    fig_perfil.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        margin=dict(l=0, r=0, t=0, b=0),
        height=280,
        showlegend=False
    )

    fig_perfil.update_traces(
        textposition="outside",
        textinfo="percent",
        marker=dict(line=dict(color="#0F172A", width=2))
    )

    leyenda_perfil = generar_leyenda_html(
        perfil_counts["Perfil_Formativo"].tolist(),
        perfil_counts["cantidad"].tolist(),
        mapa_colores_perfiles
    )

    # ==========================================
    # MAPA PCA
    # ==========================================
    fig_mapa = px.scatter(
        df,
        x="Coordenada_X",
        y="Coordenada_Y",
        color="Perfil_Formativo",
        color_discrete_map=mapa_colores_perfiles,
        hover_name="universidad",
        hover_data={
            "carrera": True,
            "tipo_institucion": True,
            "Coordenada_X": ":.3f",
            "Coordenada_Y": ":.3f",
            "Perfil_Formativo": False
        }
    )

    fig_mapa.update_traces(
        marker=dict(
            size=12,
            line=dict(width=1, color="white"),
            opacity=0.85
        )
    )

    fig_mapa.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(color="white", size=12),
        height=520,
        margin=dict(l=30, r=30, t=20, b=30),
        xaxis_title="Componente Principal 1",
        yaxis_title="Componente Principal 2",
        legend_title_text="Perfil Formativo",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        )
    )

    fig_mapa.update_xaxes(
        showgrid=True,
        gridcolor="#334155",
        zeroline=True,
        zerolinecolor="#64748B"
    )

    fig_mapa.update_yaxes(
        showgrid=True,
        gridcolor="#334155",
        zeroline=True,
        zerolinecolor="#64748B"
    )

    # ==========================================
    # CONVERSIÓN DE GRÁFICOS A HTML
    # ==========================================
    html_tipo = pio.to_html(
        fig_tipo,
        include_plotlyjs="cdn",
        full_html=False,
        config=CONFIG_PLOTLY
    )

    html_perfil = pio.to_html(
        fig_perfil,
        include_plotlyjs=False,
        full_html=False,
        config=CONFIG_PLOTLY
    )

    html_mapa = pio.to_html(
        fig_mapa,
        include_plotlyjs=False,
        full_html=False,
        config=CONFIG_PLOTLY
    )

    # ==========================================
    # TABLA CRUZADA LIMPIA
    # ==========================================
    tabla = pd.crosstab(
        df["tipo_institucion"],
        df["Perfil_Formativo"]
    )

    tabla = tabla.reindex(index=orden_instituciones, fill_value=0)
    tabla = tabla.reindex(columns=orden_perfiles, fill_value=0)

    tabla["Total"] = tabla.sum(axis=1)

    fila_total = tabla.sum(axis=0)
    tabla.loc["Total"] = fila_total

    tabla.index.name = "Tipo de institución"
    tabla.columns.name = None

    tabla = tabla.reset_index()

    tabla_html = tabla.to_html(
        classes="tabla-dashboard",
        index=False,
        border=0
    )

    # ==========================================
    # TARJETAS POR CLUSTER
    # ==========================================
    cards_clusters = ""

    for cluster_id, nombre in NOMBRES_CLUSTERS.items():
        df_cluster = df[df["Cluster"] == cluster_id]
        cantidad = len(df_cluster)
        color = COLORES_CLUSTERS[cluster_id]

        cards_clusters += f"""
        <div class="cluster-card" style="border-top: 4px solid {color};">
            <div class="cluster-header">
                <div>
                    <h3 style="color:{color};">Cluster {cluster_id}</h3>
                    <h2>{nombre}</h2>
                </div>

                <div class="cluster-number" style="background:{color}33; color:{color};">
                    {cantidad}
                </div>
            </div>

            <h4>Palabras clave</h4>

            <div class="tags">
                {generar_tags(PALABRAS_CLAVE[cluster_id], color)}
            </div>

            <h4>Ejemplos de programas</h4>

            <ul>
                {generar_ejemplos(df_cluster)}
            </ul>
        </div>
        """

    # ==========================================
    # HTML FINAL
    # ==========================================
    html = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Dashboard Clustering Semántico</title>

        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: #0F172A;
                color: #F8FAFC;
            }}

            .container {{
                padding: 28px;
                max-width: 1920px;
                margin: 0 auto;
            }}

            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }}

            .title h1 {{
                margin: 0;
                font-size: 32px;
                letter-spacing: -1px;
                color: #FFFFFF;
            }}

            .title span {{
                color: #60A5FA;
                font-size: 23px;
            }}

            .title p {{
                margin: 8px 0 0 0;
                color: #CBD5E1;
                font-size: 16px;
            }}

            .kpi-top {{
                background: #1E293B;
                border-radius: 14px;
                padding: 18px 28px;
                min-width: 170px;
                text-align: center;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
            }}

            .kpi-top small {{
                color: #CBD5E1;
                font-weight: 600;
            }}

            .kpi-top strong {{
                display: block;
                font-size: 34px;
                margin-top: 6px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: 0.9fr 2.1fr 1.35fr;
                gap: 18px;
                margin-bottom: 18px;
                align-items: stretch;
            }}

            .panel {{
                background: #1E293B;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                overflow: hidden;
            }}

            .panel h2 {{
                margin: 0 0 16px 0;
                font-size: 16px;
                text-transform: uppercase;
                color: #E2E8F0;
                letter-spacing: 0.2px;
                line-height: 1.2;
            }}

            .summary-item {{
                display: flex;
                align-items: center;
                gap: 14px;
                margin-bottom: 22px;
            }}

            .summary-item:last-child {{
                margin-bottom: 0;
            }}

            .summary-icon {{
                width: 48px;
                height: 48px;
                border-radius: 50%;
                background: #334155;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 22px;
            }}

            .summary-number {{
                font-size: 30px;
                font-weight: bold;
                color: #FFFFFF;
                line-height: 1;
            }}

            .summary-label {{
                color: #CBD5E1;
                font-size: 14px;
                margin-top: 5px;
            }}

            .chart-box {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-start;
                gap: 12px;
            }}

            .chart-box > div:first-child {{
                width: 100%;
            }}

            .custom-legend {{
                width: 100%;
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin-top: 2px;
            }}

            .legend-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 13px;
                color: #E2E8F0;
                line-height: 1.3;
            }}

            .legend-dot {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                flex-shrink: 0;
            }}

            .legend-text {{
                flex: 1;
            }}

            .legend-value {{
                color: #94A3B8;
                font-weight: 700;
                white-space: nowrap;
            }}

            .tabla-dashboard {{
                width: 100%;
                border-collapse: collapse;
                color: #F8FAFC;
                font-size: 13px;
                table-layout: fixed;
            }}

            .tabla-dashboard th {{
                background: #0F172A;
                padding: 12px 8px;
                color: #93C5FD;
                border: 1px solid #334155;
                word-break: break-word;
                overflow-wrap: anywhere;
                font-weight: 700;
                line-height: 1.2;
            }}

            .tabla-dashboard td {{
                padding: 14px 8px;
                text-align: center;
                border: 1px solid #334155;
                font-weight: 600;
                word-break: break-word;
                line-height: 1.2;
            }}

            .tabla-dashboard tbody tr:hover {{
                background: #334155;
            }}

            .tabla-dashboard tbody tr:last-child {{
                background: #111827;
                color: #FACC15;
                font-weight: bold;
            }}

            .clusters-grid {{
                display: grid;
                grid-template-columns: repeat(4, minmax(260px, 1fr));
                gap: 18px;
                margin-bottom: 18px;
            }}

            .cluster-card {{
                background: #1E293B;
                border-radius: 14px;
                padding: 18px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                min-height: 300px;
            }}

            .cluster-header {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 12px;
                margin-bottom: 16px;
            }}

            .cluster-card h3 {{
                margin: 0 0 6px 0;
                text-transform: uppercase;
                font-size: 14px;
                letter-spacing: 0.5px;
            }}

            .cluster-card h2 {{
                margin: 0;
                font-size: 17px;
                line-height: 1.25;
                color: #FFFFFF;
            }}

            .cluster-card h4 {{
                margin: 16px 0 8px 0;
                color: #CBD5E1;
                text-transform: uppercase;
                font-size: 12px;
                letter-spacing: 0.3px;
            }}

            .cluster-number {{
                min-width: 42px;
                height: 42px;
                border-radius: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 22px;
                font-weight: bold;
            }}

            .tags {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-bottom: 14px;
            }}

            .tag {{
                padding: 5px 8px;
                border-radius: 7px;
                font-size: 12px;
                border: 1px solid;
                font-weight: 600;
            }}

            ul {{
                padding-left: 18px;
                color: #E2E8F0;
                font-size: 12px;
                line-height: 1.55;
                margin: 0;
            }}

            .map-panel {{
                background: #1E293B;
                border-radius: 14px;
                padding: 20px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                margin-bottom: 16px;
            }}

            .map-panel h2 {{
                margin: 0 0 14px 0;
                font-size: 24px;
                color: #FFFFFF;
            }}

            .footer {{
                color: #94A3B8;
                font-size: 12px;
                margin-top: 15px;
            }}

            @media (max-width: 1400px) {{
                .grid {{
                    grid-template-columns: 1fr;
                }}

                .clusters-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}

            @media (max-width: 760px) {{
                .container {{
                    padding: 18px;
                }}

                .header {{
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 18px;
                }}

                .kpi-top {{
                    width: 100%;
                }}

                .clusters-grid {{
                    grid-template-columns: 1fr;
                }}

                .title h1 {{
                    font-size: 26px;
                }}

                .title span {{
                    display: block;
                    font-size: 18px;
                    margin-top: 4px;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="container">

            <div class="header">
                <div class="title">
                    <h1>
                        Distribución de Perfiles Formativos
                        <span>(Clustering Semántico)</span>
                    </h1>
                    <p>Análisis de la oferta académica en Informática en Chile</p>
                </div>

                <div class="kpi-top">
                    <small>Total de programas</small>
                    <strong>{total_programas}</strong>
                </div>
            </div>

            <div class="grid">

                <div>
                    <div class="panel">
                        <h2>Resumen general</h2>

                        <div class="summary-item">
                            <div class="summary-icon">🎓</div>
                            <div>
                                <div class="summary-number">{total_programas}</div>
                                <div class="summary-label">Programas analizados</div>
                            </div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-icon">🏛️</div>
                            <div>
                                <div class="summary-number">{total_tipos}</div>
                                <div class="summary-label">Tipos de institución</div>
                            </div>
                        </div>

                        <div class="summary-item">
                            <div class="summary-icon">🧩</div>
                            <div>
                                <div class="summary-number">{total_perfiles}</div>
                                <div class="summary-label">Perfiles formativos</div>
                            </div>
                        </div>
                    </div>

                    <div class="panel" style="margin-top:18px;">
                        <h2>Distribución por tipo de institución</h2>
                        <div class="chart-box">
                            {html_tipo}
                            {leyenda_tipo}
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Distribución de programas por tipo de institución y perfil formativo</h2>
                    {tabla_html}
                </div>

                <div class="panel">
                    <h2>Distribución general por perfil formativo</h2>
                    <div class="chart-box">
                        {html_perfil}
                        {leyenda_perfil}
                    </div>
                </div>

            </div>

            <div class="clusters-grid">
                {cards_clusters}
            </div>

            <div class="map-panel">
                <h2>Mapa semántico PCA de perfiles de egreso</h2>
                {html_mapa}
            </div>

            <div class="footer">
                Fuente: Análisis de clustering semántico sobre perfiles de egreso de programas de informática en Chile.
            </div>

        </div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(RUTA_HTML), exist_ok=True)

    with open(RUTA_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print("✅ DASHBOARD GENERADO CORRECTAMENTE")
    print(f"🌐 Abre este archivo en tu navegador:")
    print(f"--> {RUTA_HTML}")
    print("=" * 60)