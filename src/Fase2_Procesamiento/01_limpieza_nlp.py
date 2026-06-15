import pandas as pd
import spacy
import os
import csv

# 1. Configuración de Rutas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_etiquetado_v1.csv"))
RUTA_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))

# 2. Cargar el modelo NLP
print(" Cargando modelo de lenguaje de spaCy...")
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print(" Error: No se encontró el modelo. Ejecuta: python -m spacy download es_core_news_sm")
    exit()

# 3. Stopwords
STOPWORDS_CUSTOM = {
    "ufro", "duoc", "duocuc", "inacap", "aiep", "santo", "tomás", "tomas", 
    "cft", "ip", "universidad", "instituto", "profesional", "centro", "formación", "tecnológica",
    "ceduc", "ucn", "ipchile", "uss", "unab", "udp", "pucv", "usm", "usach", "uchile",
    "magíster", "magister", "director", "directora", "licenciado", "licenciatura", 
    "jefatura", "alumno", "estudiante", "egresado", "titulado"
}

def limpiar_texto_avanzado(texto):
    if not isinstance(texto, str):
        return ""
    doc = nlp(texto)
    tokens_limpios = []
    for token in doc:
        if token.ent_type_ == "ORG":
            continue
        if token.is_stop or token.is_punct or token.is_digit:
            continue
        token_lower = token.text.lower()
        if token_lower in STOPWORDS_CUSTOM:
            continue
        tokens_limpios.append(token.lemma_.lower())
    return " ".join(tokens_limpios)

def main():
    print(" Leyendo el dataset etiquetado...")
    try:
        # Leemos el archivo limpio generado por tu etiquetador.py
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except Exception as e:
        print(" Archivo no encontrado o corrompido.")
        return

    #  CORRECCIÓN AUTOMÁTICA (Validación Manual en Código) 
    # Buscamos la carrera de AIEP y la cambiamos a 'Técnico' directamente en Python
    mascara_aiep = df['carrera'] == 'Programacion y Analisis de Sistemas'
    if mascara_aiep.any():
        df.loc[mascara_aiep, 'grado'] = 'Técnico'
        print("🔧 Corrección automática aplicada: 'Programacion y Analisis de Sistemas' asignada a Técnico.")

    print("\n Análisis del confusor en consola (Grado vs Tipo de Institución):")
    print(pd.crosstab(df['grado'], df['tipo_institucion']))
    print("-" * 50)

    print(" Iniciando limpieza avanzada con spaCy (NER y Stopwords)...")
    print(" Esto tomará unos momentos...")
    df['perfil_limpio'] = df['perfil_egreso'].apply(limpiar_texto_avanzado)

    print(" Guardando el dataset limpio para la vectorización...")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    df.to_csv(RUTA_SALIDA, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f" ¡Punto 3 completado! Limpieza exitosa.")
    print(f" Archivo final guardado en: {RUTA_SALIDA}")

if __name__ == "__main__":
    main()