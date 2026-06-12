"""
01_LIMPIEZA_NLP.PY
-----------------------------------------------------------------------
Fase: 2. Procesamiento
Propósito: 
    Procesar los perfiles de egreso en bruto (raw) extraídos en la Fase 1.
    Elimina ruido estructural (caracteres, símbolos) y semántico (palabras 
    vacías) usando Inteligencia Artificial lingüística para preparar el 
    texto para los modelos matemáticos.

Metodología:
    1. Capa Estructural: Limpieza con Expresiones Regulares (Regex).
    2. Capa Lingüística: Lematización y filtrado con spaCy.
"""

# IMPORTACIÓN DE LIBRERÍAS
import os
import pandas as pd
import re
import spacy

# CONFIGURACIÓN DE RUTAS INTELIGENTES
# Ubicamos la carpeta donde vive este script (Fase2_Procesamiento)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Definimos las rutas subiendo un nivel hasta 'src' y luego entrando a 'data'
RUTA_ENTRADA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "raw", "perfiles_egreso_raw.csv"))
RUTA_SALIDA = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "processed", "perfiles_limpios.csv"))

# PREPARACIÓN DEL MODELO LINGÜÍSTICO (IA)
try:
    # IA: Cargamos el motor de procesamiento en español
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Error: Falta descargar el modelo. Ejecuta: python -m spacy download es_core_news_sm")
    exit()

# STOPWORDS PERSONALIZADAS: Palabras que la IA debe ignorar para no sesgar el análisis
stopwords_academicas = {
    "universidad", "instituto", "carrera", "estudiante", "alumno", "profesional",
    "egresado", "titulado", "malla", "semestre", "grado", "título", "formación",
    "capacidad", "conocimiento", "habilidad", "permitir", "desarrollar", "capaz",
    "preparado", "orientado", "año", "nivel", "superior", "programa",
    "chile", "católico", "católica", "pontificia", "nacional", "santiago",
    "valparaíso", "concepción", "temuco", "norte", "sur", "san", "federico",
    "santa", "maría", "andrés", "bello", "autónomo", "autónoma",
    "informático", "informática", "ingeniería", "ingeniero", "civil"
}

for word in stopwords_academicas:
    nlp.vocab[word].is_stop = True

# FUNCIONES DE PROCESAMIENTO

def limpiar_texto_estructural(texto):
    """
    IA ESTRUCTURAL: Usa Regex para estandarizar el formato.
    """
    if not isinstance(texto, str): return ""
    
    texto = texto.lower()
    
    # 1. Normalización de género y artículos
    texto = re.sub(r'\b(el|un)/la\b', r'\1', texto)
    texto = re.sub(r'\b(un)/a\b', r'\1', texto)
    texto = re.sub(r'/(as?|os?|a|o)\b', '', texto)
    
    # 2. Eliminación de símbolos y números
    texto = re.sub(r'[^\w\s]', ' ', texto) # Quita puntuación
    texto = re.sub(r'\d+', '', texto)      # Quita números
    
    # 3. Colapsar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def limpiar_texto_linguistico(texto):
    """
    IA SEMÁNTICA: Usa spaCy para entender la raíz de las palabras.
    """
    doc = nlp(texto)
    tokens_limpios = []
    
    for token in doc:
        # Filtros: No es stopword, no es pronombre y tiene más de 2 letras
        if not token.is_stop and token.pos_ != "PRON" and len(token.text) > 2:
            # LEMATIZACIÓN: "optimizando" -> "optimizar"
            tokens_limpios.append(token.lemma_)
            
    return " ".join(tokens_limpios)

# EJECUCIÓN DEL PIPELINE

if __name__ == "__main__":
    print(f"Cargando datos desde: {RUTA_ENTRADA}")
    
    if not os.path.exists(RUTA_ENTRADA):
        print("Error: No se encontró el archivo raw. Ejecuta primero la Fase 1.")
        exit()

    df = pd.read_csv(
    RUTA_ENTRADA,
    sep=",",
    encoding="utf-8-sig"
)
    print("Columnas detectadas:")
    print(df.columns.tolist())
    print("Filas cargadas:", len(df))

    print("1. Iniciando Limpieza Estructural (Regex)...")
    df['perfil_estructural'] = df['perfil'].apply(limpiar_texto_estructural)

    print("2. Iniciando Limpieza Lingüística (NLP/spaCy)...")
    df['perfil_final'] = df['perfil_estructural'].apply(limpiar_texto_linguistico)

    # GUARDADO ORGANIZADO
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    df.to_csv(
    RUTA_SALIDA,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)
    
    print("\n" + "="*40)
    print("FASE 2: PROCESAMIENTO FINALIZADO")
    print(f"Archivo guardado en: {RUTA_SALIDA}")
    print("="*40)