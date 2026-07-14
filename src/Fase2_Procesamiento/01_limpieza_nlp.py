import pandas as pd
import spacy
import os
import csv
import warnings

# Inhabilitación de advertencias
warnings.filterwarnings('ignore')

try:
    from ftfy import fix_text
except ImportError:
    fix_text = None

# =============================================================================
# 1. Configuración de Rutas
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_etiquetado_v1.csv"))
RUTA_SALIDA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))

# =============================================================================
# 2. Cargar el modelo NLP (Optimizado a 'lg' por recomendación docente)
# =============================================================================
print(" Cargando modelo de lenguaje 'es_core_news_lg' (mayor precisión técnica)...")
try:
    # Si 'lg' es muy pesado para tu máquina, puedes volver a 'sm', 
    # pero 'lg' es el estándar recomendado para el reconocimiento de entidades.
    nlp = spacy.load("es_core_news_lg")
except OSError:
    print(" Modelo 'lg' no encontrado. Intentando descargar...")
    os.system("python -m spacy download es_core_news_lg")
    nlp = spacy.load("es_core_news_lg")

# =============================================================================
# 3. Stopwords Customizadas
# =============================================================================
STOPWORDS_CUSTOM = {
    "ufro", "duoc", "duocuc", "inacap", "aiep", "santo", "tomás", "tomas", 
    "cft", "ip", "universidad", "instituto", "profesional", "centro", "formación", "tecnológica",
    "ceduc", "ucn", "ipchile", "uss", "unab", "udp", "pucv", "usm", "usach", "uchile",
    "magíster", "magister", "director", "directora", "licenciado", "licenciatura", 
    "jefatura", "alumno", "estudiante", "egresado", "titulado", "chile", "programa"
}

def limpiar_texto_avanzado(texto):
    if not isinstance(texto, str):
        return ""
    
    # fix_text antes de procesar
    if fix_text:
        texto = fix_text(texto)
        
    doc = nlp(texto)
    tokens_limpios = []
    
    for token in doc:
        # 1. Eliminar ORG (Instituciones detectadas por NER)
        if token.ent_type_ == "ORG":
            continue
        # 2. Eliminar puntuación, números y stopwords nativas
        if token.is_stop or token.is_punct or token.is_digit or token.is_space:
            continue
        # 3. Eliminar stopwords custom
        token_lower = token.text.lower()
        if token_lower in STOPWORDS_CUSTOM:
            continue
            
        # 4. Lematización
        tokens_limpios.append(token.lemma_.lower())
        
    return " ".join(tokens_limpios)

# =============================================================================
# 4. Flujo Principal
# =============================================================================
def main():
    print(" Leyendo el dataset etiquetado...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, sep=";", encoding='utf-8-sig')
    except Exception as e:
        print(f" Error al leer archivo: {e}")
        return

    print(" Iniciando limpieza avanzada con spaCy (NER + Lematización)...")
    
    # Aplicar limpieza
    df['perfil_limpio'] = df['perfil_egreso'].apply(limpiar_texto_avanzado)

    # Filtrar vacíos
    df = df[df['perfil_limpio'].str.strip() != ""]

    print(" Guardando dataset limpio...")
    df.to_csv(RUTA_SALIDA, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f" ¡Fase 2 completada! Dataset guardado en: {RUTA_SALIDA}")

if __name__ == "__main__":
    main()