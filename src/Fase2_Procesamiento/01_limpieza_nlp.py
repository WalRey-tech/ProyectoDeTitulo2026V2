import pandas as pd
import spacy
import os
import csv

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
# 2. Cargar el modelo NLP
# =============================================================================
print(" Cargando modelo de lenguaje de spaCy...")
try:
    nlp = spacy.load("es_core_news_lg")
    print(" Modelo cargado: es_core_news_lg")
except OSError:
    try:
        nlp = spacy.load("es_core_news_sm")
        print(" Advertencia: se utilizará es_core_news_sm como respaldo.")
    except OSError:
        print(
            " Error: No se encontró un modelo de spaCy en español. "
            "Ejecuta: python -m spacy download es_core_news_lg"
        )
        raise SystemExit(1)

# =============================================================================
# 3. Stopwords Customizadas (Ruido Institucional)
# =============================================================================
STOPWORDS_CUSTOM = {
    # Instituciones y siglas
    "ufro", "duoc", "duocuc", "inacap", "aiep", "santo", "tomás", "tomas",
    "cft", "ip", "ceduc", "ucn", "ipchile", "uss", "unab", "udp", "pucv",
    "usm", "usach", "uchile", "udla", "ucsc", "ubo", "uoh", "ugm",

    # Ruido institucional y académico
    "universidad", "instituto", "centro", "carrera", "programa", "malla",
    "admisión", "admision", "académico", "académica", "institucional",
    "universitario", "universitaria", "educación", "educacion", "promoción",
    "promocion", "director", "directora", "jefatura", "alumno", "estudiante",
    "egresado", "egresada", "titulado", "titulada", "título", "titulo",
    "licenciado", "licenciada", "licenciatura", "magíster", "magister",
    "doctor", "doctora", "doctorado", "phd",

    # Denominaciones de grado o carrera: se eliminan para evitar fuga de etiqueta
    "ingeniería", "ingenieria", "ingeniero", "ingeniera",
    "civil", "informática", "informatica", "informático", "informatico",
    "computación", "computacion", "ejecución", "ejecucion",

    # Expresiones genéricas sin valor discriminativo
    "profesional", "formación", "formacion", "formar",
    "ámbito", "ambito", "función", "funcion",
    "participar", "in","chile","chileno","chilena","región","region","regional",
    "departamento","cristiano","cristiana","visión","vision","identidad","vida","principio",
    "persona","humano","humana","respeto",

    # Residuos pronominales de la lematización
    "él", "ella", "ello"
}
def corregir_mojibake(texto):
    """
    Corrige problemas de codificación típicos como:
    InformÃ¡tica -> Informática
    diseÃ±ar -> diseñar
    tecnolÃ³gicas -> tecnológicas
    Ãtica -> Ética
    """
    if not isinstance(texto, str):
        return texto

    texto = texto.strip()

    # Opción recomendada: ftfy
    if fix_text is not None:
        return fix_text(texto)

    # Respaldo manual si ftfy no está instalado
    try:
        if any(marca in texto for marca in ["Ã", "Â", "â"]):
            return texto.encode("latin1").decode("utf-8")
    except Exception:
        pass

    return texto


def corregir_columnas_texto(df):
    """
    Aplica corrección de encoding a todas las columnas de texto del dataset.
    """
    columnas_texto = df.select_dtypes(include=["object"]).columns

    for col in columnas_texto:
        df[col] = df[col].apply(corregir_mojibake)

    return df


def detectar_problemas_encoding(df):
    """
    Muestra filas que todavía contienen caracteres sospechosos.
    """
    patron = r"Ã|Â|â|�"

    problemas = df[
        df.apply(
            lambda fila: fila.astype(str).str.contains(patron, regex=True).any(),
            axis=1
        )
    ]

    return problemas

def limpiar_texto_avanzado(texto):
    """
    Normaliza, lematiza y elimina ruido institucional, nombres de carrera
    y términos que podrían filtrar directamente la etiqueta del grado.
    """
    if not isinstance(texto, str):
        return ""

    texto = corregir_mojibake(texto)
    doc = nlp(texto)
    tokens_limpios = []

    for token in doc:
        # Elimina nombres de organizaciones reconocidos por spaCy.
        if token.ent_type_ == "ORG":
            continue

        # Elimina espacios, puntuación, números, URLs y correos.
        if (
            token.is_space
            or token.is_punct
            or token.is_digit
            or token.like_num
            or token.like_url
            or token.like_email
        ):
            continue

        token_lower = token.text.lower().strip()
        lema = token.lemma_.lower().strip()

        # Evita residuos muy cortos como "in".
        if len(lema) < 3:
            continue

        # Se revisa tanto la forma original como el lema. Esto evita que
        # "profesionales" sobreviva y luego se convierta en "profesional".
        if token.is_stop:
            continue

        if token_lower in STOPWORDS_CUSTOM or lema in STOPWORDS_CUSTOM:
            continue

        # Solo conserva tokens alfabéticos.
        if not lema.isalpha():
            continue

        tokens_limpios.append(lema)

    return " ".join(tokens_limpios)

# =============================================================================
# 4. Flujo Principal
# =============================================================================
def main():
    print(" Leyendo el dataset etiquetado...")
    try:
        df = pd.read_csv(RUTA_ENTRADA,sep=";", encoding='utf-8-sig')
    except Exception as e:
        print(" Archivo no encontrado o corrompido.")
        return

    print("\n Distribución de datos a limpiar (Grado vs Tipo de Institución):")
    print(pd.crosstab(df['grado'], df['tipo_institucion']))
    print("-" * 50)

    print(" Iniciando limpieza avanzada con spaCy (NER y Stopwords)...")
    print(" Esto tomará unos momentos. Por favor espera...")
    
    # Aplicamos la limpieza
    df['perfil_limpio'] = df['perfil_egreso'].apply(limpiar_texto_avanzado)

    # Filtro de seguridad: quitamos cualquier fila que haya quedado vacía tras limpiar
    df = df[df['perfil_limpio'].str.strip() != ""]

    print(" Guardando el dataset limpio para la vectorización...")
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    df.to_csv(RUTA_SALIDA, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f" ¡Fase 2 completada! Limpieza exitosa.")
    print(f" Archivo final guardado en: {RUTA_SALIDA}")

if __name__ == "__main__":
    main()