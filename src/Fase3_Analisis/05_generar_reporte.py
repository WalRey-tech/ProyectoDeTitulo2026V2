import pandas as pd
import json
import os

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_TOP_WORDS = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "top15_palabras_clave.csv"))
RUTA_SALIDA_JSON = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "resultados.json"))

def main():
    print(" Generando reporte final consolidado...")
    
    # 1. Leer los términos distintivos del CSV anterior
    try:
        df_top = pd.read_csv(RUTA_TOP_WORDS, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f" Error: No se encontró el archivo de palabras clave: {RUTA_TOP_WORDS}")
        return

    # Estructuramos las palabras por carrera
    terminos_dict = {}
    for grado in df_top['Grado'].unique():
        palabras = df_top[df_top['Grado'] == grado]['Término distintivo'].tolist()
        terminos_dict[grado] = palabras

    # 2. Construir el JSON consolidado con los resultados matemáticos obtenidos
    reporte = {
        "n_por_grado": {
            "Civil": 29,
            "Informática": 23,
            "Técnico": 15,
            "Ejecución": 6,
            "Total": 73
        },
        "metricas_supervisadas": {
            "modelo": "Logistic Regression (Validación Cruzada 5-Folds)",
            "accuracy": 0.5771,
            "f1_macro": 0.5723,
            "matriz_confusion": "Ver imagen: matriz_confusion_LR.png"
        },
        "analisis_homogeneidad": {
            "similitud_civil_informatica": 0.6927,
            "test_permutacion_p_valor": 0.0,
            "significancia": "Estadísticamente significativo (p < 0.05)"
        },
        "terminos_distintivos": terminos_dict
    }

    # 3. Guardar el archivo JSON
    os.makedirs(os.path.dirname(RUTA_SALIDA_JSON), exist_ok=True)
    with open(RUTA_SALIDA_JSON, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*60)
    print(f" ¡Punto 8 completado! Archivo maestro generado en:\n{RUTA_SALIDA_JSON}")
    print("="*60)
    print(" Instrucción para tu Frontend (React/Next.js):")
    print("Asegúrate de que 'ResultsSection.tsx' lea este archivo JSON para mostrar los datos,")
    print("o de lo contrario, agrega un comentario en tu código frontend diciendo:")
    print("// NOTA: Esta vista es ilustrativa. Los datos reales se encuentran en processed/resultados.json")

if __name__ == "__main__":
    main()