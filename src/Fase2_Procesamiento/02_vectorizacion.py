import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuración de Rutas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_ENTRADA = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "perfiles_egreso_limpio_v1.csv"))
# Aquí guardaremos la imagen gráfica de la matriz de confusión para tu tesis
RUTA_MATRIZ_IMG = os.path.normpath(os.path.join(DIRECTORIO_ACTUAL, "..", "data", "processed", "matriz_confusion.png"))

def main():
    print("⏳ Cargando dataset limpio...")
    try:
        df = pd.read_csv(RUTA_ENTRADA, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"❌ Error: No se encontró {RUTA_ENTRADA}.")
        return

    # Eliminamos cualquier fila vacía accidental
    df = df.dropna(subset=['perfil_limpio'])

    X = df['perfil_limpio']
    y = df['grado']

    print("🧮 1. Vectorizando textos (TF-IDF)...")
    # TF-IDF: Extrae hasta 1500 características combinando palabras sueltas (1) y pares de palabras (2)
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)

    print("🔀 2. Dividiendo en datos de Entrenamiento y Prueba (80/20)...")
    # stratify=y asegura que la proporción de grados (especialmente Ejecución que son 6) se mantenga equilibrada
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🤖 3. Entrenando modelo de Regresión Logística...")
    # class_weight='balanced' es crítico: penaliza más al modelo si se equivoca en las clases minoritarias (Ejecución)
    modelo = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    modelo.fit(X_train, y_train)

    print("🔮 4. Realizando predicciones...")
    y_pred = modelo.predict(X_test)

    # =========================================================================
    # MÉTRICAS Y RESULTADOS PARA LA TESIS
    # =========================================================================
    print("\n" + "="*50)
    print("📈 REPORTE DE CLASIFICACIÓN (Métricas de Separabilidad):")
    print("="*50)
    print(classification_report(y_test, y_pred))
    print("="*50)

    print("🎨 5. Generando Matriz de Confusión visual...")
    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    
    # Configuramos el gráfico
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=modelo.classes_,
                yticklabels=modelo.classes_)
    plt.title('Matriz de Confusión - Separabilidad de Grados')
    plt.ylabel('Grado Real')
    plt.xlabel('Grado Predicho por la IA')
    plt.tight_layout()
    
    # Guardamos la imagen en alta calidad
    plt.savefig(RUTA_MATRIZ_IMG, dpi=300)
    
    print(f"✅ ¡Modelo entrenado y evaluado exitosamente!")
    print(f"📁 Gráfico de la Matriz de Confusión guardado en: {RUTA_MATRIZ_IMG}")

if __name__ == "__main__":
    main()